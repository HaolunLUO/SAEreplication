#!/usr/bin/env python3
"""
sparse_encoding_lag_screen.py
=============================
**Primary** lag-resolved analysis for temporal precision in Augmented Sparse
Encoding Models (sEEG millisecond dynamics).

Builds event-locked responses from 0–800 ms in 50 ms steps using short 50 ms
bins. Feature selection and scaling stay inside each outer contiguous training
fold. A lag-length guard band around held-out blocks prevents neighboring-event
leakage.

Per-electrode peak lag, peak R, FWHM, and full−surprisal lag differences are
written alongside the lag curves. For overlap-corrected impulse responses see
``sparse_encoding_deconvolution.py``. The fixed-lag
``sparse_encoding_regression.py`` path is a simplified baseline only.

Outputs
-------
tables/sparse_encoding_lag_screen[_suffix].csv
    per-electrode wide lag × mode R (Fisher)
tables/sparse_encoding_lag_screen_long[_suffix].csv
    long form for plotting
tables/sparse_encoding_lag_screen_peaks[_suffix].csv
    per-electrode peak_lag / peak_R / FWHM / lag_diff
tables/sparse_encoding_lag_screen_subject[_suffix].csv
    subject × ROI × lag means
reports/sparse_encoding_lag_screen[_suffix].txt
figures/lag_timecourse_*[_suffix].png

Example
-------
    python -m sparse_encoding.sparse_encoding_lag_screen \\
        --feature_tag sae_qwen35_4b_mat_l15 --lang_only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_regression import (
    FEATURE_TAG, FS_TARGET, MODES, RANDOM_STATE, SECTIONS,
    _aggregate_rs, _fit_predict, _fmt_duration, _progress_line, _to_dense,
    compute_joint_support, compute_support_features, get_subjects,
    load_residual_features, load_sae_features, maybe_shuffle_features,
)
from sparse_encoding.sparse_encoding_summary import LEFT_MFG_REGION, language_mask, merge_taxonomy
from sparse_encoding.temporal_metrics import (
    electrode_lag_peaks_from_long,
    subject_mean_sign_flip_p,
    temporal_qc_report,
)


# ======================================================================
# CONFIG
# ======================================================================
LAG_START_MS = 0.0
LAG_END_MS = 800.0
LAG_STEP_MS = 50.0
RESP_WIN_MS = 50.0          # short bin matching the lag step
N_FOLDS = 5


def lag_grid_ms(
    start: float = LAG_START_MS,
    end: float = LAG_END_MS,
    step: float = LAG_STEP_MS,
) -> np.ndarray:
    return np.arange(start, end + 0.5 * step, step, dtype=float)


def guard_samples(max_lag_ms: float, resp_win_ms: float, fs: float) -> int:
    """Samples to drop on each side of a held-out block (max lag + half window)."""
    return int(np.ceil(((max_lag_ms + 0.5 * resp_win_ms) / 1000.0) * fs))


def make_buffered_contiguous_splits(
    n_words: int,
    n_folds: int,
    section_slices: List[Tuple[int, int]],
    ok: np.ndarray,
    guard_words: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous within-section folds with a word-level guard band.

    Words within ``guard_words`` of a test block boundary are dropped from
    *both* train and test so lagged neural windows cannot leak across the
    fold edge.
    """
    ok = np.asarray(ok, dtype=bool)
    fold_ranges = ec.make_folds_within_sections(
        section_slices, n_folds, min_per_section=1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fa, fb in fold_ranges:
        # Expand exclusion zone around the test block.
        excl_lo = max(0, fa - guard_words)
        excl_hi = min(n_words, fb + guard_words)
        te_mask = ok.copy()
        te_mask[:fa] = False
        te_mask[fb:] = False
        # Also drop near-boundary words inside the test block.
        if guard_words > 0:
            te_mask[fa: min(fb, fa + guard_words)] = False
            te_mask[max(fa, fb - guard_words): fb] = False
        tr_mask = ok.copy()
        tr_mask[excl_lo:excl_hi] = False
        te = np.flatnonzero(te_mask)
        tr = np.flatnonzero(tr_mask)
        if te.size == 0 or tr.size < 20:
            continue
        splits.append((tr, te))
    return splits


def compute_Y_lag_grid(
    data: ec.SubjectWordLockedData,
    lags_ms: np.ndarray,
    resp_win_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Y[lag, word, ch] and valid[lag, word] for a lag grid."""
    half_win = max(1, int(round((resp_win_ms / 1000.0) * data.fs / 2.0)))
    Ys, Vs = [], []
    for lag in lags_ms:
        lag_samp = int(round((lag / 1000.0) * data.fs))
        Y, v = ec.compute_Y_for_lag(data, lag_samp, half_win)
        Ys.append(Y)
        Vs.append(v)
    return np.stack(Ys, axis=0), np.stack(Vs, axis=0)


def regress_electrode_lag(
    X_sae: sparse.csr_matrix,
    surprisal: np.ndarray,
    y_lags: np.ndarray,
    valid_lags: np.ndarray,
    modes: Tuple[str, ...],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    joint_lasso_surprisal: bool = True,
) -> Dict[str, object]:
    """Per-lag encoding with training-only peak selection for nested eval.

    Selection is fit once per fold on the *training* lag that maximizes
    content-model r (or surprisal if content empty), then applied to held-out
    data at every lag without using test peaks.
    """
    n_lags = y_lags.shape[0]
    fold_rs = {m: [[] for _ in range(n_lags)] for m in modes}
    train_peak_lags: List[int] = []

    for tr, te in folds:
        if te.size < 5 or tr.size < 20:
            continue
        # Pick training peak lag using surprisal-only (cheap, selection-safe).
        peak_li = 0
        peak_r = -np.inf
        for li in range(n_lags):
            y = y_lags[li]
            ok_tr = np.isfinite(y[tr]) & valid_lags[li, tr]
            if ok_tr.sum() < 20:
                continue
            # Quick correlation with surprisal as a training-only screen.
            s = surprisal[tr][ok_tr]
            yy = y[tr][ok_tr]
            if np.std(s) < 1e-12 or np.std(yy) < 1e-12:
                continue
            r, _ = pearsonr(s, yy)
            if np.isfinite(r) and r > peak_r:
                peak_r = float(r)
                peak_li = li
        train_peak_lags.append(peak_li)

        # Feature selection on the training peak lag only.
        y_sel = y_lags[peak_li]
        ok_sel = np.isfinite(y_sel[tr]) & valid_lags[peak_li, tr]
        tr_sel = tr[ok_sel]
        if tr_sel.size < 20:
            continue
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(surprisal[tr_sel].reshape(-1, 1))
        surp_tr_sel = sc.transform(surprisal[tr_sel].reshape(-1, 1))
        acts_tr = X_sae[tr_sel]
        y_tr_sel = y_sel[tr_sel]

        content_support = None
        full_support = None
        if "content" in modes or ("full" in modes and not joint_lasso_surprisal):
            content_support = compute_support_features(acts_tr, y_tr_sel)
        if "full" in modes:
            if joint_lasso_surprisal:
                full_support = compute_joint_support(acts_tr, surp_tr_sel, y_tr_sel)
                if not full_support.any():
                    full_support = None
            else:
                full_support = content_support

        # Evaluate every lag with frozen selection / scaler from train peak.
        for li in range(n_lags):
            y = y_lags[li]
            ok_tr = np.isfinite(y[tr]) & valid_lags[li, tr]
            ok_te = np.isfinite(y[te]) & valid_lags[li, te]
            tr_i, te_i = tr[ok_tr], te[ok_te]
            if tr_i.size < 20 or te_i.size < 5:
                for m in modes:
                    fold_rs[m][li].append(0.0)
                continue
            surp_tr = sc.transform(surprisal[tr_i].reshape(-1, 1))
            surp_te = sc.transform(surprisal[te_i].reshape(-1, 1))
            y_tr, y_te = y[tr_i], y[te_i]
            for m in modes:
                if m == "surprisal_only":
                    Xtr, Xte = surp_tr, surp_te
                elif m == "content":
                    if content_support is None:
                        fold_rs[m][li].append(0.0)
                        continue
                    Xtr = _to_dense(X_sae[tr_i][:, content_support])
                    Xte = _to_dense(X_sae[te_i][:, content_support])
                elif m == "full":
                    if full_support is None:
                        Xtr, Xte = surp_tr, surp_te
                    else:
                        Xtr = np.hstack([
                            _to_dense(X_sae[tr_i][:, full_support]), surp_tr])
                        Xte = np.hstack([
                            _to_dense(X_sae[te_i][:, full_support]), surp_te])
                else:
                    raise ValueError(m)
                pred, _ = _fit_predict(Xtr, y_tr, Xte)
                if (np.std(pred) > 1e-12 and np.std(y_te) > 1e-12
                        and np.all(np.isfinite(pred))):
                    r, _ = pearsonr(y_te, pred)
                    fold_rs[m][li].append(0.0 if not np.isfinite(r) else float(r))
                else:
                    fold_rs[m][li].append(0.0)

    out: Dict[str, object] = {
        "train_peak_lag_idx_median": (
            float(np.median(train_peak_lags)) if train_peak_lags else np.nan),
    }
    for m in modes:
        for li in range(n_lags):
            _, r_f, nvalid = _aggregate_rs(fold_rs[m][li])
            out[f"{m}__R_fisher_lag{li}"] = r_f
            out[f"{m}__n_folds_lag{li}"] = nvalid
    return out


def process_subject(subject: str, eeg_files: Dict[int, str], args) -> pd.DataFrame:
    eeg_files_abs = {sid: str(ap.DATA_ROOT / p) for sid, p in eeg_files.items()}
    missing = [p for p in eeg_files_abs.values() if not Path(p).exists()]
    if missing:
        print(f"[{subject}] missing EEG, skip: {missing}")
        return pd.DataFrame()

    data = ec.load_subject_data(
        subject_name=subject, eeg_files=eeg_files_abs,
        features_dir=ap.FEATURES_DIR, fs_target=args.fs_target,
        sections=tuple(args.sections), feature_sets=["glove"],
    )
    feature_kind = getattr(args, "feature_kind", "sae")
    if feature_kind == "residual":
        X_sae, surprisal, feat_valid = load_residual_features(
            args.feature_tag, args.sections)
        print(
            f"  [{subject}] residual LASSO lag-screen: "
            f"{X_sae.shape[0]} words × {X_sae.shape[1]} dims",
            flush=True,
        )
    else:
        X_sae, surprisal, feat_valid = load_sae_features(
            args.feature_tag, args.sections)
    Wtot = data.section_word_slices[-1][1]
    if X_sae.shape[0] != Wtot:
        raise ValueError(f"[{subject}] SAE/EEG word mismatch")

    lags_ms = lag_grid_ms(args.lag_start_ms, args.lag_end_ms, args.lag_step_ms)
    Y_lags, V_lags = compute_Y_lag_grid(data, lags_ms, args.resp_win_ms)
    # Word-level guard: convert max lag from time to approximate words using
    # median IOI if available, else a conservative 1 word per 50 ms.
    guard_w = max(1, int(np.ceil(args.lag_end_ms / args.lag_step_ms)))

    base_ok = feat_valid & np.isfinite(surprisal)
    if args.shuffle_control == "global":
        rng = np.random.default_rng(RANDOM_STATE + sum(ord(c) for c in subject))
        X_sae, surprisal = maybe_shuffle_features(X_sae, surprisal, "global", rng)

    channels = list(data.channel_labels)
    electrode_set = getattr(args, "electrode_set", None)
    if electrode_set is None and getattr(args, "lang_only", False):
        electrode_set = "is_lang"
    if electrode_set and electrode_set != "all":
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        col = {"is_lang": "is_lang", "sig_glove": "sig_glove"}[electrode_set]
        if col not in tax.columns:
            raise ValueError(f"taxonomy missing column {col}")
        keep = set(tax.loc[
            (tax["subject"] == subject) & tax[col].fillna(False), "channel"
        ].astype(str))
        channels = [c for c in channels if c in keep]
        print(f"  [{subject}] filtered to {len(channels)} {electrode_set} channels",
              flush=True)
    label_to_idx = {str(c): i for i, c in enumerate(data.channel_labels)}
    if args.max_channels:
        channels = channels[: args.max_channels]
    modes = tuple(args.modes)
    n_ch = len(channels)
    n_jobs = int(getattr(args, "n_jobs", 1) or 1)
    t0 = time.time()
    print(
        f"  [{subject}] lag-screen {n_ch} ch × {len(lags_ms)} lags "
        f"(n_jobs={n_jobs})",
        flush=True,
    )
    out_tag = (
        f"{args.feature_tag}_resid" if feature_kind == "residual"
        else args.feature_tag
    )
    section_slices = list(data.section_word_slices)

    def run_channel(ch: str):
        eeg_ci = label_to_idx.get(str(ch))
        if eeg_ci is None:
            return None
        y_lags = Y_lags[:, :, eeg_ci]
        valid_lags = V_lags & base_ok[None, :] & np.isfinite(y_lags)
        ok = base_ok & np.any(valid_lags, axis=0)
        folds = make_buffered_contiguous_splits(
            Wtot, args.n_folds, section_slices, ok, guard_w)
        if not folds:
            return None
        res = regress_electrode_lag(
            X_sae, surprisal, y_lags, valid_lags, modes, folds,
            joint_lasso_surprisal=args.joint_lasso_surprisal,
        )
        res.update({
            "subject": subject, "channel": ch, "channel_index": eeg_ci,
            "n_words": int(ok.sum()), "cv": "contiguous_buffered",
            "shuffle_control": args.shuffle_control,
            "feature_tag": out_tag,
            "feature_kind": feature_kind,
            "electrode_set": electrode_set or "all",
        })
        for li, lag in enumerate(lags_ms):
            res[f"lag_ms_{li}"] = float(lag)
        return res

    report_every = max(1, min(10, n_ch // 10 or 1))
    if n_jobs == 1:
        rows = []
        for ci, ch in enumerate(channels):
            r = run_channel(ch)
            if r is not None:
                rows.append(r)
            done = ci + 1
            if done % report_every == 0 or done == n_ch:
                print(_progress_line(subject, done, n_ch, t0, kept=len(rows)),
                      flush=True)
    else:
        from joblib import Parallel, delayed
        results_iter = Parallel(
            n_jobs=n_jobs, backend="loky", return_as="generator",
        )(delayed(run_channel)(ch) for ch in channels)
        rows = []
        for done, r in enumerate(results_iter, 1):
            if r is not None:
                rows.append(r)
            if done % report_every == 0 or done == n_ch:
                print(_progress_line(subject, done, n_ch, t0, kept=len(rows)),
                      flush=True)
    print(
        f"  [{subject}] done: {len(rows)}/{n_ch} in "
        f"{_fmt_duration(time.time() - t0)}",
        flush=True,
    )
    return pd.DataFrame(rows)


def long_form(df: pd.DataFrame, lags_ms: np.ndarray, modes) -> pd.DataFrame:
    """Melt wide lag columns into long form for plotting / subject means."""
    records = []
    for _, r in df.iterrows():
        for li, lag in enumerate(lags_ms):
            for m in modes:
                key = f"{m}__R_fisher_lag{li}"
                if key not in r or not np.isfinite(r[key]):
                    continue
                records.append({
                    "subject": r["subject"],
                    "channel": r["channel"],
                    "channel_index": r["channel_index"],
                    "lag_ms": float(lag),
                    "mode": m,
                    "R_fisher": float(r[key]),
                    "feature_tag": r.get("feature_tag", ""),
                })
    return pd.DataFrame(records)


def write_lag_report(
    long_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    path: Path,
    peaks_df: Optional[pd.DataFrame] = None,
):
    lines = [
        "=" * 70,
        "LAG-RESOLVED ANALYSIS — TEMPORAL PRECISION (PRIMARY)",
        "=" * 70,
        "",
        "Grid: 0–800 ms in 50 ms steps; 50 ms response bins; contiguous "
        "buffered CV. Peak metrics are held-out Fisher-z R after training-only "
        "feature selection. Overlap-corrected kernels: "
        "sparse_encoding_deconvolution.py.",
        "",
    ]
    if long_df.empty:
        lines.append("(no rows)")
        path.write_text("\n".join(lines))
        return
    keep_tax = [c for c in ("subject", "channel", "region", "is_lang",
                            "hemisphere", "category") if c in tax_df.columns]
    merged = (long_df.merge(tax_df[keep_tax], on=["subject", "channel"], how="left")
              if not tax_df.empty and keep_tax else long_df.assign(is_lang=False, region=""))
    lang = language_mask(merged) if "is_lang" in merged.columns else pd.Series(False, index=merged.index)
    for label, mask in (
        ("ALL", pd.Series(True, index=merged.index)),
        ("LOCALIZER-POSITIVE", lang),
        ("LEFT MFG", lang & merged.get("region", pd.Series("", index=merged.index)).eq(LEFT_MFG_REGION)),
    ):
        sub = merged.loc[mask]
        lines.append("-" * 70)
        lines.append(f"{label} (n_electrode_rows={sub[['subject','channel']].drop_duplicates().shape[0]})")
        lines.append("-" * 70)
        if sub.empty:
            lines.append("  (empty)")
            lines.append("")
            continue
        for mode, g in sub.groupby("mode"):
            pivot = (g.groupby(["subject", "lag_ms"])["R_fisher"].mean()
                       .groupby("lag_ms").mean())
            peak_lag = float(pivot.idxmax()) if len(pivot) else np.nan
            peak_r = float(pivot.max()) if len(pivot) else np.nan
            lines.append(
                f"  {mode:15s} peak_lag={peak_lag:.0f} ms  "
                f"peak_R(subject-mean)={peak_r:+.4f}"
            )
        lines.append("")

    if peaks_df is not None and not peaks_df.empty:
        lines.append("-" * 70)
        lines.append("PER-ELECTRODE PEAK METRICS (subject-mean of electrode peaks)")
        lines.append("-" * 70)
        pmerge = peaks_df.merge(
            tax_df[keep_tax] if keep_tax else peaks_df[["subject", "channel"]],
            on=["subject", "channel"], how="left",
        ) if not tax_df.empty else peaks_df
        if "is_lang" in pmerge.columns:
            puse = pmerge.loc[language_mask(pmerge)]
            if puse.empty:
                puse = pmerge
        else:
            puse = pmerge
        for col, name in (
            ("full_peak_lag_ms", "full peak lag"),
            ("full_fwhm_ms", "full FWHM"),
            ("surprisal_peak_lag_ms", "surprisal peak lag"),
            ("lag_diff_full_vs_surprisal_ms", "full−surprisal lag diff"),
        ):
            if col not in puse.columns:
                continue
            subj = puse.groupby("subject")[col].mean()
            mu, p = subject_mean_sign_flip_p(subj.to_numpy())
            med = float(np.nanmedian(puse[col]))
            lines.append(
                f"  {name:28s} median={med:.1f}  subject-mean={mu:+.1f}  "
                f"sign-flip p={p:.3f}  nS={subj.notna().sum()}"
            )
        if "lag_diff_full_vs_surprisal_ms" in puse.columns:
            subj = puse.groupby("subject")["lag_diff_full_vs_surprisal_ms"].mean()
            mu, _ = subject_mean_sign_flip_p(subj.to_numpy())
            if np.isfinite(mu):
                interp = (
                    "SAE/full predicts later than surprisal" if mu > 0
                    else ("SAE/full predicts earlier than surprisal" if mu < 0
                          else "same timing")
                )
                lines.append(f"  SAE temporal signature: {interp}")
        lines.append("")
        lines.extend(temporal_qc_report(puse))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def fig_lag_timecourse(long_df: pd.DataFrame, tax_df: pd.DataFrame, outdir: Path, suffix: str):
    if long_df.empty:
        return
    merged = long_df.merge(
        tax_df[["subject", "channel", "region", "is_lang"]],
        on=["subject", "channel"], how="left",
    ) if not tax_df.empty else long_df
    lang = language_mask(merged) if "is_lang" in merged.columns else pd.Series(True, index=merged.index)
    use = merged.loc[lang] if lang.any() else merged
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"full": "#1f77b4", "content": "#2ca02c", "surprisal_only": "#d62728"}
    for mode, g in use.groupby("mode"):
        # subject means then mean±sem across subjects
        subj = g.groupby(["subject", "lag_ms"])["R_fisher"].mean().reset_index()
        mu = subj.groupby("lag_ms")["R_fisher"].mean()
        sem = subj.groupby("lag_ms")["R_fisher"].sem()
        ax.plot(mu.index, mu.values, label=mode, color=colors.get(mode, "k"))
        ax.fill_between(mu.index, mu - sem, mu + sem, alpha=0.2,
                         color=colors.get(mode, "k"))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("R (Fisher), subject-mean")
    ax.set_title("Lag-resolved screening (localizer-positive)")
    ax.legend()
    fig.tight_layout()
    p = outdir / f"lag_timecourse_lang{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument(
        "--feature_kind", choices=("sae", "residual"), default="sae",
        help="sae: sparse latents. residual: dense residual stream through "
             "the same F-test+LASSO→Ridge lag screen.",
    )
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Parallel workers across electrodes (-1 = all cores).")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--lag_start_ms", type=float, default=LAG_START_MS)
    p.add_argument("--lag_end_ms", type=float, default=LAG_END_MS)
    p.add_argument("--lag_step_ms", type=float, default=LAG_STEP_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--max_channels", type=int, default=0)
    p.add_argument("--joint_lasso_surprisal",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shuffle_control",
                   choices=("none", "global"), default="none")
    p.add_argument("--lang_only", action="store_true",
                   help="Alias for --electrode_set is_lang.")
    p.add_argument(
        "--electrode_set",
        choices=("is_lang", "sig_glove", "all"),
        default=None,
        help="Channel gate: is_lang, sig_glove (GloVe encoding-sig), or all.",
    )
    p.add_argument("--out_suffix", default="")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute subjects already in the wide CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    if args.electrode_set is None and args.lang_only:
        args.electrode_set = "is_lang"
    elif args.electrode_set is None:
        args.electrode_set = "all"
    lags_ms = lag_grid_ms(args.lag_start_ms, args.lag_end_ms, args.lag_step_ms)
    subjects_map = get_subjects()
    requested = args.subjects or list(subjects_map.keys())
    suffix = args.out_suffix
    if not suffix:
        kind = "resid_lasso" if args.feature_kind == "residual" else args.feature_tag
        if args.feature_kind == "residual":
            kind = f"{args.feature_tag}_resid_lasso"
        else:
            kind = args.feature_tag
        if args.electrode_set == "is_lang":
            suffix = f"_{kind}_lang"
        elif args.electrode_set == "sig_glove":
            suffix = f"_{kind}_sig_glove"
        else:
            suffix = f"_{kind}"
    if args.shuffle_control != "none" and "shuffle" not in suffix:
        suffix = f"{suffix}_shuffle_{args.shuffle_control}"

    wide_path = ap.SAE_TABLES / f"sparse_encoding_lag_screen{suffix}.csv"
    dfs = []
    done = set()
    if wide_path.exists() and not getattr(args, "overwrite", False):
        prev = pd.read_csv(wide_path)
        if not prev.empty and "subject" in prev.columns:
            done = set(prev["subject"].astype(str).unique())
            dfs.append(prev)
            print(f"[resume] {wide_path.name} has {sorted(done)}", flush=True)
    t0 = time.time()
    for i, sid in enumerate(requested, 1):
        if sid not in subjects_map:
            print(f"[skip] {sid}")
            continue
        if sid in done:
            print(f"\n=== [{i}] {sid} [resume skip] ===", flush=True)
            continue
        print(f"\n=== [{i}] {sid} ===", flush=True)
        df = process_subject(sid, subjects_map[sid]["eeg_files"], args)
        if not df.empty:
            dfs.append(df)
            done.add(sid)
            pd.concat(dfs, ignore_index=True).to_csv(wide_path, index=False)
            print(f"  checkpoint {sid} -> {wide_path}", flush=True)
    if not dfs:
        print("No lag-screen results.")
        return
    wide = pd.concat(dfs, ignore_index=True)
    wide_path = ap.SAE_TABLES / f"sparse_encoding_lag_screen{suffix}.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Saved wide -> {wide_path} ({_fmt_duration(time.time()-t0)})")

    long_df = long_form(wide, lags_ms, args.modes)
    long_path = ap.SAE_TABLES / f"sparse_encoding_lag_screen_long{suffix}.csv"
    long_df.to_csv(long_path, index=False)

    peaks_df = electrode_lag_peaks_from_long(long_df, modes=args.modes)
    peaks_path = ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks{suffix}.csv"
    if not peaks_df.empty:
        peaks_df.to_csv(peaks_path, index=False)
        print(f"Saved peaks -> {peaks_path}")

    tax = merge_taxonomy(wide[["subject", "channel"]].drop_duplicates())
    # subject × lag means for language electrodes
    if not long_df.empty and "is_lang" in tax.columns:
        m = long_df.merge(tax, on=["subject", "channel"], how="left")
        lang = m[m["is_lang"].fillna(False)]
        subj = (lang.groupby(["subject", "region", "mode", "lag_ms"], dropna=False)
                    ["R_fisher"].mean().reset_index())
        subj_path = ap.SAE_TABLES / f"sparse_encoding_lag_screen_subject{suffix}.csv"
        subj.to_csv(subj_path, index=False)
        print(f"Saved subject means -> {subj_path}")

    report = ap.SAE_REPORTS / f"sparse_encoding_lag_screen{suffix}.txt"
    write_lag_report(
        long_df,
        tax if "is_lang" in tax.columns else pd.DataFrame(),
        report,
        peaks_df=peaks_df,
    )
    print("Figures:")
    fig_lag_timecourse(long_df, tax if "is_lang" in tax.columns else pd.DataFrame(),
                       ap.SAE_FIGURES, suffix)

    # Region-specific lag contrasts (optional; skips cleanly if empty).
    if not peaks_df.empty:
        try:
            from sparse_encoding.sparse_encoding_lag_by_region import (
                assign_anat_bucket, fig_lag_by_region, pairwise_lag_tests,
                region_summary, write_report as write_region_report,
            )
            preg = peaks_df.merge(
                tax[["subject", "channel", "region", "is_lang"]]
                if "region" in tax.columns else peaks_df[["subject", "channel"]],
                on=["subject", "channel"], how="left",
            )
            if "is_lang" in preg.columns and preg["is_lang"].fillna(False).any():
                preg = preg.loc[language_mask(preg)].copy()
            if "region" not in preg.columns:
                preg["region"] = ""
            preg["anat_bucket"] = preg["region"].map(assign_anat_bucket)
            summary = region_summary(preg)
            pairwise = pairwise_lag_tests(
                preg,
                pairs=[
                    ("temporal", "frontal"),
                    ("temporal", "left_MFG"),
                    ("left_MFG", "frontal"),
                    ("deep", "temporal"),
                ],
            )
            summary.to_csv(
                ap.SAE_TABLES / f"sparse_encoding_lag_by_region{suffix}.csv",
                index=False,
            )
            pairwise.to_csv(
                ap.SAE_TABLES / f"sparse_encoding_lag_region_pairwise{suffix}.csv",
                index=False,
            )
            write_region_report(
                summary, pairwise, preg,
                ap.SAE_REPORTS / f"sparse_encoding_lag_by_region{suffix}.txt",
            )
            fig_lag_by_region(summary, ap.SAE_FIGURES, suffix)
        except Exception as exc:  # noqa: BLE001 — secondary stage must not kill lag screen
            print(f"[warn] lag_by_region skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
