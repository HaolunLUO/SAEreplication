#!/usr/bin/env python3
"""
sparse_encoding_dense_baseline.py
=================================
Dense Chinese-LM / GloVe baselines with the same word-locked CV settings as
``sparse_encoding_regression.py`` (contiguous folds, Fisher-z, paired metrics).

Default feature sets
--------------------
- glove
- gpt2cn_l24   (Chinese GPT-2 layer 24)
- sae_*_resid  (same-model residual dump from ``sae_extract_features.py --resid_only``)

These are scored with RidgeCV only (no SAE Lasso path). Surprisal from a chosen
SAE tag can optionally be concatenated for a dense+surprisal model.

Example
-------
    python sparse_encoding_dense_baseline.py --features glove gpt2cn_l24 \\
        --surprisal_tag sae_gemma2_2b_mat_l12 --lang_only
    python sparse_encoding_dense_baseline.py --features sae_qwen3_8b_l18_resid \\
        --surprisal_tag sae_qwen3_8b_l18 --with_surprisal --lang_only \\
        --out_suffix _qwen_resid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_regression import (
    FS_TARGET, RESP_WIN_MS, RESPONSE_LAG_MS, RANDOM_STATE, SECTIONS, N_FOLDS,
    _aggregate_rs, _fmt_duration, _fold_r, _progress_line,
    get_subjects, load_sae_features, make_cv_splits, select_alpha,
)
from sparse_encoding.sparse_encoding_summary import language_mask, merge_taxonomy, LEFT_MFG_REGION

# Above this width, LOO RidgeCV GCV/SVD is too slow for word-locked n.
# Use a short alpha grid with iterative LSQR instead (still full residual dim).
HIGH_D_THRESHOLD = 512
HIGH_D_ALPHAS = (100.0, 1.0e4)


def load_dense_features(feature_name: str, sections) -> np.ndarray:
    # float32: high-d residuals are ~2–4k wide; float64 doubles peak RSS and
    # has repeatedly triggered host OOM (killing the job mid-subject).
    mats = []
    for sid in sections:
        path = ap.FEATURES_DIR / f"section_{sid:03d}" / f"X_word_{feature_name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        mats.append(np.load(path).astype(np.float32, copy=False))
    return np.vstack(mats)


def select_alpha_lsqr(X, y) -> float:
    """Holdout alpha search with Ridge(solver='lsqr') for high-d residuals."""
    n = int(X.shape[0])
    if n < 40:
        return 100.0
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(n)
    cut = max(20, int(0.8 * n))
    tr, te = idx[:cut], idx[cut:]
    if te.size < 5:
        return 100.0
    y_tr, y_te = y[tr], y[te]
    best_a, best = float(HIGH_D_ALPHAS[1]), -np.inf
    for a in HIGH_D_ALPHAS:
        model = Ridge(
            alpha=float(a), fit_intercept=True,
            solver="lsqr", max_iter=4000, tol=1e-3,
        )
        model.fit(X[tr], y_tr)
        pred = model.predict(X[te]).ravel()
        if np.std(pred) < 1e-12 or np.std(y_te) < 1e-12:
            continue
        # Fisher-friendly: use pearson via fold helper
        r = _fold_r(pred, y_te, degenerate_as_zero=True)
        if r > best:
            best = r
            best_a = float(a)
    return best_a


def _fit_predict_dense(Xtr, y_tr, Xte, high_d: bool):
    if high_d:
        alpha = select_alpha_lsqr(Xtr, y_tr)
        model = Ridge(
            alpha=alpha, fit_intercept=True,
            solver="lsqr", max_iter=4000, tol=1e-3,
        )
    else:
        alpha = select_alpha(Xtr, y_tr)
        model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, y_tr)
    return model.predict(Xte).ravel(), alpha


def regress_dense_electrode(
    X: np.ndarray,
    y: np.ndarray,
    folds,
    surprisal: Optional[np.ndarray] = None,
    degenerate_as_zero: bool = True,
) -> Dict[str, float]:
    """RidgeCV on dense features; optional surprisal concatenation."""
    if not folds:
        return {}
    modes = ("dense", "dense_surprisal") if surprisal is not None else ("dense",)
    fold_rs = {m: [] for m in modes}
    alphas = {m: [] for m in modes}
    n_folds_total = 0
    high_d = X.shape[1] > HIGH_D_THRESHOLD
    for tr, te in folds:
        if te.size < 5 or tr.size < 20:
            continue
        n_folds_total += 1
        y_tr, y_te = y[tr], y[te]
        scx = StandardScaler().fit(X[tr])
        Xtr = scx.transform(X[tr])
        Xte = scx.transform(X[te])
        pred, alpha = _fit_predict_dense(Xtr, y_tr, Xte, high_d=high_d)
        fold_rs["dense"].append(_fold_r(pred, y_te, degenerate_as_zero))
        alphas["dense"].append(alpha)
        if surprisal is not None:
            scs = StandardScaler().fit(surprisal[tr].reshape(-1, 1))
            str_ = scs.transform(surprisal[tr].reshape(-1, 1))
            ste = scs.transform(surprisal[te].reshape(-1, 1))
            Xtr2 = np.hstack([Xtr, str_])
            Xte2 = np.hstack([Xte, ste])
            pred2, alpha2 = _fit_predict_dense(Xtr2, y_tr, Xte2, high_d=high_d)
            fold_rs["dense_surprisal"].append(
                _fold_r(pred2, y_te, degenerate_as_zero))
            alphas["dense_surprisal"].append(alpha2)
    out: Dict[str, float] = {
        "n_folds_total": float(n_folds_total),
        "high_d_lsqr": float(1.0 if high_d else 0.0),
    }
    for m in modes:
        r_mean, r_fisher, nvalid = _aggregate_rs(fold_rs[m])
        out[f"{m}__R"] = r_mean
        out[f"{m}__R_fisher"] = r_fisher
        out[f"{m}__n_folds_valid"] = nvalid
        out[f"{m}__alpha_median"] = (
            float(np.median(alphas[m])) if alphas[m] else np.nan)
    if len(modes) > 1 and n_folds_total > 0:
        stacked = np.vstack([np.asarray(fold_rs[m], dtype=float) for m in modes])
        paired_ok = np.all(np.isfinite(stacked), axis=0)
        out["n_folds_paired"] = int(paired_ok.sum())
        for i, m in enumerate(modes):
            _, r_f, n_p = _aggregate_rs(stacked[i, paired_ok].tolist())
            out[f"{m}__R_fisher_paired"] = r_f
            out[f"{m}__n_folds_paired"] = n_p
    return out


def process_subject(
    subject, eeg_files, args, X_dense, surprisal, feat_valid,
    skip_channels: Optional[set] = None,
    on_checkpoint=None,
):
    eeg_files_abs = {sid: str(ap.DATA_ROOT / p) for sid, p in eeg_files.items()}
    missing = [p for p in eeg_files_abs.values() if not Path(p).exists()]
    if missing:
        print(f"[{subject}] missing EEG, skip")
        return pd.DataFrame()
    data = ec.load_subject_data(
        subject_name=subject, eeg_files=eeg_files_abs,
        features_dir=ap.FEATURES_DIR, fs_target=args.fs_target,
        sections=tuple(args.sections), feature_sets=["glove"],
    )
    Wtot = data.section_word_slices[-1][1]
    if X_dense.shape[0] != Wtot:
        raise ValueError(f"[{subject}] dense/EEG word mismatch")
    half_win = max(1, int(round((args.resp_win_ms / 1000.0) * args.fs_target / 2.0)))
    lag_samp = int(round((args.lag_ms / 1000.0) * args.fs_target))
    Y, valid = ec.compute_Y_for_lag(data, lag_samp, half_win)
    section_word_slices = list(data.section_word_slices)
    channel_labels = list(data.channel_labels)
    # Drop raw EEG ASAP — peak RSS from 3-section HG is large enough that a
    # concurrent host OOM (e.g. IDE) can kill this process mid-subject.
    del data
    base_ok = valid & feat_valid & np.isfinite(surprisal) if surprisal is not None else valid

    channels = list(channel_labels)
    if args.lang_only:
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        keep = set(
            tax.loc[
                (tax["subject"] == subject)
                & tax["is_lang"].fillna(False).astype(bool),
                "channel",
            ].astype(str).str.strip()
        )
        channels = [c for c in channels if str(c).strip() in keep]
    if skip_channels:
        before = len(channels)
        channels = [c for c in channels if str(c).strip() not in skip_channels]
        print(
            f"  [{subject}] resume: skip {before - len(channels)} done ch, "
            f"{len(channels)} remaining",
            flush=True,
        )
    if args.max_channels:
        channels = channels[: args.max_channels]

    n_ch = len(channels)
    if n_ch == 0:
        return pd.DataFrame()
    t0 = time.time()
    # High-d residual matrices (~2–4k dims) blow memory under loky fan-out.
    n_jobs = int(getattr(args, "n_jobs", 1) or 1)
    if X_dense.shape[1] > HIGH_D_THRESHOLD and n_jobs != 1:
        print(
            f"  [{subject}] high-d residual (d={X_dense.shape[1]}): "
            f"forcing n_jobs=1 (was {n_jobs})",
            flush=True,
        )
        n_jobs = 1
    print(f"  [{subject}] dense baseline {args.feature_name}: {n_ch} ch "
          f"(n_jobs={n_jobs})", flush=True)

    def run_one(ch: str):
        try:
            eeg_ci = channel_labels.index(ch)
        except ValueError:
            return None
        y = Y[:, eeg_ci].astype(np.float64)
        ok = base_ok & np.isfinite(y)
        if int(ok.sum()) < 50:
            return None
        folds = make_cv_splits(
            Wtot, args.n_folds, "contiguous",
            section_word_slices, ok=ok)
        if not folds:
            return None
        res = regress_dense_electrode(
            X_dense, y, folds,
            surprisal=surprisal if args.with_surprisal else None,
        )
        if not res:
            return None
        res.update({
            "subject": subject, "channel": ch, "channel_index": eeg_ci,
            "feature_name": args.feature_name,
            "surprisal_tag": args.surprisal_tag if args.with_surprisal else "",
            "n_words": int(ok.sum()), "cv": "contiguous",
        })
        return res

    if n_jobs == 1:
        rows = []
        kept_total = 0
        for ci, ch in enumerate(channels):
            r = run_one(ch)
            if r is not None:
                rows.append(r)
                kept_total += 1
                if on_checkpoint is not None and (
                    (ci + 1) % 5 == 0 or ci + 1 == n_ch
                ):
                    on_checkpoint(pd.DataFrame(rows))
                    rows = []
            if (ci + 1) % 5 == 0 or ci + 1 == n_ch:
                print(_progress_line(subject, ci + 1, n_ch, t0, kept=kept_total),
                      flush=True)
        return pd.DataFrame(rows)

    from joblib import Parallel, delayed
    results_iter = Parallel(
        n_jobs=n_jobs, backend="loky", return_as="generator",
    )(delayed(run_one)(ch) for ch in channels)
    rows = []
    done = 0
    for r in results_iter:
        done += 1
        if r is not None:
            rows.append(r)
        if done % 20 == 0 or done == n_ch:
            print(_progress_line(subject, done, n_ch, t0, kept=len(rows)),
                  flush=True)
    return pd.DataFrame(rows)


def write_baseline_report(df: pd.DataFrame, path: Path):
    lines = ["=" * 70, "DENSE BASELINE SUMMARY", "=" * 70, ""]
    if df.empty:
        path.write_text("\n".join(lines + ["(empty)"]))
        return
    tax = merge_taxonomy(df)
    lang = language_mask(tax)
    score = ("dense__R_fisher_paired"
             if "dense__R_fisher_paired" in tax.columns else "dense__R_fisher")
    for label, mask in (
        ("ALL", pd.Series(True, index=tax.index)),
        ("LOCALIZER-POSITIVE", lang),
        ("LEFT MFG", lang & tax["region"].astype(str).eq(LEFT_MFG_REGION)),
    ):
        sub = tax.loc[mask]
        lines.append("-" * 70)
        lines.append(f"{label} n={len(sub)}")
        lines.append("-" * 70)
        if sub.empty:
            lines.append("  (empty)\n")
            continue
        for feat, g in sub.groupby("feature_name"):
            v = g[score].dropna()
            lines.append(f"  {feat:12s} mean={v.mean():+.4f}  "
                         f"median={v.median():+.4f}  n={len(v)}")
            if "dense_surprisal__R_fisher" in g.columns:
                vs = g.get("dense_surprisal__R_fisher_paired",
                           g["dense_surprisal__R_fisher"]).dropna()
                if len(vs):
                    lines.append(f"  {feat}+surp   mean={vs.mean():+.4f}  "
                                 f"median={vs.median():+.4f}")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", nargs="+", default=["glove", "gpt2cn_l24"])
    p.add_argument("--surprisal_tag", default="sae_gemma2_2b_mat_l12",
                   help="Tag used to load validity mask (+ optional surprisal).")
    p.add_argument("--with_surprisal", action="store_true",
                   help="Also fit dense+surprisal models.")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--lag_ms", type=float, default=RESPONSE_LAG_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--max_channels", type=int, default=0)
    p.add_argument("--lang_only", action="store_true")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers across electrodes "
                        "(default 1; high-d residual forces 1).")
    p.add_argument("--out_suffix", default="")
    p.add_argument(
        "--resume", action="store_true",
        help="Skip subjects already present in the output CSV (append remaining).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    subjects_map = get_subjects()
    requested = args.subjects or list(subjects_map.keys())

    # Validity / surprisal from repaired SAE tag (shared event mask).
    _, surprisal, feat_valid = load_sae_features(args.surprisal_tag, args.sections)

    all_dfs = []
    t0 = time.time()
    suffix = args.out_suffix
    if args.lang_only and "lang" not in suffix:
        suffix = f"{suffix}_lang"
    out_path = ap.SAE_TABLES / f"sparse_encoding_dense_baselines{suffix}.csv"

    # Resume at (subject, channel, feature): partial subjects keep finished electrodes.
    done_channels: Dict[Tuple[str, str], set] = {}
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        if not prev.empty and "subject" in prev.columns:
            all_dfs.append(prev)
            for (sid, feat), g in prev.groupby(
                [prev["subject"].astype(str), prev["feature_name"].astype(str)]
            ):
                done_channels[(str(sid), str(feat))] = set(
                    g["channel"].astype(str).str.strip().unique()
                )
            print(
                f"Resume: loaded {len(prev)} rows from {out_path.name}; "
                f"subject×feature pairs: {len(done_channels)}",
                flush=True,
            )

    def _flush_checkpoint(extra: Optional[pd.DataFrame] = None):
        dfs = list(all_dfs)
        if extra is not None and not extra.empty:
            dfs.append(extra)
        if not dfs:
            return
        out = pd.concat(dfs, ignore_index=True)
        # Dedup if a channel was written twice (shouldn't happen, but safe).
        if {"subject", "channel", "feature_name"}.issubset(out.columns):
            out = out.drop_duplicates(
                subset=["subject", "channel", "feature_name"], keep="last")
        out.to_csv(out_path, index=False)
        print(f"  checkpoint {len(out)} rows -> {out_path.name}", flush=True)

    for feat in args.features:
        print(f"\n===== DENSE FEATURE {feat} =====", flush=True)
        X = load_dense_features(feat, args.sections)
        args.feature_name = feat
        for i, sid in enumerate(requested, 1):
            if sid not in subjects_map:
                continue
            skip = done_channels.get((sid, feat), set())
            print(f"\n=== [{i}] {sid} / {feat} ===", flush=True)
            if args.lang_only and skip:
                tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
                lang_chs = set(
                    tax.loc[
                        (tax["subject"] == sid)
                        & tax["is_lang"].fillna(False).astype(bool),
                        "channel",
                    ].astype(str).str.strip()
                )
                remaining = lang_chs - skip
                if not remaining:
                    print(f"  [{sid}] already complete (resume)", flush=True)
                    continue

            def _on_batch(batch_df, _sid=sid, _feat=feat):
                if batch_df is None or batch_df.empty:
                    return
                all_dfs.append(batch_df)
                done_channels.setdefault((_sid, _feat), set()).update(
                    batch_df["channel"].astype(str).str.strip().tolist()
                )
                _flush_checkpoint()

            df = process_subject(
                sid, subjects_map[sid]["eeg_files"], args, X, surprisal,
                feat_valid, skip_channels=skip or None,
                on_checkpoint=_on_batch if X.shape[1] > HIGH_D_THRESHOLD else None,
            )
            if not df.empty:
                all_dfs.append(df)
                done_channels.setdefault((sid, feat), set()).update(
                    df["channel"].astype(str).str.strip().tolist()
                )
                _flush_checkpoint()
            elif skip:
                print(f"  [{sid}] already complete (resume)", flush=True)
    if not all_dfs:
        print("No dense baseline results.")
        return
    out = pd.concat(all_dfs, ignore_index=True)
    if {"subject", "channel", "feature_name"}.issubset(out.columns):
        out = out.drop_duplicates(
            subset=["subject", "channel", "feature_name"], keep="last")
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows -> {out_path} ({_fmt_duration(time.time()-t0)})")
    write_baseline_report(
        out, ap.SAE_REPORTS / f"sparse_encoding_dense_baselines{suffix}.txt")


if __name__ == "__main__":
    main()
