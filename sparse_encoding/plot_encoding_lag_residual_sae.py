#!/usr/bin/env python3
"""
Lag-resolved Residual vs SAE vs surprisal encoding curves.

Uses existing SAE lag-screen long CSV for SAE (content) and surprisal_only.
Computes same-model residual-stream Ridge on a PCA-compressed residual
(default 128 PCs) across the same 0–800 ms / 50 ms grid for speed, then
writes a combined long table + figure.

Example
-------
    python -m sparse_encoding.plot_encoding_lag_residual_sae \\
        --feature_tag sae_qwen35_4b_mat_l15 --lang_only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_dense_baseline import load_dense_features
from sparse_encoding.sparse_encoding_lag_screen import (
    LAG_END_MS, LAG_START_MS, LAG_STEP_MS, RESP_WIN_MS,
    compute_Y_lag_grid, lag_grid_ms, make_buffered_contiguous_splits,
)
from sparse_encoding.sparse_encoding_regression import (
    FS_TARGET, N_FOLDS, RANDOM_STATE, SECTIONS,
    _aggregate_rs, _fmt_duration, _progress_line, get_subjects,
    load_sae_features,
)

def _ridge_r(Xtr, y_tr, Xte, y_te, alpha: float) -> float:
    if Xtr.shape[0] < 20 or Xte.shape[0] < 5:
        return 0.0
    if np.std(y_tr) < 1e-12 or np.std(y_te) < 1e-12:
        return 0.0
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, y_tr)
    pred = model.predict(Xte).ravel()
    if np.std(pred) < 1e-12 or not np.all(np.isfinite(pred)):
        return 0.0
    r, _ = pearsonr(y_te, pred)
    return 0.0 if not np.isfinite(r) else float(r)


def _precompute_fold_features(
    X_resid: np.ndarray,
    surprisal: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    base_ok: np.ndarray,
    n_pcs: int,
    ridge_alpha: float,
) -> List[dict]:
    """PCA residual once per fold (shared across channels/lags)."""
    pack = []
    for tr, te in folds:
        tr_ok = tr[base_ok[tr]]
        if tr_ok.size < 20 or te.size < 5:
            pack.append(None)
            continue
        n_comp = int(min(n_pcs, tr_ok.size - 1, X_resid.shape[1]))
        if n_comp < 2:
            pack.append(None)
            continue
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        # Fit on training words only; transform full vocabulary once.
        pca.fit(X_resid[tr_ok])
        Z = pca.transform(X_resid).astype(np.float64, copy=False)
        sc_s = StandardScaler().fit(surprisal[tr_ok].reshape(-1, 1))
        S = sc_s.transform(surprisal.reshape(-1, 1)).astype(np.float64, copy=False)
        pack.append({
            "tr": tr,
            "te": te,
            "Z": Z,
            "S": S,
            "alpha": float(ridge_alpha),
        })
    return pack


def regress_residual_lags_fast(
    fold_pack: List[dict],
    y_lags: np.ndarray,
    valid_lags: np.ndarray,
) -> Dict[str, float]:
    """Per-lag residual / residual+surprisal / surprisal using shared PCA features."""
    n_lags = y_lags.shape[0]
    modes = ("residual", "residual_surprisal", "surprisal_only")
    fold_rs = {m: [[] for _ in range(n_lags)] for m in modes}

    for fp in fold_pack:
        if fp is None:
            continue
        tr, te = fp["tr"], fp["te"]
        Z, S, alpha = fp["Z"], fp["S"], fp["alpha"]
        for li in range(n_lags):
            y = y_lags[li]
            ok_tr = np.isfinite(y[tr]) & valid_lags[li, tr]
            ok_te = np.isfinite(y[te]) & valid_lags[li, te]
            tr_i, te_i = tr[ok_tr], te[ok_te]
            if tr_i.size < 20 or te_i.size < 5:
                for m in modes:
                    fold_rs[m][li].append(0.0)
                continue
            y_tr, y_te = y[tr_i], y[te_i]
            specs = {
                "residual": (Z[tr_i], Z[te_i]),
                "residual_surprisal": (
                    np.hstack([Z[tr_i], S[tr_i]]),
                    np.hstack([Z[te_i], S[te_i]]),
                ),
                "surprisal_only": (S[tr_i], S[te_i]),
            }
            for m, (Xtr, Xte) in specs.items():
                fold_rs[m][li].append(_ridge_r(Xtr, y_tr, Xte, y_te, alpha))

    out: Dict[str, float] = {}
    for m in modes:
        for li in range(n_lags):
            _, r_f, _ = _aggregate_rs(fold_rs[m][li])
            out[f"{m}__R_fisher_lag{li}"] = r_f
    return out


def process_subject_residual(subject: str, eeg_files, args, X_resid, surprisal, feat_valid):
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
    if X_resid.shape[0] != Wtot:
        raise ValueError(f"[{subject}] residual/EEG word mismatch")

    lags_ms = lag_grid_ms(args.lag_start_ms, args.lag_end_ms, args.lag_step_ms)
    Y_lags, V_lags = compute_Y_lag_grid(data, lags_ms, args.resp_win_ms)
    guard_w = max(1, int(np.ceil(args.lag_end_ms / args.lag_step_ms)))
    base_ok = feat_valid & np.isfinite(surprisal)

    channels = list(data.channel_labels)
    electrode_set = getattr(args, "electrode_set", None)
    if electrode_set is None and getattr(args, "lang_only", False):
        electrode_set = "is_lang"
    if electrode_set and electrode_set != "all":
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        col = {"is_lang": "is_lang", "sig_glove": "sig_glove"}[electrode_set]
        keep = set(
            tax.loc[
                (tax["subject"] == subject) & tax[col].fillna(False),
                "channel",
            ].astype(str)
        )
        channels = [c for c in channels if c in keep]
        print(f"  [{subject}] filtered to {len(channels)} {electrode_set} channels",
              flush=True)
    if args.max_channels:
        channels = channels[: args.max_channels]

    folds = make_buffered_contiguous_splits(
        Wtot, args.n_folds, list(data.section_word_slices), base_ok, guard_w)
    if not folds:
        print(f"[{subject}] no usable folds, skip")
        return pd.DataFrame()

    print(f"  [{subject}] PCA residual once × {len(folds)} folds "
          f"(n_pcs={args.n_pcs})…", flush=True)
    t_pca = time.time()
    fold_pack = _precompute_fold_features(
        X_resid, surprisal, folds, base_ok, args.n_pcs, args.ridge_alpha)
    print(f"  [{subject}] PCA done in {time.time() - t_pca:.1f}s", flush=True)

    rows = []
    t0 = time.time()
    n_ch = len(channels)
    print(f"  [{subject}] residual lag-screen {n_ch} ch × {len(lags_ms)} lags",
          flush=True)
    for ci, ch in enumerate(channels):
        try:
            eeg_ci = list(data.channel_labels).index(ch)
        except ValueError:
            continue
        y_lags = Y_lags[:, :, eeg_ci]
        valid_lags = V_lags & base_ok[None, :] & np.isfinite(y_lags)
        res = regress_residual_lags_fast(fold_pack, y_lags, valid_lags)
        for li, lag in enumerate(lags_ms):
            for mode in ("residual", "residual_surprisal", "surprisal_only"):
                key = f"{mode}__R_fisher_lag{li}"
                if key not in res:
                    continue
                rows.append({
                    "subject": subject,
                    "channel": ch,
                    "channel_index": eeg_ci,
                    "lag_ms": float(lag),
                    "mode": mode,
                    "R_fisher": float(res[key]),
                    "feature_tag": args.feature_tag,
                    "n_pcs": int(args.n_pcs),
                    "ridge_alpha": float(args.ridge_alpha),
                })
        if (ci + 1) % 5 == 0 or ci + 1 == n_ch:
            print(_progress_line(subject, ci + 1, n_ch, t0, kept=ci + 1), flush=True)
    return pd.DataFrame(rows)


def subject_mean_curve(long_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = long_df.loc[long_df["mode"] == mode]
    if sub.empty:
        return pd.DataFrame(columns=["lag_ms", "mean", "sem", "n_subjects"])
    # electrode → subject mean, then across subjects
    s = (sub.groupby(["subject", "lag_ms"])["R_fisher"].mean()
           .reset_index())
    g = s.groupby("lag_ms")["R_fisher"]
    out = pd.DataFrame({
        "lag_ms": g.mean().index.astype(float),
        "mean": g.mean().values,
        "sem": g.sem().values,
        "n_subjects": g.count().values,
    })
    return out


def plot_overlay(
    curves: Dict[str, pd.DataFrame],
    out_path: Path,
    title: str,
) -> None:
    colors = {
        "Residual": "#9467bd",
        "Residual + surprisal": "#8c564b",
        "SAE": "#1f77b4",
        "SAE + surprisal": "#2ca02c",
        "Surprisal": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, cur in curves.items():
        if cur.empty:
            continue
        ax.plot(cur["lag_ms"], cur["mean"], label=name,
                color=colors.get(name, "k"), lw=2)
        if cur["sem"].notna().any():
            ax.fill_between(
                cur["lag_ms"],
                cur["mean"] - cur["sem"],
                cur["mean"] + cur["sem"],
                alpha=0.18, color=colors.get(name, "k"),
            )
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Encoding R (Fisher-z), subject-mean ± SEM")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default="sae_qwen35_4b_mat_l15")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--lag_start_ms", type=float, default=LAG_START_MS)
    p.add_argument("--lag_end_ms", type=float, default=LAG_END_MS)
    p.add_argument("--lag_step_ms", type=float, default=LAG_STEP_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--n_pcs", type=int, default=128,
                   help="PCA dims for residual lag screen (speed).")
    p.add_argument("--ridge_alpha", type=float, default=100.0,
                   help="Fixed Ridge alpha on PCA residual features.")
    p.add_argument("--lang_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--electrode_set",
        choices=("is_lang", "sig_glove", "all"),
        default=None,
        help="Overrides --lang_only when set (is_lang / sig_glove / all).",
    )
    p.add_argument("--max_channels", type=int, default=0)
    p.add_argument("--sae_long_csv", default=None,
                   help="Existing SAE lag long CSV (default: lang table for tag).")
    p.add_argument("--reuse_residual_csv", default=None,
                   help="Skip residual recompute if this long CSV exists.")
    p.add_argument("--out_suffix", default="")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    if args.electrode_set is None:
        args.electrode_set = "is_lang" if args.lang_only else "all"
    tag = args.feature_tag
    set_suf = {
        "is_lang": "lang",
        "sig_glove": "sig_glove",
        "all": "all",
    }[args.electrode_set]
    suffix = args.out_suffix or f"_{tag}_{set_suf}"

    sae_long_path = Path(args.sae_long_csv) if args.sae_long_csv else (
        ap.SAE_TABLES / f"sparse_encoding_lag_screen_long_{tag}_{set_suf}.csv"
    )
    if not sae_long_path.exists():
        raise FileNotFoundError(
            f"Missing SAE lag long CSV: {sae_long_path}. "
            "Run sparse_encoding_lag_screen.py first."
        )
    sae_long = pd.read_csv(sae_long_path)
    print(f"Loaded SAE lag long: {sae_long_path} ({len(sae_long)} rows)")

    resid_long_path = Path(args.reuse_residual_csv) if args.reuse_residual_csv else (
        ap.SAE_TABLES / f"sparse_encoding_lag_screen_long_{tag}_resid{suffix}.csv"
    )
    if resid_long_path.exists() and args.reuse_residual_csv is None:
        # Default: reuse if present.
        print(f"Reusing residual lag long: {resid_long_path}")
        resid_long = pd.read_csv(resid_long_path)
    elif args.reuse_residual_csv and Path(args.reuse_residual_csv).exists():
        resid_long = pd.read_csv(args.reuse_residual_csv)
    else:
        resid_name = f"{tag}_resid"
        X_resid = load_dense_features(resid_name, args.sections)
        _, surprisal, feat_valid = load_sae_features(tag, args.sections)
        subjects_map = get_subjects()
        requested = args.subjects or list(subjects_map.keys())
        dfs = []
        t0 = time.time()
        for i, sid in enumerate(requested, 1):
            if sid not in subjects_map:
                continue
            print(f"\n=== [{i}] {sid} ===", flush=True)
            df = process_subject_residual(
                sid, subjects_map[sid]["eeg_files"], args,
                X_resid, surprisal, feat_valid,
            )
            if not df.empty:
                dfs.append(df)
        if not dfs:
            raise RuntimeError("No residual lag rows produced.")
        resid_long = pd.concat(dfs, ignore_index=True)
        resid_long.to_csv(resid_long_path, index=False)
        print(f"Saved residual lag long -> {resid_long_path} "
              f"({_fmt_duration(time.time() - t0)})")

    # Combined long for bookkeeping
    comb = pd.concat([
        sae_long.assign(source="sae_lag_screen"),
        resid_long.assign(source="residual_lag_screen"),
    ], ignore_index=True)
    comb_path = ap.SAE_TABLES / f"encoding_lag_residual_sae_surprisal{suffix}.csv"
    comb.to_csv(comb_path, index=False)

    curves = {
        "Residual": subject_mean_curve(resid_long, "residual"),
        "Residual + surprisal": subject_mean_curve(resid_long, "residual_surprisal"),
        "SAE": subject_mean_curve(sae_long, "content"),
        "SAE + surprisal": subject_mean_curve(sae_long, "full"),
        "Surprisal": subject_mean_curve(sae_long, "surprisal_only"),
    }
    # Prefer residual-screen surprisal if present (same CV), else SAE screen.
    if not curves["Surprisal"].empty and not subject_mean_curve(
            resid_long, "surprisal_only").empty:
        # Average of the two surprisal curves is confusing; use SAE-screen
        # surprisal for the SAE panel consistency, already set.
        pass

    fig_path = ap.SAE_FIGURES / f"encoding_lag_residual_sae_surprisal{suffix}.png"
    plot_overlay(
        curves, fig_path,
        title=("Encoding lag: Residual vs SAE vs surprisal\n"
               f"{tag} · {args.electrode_set} · subject-mean Fisher-z R"),
    )

    # Compact subject×lag means table for canvas / reports
    rows = []
    for name, cur in curves.items():
        for _, r in cur.iterrows():
            rows.append({
                "model": name,
                "lag_ms": r["lag_ms"],
                "R_fisher_subject_mean": r["mean"],
                "sem": r["sem"],
                "n_subjects": r["n_subjects"],
            })
    summary = pd.DataFrame(rows)
    summary_path = ap.SAE_TABLES / f"encoding_lag_residual_sae_surprisal_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # Peak printout
    print("\nPeak lags (subject-mean curves):")
    for name, cur in curves.items():
        if cur.empty:
            continue
        i = int(cur["mean"].values.argmax())
        print(f"  {name:22s} peak={cur['lag_ms'].iloc[i]:.0f} ms  "
              f"R={cur['mean'].iloc[i]:+.4f}")


if __name__ == "__main__":
    main()
