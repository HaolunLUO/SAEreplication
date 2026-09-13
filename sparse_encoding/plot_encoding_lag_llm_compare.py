#!/usr/bin/env python3
"""
Compare encoding-lag curves across LLM residual / SAE / classic embeddings.

Uses existing SAE lag-screen long CSVs for SAE content / full / surprisal.
Runs (or reuses) PCA-Ridge dense lag screens for residual streams and
embeddings (GloVe, GPT-2-CN L24, etc.) on the same 0–800 ms / 50 ms grid.

Example
-------
    python -m sparse_encoding.plot_encoding_lag_llm_compare --lang_only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.plot_encoding_lag_residual_sae import (
    process_subject_residual,
    subject_mean_curve,
)
from sparse_encoding.sparse_encoding_dense_baseline import load_dense_features
from sparse_encoding.sparse_encoding_lag_screen import (
    LAG_END_MS, LAG_START_MS, LAG_STEP_MS, RESP_WIN_MS,
)
from sparse_encoding.sparse_encoding_regression import (
    FS_TARGET, N_FOLDS, SECTIONS, _fmt_duration, get_subjects, load_sae_features,
)

# (label, dense_feature_npy_stem, surprisal_tag_or_None)
DEFAULT_DENSE = [
    ("Qwen3.5-4B residual", "sae_qwen35_4b_mat_l15_resid", "sae_qwen35_4b_mat_l15"),
    ("Qwen3-8B residual", "sae_qwen3_8b_l18_resid", "sae_qwen3_8b_l18"),
    ("Gemma-2-2B residual", "sae_gemma2_2b_mat_l12_resid", "sae_gemma2_2b_mat_l12"),
    ("GPT-2-CN L24", "gpt2cn_l24", "sae_qwen35_4b_mat_l15"),
    ("GloVe", "glove", "sae_qwen35_4b_mat_l15"),
]

DEFAULT_SAE = [
    ("Qwen3.5-4B SAE", "sae_qwen35_4b_mat_l15"),
    ("Qwen3-8B SAE", "sae_qwen3_8b_l18"),
    ("Gemma-2-2B SAE", "sae_gemma2_2b_mat_l12"),
]


def _set_suffix(electrode_set: str) -> str:
    return {"is_lang": "lang", "sig_glove": "sig_glove", "all": "all"}[electrode_set]


def _cache_path(dense_name: str, set_suf: str) -> Path:
    return ap.SAE_TABLES / (
        f"sparse_encoding_lag_screen_long_dense_{dense_name}_{set_suf}.csv"
    )


def run_or_load_dense_lag(
    dense_name: str,
    surprisal_tag: str,
    args,
) -> pd.DataFrame:
    set_suf = _set_suffix(args.electrode_set)
    out = _cache_path(dense_name, set_suf)
    if out.exists() and not args.force:
        print(f"Reusing dense lag: {out}")
        return pd.read_csv(out)

    print(f"\n### Dense lag-screen: {dense_name} (surprisal={surprisal_tag}) "
          f"[{args.electrode_set}] ###", flush=True)
    X = load_dense_features(dense_name, args.sections)
    _, surprisal, feat_valid = load_sae_features(surprisal_tag, args.sections)
    if X.shape[0] != surprisal.shape[0]:
        raise ValueError(
            f"{dense_name} rows {X.shape[0]} != surprisal {surprisal.shape[0]}")

    ns = SimpleNamespace(
        fs_target=args.fs_target,
        sections=args.sections,
        lag_start_ms=args.lag_start_ms,
        lag_end_ms=args.lag_end_ms,
        lag_step_ms=args.lag_step_ms,
        resp_win_ms=args.resp_win_ms,
        n_folds=args.n_folds,
        n_pcs=args.n_pcs,
        ridge_alpha=args.ridge_alpha,
        lang_only=(args.electrode_set == "is_lang"),
        electrode_set=args.electrode_set,
        max_channels=args.max_channels,
        feature_tag=dense_name,
    )
    subjects_map = get_subjects()
    requested = args.subjects or list(subjects_map.keys())
    dfs = []
    t0 = time.time()
    for i, sid in enumerate(requested, 1):
        if sid not in subjects_map:
            continue
        print(f"\n=== [{i}] {sid} / {dense_name} ===", flush=True)
        df = process_subject_residual(
            sid, subjects_map[sid]["eeg_files"], ns, X, surprisal, feat_valid)
        if not df.empty:
            df = df.assign(
                dense_name=dense_name,
                surprisal_tag=surprisal_tag,
                electrode_set=args.electrode_set,
            )
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No rows for {dense_name}")
    long = pd.concat(dfs, ignore_index=True)
    long.to_csv(out, index=False)
    print(f"Saved {out} ({_fmt_duration(time.time() - t0)})")
    return long


def load_sae_curve(tag: str, mode: str, set_suf: str) -> pd.DataFrame:
    path = ap.SAE_TABLES / f"sparse_encoding_lag_screen_long_{tag}_{set_suf}.csv"
    if not path.exists():
        print(f"[warn] missing SAE lag long: {path}")
        return pd.DataFrame()
    return subject_mean_curve(pd.read_csv(path), mode)


def plot_multi(
    curves: Dict[str, pd.DataFrame],
    out_path: Path,
    title: str,
    ylabel: str = "Encoding R (Fisher-z), subject-mean ± SEM",
) -> None:
    # Stable color cycle (tab10-ish, not purple-heavy stack)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, (name, cur) in enumerate(curves.items()):
        if cur is None or cur.empty:
            continue
        c = palette[i % len(palette)]
        ax.plot(cur["lag_ms"], cur["mean"], label=name, color=c, lw=2)
        if cur["sem"].notna().any():
            ax.fill_between(
                cur["lag_ms"],
                cur["mean"] - cur["sem"],
                cur["mean"] + cur["sem"],
                alpha=0.15, color=c,
            )
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def peaks_table(curves: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, cur in curves.items():
        if cur is None or cur.empty:
            continue
        i = int(cur["mean"].values.argmax())
        rows.append({
            "model": name,
            "peak_lag_ms": float(cur["lag_ms"].iloc[i]),
            "peak_R": float(cur["mean"].iloc[i]),
            "n_subjects": int(cur["n_subjects"].iloc[i]),
        })
    return pd.DataFrame(rows)


def summary_long(curves: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, cur in curves.items():
        if cur is None or cur.empty:
            continue
        for _, r in cur.iterrows():
            rows.append({
                "model": name,
                "lag_ms": r["lag_ms"],
                "R_fisher_subject_mean": r["mean"],
                "sem": r["sem"],
                "n_subjects": r["n_subjects"],
            })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--lag_start_ms", type=float, default=LAG_START_MS)
    p.add_argument("--lag_end_ms", type=float, default=LAG_END_MS)
    p.add_argument("--lag_step_ms", type=float, default=LAG_STEP_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--n_pcs", type=int, default=128)
    p.add_argument("--ridge_alpha", type=float, default=100.0)
    p.add_argument("--lang_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--electrode_set",
        choices=("is_lang", "sig_glove", "all"),
        default=None,
        help="Channel gate (default: is_lang if --lang_only else all).",
    )
    p.add_argument("--max_channels", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Recompute dense lag CSVs even if cached.")
    p.add_argument("--skip_dense", action="store_true",
                   help="Only plot from existing SAE / dense caches.")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    if args.electrode_set is None:
        args.electrode_set = "is_lang" if args.lang_only else "all"
    set_suf = _set_suffix(args.electrode_set)
    gate_label = {
        "is_lang": "is_lang",
        "sig_glove": "sig_glove (GloVe encoding-sig)",
        "all": "all channels",
    }[args.electrode_set]

    dense_curves: Dict[str, pd.DataFrame] = {}
    for label, dense_name, surp_tag in DEFAULT_DENSE:
        # Prefer prior qwen35 residual cache only for is_lang runs.
        special = None
        if (dense_name == "sae_qwen35_4b_mat_l15_resid"
                and args.electrode_set == "is_lang"):
            special = (
                ap.SAE_TABLES
                / ("sparse_encoding_lag_screen_long_sae_qwen35_4b_mat_l15_resid"
                   "_sae_qwen35_4b_mat_l15_lang.csv")
            )
        if special is not None and special.exists() and not args.force:
            print(f"Reusing Qwen35 residual lag: {special}")
            long = pd.read_csv(special)
        elif args.skip_dense and _cache_path(dense_name, set_suf).exists():
            long = pd.read_csv(_cache_path(dense_name, set_suf))
        elif args.skip_dense:
            print(f"[skip] no cache for {dense_name}")
            continue
        else:
            long = run_or_load_dense_lag(dense_name, surp_tag, args)
        dense_curves[label] = subject_mean_curve(long, "residual")

    sae_curves: Dict[str, pd.DataFrame] = {}
    surprisal_curves: Dict[str, pd.DataFrame] = {}
    for label, tag in DEFAULT_SAE:
        sae_curves[label] = load_sae_curve(tag, "content", set_suf)
        surprisal_curves[label.replace(" SAE", " surprisal")] = load_sae_curve(
            tag, "surprisal_only", set_suf)

    q35_resid = dense_curves.get("Qwen3.5-4B residual", pd.DataFrame())
    q35_sae = sae_curves.get("Qwen3.5-4B SAE", pd.DataFrame())
    q35_surp = surprisal_curves.get("Qwen3.5-4B surprisal", pd.DataFrame())

    plot_multi(
        dense_curves,
        ap.SAE_FIGURES / f"encoding_lag_llm_dense_compare_{set_suf}.png",
        title=("Encoding lag: LLM residuals & embeddings\n"
               f"{gate_label} · PCA≤128 Ridge · subject-mean Fisher-z R"),
    )
    plot_multi(
        sae_curves,
        ap.SAE_FIGURES / f"encoding_lag_llm_sae_compare_{set_suf}.png",
        title=("Encoding lag: SAE content across models\n"
               f"{gate_label} · subject-mean Fisher-z R"),
    )

    head: Dict[str, pd.DataFrame] = {}
    if not q35_resid.empty:
        head["Qwen3.5 residual"] = q35_resid
    if not q35_sae.empty:
        head["Qwen3.5 SAE"] = q35_sae
    if not q35_surp.empty:
        head["Qwen3.5 surprisal"] = q35_surp
    for k in ("Qwen3-8B residual", "Gemma-2-2B residual",
              "GPT-2-CN L24", "GloVe"):
        if k in dense_curves and not dense_curves[k].empty:
            head[k] = dense_curves[k]
    plot_multi(
        head,
        ap.SAE_FIGURES / f"encoding_lag_residual_sae_vs_llms_{set_suf}.png",
        title=("Encoding lag: Residual / SAE / surprisal vs other LLMs\n"
               f"{gate_label} · subject-mean Fisher-z R"),
    )

    all_curves = {**{f"[dense] {k}": v for k, v in dense_curves.items()},
                  **{f"[sae] {k}": v for k, v in sae_curves.items()},
                  **{f"[surp] {k}": v for k, v in surprisal_curves.items()}}
    summary = summary_long(all_curves)
    peaks = peaks_table(all_curves)
    summary.to_csv(
        ap.SAE_TABLES / f"encoding_lag_llm_compare_summary_{set_suf}.csv",
        index=False)
    peaks.to_csv(
        ap.SAE_TABLES / f"encoding_lag_llm_compare_peaks_{set_suf}.csv",
        index=False)
    print("\nPeak lags:")
    print(peaks.to_string(index=False))


if __name__ == "__main__":
    main()