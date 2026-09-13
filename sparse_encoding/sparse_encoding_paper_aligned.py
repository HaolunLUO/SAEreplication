#!/usr/bin/env python3
"""
sparse_encoding_paper_aligned.py
================================
Paper-faithful primary analysis for Augmented Sparse Encoding Models
(Lepori, Kay & Tuckute; arXiv:2606.06857).

Primary endpoint
----------------
SAE incremental benefit beyond surprisal::

    paper_sae_gain = full_R_fisher_paired − surprisal_only_R_fisher_paired

Positive, statistically supported subject-level SAE gain means content (SAE)
features improve prediction beyond surprisal alone. Lack of significant gain
is reported as **"no detected SAE benefit"**, not as proof that a channel is
surprisal-only.

This script does **not** use the exploratory dominance index
(``full−content``, ``surprisal_dominant`` labels). Those remain in
``sparse_encoding_surprisal_dominance.py`` and are labeled exploratory.

iEEG deviations from Lepori et al. (documented in the report)
-------------------------------------------------------------
- Word-locked Chinese iEEG (high-gamma), not sentence-level 7T fMRI.
- Contiguous within-section CV (default), not shuffled sentence folds.
- Predictivity is Fisher-z *r*, **not** noise-ceiling normalized
  (no repeated-stimulus reliability estimate available for NC normalization).

Example
-------
    python sparse_encoding_paper_aligned.py
    python sparse_encoding_paper_aligned.py --n_perm 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_summary import (
    LEFT_MFG_REGION, bootstrap_subject_means, language_mask, merge_taxonomy,
)
from sparse_encoding.sparse_encoding_surprisal_dominance import (
    FEATURE_TAG, MIN_GROUP_N, N_PERM, RNG_SEED,
    TASK_POS, classify_dominance, merge_localizers, normalize_channel,
    paper_group_summary, run_paper_sae_gain_stats,
)


# ======================================================================
# Report / figures
# ======================================================================

def write_paper_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    cov: Dict[str, int],
    path: Path,
    results_csv: Path,
    n_perm: int,
):
    lines = []
    a = lines.append
    a("=" * 70)
    a("PAPER-ALIGNED ANALYSIS — FULL vs SURPRISAL-ONLY (Lepori et al.)")
    a("=" * 70)
    a(f"Feature tag: {FEATURE_TAG}")
    a(f"Source regression CSV: {results_csv}")
    a(f"Electrodes: {len(df)}  subjects: {df['subject'].nunique()}")
    a(f"n_perm={n_perm}  min_group_n={MIN_GROUP_N}  rng_seed={RNG_SEED}")
    a("")
    a("-" * 70)
    a("PRIMARY DEFINITION")
    a("-" * 70)
    a("  paper_sae_gain = full_R_fisher_paired − surprisal_only_R_fisher_paired")
    a("  Inferential unit = subject (electrodes averaged within subject).")
    a("  Significant positive SAE gain → content features improve beyond surprisal.")
    a("  Nonsignificant SAE gain → 'no detected SAE benefit' (NOT surprisal-only).")
    a("")
    a("-" * 70)
    a("DEVIATIONS FROM LEPORI ET AL. (unavoidable / intentional)")
    a("-" * 70)
    a("  1. Modality: word-locked Chinese iEEG high-gamma, not sentence-level 7T fMRI.")
    a("  2. CV: contiguous within-section folds (default), not shuffled sentence KFold.")
    a("  3. Scoring: Fisher-z Pearson r without noise-ceiling normalization")
    a("     (no repeated-stimulus NCSNR available for NC normalization).")
    a("  4. Backbone: Qwen3-8B SAE (Chinese-primary), not gemma-2-2b Matryoshka.")
    a("")
    a("-" * 70)
    a("MERGE / COVERAGE")
    a("-" * 70)
    for k, v in cov.items():
        a(f"  {k}: {v}")
    a(f"  paper_quality_ok: {int(df['paper_quality_ok'].sum())}")
    if "is_lang" in df.columns:
        lang = language_mask(df)
        a(f"  is_lang: {int(lang.sum())}")
    a("")
    a("-" * 70)
    a("SUBJECT-LEVEL SUMMARY (paper SAE gain)")
    a("-" * 70)
    if summary.empty:
        a("  (empty)")
    else:
        show = summary.copy()
        a(show.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("INFERENTIAL TESTS (FDR within family; subject-level)")
    a("-" * 70)
    if stats.empty:
        a("  (no tests)")
    else:
        a(stats.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("EXPLORATORY EXTENSION (not Lepori et al.)")
    a("-" * 70)
    a("  full−content, dominance_index, and surprisal_dominant / sae_dominant")
    a("  labels are written by sparse_encoding_surprisal_dominance.py only.")
    a("  They are NOT used as the primary paper-aligned claim in this report.")
    a("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def fig_full_vs_surprisal(df: pd.DataFrame, outdir: Path):
    """Paired subject means: full vs surprisal_only."""
    ok = df[df["paper_quality_ok"]]
    if ok.empty:
        return
    subj = ok.groupby("subject")[["paper_full_R", "paper_surprisal_R"]].mean()
    if len(subj) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    x = subj["paper_surprisal_R"].to_numpy()
    y = subj["paper_full_R"].to_numpy()
    ax.scatter(x, y, s=50, alpha=0.85, zorder=3)
    lims = [
        min(x.min(), y.min()) - 0.01,
        max(x.max(), y.max()) + 0.01,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, label="full = surprisal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("surprisal_only R (subject-mean Fisher-z)")
    ax.set_ylabel("full R (subject-mean Fisher-z)")
    ax.set_title("Paper-aligned: full vs surprisal-only (subjects)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    p = outdir / "paper_aligned_full_vs_surprisal_subjects.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_sae_gain_by_group(summary: pd.DataFrame, outdir: Path, group_col: str, fname: str, title: str):
    sub = summary[summary["group_col"] == group_col].copy()
    if sub.empty:
        return
    sub = sub.sort_values("paper_sae_gain_subject_mean")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    ax.barh(
        y, sub["paper_sae_gain_subject_mean"],
        xerr=[
            sub["paper_sae_gain_subject_mean"] - sub["paper_sae_gain_ci_lo"],
            sub["paper_sae_gain_ci_hi"] - sub["paper_sae_gain_subject_mean"],
        ],
        capsize=3, color="#1f77b4",
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{g} (nE={int(ne)}, nS={int(ns)})"
        for g, ne, ns in zip(sub["group"], sub["n_electrodes"], sub["n_subjects"])
    ], fontsize=8)
    ax.set_xlabel("SAE gain (full − surprisal_only), subject-mean")
    ax.set_title(title)
    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_frac_positive_descriptive(summary: pd.DataFrame, outdir: Path):
    """Descriptive fraction of channels with paper_sae_gain > 0 (not a claim)."""
    sub = summary[summary["group_col"].isin(["task_group", "all"])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = [f"{r.group}" for _, r in sub.iterrows()]
    vals = sub["frac_sae_gain_positive_subject_mean"].to_numpy(float)
    ax.bar(np.arange(len(vals)), vals, color="#ff7f0e")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("subject-mean fraction (descriptive)")
    ax.set_title("Fraction of channels with SAE gain > 0 (descriptive only)")
    fig.tight_layout()
    p = outdir / "paper_aligned_frac_sae_gain_positive.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_csv",
        default=str(ap.SAE_TABLES / f"sparse_encoding_results_{FEATURE_TAG}_cohort.csv"),
    )
    p.add_argument("--n_perm", type=int, default=N_PERM)
    p.add_argument("--min_group_n", type=int, default=MIN_GROUP_N)
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    raw = pd.read_csv(results_csv)
    raw["channel"] = normalize_channel(raw["channel"])
    if ap.TAX_ELECTRODE_TABLE.exists():
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        tax["channel"] = normalize_channel(tax["channel"])
        keep = [c for c in ("subject", "channel", "region", "category",
                            "hemisphere", "is_lang") if c in tax.columns]
        raw = raw.merge(tax[keep], on=["subject", "channel"], how="left")
    else:
        raw = merge_taxonomy(raw)
    raw["channel"] = normalize_channel(raw["channel"])

    print(f"Loaded {len(raw)} electrodes from {results_csv.name}")
    d = classify_dominance(raw)
    d, cov = merge_localizers(d)

    # Summaries
    pieces = [paper_group_summary(d, None)]
    if "is_lang" in d.columns:
        lang = d[language_mask(d)]
        if not lang.empty:
            tmp = paper_group_summary(lang, None)
            if not tmp.empty:
                tmp = tmp.copy()
                tmp["group_col"] = "is_lang"
                tmp["group"] = "is_lang=True"
                pieces.append(tmp)
    if "task_group" in d.columns:
        pieces.append(paper_group_summary(d, "task_group"))
    for task, pos in TASK_POS.items():
        if pos in d.columns:
            tmp = d.copy()
            tmp["_g"] = np.where(tmp[pos], f"{task}+", f"{task}-")
            pieces.append(paper_group_summary(tmp, "_g"))
    if "region" in d.columns and "is_lang" in d.columns:
        lang = d[language_mask(d)]
        if not lang.empty:
            pieces.append(paper_group_summary(lang, "region"))
    summary = pd.concat([p for p in pieces if p is not None and not p.empty],
                        ignore_index=True)

    print(f"Running paper-aligned SAE-gain stats (n_perm={args.n_perm})…")
    stats = run_paper_sae_gain_stats(
        d, n_perm=args.n_perm, min_group_n=args.min_group_n,
    )

    # Channel table: paper fields first; exploratory columns retained but marked
    paper_cols = [
        c for c in (
            "subject", "channel", "region", "hemisphere", "is_lang", "category",
            "task_class", "task_group", "dominant_task",
            "paper_full_R", "paper_surprisal_R", "paper_sae_gain",
            "paper_sae_gain_positive", "paper_quality_ok", "paper_quality_paired",
            "full_R", "surprisal_only_R", "content_R",
            "sae_gain", "exploratory_surprisal_gain",
            "exploratory_dominance_index", "exploratory_dominance_label",
            "surprisal_gain", "dominance_index", "dominance_label",
            "all_folds_paired",
        ) if c in d.columns
    ]
    chan = d[paper_cols].copy()
    chan["analysis_role"] = "paper_primary_sae_gain; exploratory_* are not Lepori metrics"
    chan["feature_tag"] = FEATURE_TAG
    chan["source_results_csv"] = str(results_csv)
    chan.to_csv(ap.SAE_PAPER_CHANNEL_TABLE, index=False)
    print(f"Saved {len(chan)} rows -> {ap.SAE_PAPER_CHANNEL_TABLE}")

    summary["feature_tag"] = FEATURE_TAG
    summary["source_results_csv"] = str(results_csv)
    summary.to_csv(ap.SAE_PAPER_SUMMARY_TABLE, index=False)
    print(f"Saved -> {ap.SAE_PAPER_SUMMARY_TABLE}")

    stats["feature_tag"] = FEATURE_TAG
    stats["source_results_csv"] = str(results_csv)
    stats.to_csv(ap.SAE_PAPER_STATS_TABLE, index=False)
    print(f"Saved -> {ap.SAE_PAPER_STATS_TABLE}")

    write_paper_report(
        d, summary, stats, cov, ap.SAE_PAPER_REPORT, results_csv, args.n_perm,
    )

    if not args.skip_figures:
        print("Figures:")
        fig_full_vs_surprisal(d, ap.SAE_FIGURES)
        fig_sae_gain_by_group(
            summary, ap.SAE_FIGURES, "task_group",
            "paper_aligned_sae_gain_by_task_group.png",
            "Paper-aligned SAE gain by exclusive task group",
        )
        fig_sae_gain_by_group(
            summary, ap.SAE_FIGURES, "region",
            "paper_aligned_sae_gain_by_region.png",
            "Paper-aligned SAE gain by region (is_lang electrodes)",
        )
        fig_frac_positive_descriptive(summary, ap.SAE_FIGURES)


if __name__ == "__main__":
    main()
