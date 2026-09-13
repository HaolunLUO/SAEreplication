#!/usr/bin/env python3
"""
sparse_encoding_plot_kernels.py
================================
Publication-oriented temporal kernel figures for Augmented Sparse Encoding.

Reads ``sparse_encoding_kernels{suffix}.csv`` from the deconvolution stage and
writes:

1. Grand-average kernels by anatomical bucket
2. Electrode × time heatmap sorted by peak lag
3. Surprisal vs SAE kernel overlay
4. Anterior (frontal/MFG) vs posterior (temporal) kernels

Example
-------
    python -m sparse_encoding.sparse_encoding_plot_kernels \\
        --kernels_csv .../sparse_encoding_kernels_sae_qwen35_4b_mat_l15_lang.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_lag_by_region import assign_anat_bucket
from sparse_encoding.sparse_encoding_summary import language_mask, merge_taxonomy
from sparse_encoding.temporal_metrics import kernel_shape_table


def _merge_anat(kdf: pd.DataFrame) -> pd.DataFrame:
    tax = merge_taxonomy(kdf[["subject", "channel"]].drop_duplicates())
    out = kdf.merge(tax, on=["subject", "channel"], how="left")
    if "region" not in out.columns:
        out["region"] = ""
    out["anat_bucket"] = out["region"].map(assign_anat_bucket)
    return out


def fig_grand_average(kdf: pd.DataFrame, outdir: Path, suffix: str) -> None:
    buckets = [b for b in ("temporal", "left_MFG", "frontal", "other")
               if b in set(kdf["anat_bucket"])]
    if not buckets:
        buckets = sorted(kdf["anat_bucket"].dropna().unique())
    n = max(1, len(buckets))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, bucket in zip(axes, buckets):
        sub = kdf.loc[kdf["anat_bucket"] == bucket]
        if sub.empty:
            ax.set_visible(False)
            continue
        # Subject means then mean across subjects.
        for col, label, color in (
            ("kernel_surprisal", "surprisal", "#d62728"),
            ("kernel_sae_l2", "SAE L2", "#1f77b4"),
            ("kernel_nuisance_l2", "nuisance L2", "#7f7f7f"),
        ):
            if col not in sub.columns:
                continue
            subj = (sub.groupby(["subject", "lag_ms"])[col].mean()
                      .groupby("lag_ms"))
            mu = subj.mean()
            sem = subj.sem()
            ax.plot(mu.index, mu.values, label=label, color=color)
            ax.fill_between(mu.index, mu - sem, mu + sem, alpha=0.2, color=color)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(bucket)
        ax.set_xlabel("lag (ms)")
        ax.set_ylabel("kernel weight")
        ax.legend(fontsize=8)
    fig.suptitle("Grand-average temporal kernels by region", y=1.02)
    fig.tight_layout()
    p = outdir / f"kernel_grand_average{suffix}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def fig_heatmap(kdf: pd.DataFrame, outdir: Path, suffix: str,
                value_col: str = "kernel_sae_l2") -> None:
    if value_col not in kdf.columns:
        return
    # One row per electrode: pivot lag → value, sort by argmax lag.
    piv = (kdf.pivot_table(
        index=["subject", "channel"], columns="lag_ms",
        values=value_col, aggfunc="mean"))
    if piv.empty:
        return
    peak_idx = np.nanargmax(piv.to_numpy(), axis=1)
    lags = piv.columns.to_numpy(dtype=float)
    order = np.argsort(lags[peak_idx])
    mat = piv.to_numpy()[order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.04 * mat.shape[0] + 2)))
    im = ax.imshow(
        mat, aspect="auto", interpolation="nearest",
        extent=[lags.min(), lags.max(), mat.shape[0], 0],
        cmap="viridis",
    )
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("electrodes (sorted by peak lag)")
    ax.set_title(f"Kernel heatmap — {value_col}")
    fig.colorbar(im, ax=ax, shrink=0.7, label=value_col)
    fig.tight_layout()
    p = outdir / f"kernel_heatmap{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_surprisal_vs_sae(kdf: pd.DataFrame, outdir: Path, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label, color in (
        ("kernel_surprisal", "surprisal", "#d62728"),
        ("kernel_sae_l2", "SAE L2", "#1f77b4"),
    ):
        if col not in kdf.columns:
            continue
        subj = (kdf.groupby(["subject", "lag_ms"])[col].mean()
                  .groupby("lag_ms"))
        mu = subj.mean()
        sem = subj.sem()
        ax.plot(mu.index, mu.values, label=label, color=color, lw=2)
        ax.fill_between(mu.index, mu - sem, mu + sem, alpha=0.2, color=color)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("kernel weight (subject-mean)")
    ax.set_title("Surprisal vs SAE temporal profiles")
    ax.legend()
    fig.tight_layout()
    p = outdir / f"kernel_surprisal_vs_sae{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_anterior_posterior(kdf: pd.DataFrame, outdir: Path, suffix: str) -> None:
    groups = {
        "posterior (temporal)": kdf["anat_bucket"].eq("temporal"),
        "anterior (frontal/MFG)": kdf["anat_bucket"].isin(("frontal", "left_MFG")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (title, mask) in zip(axes, groups.items()):
        sub = kdf.loc[mask]
        if sub.empty:
            ax.set_title(f"{title}\n(n=0)")
            continue
        for col, label, color in (
            ("kernel_surprisal", "surprisal", "#d62728"),
            ("kernel_sae_l2", "SAE L2", "#1f77b4"),
        ):
            if col not in sub.columns:
                continue
            subj = (sub.groupby(["subject", "lag_ms"])[col].mean()
                      .groupby("lag_ms"))
            mu = subj.mean()
            sem = subj.sem()
            ax.plot(mu.index, mu.values, label=label, color=color)
            ax.fill_between(mu.index, mu - sem, mu + sem, alpha=0.2, color=color)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("lag (ms)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("kernel weight")
    fig.suptitle("Anterior vs posterior language kernels", y=1.02)
    fig.tight_layout()
    p = outdir / f"kernel_anterior_posterior{suffix}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels_csv", required=True)
    p.add_argument("--out_suffix", default="")
    p.add_argument("--lang_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write_shapes", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    kdf = pd.read_csv(args.kernels_csv)
    if kdf.empty:
        print("Empty kernels CSV.")
        return
    kdf = _merge_anat(kdf)
    if args.lang_only and "is_lang" in kdf.columns:
        elec = kdf[["subject", "channel", "is_lang"]].drop_duplicates()
        keep = set(
            zip(elec.loc[language_mask(elec), "subject"],
                elec.loc[language_mask(elec), "channel"])
        )
        kdf = kdf.loc[
            [(s, c) in keep for s, c in zip(kdf["subject"], kdf["channel"])]
        ].copy()

    suffix = args.out_suffix
    outdir = ap.SAE_FIGURES
    print("Kernel figures:")
    fig_grand_average(kdf, outdir, suffix)
    fig_heatmap(kdf, outdir, suffix)
    fig_surprisal_vs_sae(kdf, outdir, suffix)
    fig_anterior_posterior(kdf, outdir, suffix)

    if args.write_shapes:
        shapes = kernel_shape_table(kdf)
        if not shapes.empty:
            shapes = merge_taxonomy(shapes)
            if "region" in shapes.columns:
                shapes["anat_bucket"] = shapes["region"].map(assign_anat_bucket)
            sp = ap.SAE_TABLES / f"sparse_encoding_kernel_shapes{suffix}.csv"
            shapes.to_csv(sp, index=False)
            print(f"Saved kernel shapes -> {sp}")


if __name__ == "__main__":
    main()
