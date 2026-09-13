#!/usr/bin/env python3
"""Compare residual vs SAE encodings when both use LASSO→Ridge.

Paired on language-localizer channels (``is_lang``). Uses the same
full / content / surprisal_only modes as sparse_encoding_regression.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import core.analysis_paths as ap

_TABLES = getattr(ap, "SAE_TABLES", None) or ap.SAE_TABLES
_FIGURES = getattr(ap, "SAE_FIGURES", None) or ap.SAE_FIGURES
SAE_CSV = _TABLES / "sparse_encoding_results_sae_qwen35_4b_mat_l15_lang.csv"
RESID_CSV = _TABLES / "sparse_encoding_results_sae_qwen35_4b_mat_l15_resid_lasso_lang.csv"
STAT = "full__R_fisher"
CONTENT = "content__R_fisher"
SURP = "surprisal_only__R_fisher"


def _paired_report(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    d = a - b
    n = int(np.isfinite(d).sum())
    mask = np.isfinite(a) & np.isfinite(b)
    t_p = ttest_rel(a[mask], b[mask]).pvalue if mask.sum() > 1 else np.nan
    try:
        w_p = wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        w_p = np.nan
    return {
        "contrast": label,
        "n": n,
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "mean_delta_a_minus_b": float(np.nanmean(d)),
        "median_delta": float(np.nanmedian(d)),
        "a_wins": int(np.nansum(a > b)),
        "b_wins": int(np.nansum(b > a)),
        "ties": int(np.nansum(a == b)),
        "ttest_p": float(t_p) if np.isfinite(t_p) else np.nan,
        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
    }


def main() -> None:
    sae = pd.read_csv(SAE_CSV)
    resid = pd.read_csv(RESID_CSV)
    keys = ["subject", "channel"]
    m = resid.merge(
        sae[keys + [STAT, CONTENT, SURP]],
        on=keys, how="inner", suffixes=("_resid", "_sae"),
    )
    # after merge: resid cols keep original names; sae get _sae if collision.
    # residual file uses same STAT names, so suffixes apply to overlap.
    # pandas suffixes: left=resid, right=sae for overlapping non-key cols.
    r_full = m[f"{STAT}_resid"] if f"{STAT}_resid" in m.columns else m[STAT]
    s_full = m[f"{STAT}_sae"]
    r_cont = m[f"{CONTENT}_resid"] if f"{CONTENT}_resid" in m.columns else m[CONTENT]
    s_cont = m[f"{CONTENT}_sae"]

    rows = [
        _paired_report(r_full.to_numpy(), s_full.to_numpy(),
                       "resid_full vs SAE_full"),
        _paired_report(r_cont.to_numpy(), s_cont.to_numpy(),
                       "resid_content vs SAE_content"),
    ]
    # subject means (paper-style)
    subj = m.groupby("subject").agg(
        resid_full=(f"{STAT}_resid" if f"{STAT}_resid" in m.columns else STAT, "mean"),
        sae_full=(f"{STAT}_sae", "mean"),
        resid_content=(f"{CONTENT}_resid" if f"{CONTENT}_resid" in m.columns else CONTENT, "mean"),
        sae_content=(f"{CONTENT}_sae", "mean"),
        n=("channel", "size"),
    )
    subj["delta_full"] = subj["resid_full"] - subj["sae_full"]
    subj["delta_content"] = subj["resid_content"] - subj["sae_content"]
    rows.append(_paired_report(
        subj["resid_full"].to_numpy(), subj["sae_full"].to_numpy(),
        "subject-mean resid_full vs SAE_full"))
    rows.append(_paired_report(
        subj["resid_content"].to_numpy(), subj["sae_content"].to_numpy(),
        "subject-mean resid_content vs SAE_content"))

    summary = pd.DataFrame(rows)
    out_csv = _TABLES / "resid_vs_sae_lasso_lang.csv"
    subj_csv = _TABLES / "resid_vs_sae_lasso_lang_subjects.csv"
    fig_path = _FIGURES / "resid_vs_sae_lasso_lang.png"
    summary.to_csv(out_csv, index=False)
    subj.to_csv(subj_csv)
    print(summary.to_string(index=False))
    print(f"\nPaired channels: {len(m)}")
    print(subj.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, x, y, title in (
        (axes[0], s_full, r_full, "Full (features + surprisal)"),
        (axes[1], s_cont, r_cont, "Content (features only)"),
    ):
        ax.scatter(x, y, s=18, alpha=0.7, c="#1f77b4", edgecolors="none")
        lims = [
            min(np.nanmin(x), np.nanmin(y)),
            max(np.nanmax(x), np.nanmax(y)),
        ]
        pad = 0.05 * (lims[1] - lims[0] + 1e-6)
        lo, hi = lims[0] - pad, lims[1] + pad
        ax.plot([lo, hi], [lo, hi], color="#888", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("SAE LASSO→Ridge  Fisher-z r")
        ax.set_ylabel("Residual LASSO→Ridge  Fisher-z r")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        wins = int(np.nansum(y > x))
        ax.text(
            0.04, 0.96,
            f"n={len(m)}\nresid>SAE: {wins}/{len(m)}\n"
            f"Δ={np.nanmean(y-x):+.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
        )
    fig.suptitle(
        "Language-localizer channels · same LASSO→Ridge encoder · 300 ms",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
