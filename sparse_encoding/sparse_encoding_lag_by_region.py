#!/usr/bin/env python3
"""
sparse_encoding_lag_by_region.py
================================
Region-specific temporal dynamics from lag-screen peak tables.

Tests whether anatomical regions differ in peak lag (e.g. temporal early vs
frontal late) and whether SAE vs surprisal timing differs within region.

Inputs (from ``sparse_encoding_lag_screen.py``):
    tables/sparse_encoding_lag_screen_peaks{suffix}.csv
    — or long-form lag CSV, from which peaks are recomputed.

Outputs
-------
tables/sparse_encoding_lag_by_region{suffix}.csv
tables/sparse_encoding_lag_region_pairwise{suffix}.csv
reports/sparse_encoding_lag_by_region{suffix}.txt
figures/lag_by_region{suffix}.png

Example
-------
    python -m sparse_encoding.sparse_encoding_lag_by_region \\
        --peaks_csv .../sparse_encoding_lag_screen_peaks_sae_qwen35_4b_mat_l15_lang.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_summary import (
    LEFT_MFG_REGION, bootstrap_subject_means, language_mask, merge_taxonomy,
)
from sparse_encoding.temporal_metrics import (
    electrode_lag_peaks_from_long,
    subject_mean_sign_flip_p,
)

TEMPORAL_RE = r"Temp|STG|MTG|ITG|STS|Heschl|temporal|planum"
FRONTAL_RE = r"IFG|MFG|SFG|frontal|Precentral|pars|opercul"
DEEP_RE = r"Hippocamp|Amygdala|Insula|Putamen|Caudate|Thalamus|Pallidum"


def assign_anat_bucket(region: str) -> str:
    r = str(region) if pd.notna(region) else ""
    if LEFT_MFG_REGION in r or ("Middle frontal" in r and "Left" in r):
        return "left_MFG"
    if pd.Series([r]).str.contains(TEMPORAL_RE, case=False, na=False).iloc[0]:
        return "temporal"
    if pd.Series([r]).str.contains(FRONTAL_RE, case=False, na=False).iloc[0]:
        return "frontal"
    if pd.Series([r]).str.contains(DEEP_RE, case=False, na=False).iloc[0]:
        return "deep"
    return "other"


def load_peaks(peaks_csv: Optional[Path], long_csv: Optional[Path]) -> pd.DataFrame:
    if peaks_csv is not None and peaks_csv.exists():
        return pd.read_csv(peaks_csv)
    if long_csv is not None and long_csv.exists():
        long_df = pd.read_csv(long_csv)
        return electrode_lag_peaks_from_long(long_df)
    raise FileNotFoundError(
        "Need --peaks_csv or --long_csv pointing at lag-screen outputs."
    )


def region_summary(peaks: pd.DataFrame, lag_col: str = "full_peak_lag_ms") -> pd.DataFrame:
    rows = []
    for bucket, g in peaks.groupby("anat_bucket", dropna=False):
        subj = g.groupby("subject")[lag_col].mean()
        mu, lo, hi, n = bootstrap_subject_means(
            subj.rename(lag_col).reset_index(), lag_col)
        rows.append({
            "anat_bucket": bucket,
            "n_electrodes": int(len(g)),
            "n_subjects": int(n),
            "median_peak_lag_ms": float(np.nanmedian(g[lag_col])),
            "subject_mean_peak_lag_ms": mu,
            "ci_lo": lo,
            "ci_hi": hi,
            "median_fwhm_ms": (
                float(np.nanmedian(g["full_fwhm_ms"]))
                if "full_fwhm_ms" in g.columns else np.nan
            ),
            "mean_lag_diff_full_vs_surprisal_ms": (
                float(g.groupby("subject")["lag_diff_full_vs_surprisal_ms"]
                        .mean().mean())
                if "lag_diff_full_vs_surprisal_ms" in g.columns else np.nan
            ),
        })
    return pd.DataFrame(rows)


def pairwise_lag_tests(
    peaks: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    lag_col: str = "full_peak_lag_ms",
    n_perm: int = 2000,
    seed: int = 19,
) -> pd.DataFrame:
    """Within-subject mean lag(A) − lag(B); sign-flip p on subject diffs."""
    rows = []
    for a, b in pairs:
        ga = peaks.loc[peaks["anat_bucket"] == a]
        gb = peaks.loc[peaks["anat_bucket"] == b]
        if ga.empty or gb.empty:
            rows.append({
                "group_a": a, "group_b": b, "n_subjects": 0,
                "mean_diff_ms": np.nan, "perm_p": np.nan,
            })
            continue
        sa = ga.groupby("subject")[lag_col].mean()
        sb = gb.groupby("subject")[lag_col].mean()
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            # Pooled subjects that have either region — less ideal.
            diffs = []
            for subj in sorted(set(sa.index) | set(sb.index)):
                if subj in sa.index and subj in sb.index:
                    diffs.append(float(sa[subj] - sb[subj]))
            vals = np.asarray(diffs, dtype=float)
        else:
            vals = (sa.loc[common] - sb.loc[common]).to_numpy(dtype=float)
        mu, p = subject_mean_sign_flip_p(vals, n_perm=n_perm, seed=seed)
        rows.append({
            "group_a": a, "group_b": b,
            "n_subjects": int(np.isfinite(vals).sum()),
            "mean_diff_ms": mu,
            "perm_p": p,
            "lag_col": lag_col,
        })
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    peaks: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "=" * 70,
        "LAG DYNAMICS BY ANATOMICAL REGION",
        "=" * 70,
        "",
        "Peak lag = argmax of held-out Fisher-z R across 0–800 ms (50 ms steps).",
        "Inferential unit = subject. Between-region tests: sign-flip on "
        "within-subject region-mean differences.",
        "",
    ]
    if not summary.empty:
        lines.append("-" * 70)
        lines.append("PER-REGION SUBJECT-MEAN PEAK LAG")
        lines.append("-" * 70)
        lines.append(summary.round(2).to_string(index=False))
        lines.append("")

    if not pairwise.empty:
        lines.append("-" * 70)
        lines.append("PAIRWISE REGION CONTRASTS (A − B)")
        lines.append("-" * 70)
        lines.append(pairwise.round(4).to_string(index=False))
        lines.append("")
        for _, r in pairwise.iterrows():
            if not np.isfinite(r.get("mean_diff_ms", np.nan)):
                continue
            direction = (
                "earlier" if r["mean_diff_ms"] < 0
                else ("later" if r["mean_diff_ms"] > 0 else "same")
            )
            lines.append(
                f"  {r['group_a']} vs {r['group_b']}: "
                f"{r['group_a']} peaks {direction} by "
                f"{abs(r['mean_diff_ms']):.0f} ms (p={r['perm_p']:.3f})"
            )
        lines.append("")

    if "lag_diff_full_vs_surprisal_ms" in peaks.columns:
        lines.append("-" * 70)
        lines.append("SAE TEMPORAL SIGNATURE (full − surprisal peak lag)")
        lines.append("-" * 70)
        subj = peaks.groupby("subject")["lag_diff_full_vs_surprisal_ms"].mean()
        mu, p = subject_mean_sign_flip_p(subj.to_numpy())
        lines.append(
            f"  subject-mean lag_diff: {mu:+.1f} ms  p={p:.3f}  "
            f"n_subjects={subj.notna().sum()}"
        )
        interp = (
            "SAE/full peaks later than surprisal" if np.isfinite(mu) and mu > 0
            else ("SAE/full peaks earlier than surprisal"
                  if np.isfinite(mu) and mu < 0
                  else "no clear timing difference")
        )
        lines.append(f"  interpretation: {interp}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def fig_lag_by_region(summary: pd.DataFrame, outdir: Path, suffix: str) -> None:
    if summary.empty:
        return
    order = ["temporal", "left_MFG", "frontal", "deep", "other"]
    s = summary.set_index("anat_bucket").reindex(
        [b for b in order if b in set(summary["anat_bucket"])]
    ).dropna(subset=["subject_mean_peak_lag_ms"])
    if s.empty:
        s = summary.set_index("anat_bucket")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(s))
    y = s["subject_mean_peak_lag_ms"].to_numpy()
    lo = s["ci_lo"].to_numpy()
    hi = s["ci_hi"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])
    ax.bar(x, y, color="#4C78A8", alpha=0.85)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(list(s.index), rotation=20, ha="right")
    ax.set_ylabel("Peak lag (ms), subject-mean")
    ax.set_xlabel("Anatomical bucket")
    ax.set_title("Lag-resolved peak timing by region")
    fig.tight_layout()
    p = outdir / f"lag_by_region{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peaks_csv", default=None)
    p.add_argument("--long_csv", default=None)
    p.add_argument("--out_suffix", default="")
    p.add_argument("--n_perm", type=int, default=2000)
    p.add_argument("--lang_only", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    peaks_csv = Path(args.peaks_csv) if args.peaks_csv else None
    long_csv = Path(args.long_csv) if args.long_csv else None
    peaks = load_peaks(peaks_csv, long_csv)
    peaks = merge_taxonomy(peaks)
    if args.lang_only and "is_lang" in peaks.columns:
        peaks = peaks.loc[language_mask(peaks)].copy()
    if "region" not in peaks.columns:
        peaks["region"] = ""
    peaks["anat_bucket"] = peaks["region"].map(assign_anat_bucket)

    suffix = args.out_suffix
    summary = region_summary(peaks)
    pairwise = pairwise_lag_tests(
        peaks,
        pairs=[
            ("temporal", "frontal"),
            ("temporal", "left_MFG"),
            ("left_MFG", "frontal"),
            ("deep", "temporal"),
        ],
        n_perm=args.n_perm,
    )
    summary_path = ap.SAE_TABLES / f"sparse_encoding_lag_by_region{suffix}.csv"
    pair_path = ap.SAE_TABLES / f"sparse_encoding_lag_region_pairwise{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pair_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {pair_path}")

    write_report(
        summary, pairwise, peaks,
        ap.SAE_REPORTS / f"sparse_encoding_lag_by_region{suffix}.txt",
    )
    fig_lag_by_region(summary, ap.SAE_FIGURES, suffix)


if __name__ == "__main__":
    main()
