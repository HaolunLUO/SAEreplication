#!/usr/bin/env python3
"""
sparse_encoding_sae_sig_overlap.py
==================================
Define SAE encoding-significant channels from feature-shuffle nulls and test
overlap / enrichment with the MITSWJN language localizer (``is_lang``).

Pipeline
--------
1. Observed single-lag regression (``sparse_encoding_regression.py``, shuffle none).
2. Multi-draw global feature-shuffle nulls (``--null_seed_offset`` 1..N).
3. Empirical one-sided p per channel: P(null R >= obs R).
4. BH-FDR across all implanted channels -> ``sig_sae``.
5. Contingency vs ``is_lang`` (+ ``sig_glove`` reference): Fisher, Jaccard, phi.

Examples
--------
    # Run 50 null draws then overlap (slow; caches CSVs under sae_null_perm/).
    python -m sparse_encoding.sparse_encoding_sae_sig_overlap \\
        --feature_tag sae_qwen35_4b_mat_l15 --run_nulls 50

    # Overlap only from cached nulls + existing cohort table.
    python -m sparse_encoding.sparse_encoding_sae_sig_overlap \\
        --feature_tag sae_qwen35_4b_mat_l15 --n_perm 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, t as student_t

import core.analysis_paths as ap

PY = sys.executable
DEFAULT_TAG = "sae_qwen35_4b_mat_l15"
STAT_COL = "full__R_fisher"
NULL_DIR = ap.SAE_TABLES / "sae_null_perm"


def _tag_paths(feature_tag: str) -> Tuple[Path, Path, Path, Path]:
    obs = ap.SAE_TABLES / f"sparse_encoding_results_{feature_tag}_cohort.csv"
    overlap = ap.SAE_TABLES / f"sae_sig_lang_overlap_{feature_tag}.csv"
    electrode = ap.SAE_TABLES / f"sae_sig_electrodes_{feature_tag}.csv"
    fig = ap.SAE_FIGURES / f"sae_sig_lang_overlap_{feature_tag}.png"
    return obs, overlap, electrode, fig


def _null_csv(null_dir: Path, k: int) -> Path:
    return null_dir / f"sae_null_perm_{k:03d}.csv"


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def run_null_permutations(
    feature_tag: str,
    n_perm: int,
    null_dir: Path,
    start: int = 1,
    overwrite: bool = False,
) -> None:
    null_dir.mkdir(parents=True, exist_ok=True)
    for k in range(start, n_perm + 1):
        out = _null_csv(null_dir, k)
        if out.exists() and not overwrite:
            print(f"[skip] null {k}/{n_perm} exists -> {out.name}", flush=True)
            continue
        cmd = [
            PY,
            "-m",
            "sparse_encoding.sparse_encoding_regression",
            "--feature_tag",
            feature_tag,
            "--modes",
            "full",
            "--shuffle_control",
            "global",
            "--null_seed_offset",
            str(k),
            "--out",
            str(out),
            "--overwrite",
        ]
        print(f"[null] {k}/{n_perm}: {' '.join(cmd[-8:])}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ap.REPO_ROOT))


def load_null_matrix(
    obs: pd.DataFrame,
    null_dir: Path,
    n_perm: int,
    stat_col: str,
) -> Tuple[np.ndarray, List[int]]:
    """Return (n_perm_found x n_channels) null R matrix aligned to obs rows."""
    keys = obs[["subject", "channel"]].astype(str).apply(
        lambda r: f"{r['subject']}|{r['channel']}", axis=1)
    key_index = {k: i for i, k in enumerate(keys)}

    null_rows: List[np.ndarray] = []
    used: List[int] = []
    for k in range(1, n_perm + 1):
        path = _null_csv(null_dir, k)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if stat_col not in df.columns:
            raise KeyError(f"{path} missing {stat_col}")
        if len(df) < len(obs):
            print(f"[skip] null {k}: incomplete ({len(df)}/{len(obs)} rows)", flush=True)
            continue
        vec = np.full(len(obs), np.nan, dtype=float)
        for _, row in df.iterrows():
            key = f"{row['subject']}|{row['channel']}"
            if key in key_index:
                vec[key_index[key]] = float(row[stat_col])
        if np.isfinite(vec).sum() < len(obs):
            print(f"[skip] null {k}: only {np.isfinite(vec).sum()}/{len(obs)} "
                  f"channels matched obs", flush=True)
            continue
        null_rows.append(vec)
        used.append(k)
    if not null_rows:
        raise FileNotFoundError(
            f"No null CSVs in {null_dir} (expected sae_null_perm_001..)")
    return np.vstack(null_rows), used


def empirical_pvalues(obs_stat: np.ndarray, null_mat: np.ndarray) -> np.ndarray:
    """One-sided: fraction of null draws with R >= observed."""
    obs = np.asarray(obs_stat, dtype=float)
    null = np.asarray(null_mat, dtype=float)
    n_perm = null.shape[0]
    ge = (null >= obs[np.newaxis, :]).sum(axis=0)
    return (1.0 + ge) / (n_perm + 1.0)


def sae_category(sig: pd.Series, lang: pd.Series) -> pd.Series:
    sig = sig.fillna(False).astype(bool)
    lang = lang.fillna(False).astype(bool)
    out = pd.Series("neither", index=sig.index, dtype=object)
    out[sig & lang] = "both"
    out[sig & ~lang] = "enc_only"
    out[~sig & lang] = "loc_only"
    return out


def contingency_counts(sig: pd.Series, lang: pd.Series) -> Dict[str, int]:
    sig = sig.fillna(False).astype(bool)
    lang = lang.fillna(False).astype(bool)
    both = int((sig & lang).sum())
    enc_only = int((sig & ~lang).sum())
    loc_only = int((~sig & lang).sum())
    neither = int((~sig & ~lang).sum())
    n = int(len(sig))
    return {
        "n_total": n,
        "n_enc_sig": int(sig.sum()),
        "n_localizer": int(lang.sum()),
        "n_both": both,
        "n_enc_only": enc_only,
        "n_loc_only": loc_only,
        "n_neither": neither,
    }


def phi_coeff(a: int, b: int, c: int, d: int) -> float:
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den <= 0:
        return float("nan")
    return (a * d - b * c) / np.sqrt(den)


def overlap_metrics(
    sig: pd.Series,
    lang: pd.Series,
    label: str,
    obs: pd.DataFrame,
) -> Dict[str, float]:
    c = contingency_counts(sig, lang)
    a, b, c_only, d = c["n_both"], c["n_enc_only"], c["n_loc_only"], c["n_neither"]
    enc, loc = c["n_enc_sig"], c["n_localizer"]
    union = enc + loc - a
    jaccard = a / union if union else float("nan")
    dice = 2 * a / (enc + loc) if (enc + loc) else float("nan")
    sens = a / loc if loc else float("nan")
    spec = d / (d + b) if (d + b) else float("nan")
    ppv = a / enc if enc else float("nan")
    npv = d / (d + c_only) if (d + c_only) else float("nan")
    phi = phi_coeff(a, b, c_only, d)
    table = np.array([[a, b], [c_only, d]])
    _, p_two = fisher_exact(table, alternative="two-sided")
    _, p_greater = fisher_exact(table, alternative="greater")
    odds = (a * d) / (b * c_only) if b and c_only else float("inf")

    # Hypergeometric: P(>=a successes) drawing enc from pool with loc positives.
    n = c["n_total"]
    hyp_p = hypergeom.sf(a - 1, n, loc, enc) if enc and loc else float("nan")

    subj_j = []
    if "subject" in obs.columns:
        tmp = obs.assign(_sig=sig.values, _lang=lang.values)
        for _, g in tmp.groupby("subject"):
            cc = contingency_counts(g["_sig"], g["_lang"])
            u = cc["n_enc_sig"] + cc["n_localizer"] - cc["n_both"]
            if u > 0:
                subj_j.append(cc["n_both"] / u)
    mean_subj_j = float(np.mean(subj_j)) if subj_j else float("nan")
    sem_subj_j = float(np.std(subj_j, ddof=1) / np.sqrt(len(subj_j))) if len(subj_j) > 1 else float("nan")

    return {
        "feature": label,
        **c,
        "pct_enc_sig": 100.0 * enc / c["n_total"],
        "pct_localizer": 100.0 * loc / c["n_total"],
        "pct_both": 100.0 * a / c["n_total"],
        "jaccard": jaccard,
        "dice": dice,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "phi": phi,
        "fisher_odds_ratio": odds,
        "fisher_p_twosided": p_two,
        "fisher_p_greater": p_greater,
        "hypergeom_p_greater": hyp_p,
        "n_subjects": obs["subject"].nunique() if "subject" in obs.columns else float("nan"),
        "mean_subj_jaccard": mean_subj_j,
        "sem_subj_jaccard": sem_subj_j,
    }


def plot_overlap(summary: pd.DataFrame, fig_path: Path, feature_tag: str) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary.set_index("feature")
    cats = ["both", "enc_only", "loc_only", "neither"]
    labels = ["Both", "SAE only", "Lang only", "Neither"]
    x = np.arange(len(rows))
    width = 0.18
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (cat, lab, col) in enumerate(zip(cats, labels, colors)):
        colname = f"n_{cat}" if cat != "both" else "n_both"
        if cat == "enc_only":
            colname = "n_enc_only"
        elif cat == "loc_only":
            colname = "n_loc_only"
        vals = [rows.loc[f, colname] for f in rows.index]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(rows.index.tolist())
    ax.set_ylabel("Electrodes (n)")
    ax.set_title(f"Encoding significance × language localizer ({feature_tag})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default=DEFAULT_TAG)
    p.add_argument("--stat_col", default=STAT_COL,
                   help="Observed / null statistic (default full__R_fisher).")
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument(
        "--sig_method",
        choices=("empirical_bh", "empirical_pthr", "beat_all", "tfit_bh"),
        default="empirical_bh",
        help="How to turn permutation / null statistics into sig_sae.",
    )
    p.add_argument(
        "--p_thr",
        type=float,
        default=0.1,
        help="Uncorrected empirical p-value threshold when sig_method=empirical_pthr.",
    )
    p.add_argument("--n_perm", type=int, default=50,
                   help="Number of null draws to load / target when running nulls.")
    p.add_argument("--run_nulls", type=int, default=0,
                   help="If >0, run this many global shuffle null regressions first.")
    p.add_argument("--null_start", type=int, default=1)
    p.add_argument("--overwrite_nulls", action="store_true")
    p.add_argument("--null_dir", type=Path, default=NULL_DIR)
    p.add_argument("--obs_csv", type=Path, default=None)
    p.add_argument("--out_overlap", type=Path, default=None)
    p.add_argument("--out_electrodes", type=Path, default=None)
    p.add_argument("--out_fig", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ap.ensure_pipeline_dirs()
    obs_path, overlap_path, elec_path, fig_path = _tag_paths(args.feature_tag)
    if args.obs_csv:
        obs_path = args.obs_csv
    if args.out_overlap:
        overlap_path = args.out_overlap
    if args.out_electrodes:
        elec_path = args.out_electrodes
    if args.out_fig:
        fig_path = args.out_fig

    if args.sig_method != "empirical_bh" and args.out_overlap is None:
        suffix = args.sig_method
        if args.sig_method == "empirical_pthr":
            suffix = f"{suffix}_pthr{args.p_thr:g}"
        overlap_path = ap.SAE_TABLES / f"sae_sig_lang_overlap_{args.feature_tag}_{suffix}.csv"
        elec_path = ap.SAE_TABLES / f"sae_sig_electrodes_{args.feature_tag}_{suffix}.csv"
        fig_path = ap.SAE_FIGURES / f"sae_sig_lang_overlap_{args.feature_tag}_{suffix}.png"

    n_perm = max(args.run_nulls, args.n_perm)
    if args.run_nulls > 0:
        run_null_permutations(
            args.feature_tag,
            args.run_nulls,
            args.null_dir,
            start=args.null_start,
            overwrite=args.overwrite_nulls,
        )

    if not obs_path.exists():
        raise FileNotFoundError(f"Observed cohort CSV not found: {obs_path}")
    obs = pd.read_csv(obs_path)
    if args.stat_col not in obs.columns:
        raise KeyError(f"{obs_path} missing {args.stat_col}")

    null_mat, used = load_null_matrix(obs, args.null_dir, n_perm, args.stat_col)
    print(f"Loaded {null_mat.shape[0]} null draws (seeds {used[0]}..{used[-1]})", flush=True)

    obs_r = obs[args.stat_col].to_numpy()
    pvals = empirical_pvalues(obs_r, null_mat)
    padj = np.full_like(pvals, np.nan, dtype=float)
    sig_sae = None

    if args.sig_method == "empirical_bh":
        padj = _bh_fdr(pvals)
        sig_sae = pd.Series(padj <= args.fdr_alpha, index=obs.index)
    elif args.sig_method == "empirical_pthr":
        sig_sae = pd.Series(pvals <= args.p_thr, index=obs.index)
    elif args.sig_method == "beat_all":
        sig_sae = pd.Series(
            np.all(null_mat < obs_r[np.newaxis, :], axis=0),
            index=obs.index,
        )
    elif args.sig_method == "tfit_bh":
        n_draws = null_mat.shape[0]
        null_mean = null_mat.mean(axis=0)
        null_std = null_mat.std(axis=0, ddof=1)
        se = np.where(null_std == 0, np.nan, null_std / np.sqrt(n_draws))
        tstat = (obs_r - null_mean) / se
        p_tfit = student_t.sf(tstat, df=n_draws - 1)
        padj = _bh_fdr(p_tfit)
        sig_sae = pd.Series(padj <= args.fdr_alpha, index=obs.index)
    else:
        raise ValueError(f"Unknown sig_method={args.sig_method}")

    tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
    lang_cols = [c for c in ("subject", "channel", "is_lang", "sig_glove", "category", "region")
                 if c in tax.columns]
    merged = obs.merge(tax[lang_cols], on=["subject", "channel"], how="left")
    merged["p_sae"] = pvals
    merged["padj_sae"] = padj
    merged["sig_sae"] = sig_sae.values
    merged["sae_category"] = sae_category(merged["sig_sae"], merged["is_lang"])

    summaries = [
        overlap_metrics(sig_sae, merged["is_lang"], f"sae_{args.sig_method}", merged),
    ]
    if "sig_glove" in merged.columns:
        summaries.append(
            overlap_metrics(
                merged["sig_glove"].fillna(False).astype(bool),
                merged["is_lang"],
                "sig_glove",
                merged,
            )
        )
    summary = pd.DataFrame(summaries)
    overlap_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(overlap_path, index=False)
    merged.to_csv(elec_path, index=False)
    plot_overlap(summary, fig_path, args.feature_tag)

    sae_row = summary.iloc[0]
    print("\n=== SAE encoding sig × is_lang ===", flush=True)
    print(f"Null draws: {null_mat.shape[0]} | FDR α={args.fdr_alpha}", flush=True)
    print(f"sig_sae: {int(sae_row['n_enc_sig'])} / {int(sae_row['n_total'])} "
          f"({sae_row['pct_enc_sig']:.1f}%)", flush=True)
    print(f"is_lang: {int(sae_row['n_localizer'])} ({sae_row['pct_localizer']:.1f}%)", flush=True)
    print(f"both: {int(sae_row['n_both'])} | enc_only: {int(sae_row['n_enc_only'])} | "
          f"loc_only: {int(sae_row['n_loc_only'])} | neither: {int(sae_row['n_neither'])}",
          flush=True)
    print(f"Jaccard={sae_row['jaccard']:.3f} | phi={sae_row['phi']:.3f} | "
          f"Fisher p (greater)={sae_row['fisher_p_greater']:.4g} | "
          f"hypergeom p={sae_row['hypergeom_p_greater']:.4g}", flush=True)
    if "sig_glove" in summary["feature"].values:
        g = summary.loc[summary["feature"] == "sig_glove"].iloc[0]
        print(f"\nReference sig_glove: n={int(g['n_enc_sig'])}, both={int(g['n_both'])}, "
              f"Jaccard={g['jaccard']:.3f}, Fisher p={g['fisher_p_greater']:.4g}", flush=True)
    print(f"\nWrote {overlap_path.name}, {elec_path.name}", flush=True)


if __name__ == "__main__":
    main()
