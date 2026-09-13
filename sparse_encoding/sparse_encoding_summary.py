#!/usr/bin/env python3
"""
sparse_encoding_summary.py
==========================
Summarize + visualize Augmented Sparse Encoding Model results from the
**single-lag baseline** (``sparse_encoding_regression.py``).

For primary temporal dynamics (peak lag, FWHM, kernels), see
``sparse_encoding_lag_screen.py``, ``sparse_encoding_deconvolution.py``, and
``README_TEMPORAL.md``. When sibling lag-peak / kernel-shape tables exist,
this report appends a TEMPORAL DYNAMICS SUMMARY block pointing at them.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap

MODES = ("full", "content", "surprisal_only")
MODE_COLORS = {"full": "#1f77b4", "content": "#2ca02c", "surprisal_only": "#d62728"}
EXPLORATORY_LANG_CATEGORIES = ("both", "enc_only")
LEFT_MFG_REGION = "Left Middle frontal gyrus"
N_BOOT = 1000
RNG_SEED = 19


def merge_taxonomy(res: pd.DataFrame) -> pd.DataFrame:
    """Attach region / category / hemisphere from the taxonomy table if present."""
    tax_path = ap.TAX_ELECTRODE_TABLE
    if not tax_path.exists():
        print(f"[note] taxonomy table not found ({tax_path}); "
              "region/category grouping disabled.")
        return res
    t = pd.read_csv(tax_path)
    keep = [c for c in ("subject", "channel", "region", "category",
                        "hemisphere", "is_lang") if c in t.columns]
    return res.merge(t[keep], on=["subject", "channel"], how="left")


def add_contrasts(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    full = "full__R_fisher_paired" if "full__R_fisher_paired" in res.columns else "full__R_fisher"
    content = "content__R_fisher_paired" if "content__R_fisher_paired" in res.columns else "content__R_fisher"
    surp = "surprisal_only__R_fisher_paired" if "surprisal_only__R_fisher_paired" in res.columns else "surprisal_only__R_fisher"
    # Paper-aligned primary contrast (Lepori et al.): full − surprisal_only.
    res["sae_contribution"] = res[full] - res[surp]
    res["paper_sae_gain"] = res["sae_contribution"]
    # Exploratory extension (NOT defined by Lepori et al.): full − content.
    res["surprisal_contribution"] = res[full] - res[content]
    res["exploratory_surprisal_contribution"] = res["surprisal_contribution"]
    # Also keep unpaired contrasts for reference.
    res["sae_contribution_unpaired"] = (
        res["full__R_fisher"] - res["surprisal_only__R_fisher"])
    res["surprisal_contribution_unpaired"] = (
        res["full__R_fisher"] - res["content__R_fisher"])
    return res


def language_mask(res: pd.DataFrame) -> pd.Series:
    """Paper-comparable language electrodes: independent localizer ``is_lang``.

    Taxonomy categories ``both`` / ``enc_only`` are *not* used here: ``enc_only``
    is encoding-significant but not localizer-positive.
    """
    if "is_lang" in res.columns:
        return res["is_lang"].fillna(False).astype(bool)
    # Fallback only when taxonomy lacks is_lang (legacy tables).
    if "category" in res.columns and res["category"].notna().any():
        print("[warn] is_lang missing; falling back to exploratory "
              f"categories {EXPLORATORY_LANG_CATEGORIES}")
        return res["category"].isin(EXPLORATORY_LANG_CATEGORIES)
    return pd.Series(False, index=res.index)


def exploratory_category_lang_mask(res: pd.DataFrame) -> pd.Series:
    """Non-replication exploratory mask: taxonomy both / enc_only."""
    if "category" not in res.columns:
        return pd.Series(False, index=res.index)
    return res["category"].isin(EXPLORATORY_LANG_CATEGORIES)


def left_mfg_mask(res: pd.DataFrame) -> pd.Series:
    """Localizer-positive left middle frontal gyrus electrodes."""
    lang = language_mask(res)
    if "region" not in res.columns:
        return pd.Series(False, index=res.index)
    mfg = res["region"].astype(str).eq(LEFT_MFG_REGION)
    if "hemisphere" in res.columns and res["hemisphere"].notna().any():
        left = res["hemisphere"].astype(str).str.lower().eq("left")
        # Prefer explicit hemisphere when present; region name already says Left.
        return lang & mfg & (left | res["hemisphere"].isna())
    return lang & mfg


def complete_fold_mask(res: pd.DataFrame) -> pd.Series:
    """Electrodes with all outer folds valid for every mode (paired)."""
    if "all_folds_paired" in res.columns:
        return res["all_folds_paired"].fillna(0).astype(bool)
    # Fallback: every mode reported the full fold count.
    cols = [f"{m}__n_folds_valid" for m in MODES if f"{m}__n_folds_valid" in res.columns]
    if not cols or "n_folds_total" not in res.columns:
        return pd.Series(True, index=res.index)
    ok = pd.Series(True, index=res.index)
    for c in cols:
        ok &= res[c].fillna(0) >= res["n_folds_total"].fillna(0)
    return ok


def _r_col(res: pd.DataFrame, mode: str, paired: bool) -> str:
    paired_c = f"{mode}__R_fisher_paired"
    if paired and paired_c in res.columns:
        return paired_c
    return f"{mode}__R_fisher"


def _mean_table(res: pd.DataFrame, group_col: str, paired: bool) -> pd.DataFrame:
    full = _r_col(res, "full", paired)
    content = _r_col(res, "content", paired)
    surp = _r_col(res, "surprisal_only", paired)
    g = res.groupby(group_col)
    return pd.DataFrame({
        "n": g.size(),
        "full": g[full].mean(),
        "content": g[content].mean(),
        "surprisal_only": g[surp].mean(),
        "SAE_contrib": g["sae_contribution"].mean(),
        "surp_contrib": g["surprisal_contribution"].mean(),
    }).round(4)


def subject_roi_means(res: pd.DataFrame, paired: bool) -> pd.DataFrame:
    """Aggregate electrodes within subject × region (inferential unit)."""
    if "region" not in res.columns or res.empty:
        return pd.DataFrame()
    full = _r_col(res, "full", paired)
    content = _r_col(res, "content", paired)
    surp = _r_col(res, "surprisal_only", paired)
    g = res.groupby(["subject", "region"], dropna=False)
    out = pd.DataFrame({
        "n_electrodes": g.size(),
        "full": g[full].mean(),
        "content": g[content].mean(),
        "surprisal_only": g[surp].mean(),
        "SAE_contrib": g["sae_contribution"].mean(),
        "surp_contrib": g["surprisal_contribution"].mean(),
    }).reset_index()
    return out


def bootstrap_subject_means(
    subj_means: pd.DataFrame,
    value_col: str,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> tuple[float, float, float, int]:
    """Bootstrap CI over subjects for a subject-level mean table."""
    vals = subj_means[value_col].dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        v = float(vals[0])
        return v, v, v, 1
    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    return (
        float(vals.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        n,
    )


def _overall_block(res: pd.DataFrame, label: str, paired: bool) -> list[str]:
    lines = []
    add = lines.append
    add("-" * 70)
    add(label)
    add("-" * 70)
    add(f"Electrodes: {len(res)}  |  subjects: {res['subject'].nunique()}")
    for m in MODES:
        col = _r_col(res, m, paired)
        v = res[col].dropna()
        if v.empty:
            add(f"  {m:15s} R(Fisher): n_valid=0")
            continue
        add(f"  {m:15s} R(Fisher): mean={v.mean():.4f}  median={v.median():.4f}  "
            f"max={v.max():.4f}  n_valid={len(v)}")
    sae = res["sae_contribution"].dropna()
    surp_c = res["surprisal_contribution"].dropna()
    if len(sae):
        add(f"  PAPER SAE gain (full−surprisal): mean={sae.mean():+.4f}  "
            f"median={sae.median():+.4f}  frac>0={(sae > 0).mean():.3f}  "
            f"(frac>0 descriptive only)")
    if len(surp_c):
        add(f"  EXPLORATORY surprisal contrib (full−content): "
            f"mean={surp_c.mean():+.4f}  median={surp_c.median():+.4f}  "
            f"frac>0={(surp_c > 0).mean():.3f}  [NOT Lepori et al.]")
    if "content__feature_mean" in res.columns:
        content_ok = (res["content__feature_mean"] > 0).sum()
        add(f"  electrodes with >=1 selected SAE feature (content): {content_ok}")
    add("")
    return lines


def _subject_level_block(
    res: pd.DataFrame, label: str, paired: bool, note: str | None = None,
) -> list[str]:
    lines = []
    add = lines.append
    add("-" * 70)
    add(label)
    add("-" * 70)
    if note:
        add(note)
        add("")
    if res.empty:
        add("  (no electrodes)")
        add("")
        return lines
    srm = subject_roi_means(res, paired=paired)
    # Subject means pooling all regions in this subset.
    g = res.groupby("subject")
    full = _r_col(res, "full", paired)
    subj = pd.DataFrame({
        "n_electrodes": g.size(),
        "full": g[full].mean(),
        "SAE_contrib": g["sae_contribution"].mean(),
        "surp_contrib": g["surprisal_contribution"].mean(),
    })
    add("Per-subject means (electrodes pooled within subject):")
    add(subj.round(4).to_string())
    add("")
    for col, name in (
        ("full", "full"),
        ("SAE_contrib", "PAPER SAE gain (full−surprisal)"),
        ("surp_contrib", "EXPLORATORY surprisal contrib (full−content)"),
    ):
        mu, lo, hi, n = bootstrap_subject_means(subj.reset_index(), col)
        add(f"  subject-level {name}: mean={mu:+.4f}  "
            f"95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]  n_subjects={n}")
    add("")
    if not srm.empty and srm["region"].nunique() > 1:
        add("Subject × region means (top by |PAPER SAE gain|):")
        show = srm.reindex(srm["SAE_contrib"].abs().sort_values(ascending=False).index)
        add(show.head(20).round(4).to_string(index=False))
        add("")
    return lines


def output_stem_suffix(results_csv: Path) -> str:
    """Derive a report/figure suffix from the results CSV name.

    Examples
    --------
    sparse_encoding_results.csv -> ""
    sparse_encoding_results_shuffle_global.csv -> "_shuffle_global"
    """
    stem = results_csv.stem
    base = ap.SAE_REGRESSION_TABLE.stem  # sparse_encoding_results
    if stem == base:
        return ""
    if stem.startswith(base + "_"):
        return "_" + stem[len(base) + 1:]
    # Generic fallback: sanitize stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"_{safe}"


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def write_report(res: pd.DataFrame, path: Path, top_n: int, min_region_n: int):
    lines = []
    add = lines.append
    add("=" * 70)
    add("AUGMENTED SPARSE ENCODING MODELS — SUMMARY")
    add("=" * 70)
    meta = []
    if "cv" in res.columns and res["cv"].notna().any():
        meta.append(f"cv={res['cv'].dropna().iloc[0]}")
    if "shuffle_control" in res.columns and res["shuffle_control"].notna().any():
        meta.append(f"shuffle_control={res['shuffle_control'].dropna().iloc[0]}")
    if meta:
        add("Run: " + ", ".join(meta))
        add("")

    has_paired = "full__R_fisher_paired" in res.columns
    complete = complete_fold_mask(res)
    lang = language_mask(res)
    mfg = left_mfg_mask(res)
    explor = exploratory_category_lang_mask(res)

    # Feasibility note for MFG replication
    mfg_n_subj = int(res.loc[mfg, "subject"].nunique()) if mfg.any() else 0
    add("-" * 70)
    add("REPLICATION FEASIBILITY NOTE")
    add("-" * 70)
    add(
        f"Localizer-defined left MFG (is_lang & region='{LEFT_MFG_REGION}'): "
        f"n_electrodes={int(mfg.sum())}, n_subjects={mfg_n_subj}."
    )
    add(
        "Lepori et al. used ~8 participants with reliable MFG voxels from an "
        "independent language localizer. With n_subjects < 8 this dataset can "
        "only provide an underpowered extension, not a direct replication."
    )
    add(
        "Paper-comparable language mask = is_lang (independent localizer). "
        "Taxonomy categories both/enc_only are EXPLORATORY only "
        "(enc_only is not localizer-positive)."
    )
    add(
        "PRIMARY contrast (Lepori et al.) = full − surprisal_only (SAE gain). "
        "full − content / dominance labels are EXPLORATORY only; see "
        "sparse_encoding_paper_aligned.py for the dedicated paper report."
    )
    add(
        "iEEG deviations: contiguous CV (not shuffled sentence folds); "
        "Fisher-z r without noise-ceiling normalization."
    )
    add("")

    # 1) All electrodes, unpaired (backward compatible)
    lines.extend(_overall_block(res, "ALL ELECTRODES (unpaired Fisher-z)", paired=False))

    # 2) Complete-fold paired
    if has_paired:
        sub = res.loc[complete].copy()
        lines.extend(_overall_block(
            sub,
            f"COMPLETE-FOLD PAIRED (all modes valid on every fold; n={complete.sum()})",
            paired=True,
        ))

    # 3) Localizer-positive language electrodes (paper-comparable)
    lang_res = res.loc[lang].copy()
    lines.extend(_overall_block(
        lang_res,
        f"LOCALIZER-POSITIVE LANGUAGE (is_lang=True; n={int(lang.sum())})",
        paired=has_paired,
    ))
    if has_paired:
        lang_complete = res.loc[lang & complete].copy()
        lines.extend(_overall_block(
            lang_complete,
            f"LOCALIZER-POSITIVE ∩ COMPLETE-FOLD PAIRED (n={int((lang & complete).sum())})",
            paired=True,
        ))
    lines.extend(_subject_level_block(
        lang_res,
        "LOCALIZER-POSITIVE — SUBJECT-LEVEL MEANS",
        paired=has_paired,
        note="Inferential unit = subject (electrodes pooled within subject).",
    ))

    # 4) Left MFG (strict paper-comparable ROI)
    mfg_res = res.loc[mfg].copy()
    lines.extend(_overall_block(
        mfg_res,
        f"LEFT MFG (is_lang & {LEFT_MFG_REGION}; n={int(mfg.sum())})",
        paired=has_paired,
    ))
    lines.extend(_subject_level_block(
        mfg_res,
        "LEFT MFG — SUBJECT-LEVEL MEANS (UNDERPOWERED EXTENSION)",
        paired=has_paired,
        note=(
            f"n_subjects={mfg_n_subj}. Report effect sizes / bootstrap CIs only; "
            "do not claim an 8-participant replication."
        ),
    ))

    # 5) Exploratory category-based language (non-replication)
    if explor.any():
        lines.extend(_overall_block(
            res.loc[explor].copy(),
            f"EXPLORATORY (category in {EXPLORATORY_LANG_CATEGORIES}; "
            f"NOT paper-comparable; n={int(explor.sum())})",
            paired=has_paired,
        ))

    if "category" in res.columns:
        add("-" * 70)
        add("Mean R by channel category (exploratory; "
            + ("paired)" if has_paired else "unpaired)"))
        add("-" * 70)
        add(_mean_table(res, "category", paired=has_paired).to_string())
        add("")
        if has_paired and complete.any():
            add("Mean R by category — complete-fold paired only")
            add(_mean_table(res.loc[complete], "category", paired=True).to_string())
            add("")

    if "region" in res.columns:
        use = res.loc[lang] if lang.any() else res
        add("-" * 70)
        add(f"Top localizer-positive regions by SAE contribution "
            f"(min n={min_region_n})")
        add("-" * 70)
        if use.empty:
            add("  (no localizer-positive electrodes)")
        else:
            g = use.groupby("region")
            reg = pd.DataFrame({
                "n": g.size(),
                "n_subjects": g["subject"].nunique(),
                "full": g[_r_col(use, "full", has_paired)].mean(),
                "SAE_contrib": g["sae_contribution"].mean(),
                "surp_contrib": g["surprisal_contribution"].mean(),
            })
            reg = reg[reg["n"] >= min_region_n].sort_values(
                "SAE_contrib", ascending=False)
            add(reg.round(4).head(20).to_string())
        add("")

        # Coarse temporal vs frontal among localizer-positive electrodes
        reg_name = use["region"].astype(str)
        temp = use.loc[reg_name.str.contains(
            "Temp|STG|MTG|ITG|STS|temporal", case=False, na=False)]
        front = use.loc[reg_name.str.contains(
            "IFG|MFG|SFG|frontal|Precentral|pars", case=False, na=False)]
        add("Coarse anatomy on localizer-positive electrodes "
            + ("(paired)" if has_paired else ""))
        for name, sub in (("TEMPORAL-ish", temp), ("FRONTAL-ish", front)):
            if sub.empty:
                continue
            fc = _r_col(sub, "full", has_paired)
            add(f"  {name:14s} n={len(sub):4d}  subjects={sub['subject'].nunique()}  "
                f"full={sub[fc].mean():+.4f}  "
                f"SAE_contrib={sub['sae_contribution'].mean():+.4f}  "
                f"surp_contrib={sub['surprisal_contribution'].mean():+.4f}")
        add("")

    add("-" * 70)
    add(f"Top {top_n} electrodes by full-model R")
    add("-" * 70)
    score = _r_col(res, "full", has_paired)
    cols = ["subject", "channel"]
    if "region" in res.columns:
        cols.append("region")
    if "category" in res.columns:
        cols.append("category")
    if "is_lang" in res.columns:
        cols.append("is_lang")
    cols += [score, _r_col(res, "content", has_paired),
             _r_col(res, "surprisal_only", has_paired)]
    if "full__feature_mean" in res.columns:
        cols.append("full__feature_mean")
    top = res.sort_values(score, ascending=False).head(top_n)
    add(top[cols].round(4).to_string(index=False))

    # Temporal precision pointer (primary analyses live in lag/deconv outputs).
    lines.extend(_temporal_dynamics_block(res, path))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote report -> {path}")


def _temporal_dynamics_block(res: pd.DataFrame, report_path: Path) -> list[str]:
    """Append lag-peak / kernel timing if sibling temporal tables exist."""
    lines = [
        "",
        "=" * 70,
        "TEMPORAL DYNAMICS SUMMARY (primary analyses)",
        "=" * 70,
        "This summary file is from the single-lag baseline. Prefer lag-screen",
        "peaks and deconvolution kernels for temporal claims",
        "(see sparse_encoding/README_TEMPORAL.md).",
        "",
    ]
    tag = ""
    if "feature_tag" in res.columns and res["feature_tag"].notna().any():
        tag = str(res["feature_tag"].dropna().iloc[0])
    candidates = []
    if tag:
        candidates.extend([
            ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks_{tag}_lang.csv",
            ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks_{tag}.csv",
            ap.SAE_TABLES / f"sparse_encoding_kernel_shapes_{tag}_lang.csv",
            ap.SAE_TABLES / f"sparse_encoding_kernel_shapes_{tag}.csv",
        ])
    # Also try suffix inferred from the summary report name.
    stem = report_path.stem.replace("sparse_encoding_summary", "")
    if stem:
        candidates.extend([
            ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks{stem}.csv",
            ap.SAE_TABLES / f"sparse_encoding_kernel_shapes{stem}.csv",
        ])
    seen = set()
    found = False
    for p in candidates:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        found = True
        df = pd.read_csv(p)
        lines.append(f"[{p.name}] nE={len(df)}")
        for col in (
            "full_peak_lag_ms", "surprisal_peak_lag_ms", "full_fwhm_ms",
            "lag_diff_full_vs_surprisal_ms",
            "surprisal_peak_lag_ms", "sae_l2_peak_lag_ms",
        ):
            if col not in df.columns or "subject" not in df.columns:
                continue
            subj = df.groupby("subject")[col].mean()
            lines.append(
                f"  {col}: median={df[col].median():.1f}  "
                f"subject-mean={subj.mean():+.1f}  nS={subj.notna().sum()}"
            )
        lines.append("")
    if not found:
        lines.append(
            "(no lag-screen peaks / kernel-shape tables found yet — "
            "run sparse_encoding_lag_screen.py and "
            "sparse_encoding_deconvolution.py)"
        )
        lines.append("")
    return lines


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _fig_path(outdir: Path, stem: str, suffix: str) -> Path:
    return outdir / f"{stem}{suffix}.png"


def fig_r_distribution(res, outdir, suffix: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODES:
        col = _r_col(res, m, paired=False)
        v = res[col].dropna()
        ax.hist(v, bins=40, alpha=0.5, label=m, color=MODE_COLORS[m])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("R (Fisher-averaged across folds)")
    ax.set_ylabel("electrodes")
    ax.set_title("Encoding performance by model variant")
    ax.legend()
    fig.tight_layout()
    p = _fig_path(outdir, "r_distribution", suffix)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  {p}")


def fig_content_vs_surprisal(res, outdir, suffix: str = ""):
    c_col = _r_col(res, "content", paired="content__R_fisher_paired" in res.columns)
    s_col = _r_col(res, "surprisal_only", paired="surprisal_only__R_fisher_paired" in res.columns)
    d = res.dropna(subset=[c_col, s_col])
    # Prefer localizer-positive electrodes in the scatter when available.
    lang = language_mask(d)
    plot = d.loc[lang] if lang.any() else d
    fig, ax = plt.subplots(figsize=(6, 6))
    if "category" in plot.columns and plot["category"].notna().any():
        for cat, sub in plot.groupby("category"):
            ax.scatter(sub[s_col], sub[c_col], s=14, alpha=0.6, label=str(cat))
        ax.legend(title="category", fontsize=8)
    else:
        ax.scatter(plot[s_col], plot[c_col], s=14, alpha=0.6, color="#1f77b4")
    lo = float(np.nanmin([plot[s_col].min(), plot[c_col].min(), 0]))
    hi = float(np.nanmax([plot[s_col].max(), plot[c_col].max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("surprisal-only  R (Fisher)")
    ax.set_ylabel("content (SAE-only)  R (Fisher)")
    ax.set_title("Content vs. surprisal encoding\n(localizer-positive when available)")
    fig.tight_layout()
    p = _fig_path(outdir, "content_vs_surprisal", suffix)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  {p}")


def fig_mean_R_by_category(res, outdir, suffix: str = ""):
    if "category" not in res.columns or res["category"].notna().sum() == 0:
        return
    cats = [c for c in res["category"].dropna().unique()]
    x = np.arange(len(cats)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cats)), 4.5))
    paired = "full__R_fisher_paired" in res.columns
    for i, m in enumerate(MODES):
        col = _r_col(res, m, paired)
        means = [res.loc[res["category"] == c, col].mean() for c in cats]
        sems = [res.loc[res["category"] == c, col].sem() for c in cats]
        ax.bar(x + (i - 1) * w, means, w, yerr=sems, capsize=3,
               label=m, color=MODE_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean R (Fisher)")
    ax.set_title("Encoding by channel category (exploratory)"
                 + (" (paired)" if paired else ""))
    ax.legend()
    fig.tight_layout()
    p = _fig_path(outdir, "mean_R_by_category", suffix)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  {p}")


def fig_sae_contribution_by_region(res, outdir, min_region_n, suffix: str = ""):
    if "region" not in res.columns or res["region"].notna().sum() == 0:
        return
    use = res.loc[language_mask(res)] if language_mask(res).any() else res
    g = use.groupby("region")["sae_contribution"]
    reg = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sem": g.sem()})
    reg = reg[reg["n"] >= min_region_n].sort_values("mean")
    if reg.empty:
        return
    reg = reg.tail(20)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(reg))))
    ax.barh(np.arange(len(reg)), reg["mean"], xerr=reg["sem"],
            capsize=2, color="#2ca02c")
    ax.set_yticks(np.arange(len(reg)))
    ax.set_yticklabels([f"{r} (n={int(n)})" for r, n in zip(reg.index, reg["n"])],
                       fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SAE contribution  (full − surprisal_only), R Fisher")
    ax.set_title("Where SAE content features add predictive power"
                 "\n(localizer-positive electrodes)")
    fig.tight_layout()
    p = _fig_path(outdir, "sae_contribution_by_region", suffix)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  {p}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_csv", default=None)
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--min_region_n", type=int, default=5)
    p.add_argument(
        "--suffix", default=None,
        help="Override report/figure filename suffix. Default: inferred from "
             "results CSV stem (e.g. shuffle_global → _shuffle_global).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()

    csv = Path(args.results_csv) if args.results_csv else ap.SAE_REGRESSION_TABLE
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. Run sparse_encoding_regression.py first.")
    res = add_contrasts(merge_taxonomy(pd.read_csv(csv)))

    suffix = args.suffix if args.suffix is not None else output_stem_suffix(csv)
    report_path = ap.SAE_REPORTS / f"sparse_encoding_summary{suffix}.txt"
    write_report(res, report_path, args.top_n, args.min_region_n)

    print("Figures:")
    fig_r_distribution(res, ap.SAE_FIGURES, suffix)
    fig_content_vs_surprisal(res, ap.SAE_FIGURES, suffix)
    fig_mean_R_by_category(res, ap.SAE_FIGURES, suffix)
    fig_sae_contribution_by_region(res, ap.SAE_FIGURES, args.min_region_n, suffix)


if __name__ == "__main__":
    main()
