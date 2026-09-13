#!/usr/bin/env python3
"""
sparse_encoding_surprisal_dominance.py
======================================
Qwen all-channel encoding census + localizer association.

Paper-aligned primary contrast (Lepori et al. Augmented Sparse Encoding)
------------------------------------------------------------------------
- ``paper_sae_gain`` = full − surprisal_only   (SAE benefit beyond surprisal)
- Lack of significant SAE gain is reported as "no detected SAE benefit",
  **not** as proof that a channel is surprisal-only.
- Dedicated paper-aligned tables / report / figures are written by
  ``sparse_encoding_paper_aligned.py``.

Exploratory extension (NOT defined by Lepori et al.)
----------------------------------------------------
- surprisal gain  = full − content
- dominance index = surprisal_gain − SAE_gain
- surprisal_dominant / sae_dominant binary labels
- quality_ok (for confirmatory ranking) =
      all_folds_paired AND full > 0 AND content feature_mean > 0

Example
-------
    python sparse_encoding_surprisal_dominance.py
    python sparse_encoding_paper_aligned.py
    python sparse_encoding_surprisal_dominance.py --n_perm 2000 --top_n 25
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_summary import (
    LEFT_MFG_REGION, bootstrap_subject_means, language_mask, merge_taxonomy,
)

FEATURE_TAG = "sae_qwen3_8b_l18"
RNG_SEED = 19
N_PERM = 2000
N_BOOT = 1000
MIN_GROUP_N = 3

TASK_POS = {
    "MITSWJNTask": "is_pos_MITSWJNTask",
    "WM": "is_pos_WM",
    "DMN": "is_pos_DMN",
}
TASK_CONTRAST = {
    "MITSWJNTask": "resp_MITSWJNTask_SN_diff",
    "WM": "resp_WM_SN_diff",
    "DMN": "resp_DMN_SN_diff",
}
MITSWJN_CONDS = {
    "S": "resp_MITSWJNTask_S",
    "N": "resp_MITSWJNTask_N",
    "W": "resp_MITSWJNTask_W",
    "J": "resp_MITSWJNTask_J",
}
COND_LABELS = {
    "S": "Sentences",
    "N": "Nonwords",
    "W": "Word-lists",
    "J": "Jabberwocky",
}


# ======================================================================
# Helpers
# ======================================================================

def normalize_channel(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def empirical_p(obs: float, nulls: np.ndarray, alternative: str = "greater") -> float:
    """Phipson–Smyth style corrected empirical p-value."""
    nulls = np.asarray(nulls, dtype=float)
    nulls = nulls[np.isfinite(nulls)]
    if not np.isfinite(obs) or nulls.size == 0:
        return np.nan
    if alternative == "greater":
        exceed = int(np.sum(nulls >= obs))
    elif alternative == "less":
        exceed = int(np.sum(nulls <= obs))
    else:
        exceed = int(np.sum(np.abs(nulls) >= abs(obs)))
    return float((exceed + 1) / (nulls.size + 1))


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_ok = np.empty(n)
    out_ok[order] = q
    out[ok] = out_ok
    return out


def assign_task_class(row: pd.Series) -> str:
    m = bool(row.get("is_pos_MITSWJNTask", False))
    w = bool(row.get("is_pos_WM", False))
    d = bool(row.get("is_pos_DMN", False))
    n = int(m) + int(w) + int(d)
    if n == 0:
        return "none"
    if n == 3:
        return "all_three"
    if n == 1:
        if m:
            return "mitswjn_only"
        if w:
            return "wm_only"
        return "dmn_only"
    if m and w:
        return "mitswjn_wm"
    if m and d:
        return "mitswjn_dmn"
    return "wm_dmn"


def collapse_task_group(task_class: str) -> str:
    """Map fine ``task_class`` to exclusive analysis groups (+ none).

    Overlaps (mitswjn_wm / mitswjn_dmn / wm_dmn / all_three) → ``multi_task``.
    """
    if task_class in ("mitswjn_only", "wm_only", "dmn_only", "none"):
        return task_class
    if task_class in ("mitswjn_wm", "mitswjn_dmn", "wm_dmn", "all_three"):
        return "multi_task"
    if task_class in ("unknown", "", None) or (
        isinstance(task_class, float) and np.isnan(task_class)
    ):
        return "unknown"
    return str(task_class)


EXCLUSIVE_TASK_GROUPS = ("mitswjn_only", "wm_only", "dmn_only", "multi_task")
OVERLAP_TASK_GROUPS = {
    "MITSWJN+": "is_pos_MITSWJNTask",
    "WM+": "is_pos_WM",
    "DMN+": "is_pos_DMN",
}


def assign_dominant_task(row: pd.Series) -> str:
    cands = []
    for task, pos_col in TASK_POS.items():
        if bool(row.get(pos_col, False)):
            cands.append((float(row.get(TASK_CONTRAST[task], np.nan)), task))
    cands = [(v, t) for v, t in cands if np.isfinite(v)]
    if not cands:
        return "none"
    return max(cands)[1]


def classify_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """Add paper-aligned SAE-gain fields plus exploratory dominance labels.

    Paper-aligned (Lepori et al.)
    -----------------------------
    ``paper_sae_gain`` = full − surprisal_only. Positive subject-level SAE gain
    with statistical support means content features improve prediction beyond
    surprisal. Nonsignificance must **not** be labeled "surprisal-only".

    Exploratory (not in Lepori et al.)
    ----------------------------------
    ``surprisal_gain``, ``dominance_index``, and ``surprisal_dominant`` /
    ``sae_dominant`` binary labels remain for post-hoc exploration only.
    """
    d = df.copy()
    full = "full__R_fisher_paired" if "full__R_fisher_paired" in d.columns else "full__R_fisher"
    content = "content__R_fisher_paired" if "content__R_fisher_paired" in d.columns else "content__R_fisher"
    surp = ("surprisal_only__R_fisher_paired"
            if "surprisal_only__R_fisher_paired" in d.columns
            else "surprisal_only__R_fisher")
    d["full_R"] = d[full]
    d["content_R"] = d[content]
    d["surprisal_only_R"] = d[surp]

    # --- Paper-aligned primary fields ---
    d["paper_full_R"] = d["full_R"]
    d["paper_surprisal_R"] = d["surprisal_only_R"]
    d["paper_sae_gain"] = d["paper_full_R"] - d["paper_surprisal_R"]
    d["sae_gain"] = d["paper_sae_gain"]  # alias used throughout this module
    paired = d["all_folds_paired"].fillna(0).astype(bool) if "all_folds_paired" in d.columns else True
    d["paper_quality_paired"] = paired
    d["paper_quality_ok"] = (
        d["paper_quality_paired"]
        & np.isfinite(d["paper_full_R"])
        & np.isfinite(d["paper_surprisal_R"])
    )
    # Descriptive only — never used as a paper-derived binary claim.
    d["paper_sae_gain_positive"] = d["paper_sae_gain"] > 0

    # --- Exploratory extension (full − content; dominance index) ---
    d["surprisal_gain"] = d["full_R"] - d["content_R"]
    d["dominance_index"] = d["surprisal_gain"] - d["sae_gain"]
    d["exploratory_surprisal_gain"] = d["surprisal_gain"]
    d["exploratory_dominance_index"] = d["dominance_index"]

    content_support = d["content__feature_mean"].fillna(0) > 0 if "content__feature_mean" in d.columns else True
    d["quality_paired"] = paired
    d["quality_full_positive"] = d["full_R"] > 0
    d["quality_content_support"] = content_support
    d["quality_ok"] = (
        d["quality_paired"] & d["quality_full_positive"] & d["quality_content_support"]
    )

    # Exploratory descriptive / confirmatory labels (NOT Lepori definitions)
    d["surprisal_dominant_desc"] = (
        (d["surprisal_gain"] > 0) & (d["surprisal_gain"] > d["sae_gain"])
    )
    d["sae_dominant_desc"] = (
        (d["sae_gain"] > 0) & (d["sae_gain"] > d["surprisal_gain"])
    )
    d["surprisal_dominant"] = d["surprisal_dominant_desc"] & d["quality_ok"]
    d["sae_dominant"] = d["sae_dominant_desc"] & d["quality_ok"]
    d["balanced"] = (
        d["quality_ok"]
        & ~d["surprisal_dominant"]
        & ~d["sae_dominant"]
        & (d["full_R"] > 0)
    )
    d["low_performing"] = ~d["quality_full_positive"]
    d["dominance_label"] = np.select(
        [
            d["surprisal_dominant"],
            d["sae_dominant"],
            d["balanced"],
            d["low_performing"],
        ],
        ["surprisal_dominant", "sae_dominant", "balanced", "low_performing"],
        default="unclassified",
    )
    d["exploratory_dominance_label"] = d["dominance_label"]
    # Failed-content collapse: content empty but surprisal looks dominant
    d["content_collapsed"] = (
        ~d["quality_content_support"] & d["surprisal_dominant_desc"]
    )
    return d


def merge_localizers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach cross-task localizer responses; return coverage stats."""
    cov = {
        "n_encoding": len(df),
        "n_with_localizer_table": 0,
        "n_mitswjn_measured": 0,
        "n_wm_measured": 0,
        "n_dmn_measured": 0,
        "n_dropped_no_localizer_row": 0,
    }
    path = ap.CROSS_TASK_RESPONSES
    if not path.exists():
        print(f"[warn] {path} missing; localizer merge skipped")
        df = df.copy()
        df["task_class"] = "unknown"
        df["dominant_task"] = "unknown"
        return df, cov

    loc = pd.read_csv(path)
    loc["channel"] = normalize_channel(loc["channel"])
    df = df.copy()
    df["channel"] = normalize_channel(df["channel"])

    keep = ["subject", "channel"]
    keep += [c for c in TASK_POS.values() if c in loc.columns]
    keep += [c for c in TASK_CONTRAST.values() if c in loc.columns]
    keep += [c for c in MITSWJN_CONDS.values() if c in loc.columns]
    keep += [c for c in ("is_lang", "region", "category", "hemisphere",
                         "gamma_S", "gamma_N", "gamma_W", "gamma_J",
                         "gamma_SN_diff") if c in loc.columns]
    # Prefer localizer table annotations when taxonomy merge already present
    overlap = [c for c in keep if c not in ("subject", "channel") and c in df.columns]
    loc_keep = loc[keep].drop_duplicates(["subject", "channel"])
    # Avoid duplicate annotation columns: keep encoding-side taxonomy if present
    drop_from_loc = [c for c in overlap if c in ("region", "category", "hemisphere", "is_lang")]
    loc_use = loc_keep.drop(columns=drop_from_loc, errors="ignore")
    merged = df.merge(loc_use, on=["subject", "channel"], how="left", indicator=True)
    cov["n_with_localizer_table"] = int((merged["_merge"] == "both").sum())
    cov["n_dropped_no_localizer_row"] = int((merged["_merge"] == "left_only").sum())
    merged = merged.drop(columns=["_merge"])

    for pos in TASK_POS.values():
        if pos in merged.columns:
            merged[pos] = merged[pos].fillna(False).astype(bool)
    for task, col in TASK_CONTRAST.items():
        if col in merged.columns:
            cov[f"n_{task.lower()}_measured"] = int(merged[col].notna().sum())
    if "resp_MITSWJNTask_S" in merged.columns:
        cov["n_mitswjn_measured"] = int(merged["resp_MITSWJNTask_S"].notna().sum())
    if "resp_WM_SN_diff" in merged.columns:
        cov["n_wm_measured"] = int(merged["resp_WM_SN_diff"].notna().sum())
    if "resp_DMN_SN_diff" in merged.columns:
        cov["n_dmn_measured"] = int(merged["resp_DMN_SN_diff"].notna().sum())

    # If taxonomy merge didn't provide is_lang/region, pull from loc
    if "is_lang" not in merged.columns and "is_lang" in loc.columns:
        merged = merged.merge(
            loc[["subject", "channel", "is_lang", "region", "category"]].drop_duplicates(),
            on=["subject", "channel"], how="left",
        )

    merged["task_class"] = merged.apply(assign_task_class, axis=1)
    merged["task_group"] = merged["task_class"].map(collapse_task_group)
    merged["dominant_task"] = merged.apply(assign_dominant_task, axis=1)
    return merged, cov


def merge_deconv(df: pd.DataFrame, deconv_csv: Optional[Path]) -> pd.DataFrame:
    """Attach existing Qwen lang-only deconv scores when available."""
    d = df.copy()
    d["deconv_available"] = False
    d["deconv_sae_gain"] = np.nan
    d["deconv_surprisal_gain"] = np.nan
    d["deconv_full_R"] = np.nan
    d["method_agree_surprisal"] = False
    if deconv_csv is None or not Path(deconv_csv).exists():
        return d
    dd = pd.read_csv(deconv_csv)
    dd["channel"] = normalize_channel(dd["channel"])
    keep = dd[["subject", "channel", "full__R", "sae_gain", "surprisal_gain"]].rename(
        columns={
            "full__R": "deconv_full_R",
            "sae_gain": "deconv_sae_gain",
            "surprisal_gain": "deconv_surprisal_gain",
        }
    )
    # Drop placeholders then merge
    d = d.drop(columns=["deconv_full_R", "deconv_sae_gain", "deconv_surprisal_gain"],
               errors="ignore")
    d = d.merge(keep, on=["subject", "channel"], how="left")
    d["deconv_available"] = d["deconv_full_R"].notna()
    d["method_agree_surprisal"] = (
        d["surprisal_dominant"]
        & d["deconv_available"]
        & (d["deconv_surprisal_gain"] > 0)
        & (d["deconv_surprisal_gain"] > d["deconv_sae_gain"])
    )
    return d


# ======================================================================
# Subject-aware inference
# ======================================================================

def subject_stratified_perm_diff(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
) -> Dict:
    """Permute labels within subject; compare mean(group_a) − mean(group_b)."""
    sub = df[df[group_col].isin([group_a, group_b])].dropna(subset=[value_col]).copy()
    if len(sub) < 4:
        return {
            "group_a": group_a, "group_b": group_b,
            "n_a": 0, "n_b": 0, "mean_diff": np.nan,
            "perm_p": np.nan, "n_subjects_used": 0, "method": "insufficient",
        }
    vals_a = sub.loc[sub[group_col] == group_a, value_col].to_numpy(float)
    vals_b = sub.loc[sub[group_col] == group_b, value_col].to_numpy(float)
    rng = np.random.default_rng(rng_seed)
    dual = []
    for subj, sdf in sub.groupby("subject"):
        if set(sdf[group_col].unique()) >= {group_a, group_b}:
            dual.append(subj)
    if len(dual) >= 2:
        method = "within_subject"
        packs = []
        for subj in dual:
            sdf = sub[sub["subject"] == subj]
            packs.append((
                sdf[value_col].to_numpy(float),
                sdf[group_col].astype(str).to_numpy(),
            ))
        pa0 = np.concatenate([v[lab == group_a] for v, lab in packs])
        pb0 = np.concatenate([v[lab == group_b] for v, lab in packs])
        obs = float(np.mean(pa0) - np.mean(pb0))
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            pa_l, pb_l = [], []
            for vals, labels in packs:
                lab = labels.copy()
                rng.shuffle(lab)
                pa_l.append(vals[lab == group_a])
                pb_l.append(vals[lab == group_b])
            pa = np.concatenate(pa_l)
            pb = np.concatenate(pb_l)
            nulls[i] = 0.0 if (len(pa) == 0 or len(pb) == 0) else float(np.mean(pa) - np.mean(pb))
        n_subj = len(dual)
        vals_a, vals_b = pa0, pb0
    else:
        method = "pooled_shuffle"
        obs = float(np.mean(vals_a) - np.mean(vals_b))
        combined = np.concatenate([vals_a, vals_b])
        n_a = len(vals_a)
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            idx = rng.permutation(len(combined))
            nulls[i] = float(np.mean(combined[idx[:n_a]]) - np.mean(combined[idx[n_a:]]))
        n_subj = int(sub["subject"].nunique())
    return {
        "group_a": group_a, "group_b": group_b,
        "n_a": int(len(vals_a)), "n_b": int(len(vals_b)),
        "mean_diff": obs,
        "perm_p": empirical_p(obs, nulls, alternative="two-sided"),
        "n_subjects_used": n_subj, "method": method,
    }


def subject_mean_corr_perm(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
) -> Dict:
    """Spearman on subject-mean (x,y); null = shuffle y across subjects."""
    g = (df.dropna(subset=[x_col, y_col])
           .groupby("subject")[[x_col, y_col]].mean())
    if len(g) < 4:
        return {"n_subjects": len(g), "rho": np.nan, "perm_p": np.nan}
    x = g[x_col].to_numpy(float)
    y = g[y_col].to_numpy(float)
    rho, _ = sp_stats.spearmanr(x, y)
    rng = np.random.default_rng(rng_seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        yy = rng.permutation(y)
        nulls[i], _ = sp_stats.spearmanr(x, yy)
    return {
        "n_subjects": int(len(g)),
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "perm_p": empirical_p(float(rho), nulls, alternative="two-sided"),
    }


def group_summary(df: pd.DataFrame, group_col: str, value_col: str = "dominance_index") -> pd.DataFrame:
    rows = []
    for gname, g in df.groupby(group_col, dropna=False):
        subj = g.groupby("subject")[value_col].mean()
        mu, lo, hi, n = bootstrap_subject_means(
            subj.reset_index(), value_col, n_boot=N_BOOT, seed=RNG_SEED)
        rows.append({
            "group_col": group_col,
            "group": gname,
            "n_electrodes": len(g),
            "n_subjects": n,
            "frac_surprisal_dominant": float(g["surprisal_dominant"].mean()) if len(g) else np.nan,
            "frac_sae_dominant": float(g["sae_dominant"].mean()) if len(g) else np.nan,
            "dominance_index_subject_mean": mu,
            "dominance_index_ci_lo": lo,
            "dominance_index_ci_hi": hi,
            "surprisal_gain_subject_mean": float(
                g.groupby("subject")["surprisal_gain"].mean().mean()),
            "sae_gain_subject_mean": float(
                g.groupby("subject")["sae_gain"].mean().mean()),
        })
    return pd.DataFrame(rows)


def condition_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-aggregated MITSWJN condition responses by dominance label."""
    rows = []
    for label, g in df.groupby("dominance_label"):
        for code, col in MITSWJN_CONDS.items():
            if col not in g.columns:
                continue
            subj = g.dropna(subset=[col]).groupby("subject")[col].mean()
            mu, lo, hi, n = bootstrap_subject_means(
                subj.reset_index(), col, n_boot=N_BOOT, seed=RNG_SEED)
            rows.append({
                "dominance_label": label,
                "condition": COND_LABELS[code],
                "condition_code": code,
                "n_subjects": n,
                "n_electrodes": int(g[col].notna().sum()),
                "mean": mu, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def run_localizer_stats(df: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    """Group contrasts + continuous correlations with FDR."""
    rows = []
    # Binary task-positive vs negative
    for task, pos_col in TASK_POS.items():
        if pos_col not in df.columns:
            continue
        tmp = df.copy()
        tmp["_grp"] = np.where(tmp[pos_col], f"{task}+", f"{task}-")
        res = subject_stratified_perm_diff(
            tmp, "dominance_index", "_grp", f"{task}+", f"{task}-",
            n_perm=n_perm, rng_seed=RNG_SEED + hash(task) % 1000,
        )
        rows.append({
            "test": "group_contrast",
            "task": task,
            "predictor": pos_col,
            **res,
        })
        # Continuous contrast correlation (subject means)
        ccol = TASK_CONTRAST[task]
        if ccol in df.columns:
            cres = subject_mean_corr_perm(
                df, ccol, "dominance_index",
                n_perm=n_perm, rng_seed=RNG_SEED + 17 + hash(task) % 1000,
            )
            rows.append({
                "test": "subject_mean_spearman",
                "task": task,
                "predictor": ccol,
                "group_a": "", "group_b": "",
                "n_a": cres["n_subjects"], "n_b": cres["n_subjects"],
                "mean_diff": cres["rho"],
                "perm_p": cres["perm_p"],
                "n_subjects_used": cres["n_subjects"],
                "method": "subject_mean_shuffle",
            })
    # task_class: each non-none vs none
    if "task_class" in df.columns:
        for cls in sorted(df["task_class"].dropna().unique()):
            if cls in ("none", "unknown"):
                continue
            res = subject_stratified_perm_diff(
                df, "dominance_index", "task_class", cls, "none",
                n_perm=n_perm, rng_seed=RNG_SEED + hash(cls) % 1000,
            )
            rows.append({
                "test": "task_class_vs_none",
                "task": cls,
                "predictor": "task_class",
                **res,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = fdr_bh(out["perm_p"].to_numpy())
    return out


def _gate_min_subjects(res: Dict, min_group_n: int = MIN_GROUP_N) -> Dict:
    """Mark tests with too few subjects as insufficient."""
    n = int(res.get("n_subjects_used", 0) or 0)
    if n < min_group_n:
        out = dict(res)
        out["perm_p"] = np.nan
        out["method"] = "insufficient"
        return out
    return res


def subject_mean_one_sample_perm(
    df: pd.DataFrame,
    value_col: str = "paper_sae_gain",
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
    min_group_n: int = MIN_GROUP_N,
) -> Dict:
    """Sign-flip test of subject-mean ``value_col`` against zero.

    Electrodes are averaged within subject first (inferential unit = subject).
    """
    subj = (
        df.dropna(subset=[value_col])
          .groupby("subject")[value_col]
          .mean()
          .dropna()
    )
    vals = subj.to_numpy(float)
    n = len(vals)
    if n < min_group_n:
        return {
            "n_subjects_used": n,
            "mean_diff": float(np.mean(vals)) if n else np.nan,
            "perm_p": np.nan,
            "method": "insufficient",
        }
    obs = float(np.mean(vals))
    rng = np.random.default_rng(rng_seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        nulls[i] = float(np.mean(vals * signs))
    return {
        "n_subjects_used": n,
        "mean_diff": obs,
        "perm_p": empirical_p(obs, nulls, alternative="two-sided"),
        "method": "sign_flip_subjects",
    }


def paper_group_summary(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    value_col: str = "paper_sae_gain",
) -> pd.DataFrame:
    """Subject-level bootstrap summary of paper SAE gain (and full / surprisal)."""
    rows = []
    if group_col is None:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(group_col, dropna=False))
    for gname, g in groups:
        ok = g[g["paper_quality_ok"]] if "paper_quality_ok" in g.columns else g
        if ok.empty:
            continue
        subj = ok.groupby("subject").agg(
            paper_sae_gain=("paper_sae_gain", "mean"),
            paper_full_R=("paper_full_R", "mean"),
            paper_surprisal_R=("paper_surprisal_R", "mean"),
            frac_sae_gain_positive=("paper_sae_gain_positive", "mean"),
            n_electrodes=("channel", "size"),
        ).reset_index()
        mu, lo, hi, n = bootstrap_subject_means(
            subj, value_col, n_boot=N_BOOT, seed=RNG_SEED)
        mu_f, lo_f, hi_f, _ = bootstrap_subject_means(
            subj, "paper_full_R", n_boot=N_BOOT, seed=RNG_SEED)
        mu_s, lo_s, hi_s, _ = bootstrap_subject_means(
            subj, "paper_surprisal_R", n_boot=N_BOOT, seed=RNG_SEED)
        rows.append({
            "group_col": group_col or "all",
            "group": "all" if gname is None else gname,
            "n_electrodes": int(ok.shape[0]),
            "n_subjects": n,
            "paper_sae_gain_subject_mean": mu,
            "paper_sae_gain_ci_lo": lo,
            "paper_sae_gain_ci_hi": hi,
            "paper_full_R_subject_mean": mu_f,
            "paper_full_R_ci_lo": lo_f,
            "paper_full_R_ci_hi": hi_f,
            "paper_surprisal_R_subject_mean": mu_s,
            "paper_surprisal_R_ci_lo": lo_s,
            "paper_surprisal_R_ci_hi": hi_s,
            "frac_sae_gain_positive_subject_mean": float(
                subj["frac_sae_gain_positive"].mean()),
            "note": "frac_sae_gain_positive is descriptive only",
        })
    return pd.DataFrame(rows)


def run_paper_sae_gain_stats(
    df: pd.DataFrame,
    n_perm: int = N_PERM,
    min_group_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """Subject-level SAE-gain inference aligned with Lepori full-vs-surprisal.

    Families (FDR within each):
      1. overall / language / left-MFG one-sample SAE gain vs 0
      2. exclusive task_group one-sample SAE gain vs 0
      3. exclusive task_group pairwise SAE-gain differences
      4. overlapping task-positive one-sample + pairwise SAE gain
      5. ROI (region) one-sample SAE gain vs 0 for language electrodes
    """
    import itertools

    rows: List[Dict] = []
    d = df.copy()
    if "task_group" not in d.columns and "task_class" in d.columns:
        d["task_group"] = d["task_class"].map(collapse_task_group)
    lang = language_mask(d) if "is_lang" in d.columns else pd.Series(False, index=d.index)

    def _add_one_sample(sub: pd.DataFrame, family: str, label: str, seed_off: int):
        res = subject_mean_one_sample_perm(
            sub, "paper_sae_gain", n_perm=n_perm,
            rng_seed=RNG_SEED + seed_off, min_group_n=min_group_n,
        )
        rows.append({
            "family": family,
            "test": "sae_gain_vs_zero",
            "group_a": label,
            "group_b": "zero",
            "value_col": "paper_sae_gain",
            "n_a": int(sub.shape[0]),
            "n_b": 0,
            "mean_diff": res["mean_diff"],
            "perm_p": res["perm_p"],
            "n_subjects_used": res["n_subjects_used"],
            "method": res["method"],
        })

    # Family 1: overall / language / MFG
    _add_one_sample(d[d["paper_quality_ok"]], "overall", "all_channels", 1)
    if lang.any():
        _add_one_sample(d[lang & d["paper_quality_ok"]], "overall", "is_lang", 2)
    if "region" in d.columns:
        mfg = (
            lang
            & d["region"].astype(str).eq(LEFT_MFG_REGION)
            & d["paper_quality_ok"]
        )
        if mfg.any():
            _add_one_sample(d[mfg], "overall", "left_mfg_is_lang", 3)

    # Family 2–3: exclusive task groups
    if "task_group" in d.columns:
        present = [
            g for g in EXCLUSIVE_TASK_GROUPS
            if (d["task_group"] == g).any()
        ]
        for i, g in enumerate(present):
            sub = d[(d["task_group"] == g) & d["paper_quality_ok"]]
            _add_one_sample(sub, "exclusive_onesample", g, 100 + i)
        for a, b in itertools.combinations(present, 2):
            res = subject_stratified_perm_diff(
                d[d["paper_quality_ok"]], "paper_sae_gain", "task_group", a, b,
                n_perm=n_perm, rng_seed=RNG_SEED + hash((a, b, "paper")) % 10000,
            )
            res = _gate_min_subjects(res, min_group_n)
            rows.append({
                "family": "exclusive_pairwise",
                "test": "task_group_sae_gain",
                "group_a": a, "group_b": b,
                "value_col": "paper_sae_gain",
                **{k: res[k] for k in (
                    "n_a", "n_b", "mean_diff", "perm_p",
                    "n_subjects_used", "method",
                )},
            })

    # Family 4: overlapping task-positive
    for i, (gname, col) in enumerate(OVERLAP_TASK_GROUPS.items()):
        if col not in d.columns:
            continue
        sub = d[d[col].fillna(False).astype(bool) & d["paper_quality_ok"]]
        _add_one_sample(sub, "overlap_onesample", gname, 200 + i)
    for a, b in itertools.combinations(OVERLAP_TASK_GROUPS.keys(), 2):
        res = overlap_paired_subject_value_diff(
            d[d["paper_quality_ok"]], "paper_sae_gain", a, b,
            n_perm=n_perm,
            rng_seed=RNG_SEED + 301 + hash((a, b)) % 10000,
        )
        res = _gate_min_subjects(res, min_group_n)
        rows.append({
            "family": "overlap_pairwise",
            "test": "overlap_sae_gain",
            "group_a": a, "group_b": b,
            "value_col": "paper_sae_gain",
            **{k: res[k] for k in (
                "n_a", "n_b", "mean_diff", "perm_p",
                "n_subjects_used", "method",
            )},
        })

    # Family 5: ROI one-sample among language electrodes
    if "region" in d.columns and lang.any():
        lang_df = d[lang & d["paper_quality_ok"]]
        for i, (reg, sub) in enumerate(lang_df.groupby("region")):
            if sub["subject"].nunique() < min_group_n:
                rows.append({
                    "family": "roi_onesample",
                    "test": "sae_gain_vs_zero",
                    "group_a": str(reg), "group_b": "zero",
                    "value_col": "paper_sae_gain",
                    "n_a": int(len(sub)), "n_b": 0,
                    "mean_diff": np.nan, "perm_p": np.nan,
                    "n_subjects_used": int(sub["subject"].nunique()),
                    "method": "insufficient",
                })
                continue
            _add_one_sample(sub, "roi_onesample", str(reg), 400 + i)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_fdr"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "q_fdr"] = fdr_bh(out.loc[idx, "perm_p"].to_numpy())
    return out


def overlap_paired_subject_value_diff(
    df: pd.DataFrame,
    value_col: str,
    group_a: str,
    group_b: str,
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
) -> Dict:
    """Paired subject-mean difference for overlapping task-positive groups.

    For each subject with ≥1 electrode in both groups, compute
    mean(value|group_a) − mean(value|group_b), then sign-flip permute.
    """
    col_a = OVERLAP_TASK_GROUPS[group_a]
    col_b = OVERLAP_TASK_GROUPS[group_b]
    if col_a not in df.columns or col_b not in df.columns:
        return {
            "group_a": group_a, "group_b": group_b,
            "n_a": 0, "n_b": 0, "mean_diff": np.nan,
            "perm_p": np.nan, "n_subjects_used": 0, "method": "insufficient",
        }
    diffs = []
    n_a = n_b = 0
    for _, sdf in df.groupby("subject"):
        a = sdf.loc[sdf[col_a], value_col].dropna()
        b = sdf.loc[sdf[col_b], value_col].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        n_a += len(a)
        n_b += len(b)
        diffs.append(float(a.mean() - b.mean()))
    n_subj = len(diffs)
    if n_subj < MIN_GROUP_N:
        return {
            "group_a": group_a, "group_b": group_b,
            "n_a": n_a, "n_b": n_b, "mean_diff": np.nan,
            "perm_p": np.nan, "n_subjects_used": n_subj, "method": "insufficient",
        }
    diffs_arr = np.asarray(diffs, dtype=float)
    obs = float(np.mean(diffs_arr))
    rng = np.random.default_rng(rng_seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subj)
        nulls[i] = float(np.mean(diffs_arr * signs))
    return {
        "group_a": group_a, "group_b": group_b,
        "n_a": n_a, "n_b": n_b, "mean_diff": obs,
        "perm_p": empirical_p(obs, nulls, alternative="two-sided"),
        "n_subjects_used": n_subj, "method": "paired_subject_sign_flip",
    }


def run_task_type_pairwise_stats(
    df: pd.DataFrame,
    n_perm: int = N_PERM,
    min_group_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """Pairwise exclusive / overlapping task-type dominance comparisons.

    Families (FDR within family):
      1. exclusive continuous: dominance_index / sae_gain / surprisal_gain
      2. exclusive binary: surprisal_dominant / sae_dominant proportions
      3. overlap paired continuous on the same value columns
      4. overlap paired binary proportions
    """
    import itertools

    rows: List[Dict] = []
    value_cols = ("dominance_index", "sae_gain", "surprisal_gain")
    binary_cols = ("surprisal_dominant", "sae_dominant")

    # Ensure task_group present
    d = df.copy()
    if "task_group" not in d.columns and "task_class" in d.columns:
        d["task_group"] = d["task_class"].map(collapse_task_group)

    # Family 1–2: exclusive pairwise
    if "task_group" in d.columns:
        present = [g for g in EXCLUSIVE_TASK_GROUPS if (d["task_group"] == g).any()]
        for a, b in itertools.combinations(present, 2):
            for vcol in value_cols:
                res = subject_stratified_perm_diff(
                    d, vcol, "task_group", a, b,
                    n_perm=n_perm,
                    rng_seed=RNG_SEED + hash((a, b, vcol)) % 10000,
                )
                res = _gate_min_subjects(res, min_group_n)
                rows.append({
                    "family": "exclusive_continuous",
                    "test": "task_group_pairwise",
                    "value_col": vcol,
                    "predictor": "task_group",
                    **res,
                })
            for vcol in binary_cols:
                # Cast bool → float for mean = proportion
                tmp = d.copy()
                tmp[vcol] = tmp[vcol].astype(float)
                res = subject_stratified_perm_diff(
                    tmp, vcol, "task_group", a, b,
                    n_perm=n_perm,
                    rng_seed=RNG_SEED + 31 + hash((a, b, vcol)) % 10000,
                )
                res = _gate_min_subjects(res, min_group_n)
                rows.append({
                    "family": "exclusive_binary",
                    "test": "task_group_prop_pairwise",
                    "value_col": vcol,
                    "predictor": "task_group",
                    **res,
                })

    # Family 3–4: overlapping paired
    for a, b in itertools.combinations(OVERLAP_TASK_GROUPS.keys(), 2):
        for vcol in value_cols:
            res = overlap_paired_subject_value_diff(
                d, vcol, a, b, n_perm=n_perm,
                rng_seed=RNG_SEED + 97 + hash((a, b, vcol)) % 10000,
            )
            res = _gate_min_subjects(res, min_group_n)
            rows.append({
                "family": "overlap_continuous",
                "test": "overlap_paired_continuous",
                "value_col": vcol,
                "predictor": "is_pos_*",
                **res,
            })
        for vcol in binary_cols:
            tmp = d.copy()
            tmp[vcol] = tmp[vcol].astype(float)
            res = overlap_paired_subject_value_diff(
                tmp, vcol, a, b, n_perm=n_perm,
                rng_seed=RNG_SEED + 131 + hash((a, b, vcol)) % 10000,
            )
            res = _gate_min_subjects(res, min_group_n)
            rows.append({
                "family": "overlap_binary",
                "test": "overlap_paired_prop",
                "value_col": vcol,
                "predictor": "is_pos_*",
                **res,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_fdr"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "q_fdr"] = fdr_bh(out.loc[idx, "perm_p"].to_numpy())
    return out


def dominance_label_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-bootstrapable dominance-label proportions by task grouping."""
    rows = []
    # Exclusive
    if "task_group" in df.columns:
        for g, sub in df.groupby("task_group"):
            for label in ("surprisal_dominant", "sae_dominant", "balanced",
                          "low_performing", "unclassified"):
                # per-subject fraction then mean
                frac = sub.assign(is_lab=sub["dominance_label"] == label).groupby(
                    "subject")["is_lab"].mean()
                mu, lo, hi, n = bootstrap_subject_means(
                    frac.reset_index().rename(columns={"is_lab": "frac"}),
                    "frac", n_boot=N_BOOT, seed=RNG_SEED,
                )
                rows.append({
                    "grouping": "exclusive",
                    "group": g,
                    "dominance_label": label,
                    "n_electrodes": int((sub["dominance_label"] == label).sum()),
                    "n_subjects": n,
                    "frac_subject_mean": mu,
                    "frac_ci_lo": lo,
                    "frac_ci_hi": hi,
                })
    # Overlap
    for gname, col in OVERLAP_TASK_GROUPS.items():
        if col not in df.columns:
            continue
        sub = df[df[col]]
        for label in ("surprisal_dominant", "sae_dominant", "balanced",
                      "low_performing", "unclassified"):
            if sub.empty:
                rows.append({
                    "grouping": "overlap", "group": gname,
                    "dominance_label": label, "n_electrodes": 0,
                    "n_subjects": 0, "frac_subject_mean": np.nan,
                    "frac_ci_lo": np.nan, "frac_ci_hi": np.nan,
                })
                continue
            frac = sub.assign(is_lab=sub["dominance_label"] == label).groupby(
                "subject")["is_lab"].mean()
            mu, lo, hi, n = bootstrap_subject_means(
                frac.reset_index().rename(columns={"is_lab": "frac"}),
                "frac", n_boot=N_BOOT, seed=RNG_SEED,
            )
            rows.append({
                "grouping": "overlap",
                "group": gname,
                "dominance_label": label,
                "n_electrodes": int((sub["dominance_label"] == label).sum()),
                "n_subjects": n,
                "frac_subject_mean": mu,
                "frac_ci_lo": lo,
                "frac_ci_hi": hi,
            })
    return pd.DataFrame(rows)


# ======================================================================
# Candidate ranking
# ======================================================================

def rank_candidates(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Preregistered confirmatory ranking (Qwen-only, unrestricted ROI).

    Criteria: quality-gated surprisal dominance, positive full R, content
    support, fold agreement (via quality_ok). Method agreement is a bonus
    when prior deconv exists, but does not exclude non-language channels.
    Explicitly mixes top non-language candidates so confirmatory deconv is
    not restricted to ``is_lang`` ROIs.
    """
    c = df[df["surprisal_dominant"]].copy()
    if c.empty:
        return c
    c["is_lang_bool"] = (
        c["is_lang"].fillna(False).astype(bool) if "is_lang" in c.columns
        else False
    )
    c["rank_score"] = c["surprisal_gain"].astype(float)
    if "method_agree_surprisal" in c.columns:
        c["rank_score"] = (
            c["rank_score"]
            + 0.25 * c["method_agree_surprisal"].astype(float)
            + c["deconv_surprisal_gain"].fillna(0).clip(lower=0)
        )
    c = c.sort_values(
        ["rank_score", "surprisal_gain", "full_R"],
        ascending=[False, False, False],
    )
    # Reserve ~40% of slots for non-language channels (outside is_lang ROIs).
    n_nonlang = max(8, int(round(0.4 * top_n)))
    nonlang = c[~c["is_lang_bool"]].head(n_nonlang)
    remaining = top_n - len(nonlang)
    lang_or_rest = c[~c.index.isin(nonlang.index)].head(max(0, remaining))
    out = pd.concat([lang_or_rest, nonlang], ignore_index=False)
    out = out.sort_values(
        ["rank_score", "surprisal_gain", "full_R"],
        ascending=[False, False, False],
    ).head(top_n)
    out = out.copy()
    out["candidate_rank"] = np.arange(1, len(out) + 1)
    return out


# ======================================================================
# Reporting / figures
# ======================================================================

def write_report(
    df: pd.DataFrame,
    by_loc: pd.DataFrame,
    stats: pd.DataFrame,
    profiles: pd.DataFrame,
    candidates: pd.DataFrame,
    cov: Dict[str, int],
    path: Path,
    task_stats: Optional[pd.DataFrame] = None,
    label_props: Optional[pd.DataFrame] = None,
):
    lines = []
    a = lines.append
    a("=" * 70)
    a("QWEN ALL-CHANNEL ANALYSIS — EXPLORATORY DOMINANCE EXTENSION")
    a("=" * 70)
    a(f"Feature tag: {FEATURE_TAG}")
    a(f"Electrodes: {len(df)}  subjects: {df['subject'].nunique()}")
    a("")
    a("NOTE: Lepori et al. primary contrast is full vs surprisal-only")
    a("(paper_sae_gain). See sparse_encoding_paper_aligned.py for that report.")
    a("This file retains exploratory dominance_index / surprisal_dominant")
    a("labels (full−content), which are NOT defined by Lepori et al.")
    a("")
    a("-" * 70)
    a("MERGE / COVERAGE")
    a("-" * 70)
    for k, v in cov.items():
        a(f"  {k}: {v}")
    a("")
    a("-" * 70)
    a("DOMINANCE CENSUS (quality-gated unless noted)")
    a("-" * 70)
    a(f"  surprisal_dominant (quality-gated): {int(df['surprisal_dominant'].sum())}")
    a(f"  sae_dominant (quality-gated):       {int(df['sae_dominant'].sum())}")
    a(f"  balanced:                           {int(df['balanced'].sum())}")
    a(f"  low_performing:                     {int(df['low_performing'].sum())}")
    a(f"  descriptive surprisal_dominant:     {int(df['surprisal_dominant_desc'].sum())}")
    a(f"  content_collapsed (excluded from confirmatory): "
      f"{int(df['content_collapsed'].sum())}")
    a(f"  quality_ok:                         {int(df['quality_ok'].sum())}")
    if "is_lang" in df.columns:
        lang = language_mask(df)
        a(f"  surprisal_dominant ∩ is_lang:       "
          f"{int((df['surprisal_dominant'] & lang).sum())}")
        a(f"  surprisal_dominant ∩ non-lang:      "
          f"{int((df['surprisal_dominant'] & ~lang).sum())}")
    if "method_agree_surprisal" in df.columns:
        a(f"  word-locked ∩ deconv surprisal agree: "
          f"{int(df['method_agree_surprisal'].sum())}")
    a("")
    # Subject-level means
    a("-" * 70)
    a("SUBJECT-LEVEL MEANS (all channels)")
    a("-" * 70)
    subj = df.groupby("subject")[["full_R", "sae_gain", "surprisal_gain",
                                  "dominance_index"]].mean()
    a(subj.round(4).to_string())
    a("")
    for col in ("dominance_index", "surprisal_gain", "sae_gain"):
        mu, lo, hi, n = bootstrap_subject_means(
            subj.reset_index(), col, n_boot=N_BOOT, seed=RNG_SEED)
        a(f"  {col}: mean={mu:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  n={n}")
    a("")
    a("-" * 70)
    a("BY LOCALIZER TASK CLASS / DOMINANT TASK / TASK GROUP")
    a("-" * 70)
    if not by_loc.empty:
        show = by_loc[by_loc["group_col"].isin(
            ["task_class", "task_group", "dominant_task"])]
        a(show.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("SUBJECT-AWARE LOCALIZER ASSOCIATION (FDR across planned tests)")
    a("-" * 70)
    a("Inferential unit = subject. Descriptive channel rankings are separate.")
    if stats.empty:
        a("  (no tests)")
    else:
        a(stats.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("TASK-TYPE PAIRWISE DOMINANCE COMPARISONS")
    a("-" * 70)
    a(f"MIN_GROUP_N={MIN_GROUP_N}; FDR within each family separately.")
    a("Exclusive: mutually exclusive task_group. Overlap: paired subject means.")
    if task_stats is None or task_stats.empty:
        a("  (no tests)")
    else:
        a(task_stats.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("DOMINANCE-LABEL PROPORTIONS BY TASK GROUP")
    a("-" * 70)
    if label_props is None or label_props.empty:
        a("  (none)")
    else:
        a(label_props.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("MITSWJN CONDITION PROFILES BY DOMINANCE LABEL (subject means)")
    a("-" * 70)
    if not profiles.empty:
        a(profiles.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("TOP CONFIRMATORY CANDIDATES (Qwen quality-gated)")
    a("-" * 70)
    if candidates.empty:
        a("  (none)")
    else:
        cols = [c for c in (
            "candidate_rank", "subject", "channel", "region", "task_class",
            "task_group", "dominant_task", "is_lang", "full_R", "surprisal_gain",
            "sae_gain", "dominance_index", "method_agree_surprisal",
            "deconv_surprisal_gain", "deconv_sae_gain",
        ) if c in candidates.columns]
        a(candidates[cols].round(4).to_string(index=False))
    a("")
    a("NOTE: Channel rankings are descriptive. Population inference uses")
    a("subject-level tests above. Confirmatory deconvolution nulls are required")
    a("before channel-level significance claims.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def fig_dominance_by_task(by_loc: pd.DataFrame, outdir: Path):
    sub = by_loc[by_loc["group_col"] == "task_class"].copy()
    if sub.empty:
        return
    sub = sub.sort_values("dominance_index_subject_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["dominance_index_subject_mean"],
            xerr=[
                sub["dominance_index_subject_mean"] - sub["dominance_index_ci_lo"],
                sub["dominance_index_ci_hi"] - sub["dominance_index_subject_mean"],
            ],
            capsize=3, color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{g} (nE={int(n)}, nS={int(s)})"
        for g, n, s in zip(sub["group"], sub["n_electrodes"], sub["n_subjects"])
    ], fontsize=8)
    ax.set_xlabel("dominance index (surprisal_gain − SAE_gain), subject-mean")
    ax.set_title("Qwen dominance index by localizer task class")
    fig.tight_layout()
    p = outdir / "qwen_dominance_by_task_class.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_dominance_vs_contrast(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (task, col) in zip(axes, TASK_CONTRAST.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        g = df.dropna(subset=[col, "dominance_index"]).groupby("subject")[
            [col, "dominance_index"]].mean()
        ax.scatter(g[col], g["dominance_index"], s=40, alpha=0.8)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        if len(g) >= 3:
            r, _ = sp_stats.spearmanr(g[col], g["dominance_index"])
            ax.set_title(f"{task}\nρ={r:+.2f} (n={len(g)} subjects)")
        else:
            ax.set_title(task)
        ax.set_xlabel("task contrast (subject-mean)")
    axes[0].set_ylabel("dominance index (subject-mean)")
    fig.suptitle("Qwen dominance vs localizer contrasts", y=1.02)
    fig.tight_layout()
    p = outdir / "qwen_dominance_vs_localizer_contrasts.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def fig_mitswjn_profiles(profiles: pd.DataFrame, outdir: Path):
    if profiles.empty:
        return
    labels = ["surprisal_dominant", "sae_dominant", "balanced", "low_performing"]
    conds = ["Sentences", "Nonwords", "Word-lists", "Jabberwocky"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(conds))
    width = 0.2
    for i, lab in enumerate(labels):
        sub = profiles[profiles["dominance_label"] == lab]
        if sub.empty:
            continue
        means = [float(sub.loc[sub["condition"] == c, "mean"].iloc[0])
                 if (sub["condition"] == c).any() else np.nan for c in conds]
        ax.bar(x + (i - 1.5) * width, means, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15, ha="right")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("MITSWJN high-gamma (subject-mean)")
    ax.set_title("MITSWJN condition profiles by Qwen dominance label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "qwen_mitswjn_profiles_by_dominance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_anatomy_top(df: pd.DataFrame, outdir: Path, min_n: int = 5):
    if "region" not in df.columns:
        return
    g = df[df["surprisal_dominant"]].groupby("region").agg(
        n=("channel", "size"),
        n_subj=("subject", "nunique"),
        mean_surp=("surprisal_gain", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_n].sort_values("mean_surp", ascending=False).head(15)
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(g))
    ax.barh(y, g["mean_surp"], color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{r} (n={n}, s={s})" for r, n, s in zip(g["region"], g["n"], g["n_subj"])
    ], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean surprisal gain (electrode-level, descriptive)")
    ax.set_title("Regions enriched for Qwen quality-gated surprisal-dominant channels")
    fig.tight_layout()
    p = outdir / "qwen_surprisal_dominant_by_region.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  {p}")


def fig_dominance_label_by_task(label_props: pd.DataFrame, outdir: Path):
    """Stacked bars of dominance-label composition by exclusive / overlap groups."""
    if label_props is None or label_props.empty:
        return
    labels = ["surprisal_dominant", "sae_dominant", "balanced",
              "low_performing", "unclassified"]
    colors = {
        "surprisal_dominant": "#d62728",
        "sae_dominant": "#2ca02c",
        "balanced": "#1f77b4",
        "low_performing": "#7f7f7f",
        "unclassified": "#c7c7c7",
    }
    for grouping, title, fname in (
        ("exclusive", "exclusive task_group",
         "qwen_dominance_label_by_task_group.png"),
        ("overlap", "overlapping task-positive",
         "qwen_dominance_label_by_task_positive.png"),
    ):
        sub = label_props[label_props["grouping"] == grouping]
        if sub.empty:
            continue
        groups = list(sub["group"].unique())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        for lab in labels:
            vals = []
            for g in groups:
                row = sub[(sub["group"] == g) & (sub["dominance_label"] == lab)]
                vals.append(float(row["frac_subject_mean"].iloc[0])
                            if len(row) else 0.0)
            vals = np.asarray(vals, dtype=float)
            ax.bar(x, vals, bottom=bottom, label=lab, color=colors[lab])
            bottom = bottom + np.nan_to_num(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("subject-mean fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Qwen dominance-label composition ({title})")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        p = outdir / fname
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
    p.add_argument(
        "--deconv_csv",
        default=str(ap.SAE_TABLES / f"sparse_encoding_deconv_{FEATURE_TAG}_lang.csv"),
    )
    p.add_argument("--n_perm", type=int, default=N_PERM)
    p.add_argument("--top_n", type=int, default=25)
    p.add_argument("--write_candidates", action="store_true", default=True,
                   help="Write candidate channel list CSV for deconv confirmation.")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    raw = pd.read_csv(results_csv)
    raw["channel"] = normalize_channel(raw["channel"])
    # Taxonomy annotations (region / is_lang / category); normalize keys first
    if ap.TAX_ELECTRODE_TABLE.exists():
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        tax["channel"] = normalize_channel(tax["channel"])
        keep = [c for c in ("subject", "channel", "region", "category",
                            "hemisphere", "is_lang") if c in tax.columns]
        raw = raw.merge(tax[keep], on=["subject", "channel"], how="left")
    else:
        raw = merge_taxonomy(raw)
    raw["channel"] = normalize_channel(raw["channel"])

    print(f"Loaded {len(raw)} Qwen electrodes from {results_csv.name}")
    d = classify_dominance(raw)
    d, cov = merge_localizers(d)
    d = merge_deconv(d, Path(args.deconv_csv) if args.deconv_csv else None)

    # Group summaries
    pieces = [group_summary(d, "dominance_label")]
    if "task_class" in d.columns:
        pieces.append(group_summary(d, "task_class"))
    if "task_group" in d.columns:
        pieces.append(group_summary(d, "task_group"))
    if "dominant_task" in d.columns:
        pieces.append(group_summary(d, "dominant_task"))
    for task, pos in TASK_POS.items():
        if pos in d.columns:
            tmp = d.copy()
            tmp["_g"] = np.where(tmp[pos], f"{task}+", f"{task}-")
            pieces.append(group_summary(tmp, "_g"))
    by_loc = pd.concat(pieces, ignore_index=True)

    print(f"Running subject-aware localizer stats (n_perm={args.n_perm})…")
    stats = run_localizer_stats(d, n_perm=args.n_perm)
    print(f"Running task-type pairwise dominance stats (n_perm={args.n_perm})…")
    task_stats = run_task_type_pairwise_stats(d, n_perm=args.n_perm)
    label_props = dominance_label_by_task(d)
    profiles = condition_profiles(d)
    candidates = rank_candidates(d, top_n=args.top_n)

    # Save tables
    out_all = ap.SAE_DOMINANCE_TABLE
    d.to_csv(out_all, index=False)
    print(f"Saved {len(d)} rows -> {out_all}")

    by_loc.to_csv(ap.SAE_DOMINANCE_BY_LOCALIZER, index=False)
    print(f"Saved -> {ap.SAE_DOMINANCE_BY_LOCALIZER}")

    stats.to_csv(ap.SAE_DOMINANCE_LOCALIZER_STATS, index=False)
    print(f"Saved -> {ap.SAE_DOMINANCE_LOCALIZER_STATS}")

    task_stats.to_csv(ap.SAE_DOMINANCE_TASK_TYPE_STATS, index=False)
    print(f"Saved -> {ap.SAE_DOMINANCE_TASK_TYPE_STATS}")

    label_props_path = ap.SAE_TABLES / "sparse_encoding_dominance_label_by_task.csv"
    label_props.to_csv(label_props_path, index=False)
    print(f"Saved -> {label_props_path}")

    profiles_path = ap.SAE_TABLES / "sparse_encoding_dominance_mitswjn_profiles.csv"
    profiles.to_csv(profiles_path, index=False)

    cand_path = ap.SAE_TABLES / "sparse_encoding_dominance_candidates_qwen.csv"
    candidates.to_csv(cand_path, index=False)
    # Minimal channel list for deconv --channels_from_csv
    chan_list = candidates[["subject", "channel"]].drop_duplicates()
    chan_list_path = ap.SAE_TABLES / "sparse_encoding_dominance_candidate_channels.csv"
    chan_list.to_csv(chan_list_path, index=False)
    print(f"Saved candidates -> {cand_path} ({len(candidates)} rows)")
    print(f"Saved channel list -> {chan_list_path}")

    report = ap.SAE_REPORTS / "sparse_encoding_surprisal_dominance.txt"
    write_report(
        d, by_loc, stats, profiles, candidates, cov, report,
        task_stats=task_stats, label_props=label_props,
    )

    print("Figures:")
    fig_dominance_by_task(by_loc, ap.SAE_FIGURES)
    fig_dominance_vs_contrast(d, ap.SAE_FIGURES)
    fig_mitswjn_profiles(profiles, ap.SAE_FIGURES)
    fig_anatomy_top(d, ap.SAE_FIGURES)
    fig_dominance_label_by_task(label_props, ap.SAE_FIGURES)


if __name__ == "__main__":
    main()
