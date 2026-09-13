#!/usr/bin/env python3
"""
sparse_encoding_task_type_analysis.py
=====================================
Compare Qwen SAE feature sets across MITSWJN / WM / DMN task types using
unsigned and signed Jaccard indices, with subject-aware inference.

Primary grouping (mutually exclusive)
-------------------------------------
``mitswjn_only``, ``wm_only``, ``dmn_only``, collapsed ``multi_task``;
``none`` retained as a reference / sensitivity group.

Sensitivity grouping (overlapping)
----------------------------------
``MITSWJN+``, ``WM+``, ``DMN+``. Electrode self-pairs that belong to both
groups are excluded from between-group pairwise Jaccard.

Inference
---------
Group-union Jaccard is descriptive (sample-size dependent). Inferential
tests use electrode-pair similarities aggregated / permuted within subject,
with ``MIN_GROUP_N`` gating and FDR within predeclared test families.

Prerequisite
------------
Run qualitative extraction for all eligible Qwen channels first::

    python sparse_encoding_qualitative.py \\
        --feature_tag sae_qwen3_8b_l18 \\
        --results_csv .../sparse_encoding_results_sae_qwen3_8b_l18_cohort.csv \\
        --all_channels --top_n 0 \\
        --suffix _sae_qwen3_8b_l18_all_channels

Example
-------
    python sparse_encoding_task_type_analysis.py
    python sparse_encoding_task_type_analysis.py --n_perm 500 --min_group_n 3
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_qualitative import (
    parse_signed_features,
    parse_unsigned_features,
)
from sparse_encoding.sparse_encoding_surprisal_dominance import (
    FEATURE_TAG as QWEN_TAG,
    EXCLUSIVE_TASK_GROUPS,
    MIN_GROUP_N,
    N_PERM,
    OVERLAP_TASK_GROUPS,
    RNG_SEED,
    TASK_POS,
    assign_task_class,
    collapse_task_group,
    empirical_p,
    fdr_bh,
    normalize_channel,
)

FEATURE_TAG = QWEN_TAG
EXCLUSIVE_GROUPS = EXCLUSIVE_TASK_GROUPS
EXCLUSIVE_WITH_NONE = EXCLUSIVE_GROUPS + ("none",)
OVERLAP_GROUPS = tuple(OVERLAP_TASK_GROUPS.keys())
OVERLAP_POS = OVERLAP_TASK_GROUPS


def attach_task_metadata(feat: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Merge cross-task localizer flags onto qualitative feature rows."""
    cov = {
        "n_feature_rows": len(feat),
        "n_merged_localizer": 0,
        "n_unmatched": 0,
        "n_unknown_task_group": 0,
    }
    d = feat.copy()
    d["channel"] = normalize_channel(d["channel"])
    d["subject"] = d["subject"].astype(str)

    path = ap.CROSS_TASK_RESPONSES
    if not path.exists():
        warnings.warn(f"{path} missing; task metadata unavailable")
        d["task_class"] = "unknown"
        d["task_group"] = "unknown"
        for col in TASK_POS.values():
            d[col] = False
        cov["n_unmatched"] = len(d)
        cov["n_unknown_task_group"] = len(d)
        return d, cov

    loc = pd.read_csv(path)
    loc["channel"] = normalize_channel(loc["channel"])
    loc["subject"] = loc["subject"].astype(str)
    keep = ["subject", "channel"] + [c for c in TASK_POS.values() if c in loc.columns]
    loc = loc[keep].drop_duplicates(["subject", "channel"])
    merged = d.merge(loc, on=["subject", "channel"], how="left", indicator=True)
    cov["n_merged_localizer"] = int((merged["_merge"] == "both").sum())
    cov["n_unmatched"] = int((merged["_merge"] == "left_only").sum())
    merged = merged.drop(columns=["_merge"])
    for col in TASK_POS.values():
        if col not in merged.columns:
            merged[col] = False
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["task_class"] = merged.apply(assign_task_class, axis=1)
    merged["task_group"] = merged["task_class"].map(collapse_task_group)
    cov["n_unknown_task_group"] = int((merged["task_group"] == "unknown").sum())
    return merged, cov


def channels_in_overlap_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    col = OVERLAP_POS[group]
    return df[df[col]].copy()


# ======================================================================
# Jaccard helpers
# ======================================================================

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return np.nan
    union = a | b
    if not union:
        return np.nan
    return float(len(a & b) / len(union))


def feature_sets_from_row(row: pd.Series) -> Tuple[Set[int], Set[Tuple[int, int]]]:
    raw = row.get("feature_indices", "")
    signed = parse_signed_features(raw)
    unsigned = {idx for idx, _ in signed}
    return unsigned, signed


def group_union_sets(
    df: pd.DataFrame, group_col: str, groups: Sequence[str],
) -> Dict[str, Dict[str, Set]]:
    """Return unsigned/signed unions per group."""
    out: Dict[str, Dict[str, Set]] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        u: Set[int] = set()
        s: Set[Tuple[int, int]] = set()
        for _, row in sub.iterrows():
            uu, ss = feature_sets_from_row(row)
            u |= uu
            s |= ss
        out[g] = {
            "unsigned": u,
            "signed": s,
            "n_electrodes": len(sub),
            "n_subjects": int(sub["subject"].nunique()) if len(sub) else 0,
        }
    return out


def compute_union_jaccard_table(
    df: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    grouping_label: str,
    feature_tag: str,
) -> pd.DataFrame:
    unions = group_union_sets(df, group_col, groups)
    rows = []
    for a, b in itertools.combinations(groups, 2):
        ua, ub = unions[a]["unsigned"], unions[b]["unsigned"]
        sa, sb = unions[a]["signed"], unions[b]["signed"]
        rows.append({
            "grouping": grouping_label,
            "group_col": group_col,
            "group_a": a,
            "group_b": b,
            "jaccard_unsigned": jaccard(ua, ub),
            "jaccard_signed": jaccard(sa, sb),
            "n_union_a_unsigned": len(ua),
            "n_union_b_unsigned": len(ub),
            "n_intersection_unsigned": len(ua & ub),
            "n_union_a_signed": len(sa),
            "n_union_b_signed": len(sb),
            "n_intersection_signed": len(sa & sb),
            "n_electrodes_a": unions[a]["n_electrodes"],
            "n_electrodes_b": unions[b]["n_electrodes"],
            "n_subjects_a": unions[a]["n_subjects"],
            "n_subjects_b": unions[b]["n_subjects"],
            "feature_tag": feature_tag,
            "note": "descriptive_group_union_sample_size_dependent",
        })
    return pd.DataFrame(rows)


def _channel_key(row: pd.Series) -> Tuple[str, str]:
    return str(row["subject"]), str(row["channel"]).strip()


def compute_pairwise_electrode_jaccard(
    df: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    grouping_label: str,
    feature_tag: str,
    exclude_self_pairs: bool = False,
) -> pd.DataFrame:
    """All within-/between-group electrode pairs (same-subject pairs only).

    Between-group pairs are restricted to the same subject to avoid
    cross-subject pseudo-replication in the primary similarity summary.
    """
    packs: Dict[str, List[Tuple[Tuple[str, str], Set[int], Set[Tuple[int, int]]]]] = {
        g: [] for g in groups
    }
    for g in groups:
        sub = df[df[group_col] == g]
        for _, row in sub.iterrows():
            u, s = feature_sets_from_row(row)
            packs[g].append((_channel_key(row), u, s))

    rows = []
    # Within-group
    for g in groups:
        items = packs[g]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (sa, ca), ua, sga = items[i]
                (sb, cb), ub, sgb = items[j]
                if sa != sb:
                    continue  # within-subject pairs only
                if exclude_self_pairs and (sa, ca) == (sb, cb):
                    continue
                rows.append({
                    "grouping": grouping_label,
                    "pair_type": "within",
                    "group_a": g,
                    "group_b": g,
                    "subject": sa,
                    "channel_a": ca,
                    "channel_b": cb,
                    "jaccard_unsigned": jaccard(ua, ub),
                    "jaccard_signed": jaccard(sga, sgb),
                    "n_shared_unsigned": len(ua & ub),
                    "n_shared_signed": len(sga & sgb),
                    "feature_tag": feature_tag,
                })
    # Between-group
    for a, b in itertools.combinations(groups, 2):
        for (sa, ca), ua, sga in packs[a]:
            for (sb, cb), ub, sgb in packs[b]:
                if sa != sb:
                    continue
                if exclude_self_pairs and (sa, ca) == (sb, cb):
                    continue
                rows.append({
                    "grouping": grouping_label,
                    "pair_type": "between",
                    "group_a": a,
                    "group_b": b,
                    "subject": sa,
                    "channel_a": ca,
                    "channel_b": cb,
                    "jaccard_unsigned": jaccard(ua, ub),
                    "jaccard_signed": jaccard(sga, sgb),
                    "n_shared_unsigned": len(ua & ub),
                    "n_shared_signed": len(sga & sgb),
                    "feature_tag": feature_tag,
                })
    return pd.DataFrame(rows)


def summarize_pairwise_by_subject(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Mean within/between Jaccard per subject × grouping × metric family."""
    if pair_df.empty:
        return pd.DataFrame()
    rows = []
    for (grouping, subject), g in pair_df.groupby(["grouping", "subject"]):
        for metric in ("jaccard_unsigned", "jaccard_signed"):
            within = g.loc[g["pair_type"] == "within", metric].dropna()
            between = g.loc[g["pair_type"] == "between", metric].dropna()
            rows.append({
                "grouping": grouping,
                "subject": subject,
                "metric": metric,
                "within_mean": float(within.mean()) if len(within) else np.nan,
                "between_mean": float(between.mean()) if len(between) else np.nan,
                "within_minus_between": (
                    float(within.mean() - between.mean())
                    if len(within) and len(between) else np.nan
                ),
                "n_within_pairs": int(len(within)),
                "n_between_pairs": int(len(between)),
            })
    return pd.DataFrame(rows)


# ======================================================================
# Subject-aware inference
# ======================================================================

def subject_mean_diff_perm(
    subject_df: pd.DataFrame,
    value_col: str,
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
) -> Dict:
    """One-sample permutation test of subject-mean ``value_col`` vs 0."""
    vals = subject_df[value_col].dropna().to_numpy(float)
    n = len(vals)
    if n < MIN_GROUP_N:
        return {
            "n_subjects": n,
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
        "n_subjects": n,
        "mean_diff": obs,
        "perm_p": empirical_p(obs, nulls, alternative="two-sided"),
        "method": "sign_flip_subjects",
    }


def exclusive_pair_similarity_perm(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    metric: str,
    n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED,
) -> Dict:
    """Within-subject label shuffle for exclusive group pair similarity.

    For each subject with electrodes in both groups, compute mean between-group
    Jaccard; observed statistic = mean across subjects. Null: shuffle group
    labels within subject among electrodes in {a,b}.
    """
    sub = df[df["task_group"].isin([group_a, group_b])].copy()
    packs = []
    for subj, sdf in sub.groupby("subject"):
        if set(sdf["task_group"].unique()) < {group_a, group_b}:
            continue
        sets = []
        labels = []
        for _, row in sdf.iterrows():
            u, s = feature_sets_from_row(row)
            sets.append(u if metric == "jaccard_unsigned" else s)
            labels.append(row["task_group"])
        packs.append((subj, sets, np.asarray(labels, dtype=object)))
    if len(packs) < MIN_GROUP_N:
        return {
            "group_a": group_a, "group_b": group_b, "metric": metric,
            "n_subjects_used": len(packs),
            "mean_between": np.nan, "perm_p": np.nan, "method": "insufficient",
        }

    def _mean_between(sets, labels) -> float:
        vals = []
        idx_a = np.flatnonzero(labels == group_a)
        idx_b = np.flatnonzero(labels == group_b)
        for i in idx_a:
            for j in idx_b:
                vals.append(jaccard(sets[i], sets[j]))
        return float(np.nanmean(vals)) if vals else np.nan

    obs_per = [_mean_between(sets, labs) for _, sets, labs in packs]
    obs = float(np.nanmean(obs_per))
    rng = np.random.default_rng(rng_seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        per = []
        for _, sets, labs in packs:
            lab = labs.copy()
            rng.shuffle(lab)
            per.append(_mean_between(sets, lab))
        nulls[i] = float(np.nanmean(per))
    return {
        "group_a": group_a, "group_b": group_b, "metric": metric,
        "n_subjects_used": len(packs),
        "mean_between": obs,
        "perm_p": empirical_p(obs, nulls, alternative="two-sided"),
        "method": "within_subject_label_shuffle",
    }


def overlap_paired_subject_means(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    metric: str = "jaccard_unsigned",
) -> pd.DataFrame:
    """Per-subject mean within-group Jaccard for two overlapping groups."""
    rows = []
    for subj, sdf in df.groupby("subject"):
        ga = channels_in_overlap_group(sdf, group_a)
        gb = channels_in_overlap_group(sdf, group_b)
        if len(ga) < 2 and len(gb) < 2:
            continue

        def _within_mean(sub: pd.DataFrame) -> float:
            vals = []
            items = [feature_sets_from_row(r) for _, r in sub.iterrows()]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    ua, sa = items[i]
                    ub, sb = items[j]
                    vals.append(jaccard(ua, ub) if metric == "jaccard_unsigned"
                                else jaccard(sa, sb))
            return float(np.nanmean(vals)) if vals else np.nan

        wa = _within_mean(ga) if len(ga) >= 2 else np.nan
        wb = _within_mean(gb) if len(gb) >= 2 else np.nan
        if not (np.isfinite(wa) and np.isfinite(wb)):
            continue
        rows.append({
            "subject": subj,
            "group_a": group_a,
            "group_b": group_b,
            "metric": metric,
            "within_a": wa,
            "within_b": wb,
            "diff_a_minus_b": wa - wb,
            "n_a": len(ga),
            "n_b": len(gb),
        })
    return pd.DataFrame(rows)


def run_jaccard_stats(
    df: pd.DataFrame,
    subject_summary: pd.DataFrame,
    n_perm: int,
    min_group_n: int,
    feature_tag: str,
    rng_seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Predeclared Jaccard test families with within-family FDR."""
    rows: List[Dict] = []

    # Family 1: exclusive within > between (subject means)
    excl = subject_summary[subject_summary["grouping"] == "exclusive"]
    for metric in ("jaccard_unsigned", "jaccard_signed"):
        sub = excl[excl["metric"] == metric].dropna(subset=["within_minus_between"])
        if sub["subject"].nunique() < min_group_n:
            rows.append({
                "family": "exclusive_within_vs_between",
                "grouping": "exclusive",
                "metric": metric,
                "group_a": "within", "group_b": "between",
                "n_subjects_used": int(sub["subject"].nunique()),
                "mean_diff": np.nan, "perm_p": np.nan,
                "method": "insufficient", "feature_tag": feature_tag,
            })
            continue
        res = subject_mean_diff_perm(
            sub, "within_minus_between", n_perm=n_perm,
            rng_seed=rng_seed + hash(metric) % 1000,
        )
        rows.append({
            "family": "exclusive_within_vs_between",
            "grouping": "exclusive",
            "metric": metric,
            "group_a": "within", "group_b": "between",
            "n_subjects_used": res["n_subjects"],
            "mean_diff": res["mean_diff"],
            "perm_p": res["perm_p"],
            "method": res["method"],
            "feature_tag": feature_tag,
        })

    # Family 2: exclusive pairwise between-group similarity
    for a, b in itertools.combinations(EXCLUSIVE_GROUPS, 2):
        n_a = int(df.loc[df["task_group"] == a, "subject"].nunique())
        n_b = int(df.loc[df["task_group"] == b, "subject"].nunique())
        for metric in ("jaccard_unsigned", "jaccard_signed"):
            if min(n_a, n_b) < min_group_n:
                rows.append({
                    "family": "exclusive_pairwise_between",
                    "grouping": "exclusive",
                    "metric": metric,
                    "group_a": a, "group_b": b,
                    "n_subjects_used": min(n_a, n_b),
                    "mean_diff": np.nan, "perm_p": np.nan,
                    "method": "insufficient", "feature_tag": feature_tag,
                })
                continue
            res = exclusive_pair_similarity_perm(
                df, a, b, metric, n_perm=n_perm,
                rng_seed=rng_seed + hash((a, b, metric)) % 10000,
            )
            rows.append({
                "family": "exclusive_pairwise_between",
                "grouping": "exclusive",
                "metric": metric,
                "group_a": a, "group_b": b,
                "n_subjects_used": res["n_subjects_used"],
                "mean_diff": res["mean_between"],
                "perm_p": res["perm_p"],
                "method": res["method"],
                "feature_tag": feature_tag,
            })

    # Family 3: overlapping paired within-group similarity differences
    for a, b in itertools.combinations(OVERLAP_GROUPS, 2):
        for metric in ("jaccard_unsigned", "jaccard_signed"):
            paired = overlap_paired_subject_means(df, a, b, metric=metric)
            if paired.empty or paired["subject"].nunique() < min_group_n:
                rows.append({
                    "family": "overlap_paired_within",
                    "grouping": "overlap",
                    "metric": metric,
                    "group_a": a, "group_b": b,
                    "n_subjects_used": int(paired["subject"].nunique()) if not paired.empty else 0,
                    "mean_diff": np.nan, "perm_p": np.nan,
                    "method": "insufficient", "feature_tag": feature_tag,
                })
                continue
            res = subject_mean_diff_perm(
                paired, "diff_a_minus_b", n_perm=n_perm,
                rng_seed=rng_seed + hash((a, b, metric, "ov")) % 10000,
            )
            rows.append({
                "family": "overlap_paired_within",
                "grouping": "overlap",
                "metric": metric,
                "group_a": a, "group_b": b,
                "n_subjects_used": res["n_subjects"],
                "mean_diff": res["mean_diff"],
                "perm_p": res["perm_p"],
                "method": res["method"],
                "feature_tag": feature_tag,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # FDR within each family separately
    out["q_fdr"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        q = fdr_bh(out.loc[idx, "perm_p"].to_numpy())
        out.loc[idx, "q_fdr"] = q
    return out


# ======================================================================
# Feature frequency / enrichment
# ======================================================================

def feature_frequency_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
    grouping_label: str,
    feature_tag: str,
) -> pd.DataFrame:
    rows = []
    for g in groups:
        sub = df[df[group_col] == g]
        n_e = len(sub)
        if n_e == 0:
            continue
        counts: Dict[int, int] = {}
        sign_pos: Dict[int, int] = {}
        sign_neg: Dict[int, int] = {}
        for _, row in sub.iterrows():
            signed = parse_signed_features(row.get("feature_indices", ""))
            seen = set()
            for idx, sgn in signed:
                if idx not in seen:
                    counts[idx] = counts.get(idx, 0) + 1
                    seen.add(idx)
                if sgn >= 0:
                    sign_pos[idx] = sign_pos.get(idx, 0) + 1
                else:
                    sign_neg[idx] = sign_neg.get(idx, 0) + 1
        for idx, cnt in counts.items():
            rows.append({
                "grouping": grouping_label,
                "group_col": group_col,
                "group": g,
                "feature_index": idx,
                "n_electrodes": n_e,
                "n_subjects": int(sub["subject"].nunique()),
                "n_selected": cnt,
                "pct_electrodes": 100.0 * cnt / n_e,
                "n_pos_sign": sign_pos.get(idx, 0),
                "n_neg_sign": sign_neg.get(idx, 0),
                "feature_tag": feature_tag,
            })
    return pd.DataFrame(rows)


# ======================================================================
# Figures / report
# ======================================================================

def _heatmap(mat: np.ndarray, labels: Sequence[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def fig_union_heatmaps(union_df: pd.DataFrame, outdir: Path, feature_tag: str):
    for grouping, groups, label in (
        ("exclusive", EXCLUSIVE_WITH_NONE, "exclusive"),
        ("overlap", OVERLAP_GROUPS, "overlap"),
    ):
        sub = union_df[union_df["grouping"] == grouping]
        if sub.empty:
            continue
        for metric, col in (("unsigned", "jaccard_unsigned"),
                            ("signed", "jaccard_signed")):
            n = len(groups)
            mat = np.eye(n)
            idx = {g: i for i, g in enumerate(groups)}
            for _, r in sub.iterrows():
                i, j = idx[r["group_a"]], idx[r["group_b"]]
                mat[i, j] = mat[j, i] = float(r[col]) if np.isfinite(r[col]) else np.nan
            _heatmap(
                mat, groups,
                f"{feature_tag} group-union Jaccard ({metric}, {label})",
                outdir / f"task_type_jaccard_union_{label}_{metric}.png",
            )


def fig_within_between(subject_df: pd.DataFrame, outdir: Path, feature_tag: str):
    if subject_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("jaccard_unsigned", "jaccard_signed"),
        ("Unsigned", "Signed"),
    ):
        sub = subject_df[
            (subject_df["grouping"] == "exclusive")
            & (subject_df["metric"] == metric)
        ]
        if sub.empty:
            ax.set_visible(False)
            continue
        data = [sub["within_mean"].dropna(), sub["between_mean"].dropna()]
        ax.boxplot(data, tick_labels=["within", "between"], showfliers=False)
        for i, col in enumerate(("within_mean", "between_mean"), start=1):
            vals = sub[col].dropna().to_numpy()
            ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(
                -0.08, 0.08, size=len(vals)), vals, s=28, alpha=0.7, zorder=3)
        ax.set_title(f"{title} Jaccard (exclusive)")
        ax.set_ylabel("subject-mean electrode-pair Jaccard")
    fig.suptitle(f"{feature_tag}: within- vs between-task feature overlap", y=1.02)
    fig.tight_layout()
    p = outdir / "task_type_jaccard_within_vs_between.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def write_report(
    feat: pd.DataFrame,
    cov: Dict[str, int],
    union_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    stats: pd.DataFrame,
    path: Path,
    feature_tag: str,
    n_perm: int,
    min_group_n: int,
    features_csv: Path,
):
    lines = []
    a = lines.append
    a("=" * 70)
    a("SAE TASK-TYPE FEATURE OVERLAP (JACCARD)")
    a("=" * 70)
    a(f"Feature tag: {feature_tag}")
    a(f"Features CSV: {features_csv}")
    a(f"Electrodes: {len(feat)}  subjects: {feat['subject'].nunique()}")
    a(f"n_perm={n_perm}  min_group_n={min_group_n}  rng_seed={RNG_SEED}")
    a("")
    a("NOTE: Qualitative feature selection is a full-data Lasso refit")
    a("(exploratory). Do not treat selected sets as fold-honest CV supports.")
    a("Group-union Jaccard is descriptive (sample-size dependent).")
    a("Inference uses within-subject electrode-pair / label permutations.")
    a("")
    a("-" * 70)
    a("MERGE / COVERAGE")
    a("-" * 70)
    for k, v in cov.items():
        a(f"  {k}: {v}")
    a("")
    a("-" * 70)
    a("EXCLUSIVE TASK GROUP COUNTS")
    a("-" * 70)
    if "task_group" in feat.columns:
        ct = (feat.groupby("task_group")
                  .agg(n_electrodes=("channel", "size"),
                       n_subjects=("subject", "nunique"))
                  .reset_index())
        a(ct.to_string(index=False))
    a("")
    a("-" * 70)
    a("OVERLAPPING TASK-POSITIVE COUNTS")
    a("-" * 70)
    for g, col in OVERLAP_POS.items():
        if col in feat.columns:
            sub = feat[feat[col]]
            a(f"  {g}: nE={len(sub)}  nS={sub['subject'].nunique()}")
    a("")
    a("-" * 70)
    a("GROUP-UNION JACCARD (descriptive)")
    a("-" * 70)
    if not union_df.empty:
        a(union_df.round(4).to_string(index=False))
    a("")
    a("-" * 70)
    a("SUBJECT-MEAN WITHIN − BETWEEN (exclusive)")
    a("-" * 70)
    if not subject_df.empty:
        excl = subject_df[subject_df["grouping"] == "exclusive"]
        for metric in ("jaccard_unsigned", "jaccard_signed"):
            sub = excl[excl["metric"] == metric]["within_minus_between"].dropna()
            a(f"  {metric}: mean={sub.mean():+.4f}  n_subjects={len(sub)}")
    a("")
    a("-" * 70)
    a("INFERENTIAL TESTS (FDR within family)")
    a("-" * 70)
    if stats.empty:
        a("  (no tests)")
    else:
        a(stats.round(4).to_string(index=False))
    a("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ======================================================================
# Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features_csv",
        default=str(ap.SAE_FEATURES_QWEN_ALL),
        help="Qualitative per-electrode feature CSV "
             "(default: all-channel Qwen table).",
    )
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument("--n_perm", type=int, default=N_PERM)
    p.add_argument("--min_group_n", type=int, default=MIN_GROUP_N)
    p.add_argument("--rng_seed", type=int, default=RNG_SEED)
    p.add_argument(
        "--skip_figures", action="store_true",
        help="Skip figure writing (useful for fast tests).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(
            f"{features_csv} not found. Run sparse_encoding_qualitative.py "
            f"--all_channels --feature_tag {args.feature_tag} first."
        )

    feat = pd.read_csv(features_csv)
    if "feature_tag" in feat.columns:
        tags = set(feat["feature_tag"].dropna().astype(str).unique())
        if tags and tags != {args.feature_tag}:
            raise ValueError(
                f"Feature tag mismatch: CSV has {tags}, expected {args.feature_tag}"
            )
    feat, cov = attach_task_metadata(feat)
    print(
        f"Loaded {len(feat)} electrodes; "
        f"merged localizer={cov['n_merged_localizer']}, "
        f"unmatched={cov['n_unmatched']}",
        flush=True,
    )

    # Build exclusive / overlap analysis frames
    excl = feat[feat["task_group"].isin(EXCLUSIVE_WITH_NONE)].copy()
    # Overlap frame: duplicate rows into group labels via long form helper
    # For pairwise/union we filter with boolean columns directly.
    overlap_long_rows = []
    for g, col in OVERLAP_POS.items():
        sub = feat[feat[col]].copy()
        sub["overlap_group"] = g
        overlap_long_rows.append(sub)
    overlap = pd.concat(overlap_long_rows, ignore_index=True) if overlap_long_rows else pd.DataFrame()

    union_excl = compute_union_jaccard_table(
        excl, "task_group", EXCLUSIVE_WITH_NONE, "exclusive", args.feature_tag)
    union_ov = (
        compute_union_jaccard_table(
            overlap, "overlap_group", OVERLAP_GROUPS, "overlap", args.feature_tag)
        if not overlap.empty else pd.DataFrame()
    )
    union_df = pd.concat([union_excl, union_ov], ignore_index=True)

    pair_excl = compute_pairwise_electrode_jaccard(
        excl, "task_group", EXCLUSIVE_GROUPS, "exclusive", args.feature_tag)
    pair_ov = (
        compute_pairwise_electrode_jaccard(
            overlap, "overlap_group", OVERLAP_GROUPS, "overlap",
            args.feature_tag, exclude_self_pairs=True,
        )
        if not overlap.empty else pd.DataFrame()
    )
    pair_df = pd.concat([pair_excl, pair_ov], ignore_index=True)

    subject_df = summarize_pairwise_by_subject(pair_df)
    # One call: exclusive families use task_group; overlap family uses is_pos_*.
    stats = run_jaccard_stats(
        feat, subject_df, n_perm=args.n_perm, min_group_n=args.min_group_n,
        feature_tag=args.feature_tag, rng_seed=args.rng_seed,
    )

    freq_excl = feature_frequency_by_group(
        excl, "task_group", EXCLUSIVE_WITH_NONE, "exclusive", args.feature_tag)
    freq_ov = (
        feature_frequency_by_group(
            overlap, "overlap_group", OVERLAP_GROUPS, "overlap", args.feature_tag)
        if not overlap.empty else pd.DataFrame()
    )
    freq_df = pd.concat([freq_excl, freq_ov], ignore_index=True)

    # Save tables
    union_df.to_csv(ap.SAE_TASK_TYPE_JACCARD_UNION, index=False)
    pair_df.to_csv(ap.SAE_TASK_TYPE_JACCARD_PAIRWISE, index=False)
    subject_df.to_csv(ap.SAE_TASK_TYPE_JACCARD_SUBJECT, index=False)
    stats.to_csv(ap.SAE_TASK_TYPE_JACCARD_STATS, index=False)
    freq_df.to_csv(ap.SAE_TASK_TYPE_FEATURE_FREQ, index=False)
    # Also save the merged feature+task table for provenance
    merged_out = ap.SAE_TABLES / f"sparse_encoding_features_{args.feature_tag}_with_task.csv"
    feat.to_csv(merged_out, index=False)
    print(f"Saved union Jaccard -> {ap.SAE_TASK_TYPE_JACCARD_UNION}")
    print(f"Saved pairwise Jaccard -> {ap.SAE_TASK_TYPE_JACCARD_PAIRWISE}")
    print(f"Saved subject summary -> {ap.SAE_TASK_TYPE_JACCARD_SUBJECT}")
    print(f"Saved stats -> {ap.SAE_TASK_TYPE_JACCARD_STATS}")
    print(f"Saved feature freq -> {ap.SAE_TASK_TYPE_FEATURE_FREQ}")
    print(f"Saved merged features+task -> {merged_out}")

    write_report(
        feat, cov, union_df, subject_df, stats, ap.SAE_TASK_TYPE_REPORT,
        args.feature_tag, args.n_perm, args.min_group_n, features_csv,
    )

    if not args.skip_figures:
        print("Figures:")
        fig_union_heatmaps(union_df, ap.SAE_FIGURES, args.feature_tag)
        fig_within_between(subject_df, ap.SAE_FIGURES, args.feature_tag)


if __name__ == "__main__":
    main()
