#!/usr/bin/env python3
"""
sparse_encoding_lepori_study3.py
================================
iEEG analogue of Lepori, Kay & Tuckute (arXiv:2606.06857) **Study 3**:
language-network fROIs, SAE gain beyond surprisal, dense LM baseline,
Matryoshka general-vs-idiosyncratic bins, signed-feature sharing.

Paper-comparable mask is MITSWJN ``is_lang`` ∩ left hemisphere, mapped to
the five Fedorenko language parcels (IFGorb, IFG, MFG, AntTemp, PostTemp).

Unavoidable deviations (also written into the report)
-----------------------------------------------------
- Word-locked Chinese iEEG high-gamma, not sentence-level English 7T fMRI.
- Contiguous within-section CV, not shuffled sentence KFold.
- Fisher-z *r* without noise-ceiling normalization (no stimulus repeats).
- Electrodes, not voxels. Left-MFG n_subjects is typically < 8.

Example
-------
    python -m sparse_encoding.sparse_encoding_lepori_study3
    python -m sparse_encoding.sparse_encoding_lepori_study3 --n_perm 2000
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_qualitative import (
    parse_signed_features, parse_unsigned_features,
)
from sparse_encoding.sparse_encoding_summary import LEFT_MFG_REGION, language_mask
from sparse_encoding.sparse_encoding_surprisal_dominance import (
    N_PERM, RNG_SEED, MIN_GROUP_N,
    classify_dominance, fdr_bh, normalize_channel,
    paper_group_summary, subject_mean_one_sample_perm,
    subject_stratified_perm_diff,
)

GEMMA_TAG = "sae_gemma2_2b_mat_l12"
QWEN_TAG = "sae_qwen3_8b_l18"
QWEN35_TAG = "sae_qwen35_4b_mat_l15"
TEMPORAL_Y_CUT = -20.0  # MNI y: anterior if y >= cut
# Extra-parcel temporal: AAL3 midpoint is not STG/MTG, but a bipolar pole,
# Schaefer Aud/TempPole, insula/hippocampus, or a lateral-temporal MNI box
# still puts the contact in temporal cortex. These are NOT Fedorenko parcels.
TEMPORAL_AAL_KEYS = ("temporal", "heschl", "planum")
TEMPORAL_ADJ_KEYS = (
    "insula", "hippocamp", "rolandic", "fusiform", "parahippocamp", "amygdala",
)
TEMPORAL_SECONDARY_KEYS = ("_aud", "temppole")
# Very lateral, not motor strip (z typically > 30 for precentral).
TEMPORAL_MNI_ABS_X = 50.0
TEMPORAL_MNI_Y = (-55.0, 15.0)
TEMPORAL_MNI_Z = (-15.0, 20.0)
# Stimulus-grounded analogues of paper Table 1 (people / scenery / token count).
# Qwen3.5 Neuronpedia ids are not Gemma 79/44/71; match top activating words.
NAMED_LATENT_FAMILIES = {
    "people": (
        "人", "他", "她", "我", "你", "王子", "大人", "国王", "狐狸",
        "飞行员", "地理学家", "商人", "小人", "孩子", "朋友", "先生",
        "女人", "男人", "姑娘",
    ),
    "scenery": (
        "星球", "行星", "沙漠", "火山", "花", "玫瑰", "羊", "星星",
        "太阳", "地球", "花园", "猴面包", "B612", "木星", "火焰",
    ),
    "token_count": (
        "一", "二", "三", "两", "四", "五", "六", "七", "八", "九", "十",
        "岁", "天", "颗", "数字", "编号",
    ),
}
MATRYOSHKA_BINS = (
    ("bin0_128", 0, 128),
    ("bin128_512", 128, 512),
    ("bin512_2048", 512, 2048),
    ("bin2048_8192", 2048, 8192),
    ("bin8192_plus", 8192, None),
)
# Chanin Qwen3.5-4B-Base Matryoshka (runner_cfg.matryoshka_widths).
QWEN35_MATRYOSHKA_BINS = (
    ("bin0_2048", 0, 2048),
    ("bin2048_16384", 2048, 16384),
    ("bin16384_plus", 16384, None),
)
FROI_ORDER = ("IFGorb", "IFG", "MFG", "AntTemp", "PostTemp")


# ======================================================================
# fROI mapping
# ======================================================================

def _is_left(row: pd.Series) -> bool:
    hemi = str(row.get("hemisphere", "")).strip().lower()
    if hemi == "left":
        return True
    if hemi == "right":
        return False
    region = str(row.get("region", ""))
    return region.startswith("Left ")


def _text_has(text: str, keys: Sequence[str]) -> bool:
    r = str(text).lower()
    return any(k in r for k in keys)


def _region_blob(row: pd.Series) -> str:
    return " ".join(
        str(row.get(c, "")) for c in ("region", "region_a", "region_b")
    )


def _in_lateral_temporal_box(row: pd.Series) -> bool:
    x, y, z = row.get("MNI_x", np.nan), row.get("MNI_y", np.nan), row.get("MNI_z", np.nan)
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return False
    y_lo, y_hi = TEMPORAL_MNI_Y
    z_lo, z_hi = TEMPORAL_MNI_Z
    return (
        abs(float(x)) >= TEMPORAL_MNI_ABS_X
        and y_lo <= float(y) <= y_hi
        and z_lo <= float(z) <= z_hi
    )


def _is_extra_parcel_temporal(row: pd.Series) -> bool:
    """Temporal cortex outside the midpoint-AAL AntTemp/PostTemp rule."""
    region = str(row.get("region", ""))
    blob = _region_blob(row)
    secondary = str(row.get("region_secondary", ""))
    if _text_has(blob, TEMPORAL_AAL_KEYS):
        return True
    if _text_has(region, TEMPORAL_ADJ_KEYS):
        return True
    if _text_has(secondary, TEMPORAL_SECONDARY_KEYS):
        return True
    return _in_lateral_temporal_box(row)


def _map_anatomy(
    row: pd.Series,
    y_cut: float,
    left_other: str,
    right_other: str,
) -> str:
    if not _is_left(row):
        return right_other
    region = str(row.get("region", "Unknown"))
    r = region.lower()
    y = row.get("MNI_y", np.nan)

    if "orbital" in r and "inferior frontal" in r:
        return "IFGorb"
    if "inferior frontal" in r and ("triangular" in r or "opercular" in r):
        return "IFG"
    if region == LEFT_MFG_REGION:
        return "MFG"

    temporal = _text_has(r, TEMPORAL_AAL_KEYS)
    if temporal:
        if not np.isfinite(y):
            return "temporal_y_unknown"
        return "AntTemp" if float(y) >= y_cut else "PostTemp"
    if _is_extra_parcel_temporal(row):
        return "temporal_extra"
    return left_other


def assign_lepori_froi(row: pd.Series, y_cut: float = TEMPORAL_Y_CUT) -> str:
    """Map AAL3 region + MNI y to a Lepori Study 3 fROI, or a leftover bucket.

    Gated on MITSWJN ``is_lang``. AntTemp/PostTemp stay midpoint-AAL temporal
    only (parcel analogue). ``temporal_extra`` is left-hemisphere temporal
    cortex that is *not* in those parcels.
    """
    if not bool(row.get("is_lang", False)):
        return ""
    return _map_anatomy(row, y_cut, "lang_other_left", "lang_right")


def assign_anat_bucket(row: pd.Series, y_cut: float = TEMPORAL_Y_CUT) -> str:
    """Same AAL3 mapping as ``assign_lepori_froi``, but for every channel."""
    return _map_anatomy(row, y_cut, "other_left", "right")


def attach_froi(df: pd.DataFrame, y_cut: float = TEMPORAL_Y_CUT) -> pd.DataFrame:
    out = df.copy()
    out["lepori_froi"] = out.apply(lambda r: assign_lepori_froi(r, y_cut), axis=1)
    out["anat_bucket"] = out.apply(lambda r: assign_anat_bucket(r, y_cut), axis=1)
    return out


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    lang = df[language_mask(df)].copy()
    non = df[~language_mask(df)].copy()
    rows = []
    for froi in FROI_ORDER + (
        "temporal_extra", "lang_other_left", "lang_right", "temporal_y_unknown",
    ):
        sub = lang[lang["lepori_froi"] == froi]
        rows.append({
            "froi": froi,
            "n_electrodes": int(len(sub)),
            "n_subjects": int(sub["subject"].nunique()) if len(sub) else 0,
            "underpowered_vs_paper": bool(sub["subject"].nunique() < 8) if len(sub) else True,
            "regions": "; ".join(
                sub["region"].astype(str).value_counts().head(6).index.tolist()
            ) if len(sub) else "",
        })
    rows.append({
        "froi": "is_lang_all",
        "n_electrodes": int(len(lang)),
        "n_subjects": int(lang["subject"].nunique()) if len(lang) else 0,
        "underpowered_vs_paper": bool(lang["subject"].nunique() < 8) if len(lang) else True,
        "regions": "",
    })
    rows.append({
        "froi": "not_is_lang",
        "n_electrodes": int(len(non)),
        "n_subjects": int(non["subject"].nunique()) if len(non) else 0,
        "underpowered_vs_paper": False,
        "regions": "",
    })
    rows.append({
        "froi": "all_channels",
        "n_electrodes": int(len(df)),
        "n_subjects": int(df["subject"].nunique()) if len(df) else 0,
        "underpowered_vs_paper": bool(df["subject"].nunique() < 8) if len(df) else True,
        "regions": "",
    })
    return pd.DataFrame(rows)


# ======================================================================
# Load / merge
# ======================================================================

def try_load_sae_cohort(tag: str) -> Optional[pd.DataFrame]:
    path = ap.SAE_TABLES / f"sparse_encoding_results_{tag}_cohort.csv"
    if not path.exists():
        # lang-only extract/regression writes *_lang.csv before a full cohort.
        alt = ap.SAE_TABLES / f"sparse_encoding_results_{tag}_lang.csv"
        if not alt.exists():
            print(f"[study3] skip {tag}: no cohort/lang CSV yet")
            return None
        path = alt
    raw = pd.read_csv(path)
    raw["channel"] = normalize_channel(raw["channel"])
    tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
    tax["channel"] = normalize_channel(tax["channel"])
    keep = [c for c in (
        "subject", "channel", "region", "region_a", "region_b",
        "region_secondary", "category", "hemisphere",
        "is_lang", "MNI_x", "MNI_y", "MNI_z",
    ) if c in tax.columns]
    drop = [c for c in keep if c in raw.columns and c not in ("subject", "channel")]
    raw = raw.drop(columns=drop, errors="ignore").merge(
        tax[keep], on=["subject", "channel"], how="left")
    raw = classify_dominance(raw)
    raw = attach_froi(raw)
    raw["feature_tag"] = tag
    return raw


def load_sae_cohort(tag: str) -> pd.DataFrame:
    out = try_load_sae_cohort(tag)
    if out is None:
        path = ap.SAE_TABLES / f"sparse_encoding_results_{tag}_cohort.csv"
        raise FileNotFoundError(path)
    return out


def froi_summary(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    ok = df[df["paper_quality_ok"]].copy()
    lang = ok[language_mask(ok)].copy()
    pieces = [paper_group_summary(ok, None)]
    pieces[-1] = pieces[-1].copy()
    pieces[-1]["group_col"] = "sample"
    pieces[-1]["group"] = "all_channels"
    non = ok[~language_mask(ok)].copy()
    if not non.empty:
        p_non = paper_group_summary(non, None)
        p_non = p_non.copy()
        p_non["group_col"] = "sample"
        p_non["group"] = "not_is_lang"
        pieces.append(p_non)
    p_lang = paper_group_summary(lang, None)
    p_lang = p_lang.copy()
    p_lang["group_col"] = "is_lang"
    p_lang["group"] = "is_lang"
    pieces.append(p_lang)
    if "anat_bucket" in ok.columns:
        pieces.append(paper_group_summary(ok, "anat_bucket"))
    mapped = lang[lang["lepori_froi"].isin(FROI_ORDER)]
    if not mapped.empty:
        pieces.append(paper_group_summary(mapped, "lepori_froi"))
    leftover = lang[lang["lepori_froi"].isin(
        ("temporal_extra", "lang_other_left", "lang_right"))]
    if not leftover.empty:
        pieces.append(paper_group_summary(leftover, "lepori_froi"))
    # Parcelled temporal vs frontal (Fedorenko analogue)
    tmp = mapped.copy()
    if not tmp.empty:
        tmp["_lobe"] = np.where(
            tmp["lepori_froi"].isin(("AntTemp", "PostTemp")), "temporal", "frontal")
        pieces.append(paper_group_summary(tmp, "_lobe"))
    # Broad temporal = parcels + extra-parcel temporal cortex
    broad = lang[lang["lepori_froi"].isin(
        FROI_ORDER + ("temporal_extra",))].copy()
    if not broad.empty:
        broad["_lobe"] = np.where(
            broad["lepori_froi"].isin(("AntTemp", "PostTemp", "temporal_extra")),
            "broad_temporal", "frontal")
        pieces.append(paper_group_summary(broad, "_lobe"))
    out = pd.concat([p for p in pieces if p is not None and not p.empty],
                    ignore_index=True)
    out["feature_tag"] = tag
    return out


def froi_stats(df: pd.DataFrame, tag: str, n_perm: int) -> pd.DataFrame:
    ok = df[df["paper_quality_ok"]].copy()
    lang = ok[language_mask(ok)].copy()
    rows: List[Dict] = []
    seed = RNG_SEED

    def add_one(sub: pd.DataFrame, family: str, label: str):
        nonlocal seed
        seed += 1
        res = subject_mean_one_sample_perm(
            sub, "paper_sae_gain", n_perm=n_perm, rng_seed=seed,
            min_group_n=MIN_GROUP_N)
        rows.append({
            "family": family, "test": "sae_gain_vs_zero",
            "group_a": label, "group_b": "zero",
            "n_a": int(len(sub)), "n_b": 0,
            "mean_diff": res["mean_diff"], "perm_p": res["perm_p"],
            "n_subjects_used": res["n_subjects_used"], "method": res["method"],
            "feature_tag": tag,
        })

    add_one(lang, "overall", "is_lang")
    add_one(ok, "overall", "all_channels")
    non = ok[~language_mask(ok)].copy()
    if not non.empty:
        add_one(non, "overall", "not_is_lang")
    mapped = lang[lang["lepori_froi"].isin(FROI_ORDER)]
    leftover = lang[lang["lepori_froi"] == "lang_other_left"]
    if not leftover.empty:
        add_one(leftover, "leftover_onesample", "lang_other_left")
    extra = lang[lang["lepori_froi"] == "temporal_extra"]
    if not extra.empty:
        add_one(extra, "leftover_onesample", "temporal_extra")
    for froi in FROI_ORDER:
        add_one(mapped[mapped["lepori_froi"] == froi], "froi_onesample", froi)
    if not mapped.empty:
        tmp = mapped.copy()
        tmp["_lobe"] = np.where(
            tmp["lepori_froi"].isin(("AntTemp", "PostTemp")), "temporal", "frontal")
        add_one(tmp[tmp["_lobe"] == "temporal"], "lobe_onesample", "temporal")
        add_one(tmp[tmp["_lobe"] == "frontal"], "lobe_onesample", "frontal")
        res = subject_stratified_perm_diff(
            tmp, "paper_sae_gain", "_lobe", "temporal", "frontal",
            n_perm=n_perm, rng_seed=RNG_SEED + 77)
        rows.append({
            "family": "lobe_pairwise", "test": "sae_gain",
            "group_a": "temporal", "group_b": "frontal",
            "n_a": int((tmp["_lobe"] == "temporal").sum()),
            "n_b": int((tmp["_lobe"] == "frontal").sum()),
            "mean_diff": res["mean_diff"], "perm_p": res["perm_p"],
            "n_subjects_used": res["n_subjects_used"], "method": res["method"],
            "feature_tag": tag,
        })
        for a, b in (("IFG", "MFG"), ("PostTemp", "MFG"), ("AntTemp", "MFG"),
                     ("PostTemp", "IFG")):
            if a not in tmp["lepori_froi"].values or b not in tmp["lepori_froi"].values:
                continue
            seed_ab = RNG_SEED + 100 + sum(ord(c) for c in f"{a}|{b}|{tag}") % 9000
            res = subject_stratified_perm_diff(
                tmp, "paper_sae_gain", "lepori_froi", a, b,
                n_perm=n_perm, rng_seed=seed_ab)
            rows.append({
                "family": "froi_pairwise", "test": "sae_gain",
                "group_a": a, "group_b": b,
                "n_a": int((tmp["lepori_froi"] == a).sum()),
                "n_b": int((tmp["lepori_froi"] == b).sum()),
                "mean_diff": res["mean_diff"], "perm_p": res["perm_p"],
                "n_subjects_used": res["n_subjects_used"], "method": res["method"],
                "feature_tag": tag,
            })
    broad = lang[lang["lepori_froi"].isin(FROI_ORDER + ("temporal_extra",))].copy()
    if not broad.empty:
        broad["_lobe"] = np.where(
            broad["lepori_froi"].isin(("AntTemp", "PostTemp", "temporal_extra")),
            "broad_temporal", "frontal")
        add_one(broad[broad["_lobe"] == "broad_temporal"],
                "lobe_onesample", "broad_temporal")
        res = subject_stratified_perm_diff(
            broad, "paper_sae_gain", "_lobe", "broad_temporal", "frontal",
            n_perm=n_perm, rng_seed=RNG_SEED + 88)
        rows.append({
            "family": "lobe_pairwise", "test": "sae_gain",
            "group_a": "broad_temporal", "group_b": "frontal",
            "n_a": int((broad["_lobe"] == "broad_temporal").sum()),
            "n_b": int((broad["_lobe"] == "frontal").sum()),
            "mean_diff": res["mean_diff"], "perm_p": res["perm_p"],
            "n_subjects_used": res["n_subjects_used"], "method": res["method"],
            "feature_tag": tag,
        })
    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    for fam, idx in stats.groupby("family").groups.items():
        stats.loc[idx, "q_fdr"] = fdr_bh(stats.loc[idx, "perm_p"].to_numpy())
    return stats


# ======================================================================
# Dense baseline (existing glove / gpt2cn_l24 lang table)
# ======================================================================

def residual_vs_sae(sae_df: pd.DataFrame, resid_csv: Path, tag: str) -> pd.DataFrame:
    """Same-model residual Ridge vs SAE full, on overlapping is_lang electrodes."""
    if not resid_csv.exists():
        return pd.DataFrame()
    dense = pd.read_csv(resid_csv)
    dense["channel"] = normalize_channel(dense["channel"])
    g = sae_df[language_mask(sae_df) & sae_df["paper_quality_ok"]][
        ["subject", "channel", "lepori_froi", "paper_full_R",
         "paper_surprisal_R", "paper_sae_gain"]
    ].copy()
    keep = [c for c in (
        "subject", "channel", "dense__R_fisher_paired",
        "dense_surprisal__R_fisher_paired",
    ) if c in dense.columns]
    m = g.merge(dense[keep], on=["subject", "channel"], how="inner")
    if m.empty:
        return pd.DataFrame()
    r_col = ("dense_surprisal__R_fisher_paired"
             if "dense_surprisal__R_fisher_paired" in m.columns
             else "dense__R_fisher_paired")
    m["resid_R"] = m[r_col]
    m["resid_minus_sae"] = m["resid_R"] - m["paper_full_R"]
    m["resid_minus_surp"] = m["resid_R"] - m["paper_surprisal_R"]
    rows = []
    slices = [("is_lang", m)]
    leftover = m[m["lepori_froi"] == "lang_other_left"]
    if not leftover.empty:
        slices.append(("lang_other_left", leftover))
    extra = m[m["lepori_froi"] == "temporal_extra"]
    if not extra.empty:
        slices.append(("temporal_extra", extra))
    for froi in FROI_ORDER:
        mf = m[m["lepori_froi"] == froi]
        if not mf.empty:
            slices.append((froi, mf))
    for label, sub in slices:
        sf = sub.groupby("subject").mean(numeric_only=True)
        nS = int(sub["subject"].nunique())
        rows.append({
            "sae_tag": tag,
            "resid_csv": resid_csv.name,
            "group": label,
            "n_electrodes": int(len(sub)),
            "n_subjects": nS,
            "resid_R_subject_mean": float(sf["resid_R"].mean()),
            "sae_full_R_subject_mean": float(sf["paper_full_R"].mean()),
            "surprisal_R_subject_mean": float(sf["paper_surprisal_R"].mean()),
            "sae_gain_subject_mean": float(sf["paper_sae_gain"].mean()),
            "resid_minus_sae_subject_mean": float(sf["resid_minus_sae"].mean()),
            "resid_minus_surp_subject_mean": float(sf["resid_minus_surp"].mean()),
            "underpowered_vs_paper": nS < 8,
        })
    return pd.DataFrame(rows)


def dense_vs_sae(gemma: pd.DataFrame) -> pd.DataFrame:
    path = ap.SAE_TABLES / "sparse_encoding_dense_baselines_cohort_lang.csv"
    if not path.exists():
        return pd.DataFrame()
    dense = pd.read_csv(path)
    dense["channel"] = normalize_channel(dense["channel"])
    g = gemma[language_mask(gemma)][
        ["subject", "channel", "lepori_froi", "paper_full_R",
         "paper_surprisal_R", "paper_sae_gain"]
    ].copy()
    rows = []
    for feat, dsub in dense.groupby("feature_name"):
        m = g.merge(
            dsub[["subject", "channel", "dense__R_fisher_paired",
                  "dense_surprisal__R_fisher_paired"]],
            on=["subject", "channel"], how="inner")
        if m.empty:
            continue
        m["dense_minus_sae"] = m["dense__R_fisher_paired"] - m["paper_full_R"]
        m["dense_surp_minus_sae"] = (
            m["dense_surprisal__R_fisher_paired"] - m["paper_full_R"])
        subj = m.groupby("subject").mean(numeric_only=True)
        rows.append({
            "dense_feature": feat,
            "n_electrodes": int(len(m)),
            "n_subjects": int(m["subject"].nunique()),
            "dense_R_subject_mean": float(subj["dense__R_fisher_paired"].mean()),
            "dense_surprisal_R_subject_mean": float(
                subj["dense_surprisal__R_fisher_paired"].mean()),
            "gemma_sae_full_R_subject_mean": float(subj["paper_full_R"].mean()),
            "gemma_sae_gain_subject_mean": float(subj["paper_sae_gain"].mean()),
            "dense_minus_sae_subject_mean": float(subj["dense_minus_sae"].mean()),
        })
        for froi in FROI_ORDER:
            mf = m[m["lepori_froi"] == froi]
            if mf.empty:
                continue
            sf = mf.groupby("subject").mean(numeric_only=True)
            rows.append({
                "dense_feature": feat,
                "n_electrodes": int(len(mf)),
                "n_subjects": int(mf["subject"].nunique()),
                "dense_R_subject_mean": float(sf["dense__R_fisher_paired"].mean()),
                "dense_surprisal_R_subject_mean": float(
                    sf["dense_surprisal__R_fisher_paired"].mean()),
                "gemma_sae_full_R_subject_mean": float(sf["paper_full_R"].mean()),
                "gemma_sae_gain_subject_mean": float(sf["paper_sae_gain"].mean()),
                "dense_minus_sae_subject_mean": float(sf["dense_minus_sae"].mean()),
                "froi": froi,
            })
    return pd.DataFrame(rows)


# ======================================================================
# Feature sharing / Matryoshka bin occupancy
# ======================================================================

def _bin_name(idx: int, bins: Sequence[Tuple[str, int, Optional[int]]] = MATRYOSHKA_BINS) -> str:
    for name, lo, hi in bins:
        if hi is None:
            if idx >= lo:
                return name
        elif lo <= idx < hi:
            return name
    return "out_of_range"


def feature_sharing(
    gemma: pd.DataFrame,
    feat_csv: Path,
    unsigned: bool = False,
    matryoshka: bool = True,
    bins: Sequence[Tuple[str, int, Optional[int]]] = MATRYOSHKA_BINS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Prevalence / entropy plus within- vs cross-fROI Jaccard."""
    empty_meta = {
        "jaccard_within": np.nan, "jaccard_cross": np.nan,
        "n_pairs_within": 0, "n_pairs_cross": 0,
        "source_features_csv": str(feat_csv),
    }
    if not feat_csv.exists():
        return pd.DataFrame(), pd.DataFrame(), empty_meta
    feat = pd.read_csv(feat_csv)
    feat["channel"] = normalize_channel(feat["channel"])
    lang = gemma[language_mask(gemma)][
        ["subject", "channel", "lepori_froi"]].copy()
    m = feat.merge(lang, on=["subject", "channel"], how="inner")
    if m.empty:
        return pd.DataFrame(), pd.DataFrame(), empty_meta

    sets: Dict[Tuple[str, str], set] = {}
    feat_subj: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    feat_froi: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    bin_counts = Counter()
    n_with_support = 0
    for _, r in m.iterrows():
        parsed = parse_signed_features(r.get("feature_indices"))
        if unsigned:
            parsed = {(i, 1) for i, _ in parsed}
        if not parsed:
            sets[(r["subject"], r["channel"])] = set()
            continue
        n_with_support += 1
        sets[(r["subject"], r["channel"])] = parsed
        for idx, sgn in parsed:
            feat_subj[(idx, sgn)][r["subject"]] += 1
            feat_froi[(idx, sgn)][r["lepori_froi"]] += 1
            if matryoshka:
                bin_counts[_bin_name(idx, bins)] += 1

    n_subj = int(m["subject"].nunique())
    prev_rows = []
    for (idx, sgn), subj_c in feat_subj.items():
        counts = np.array(list(subj_c.values()), dtype=float)
        full = np.zeros(n_subj)
        full[: len(counts)] = counts
        p = full / full.sum() if full.sum() else full
        p = p[p > 0]
        ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
        prev_rows.append({
            "feature_idx": idx, "sign": int(sgn),
            "n_electrodes": int(sum(subj_c.values())),
            "n_subjects": int(len(subj_c)),
            "subject_entropy_bits": ent,
            "effective_n_subjects": float(2 ** ent) if ent else 1.0,
            "matryoshka_bin": _bin_name(idx, bins) if matryoshka else "not_matryoshka",
            "source_features_csv": str(feat_csv),
        })
    prev = pd.DataFrame(prev_rows)
    if not prev.empty:
        prev = prev.sort_values("n_electrodes", ascending=False)

    bin_rows = []
    if matryoshka:
        bin_rows = [{
            "kind": "occupancy_selected_tokens",
            "bin": name, "index_lo": lo,
            "index_hi": hi if hi is not None else "end",
            "n_selected_tokens": int(bin_counts[name]),
            "frac_selected_tokens": (
                bin_counts[name] / sum(bin_counts.values()) if bin_counts else np.nan),
            "n_electrodes_with_support": n_with_support,
            "n_feature_rows_merged": int(len(m)),
            "source_features_csv": str(feat_csv),
        } for name, lo, hi in bins]

    mapped = m[m["lepori_froi"].isin(FROI_ORDER)]
    j_within, j_cross = [], []
    for subj, sdf in mapped.groupby("subject"):
        items = [
            (r["lepori_froi"], sets.get((r["subject"], r["channel"]), set()))
            for _, r in sdf.iterrows()
        ]
        items = [(f, s) for f, s in items if s]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i][1], items[j][1]
                union = len(a | b)
                jac = (len(a & b) / union) if union else np.nan
                if items[i][0] == items[j][0]:
                    j_within.append(jac)
                else:
                    j_cross.append(jac)
    meta = {
        "jaccard_within": float(np.nanmean(j_within)) if j_within else np.nan,
        "jaccard_cross": float(np.nanmean(j_cross)) if j_cross else np.nan,
        "n_pairs_within": len(j_within),
        "n_pairs_cross": len(j_cross),
        "n_electrodes_merged": int(len(m)),
        "n_electrodes_with_support": n_with_support,
        "source_features_csv": str(feat_csv),
    }
    return prev, pd.DataFrame(bin_rows), meta


_WORD_ACT_RE = re.compile(r"^(?P<w>.+)\((?P<a>-?\d+(?:\.\d+)?)\)$")


def parse_top_words(s) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    out = []
    for tok in str(s).split():
        m = _WORD_ACT_RE.match(tok)
        out.append(m.group("w") if m else tok)
    return out


def score_named_families(words: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Hit counts / rates of stimulus words against Table-1 analogue families."""
    uniq = list(dict.fromkeys(words))
    n = max(len(uniq), 1)
    out = {}
    for fam, keys in NAMED_LATENT_FAMILIES.items():
        hits = [w for w in uniq if _word_in_family(w, keys, fam)]
        out[fam] = {
            "n_hits": float(len(hits)),
            "hit_rate": float(len(hits) / n),
            "hit_words": " ".join(hits[:8]),
        }
    return out


def _word_in_family(word: str, keys: Sequence[str], family: str) -> bool:
    """Exact match for 1-char count tokens; people 1-char may prefix (他们)."""
    for k in keys:
        if family == "token_count" and len(k) <= 1:
            if word == k:
                return True
        elif family == "people" and len(k) == 1:
            if word == k or word.startswith(k):
                return True
        elif k in word:
            return True
    return False


def named_latent_table(
    lang: pd.DataFrame,
    feat_csv: Path,
    words_csv: Path,
    tag: str,
    hit_rate_cut: float = 0.20,
    min_hits: int = 3,
) -> pd.DataFrame:
    """Paper Fig. 4B–C analogue: people / scenery / token-count occupancy.

    Families are stimulus-word grounded (Chinese Little Prince). Gemma
    Neuronpedia ids 79/44/71 do not transfer to Qwen3.5.
    """
    if not feat_csv.exists() or not words_csv.exists():
        return pd.DataFrame()
    feat = pd.read_csv(feat_csv)
    feat["channel"] = normalize_channel(feat["channel"])
    words = pd.read_csv(words_csv)
    fam_idx: Dict[str, Set[int]] = {k: set() for k in NAMED_LATENT_FAMILIES}
    feat_rows = []
    for _, r in words.iterrows():
        idx = int(r["feature_index"])
        scored = score_named_families(parse_top_words(r.get("top_words")))
        for fam, sc in scored.items():
            if sc["hit_rate"] >= hit_rate_cut or sc["n_hits"] >= min_hits:
                fam_idx[fam].add(idx)
                feat_rows.append({
                    "kind": "feature",
                    "feature_tag": tag,
                    "family": fam,
                    "feature_idx": idx,
                    "n_electrodes_selected": int(r.get("n_electrodes_selected", 0)),
                    "hit_rate": sc["hit_rate"],
                    "n_hits": sc["n_hits"],
                    "hit_words": sc["hit_words"],
                    "top_words": r.get("top_words"),
                })
    m = feat.merge(
        lang[["subject", "channel", "lepori_froi"]
             + (["is_lang"] if "is_lang" in lang.columns else [])],
        on=["subject", "channel"], how="inner")
    occ_rows = []
    groups = [("all_channels", m)]
    if "is_lang" in m.columns:
        lang_m = m[m["is_lang"].fillna(False).astype(bool)]
        non_m = m[~m["is_lang"].fillna(False).astype(bool)]
        groups.append(("is_lang", lang_m))
        if not non_m.empty:
            groups.append(("not_is_lang", non_m))
    else:
        groups.append(("is_lang", m))
    for froi, g in m.groupby("lepori_froi"):
        if str(froi).strip() in ("", "nan"):
            continue
        groups.append((str(froi), g))
    for label, sub in groups:
        nE = int(len(sub))
        nS = int(sub["subject"].nunique()) if nE else 0
        for fam, idxs in fam_idx.items():
            has = []
            subj_hit: Dict[str, int] = defaultdict(int)
            for _, r in sub.iterrows():
                parsed = parse_unsigned_features(r.get("feature_indices"))
                hit = bool(parsed & idxs)
                has.append(hit)
                if hit:
                    subj_hit[str(r["subject"])] += 1
            occ_rows.append({
                "kind": "occupancy",
                "feature_tag": tag,
                "family": fam,
                "group": label,
                "n_electrodes": nE,
                "n_subjects": nS,
                "n_matching_features": int(len(idxs)),
                "n_electrodes_with_family": int(sum(has)),
                "frac_electrodes_with_family": (
                    float(np.mean(has)) if has else np.nan),
                "n_subjects_with_family": int(len(subj_hit)),
                "source_features_csv": str(feat_csv),
                "source_words_csv": str(words_csv),
            })
    return pd.concat(
        [pd.DataFrame(feat_rows), pd.DataFrame(occ_rows)],
        ignore_index=True, sort=False,
    )


def _tag_from_bin_csv(name: str) -> str:
    """Infer feature_tag from a lang_bin results filename."""
    stem = name
    if stem.startswith("sparse_encoding_results_"):
        stem = stem[len("sparse_encoding_results_"):]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.startswith("lang_bin"):
        return GEMMA_TAG
    if "_lang_bin" in stem:
        return stem.split("_lang_bin")[0]
    return stem


def load_bin_refits() -> pd.DataFrame:
    """Collect any completed lang-only Matryoshka-bin regression CSVs."""
    rows = []
    for csv in sorted(ap.SAE_TABLES.glob("sparse_encoding_results_*lang_bin*.csv")):
        df = pd.read_csv(csv)
        if "full__R_fisher_paired" not in df.columns:
            continue
        df["channel"] = normalize_channel(df["channel"])
        tag = _tag_from_bin_csv(csv.name)
        rows.append({
            "file": csv.name,
            "feature_tag": tag,
            "n_electrodes": int(len(df)),
            "n_subjects": int(df["subject"].nunique()),
            "full_R_mean": float(df["full__R_fisher_paired"].mean()),
            "surprisal_R_mean": float(df["surprisal_only__R_fisher_paired"].mean())
            if "surprisal_only__R_fisher_paired" in df.columns else np.nan,
            "sae_gain_mean": float(
                (df["full__R_fisher_paired"] - df["surprisal_only__R_fisher_paired"]).mean()
            ) if "surprisal_only__R_fisher_paired" in df.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ======================================================================
# Report
# ======================================================================

def _fmt_row(r: pd.Series) -> str:
    nE, nS = int(r["n_electrodes"]), int(r["n_subjects"])
    mu, lo, hi = r["paper_sae_gain_subject_mean"], r["paper_sae_gain_ci_lo"], r["paper_sae_gain_ci_hi"]
    full = r["paper_full_R_subject_mean"]
    surp = r["paper_surprisal_R_subject_mean"]
    flag = "  [underpowered nS<8]" if nS < 8 else ""
    return (
        f"  {str(r['group']):12s} nE={nE:3d} nS={nS:2d}  "
        f"full={full:+.4f}  surp={surp:+.4f}  "
        f"SAE gain={mu:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}"
    )


def claim_status(stats: pd.DataFrame, tag: str, group: str, family: str) -> str:
    sub = stats[(stats["feature_tag"] == tag) & (stats["group_a"] == group)
                & (stats["family"] == family)]
    if sub.empty:
        return "not tested"
    r = sub.iloc[0]
    nS = r["n_subjects_used"]
    p = r["perm_p"]
    q = r.get("q_fdr", np.nan)
    mu = r["mean_diff"]
    p_s = f"{p:.3f}" if pd.notna(p) else "NA"
    q_s = f"{q:.3f}" if pd.notna(q) else "NA"
    n_s = int(nS) if pd.notna(nS) else 0
    if r["method"] == "insufficient" or n_s < 8:
        return (
            f"UNDERPOWERED (nS={n_s}; gain={mu:+.4f}, p={p_s})"
        )
    if pd.isna(p):
        return f"no test (gain={mu:+.4f})"
    sig = (q < 0.05) if pd.notna(q) else (p < 0.05)
    if group in ("lang_other_left", "temporal_extra"):
        if sig and mu > 0:
            return f"extension (not a paper fROI): SAE gain {mu:+.4f}, p={p_s}, q={q_s}"
        return f"extension (not a paper fROI): {mu:+.4f}, p={p_s}"
    if group == "MFG":
        # Paper claim: no SAE gain in MFG.
        if sig and mu > 0:
            return f"FAIL TO REPLICATE paper (SAE gain {mu:+.4f}, p={p_s}, q={q_s})"
        return f"consistent with paper (no detected SAE gain; {mu:+.4f}, p={p_s})"
    if sig and mu > 0:
        return f"REPLICATE direction (SAE gain {mu:+.4f}, p={p_s}, q={q_s})"
    return f"no detected SAE gain ({mu:+.4f}, p={p_s}, q={q_s})"


def write_report(
    cov: pd.DataFrame,
    summaries: pd.DataFrame,
    stats: pd.DataFrame,
    dense: pd.DataFrame,
    prev: pd.DataFrame,
    bins: pd.DataFrame,
    bin_refits: pd.DataFrame,
    jaccard_meta: Dict,
    path: Path,
    resid: Optional[pd.DataFrame] = None,
    named: Optional[pd.DataFrame] = None,
    model_tags: Optional[Sequence[str]] = None,
):
    a: List[str] = []
    add = a.append
    add("=" * 72)
    add("LEPORI ET AL. STUDY 3 — iEEG ANALOGUE")
    add("Lepori, Kay & Tuckute, arXiv:2606.06857  |  Interpretable Encoding Models")
    add("=" * 72)
    add("")
    add("PRIMARY ESTIMAND")
    add("  paper_sae_gain = full_R_fisher_paired − surprisal_only_R_fisher_paired")
    add("  Inferential unit = subject. Primary sample = all implanted channels")
    add("  (cohort CSV). MITSWJN is_lang is a subset, not the only mask.")
    add("  Qwen3.5-4B-Base Matryoshka L15 = main hierarchical SAE")
    add("  (prefixes 2048 / 16384 / 65536). Gemma-2-2B L12 = paper backbone;")
    add("  Qwen3-8B L18 = Chinese companion (non-nested SAE).")
    add("")
    add("DEVIATIONS FROM LEPORI ET AL.")
    add("  1. Word-locked Chinese iEEG high-gamma, not sentence-level English 7T fMRI.")
    add("  2. Contiguous within-section 5-fold CV, not shuffled sentence KFold.")
    add("  3. Fisher-z r without noise-ceiling normalization (no NCSNR).")
    add("  4. Electrodes, not voxels; surgical coverage, not whole-brain fROIs.")
    add("  5. AntTemp/PostTemp split by MNI y >= -20 on midpoint AAL3 temporal")
    add("     labels only (not Fedorenko parcels). temporal_extra = left")
    add("     temporal cortex outside that rule (bipolar STG/MTG poles,")
    add("     insula/hippocampus/rolandic, Schaefer Aud/TempPole, or a")
    add("     lateral-temporal MNI box).")
    add("  6. Same-model residual is Ridge on X_word_<tag>_resid.npy (owned-token")
    add("     mean, SAE valid mask). GloVe / gpt2cn_l24 remain a cross-model check.")
    add("  7. n_subjects < 8 in some fROIs → extension, not an 8-participant replication.")
    add("")
    add("-" * 72)
    add("COVERAGE (all channels + is_lang ∩ mapping)")
    add("-" * 72)
    add(cov.to_string(index=False))
    add("")
    add(f"Temporal y cut = {TEMPORAL_Y_CUT:.0f} (anterior if MNI_y >= cut).")
    add("")
    add("-" * 72)
    add("SAE GAIN BY fROI (subject-mean, 95% bootstrap CI)")
    add("-" * 72)
    for tag, g in summaries.groupby("feature_tag"):
        add(f"\n[{tag}]")
        for _, r in g.iterrows():
            add(_fmt_row(r))
    add("")
    add("-" * 72)
    add("INFERENCE (sign-flip / within-subject; FDR within family)")
    add("-" * 72)
    if stats.empty:
        add("  (no tests)")
    else:
        show = stats.copy()
        add(show.round(4).to_string(index=False))
    add("")
    add("-" * 72)
    add("CLAIM STATUS vs Lepori Fig. 4A")
    add("-" * 72)
    tags = tuple(model_tags) if model_tags else (GEMMA_TAG, QWEN_TAG, QWEN35_TAG)
    for tag in tags:
        if stats.empty or tag not in set(stats["feature_tag"]):
            continue
        add(f"\n[{tag}]")
        add(f"  all_channels: {claim_status(stats, tag, 'all_channels', 'overall')}")
        add(f"  not_is_lang:  {claim_status(stats, tag, 'not_is_lang', 'overall')}")
        add(f"  is_lang:      {claim_status(stats, tag, 'is_lang', 'overall')}")
        add(f"  MFG:       {claim_status(stats, tag, 'MFG', 'froi_onesample')}")
        add(f"  IFG:       {claim_status(stats, tag, 'IFG', 'froi_onesample')}")
        add(f"  AntTemp:   {claim_status(stats, tag, 'AntTemp', 'froi_onesample')}")
        add(f"  PostTemp:  {claim_status(stats, tag, 'PostTemp', 'froi_onesample')}")
        add(f"  temporal:  {claim_status(stats, tag, 'temporal', 'lobe_onesample')}")
        add(f"  frontal:   {claim_status(stats, tag, 'frontal', 'lobe_onesample')}")
        add(f"  broad_temporal: {claim_status(stats, tag, 'broad_temporal', 'lobe_onesample')}")
        add(f"  temporal_extra: {claim_status(stats, tag, 'temporal_extra', 'leftover_onesample')}")
        add(f"  lang_other_left: {claim_status(stats, tag, 'lang_other_left', 'leftover_onesample')}")
    add("")
    add("-" * 72)
    add("SAME-MODEL RESIDUAL vs SAE (Fig. 4A reconstruction; primary dense test)")
    add("-" * 72)
    if resid is None or resid.empty:
        add("  Not yet available. After residual dumps:")
        add("    python -m sparse_encoding.sparse_encoding_dense_baseline \\")
        add("        --features sae_qwen35_4b_mat_l15_resid --surprisal_tag sae_qwen35_4b_mat_l15 \\")
        add("        --with_surprisal --lang_only --out_suffix _qwen35_resid")
        add("    python -m sparse_encoding.sparse_encoding_dense_baseline \\")
        add("        --features sae_qwen3_8b_l18_resid --surprisal_tag sae_qwen3_8b_l18 \\")
        add("        --with_surprisal --lang_only --out_suffix _qwen_resid")
        add("    python -m sparse_encoding.sparse_encoding_dense_baseline \\")
        add("        --features sae_gemma2_2b_mat_l12_resid --surprisal_tag sae_gemma2_2b_mat_l12 \\")
        add("        --with_surprisal --lang_only --out_suffix _gemma_resid")
    else:
        add(resid.round(4).to_string(index=False))
        add("")
        add("  Paper claim: residual+surprisal ≈ SAE+surprisal, both beat surprisal-only.")
        add("  Qwen3.5 residual is the main-model paper analogue (same LM as the SAE).")
        add("  Qwen3-8B residual is the Chinese companion; Gemma residual is gemma-2-2b L12.")
        add("  NOTE: residual Ridge uses solver=lsqr + short alpha grid (not LOO GCV)")
        add("  because word-locked n×d makes full-d RidgeCV intractable. Full residual dim kept.")
    add("")
    add("-" * 72)
    add("CROSS-MODEL DENSE (GloVe / gpt2cn_l24; not the paper contrast)")
    add("-" * 72)
    if dense.empty:
        add("  (dense baseline CSV not found)")
    else:
        add(dense.round(4).to_string(index=False))
        add("  These are different encoding models, not same-hidden-state reconstruction.")
    add("")
    add("-" * 72)
    add("MATRYOSHKA BIN OCCUPANCY (selected signed features; Fig. 5A analogue)")
    add("-" * 72)
    if bins.empty:
        add("  (no Gemma qualitative feature table merged onto is_lang)")
    else:
        add(bins.round(4).to_string(index=False))
        add("  Paper claim (Fig. 5): encoding loads on indices < 128.")
        if "frac_selected_tokens" in bins.columns and not bins.empty:
            for tag, g in bins.groupby("feature_tag") if "feature_tag" in bins.columns else [(None, bins)]:
                top = g.sort_values("frac_selected_tokens", ascending=False).iloc[0]
                label = f" [{tag}]" if tag else ""
                add(f"  Modal bin{label}: {top['bin']} "
                    f"({100 * top['frac_selected_tokens']:.1f}% of selected tokens).")
    add("")
    if not prev.empty:
        add("Top 15 signed features by electrode prevalence (pooled tags):")
        cols = [c for c in (
            "feature_tag", "feature_idx", "sign", "n_electrodes", "n_subjects",
            "subject_entropy_bits", "effective_n_subjects", "matryoshka_bin",
        ) if c in prev.columns]
        add(prev.head(15)[cols].round(3).to_string(index=False))
        add("")
    add("  Gemma qualitative CSV is top-R electrodes only (selection-biased).")
    add("  Qwen all-channel CSV is the less biased sharing table.")
    for tag, meta in (jaccard_meta or {}).items():
        jw = meta.get("jaccard_within", np.nan)
        jc = meta.get("jaccard_cross", np.nan)
        add(f"  [{tag}] within-fROI Jaccard: {jw:.3f} "
            f"(n_pairs={meta.get('n_pairs_within', 0)}; "
            f"n_merged={meta.get('n_electrodes_merged', '?')})")
        add(f"  [{tag}] cross-fROI Jaccard:  {jc:.3f} "
            f"(n_pairs={meta.get('n_pairs_cross', 0)})")
    add("  Paper claim (Fig. 4B): within-fROI ≈ cross-fROI transfer.")
    add("")
    add("-" * 72)
    add("NAMED LATENTS (people / scenery / token count; Fig. 4B–C analogue)")
    add("-" * 72)
    add("  Families are matched on Chinese top-activating words, not Gemma")
    add("  Neuronpedia ids 79/44/71/94/40 (those ids do not transfer).")
    if named is None or named.empty:
        add("  Not extracted yet for this run. Launch:")
        add("    python -m sparse_encoding.sparse_encoding_qualitative \\")
        add(f"        --feature_tag {QWEN35_TAG} --all_channels --top_n 0 \\")
        add(f"        --suffix _{QWEN35_TAG}_all_channels \\")
        add("        --results_csv .../sparse_encoding_results_"
            f"{QWEN35_TAG}_lang.csv")
    else:
        occ = named[named["kind"] == "occupancy"] if "kind" in named.columns else named
        show = occ[occ["group"] == "is_lang"] if "group" in occ.columns else occ
        if show.empty:
            add(named.head(20).to_string(index=False))
        else:
            cols = [c for c in (
                "feature_tag", "family", "n_matching_features",
                "n_electrodes_with_family", "frac_electrodes_with_family",
                "n_subjects_with_family", "n_electrodes", "n_subjects",
            ) if c in show.columns]
            add(show[cols].round(3).to_string(index=False))
            feats = named[named["kind"] == "feature"] if "kind" in named.columns else pd.DataFrame()
            if not feats.empty:
                add("")
                add("  Top matching features per family (by electrode prevalence):")
                for tag, fg in feats.groupby("feature_tag"):
                    add(f"  [{tag}]")
                    top = (fg.sort_values("n_electrodes_selected", ascending=False)
                           .groupby("family", group_keys=False).head(3))
                    fcols = [c for c in (
                        "family", "feature_idx", "n_electrodes_selected",
                        "hit_rate", "hit_words",
                    ) if c in top.columns]
                    add(top[fcols].round(3).to_string(index=False))
    add("")
    add("-" * 72)
    add("BIN-RESTRICTED REFITS (Fig. 5C / Appendix I)")
    add("-" * 72)
    add("  Gemma bins: 0–128 vs 128–end (paper nested prefixes).")
    add("  Qwen3.5-4B Matryoshka bins: 0–2048 vs 2048–end (hosted nested prefixes).")
    add("  Qwen3-8B L18 is not nested; do not slice it as Matryoshka.")
    if bin_refits.empty:
        add("  Not yet available. Launch with:")
        add("    python -m sparse_encoding.sparse_encoding_regression \\")
        add(f"        --feature_tag {GEMMA_TAG} --lang_only \\")
        add("        --sae_col_start 0 --sae_col_end 128 --out .../lang_bin0-128.csv")
        add("    python -m sparse_encoding.sparse_encoding_regression \\")
        add(f"        --feature_tag {GEMMA_TAG} --lang_only \\")
        add("        --sae_col_start 128 --out .../lang_bin128-end.csv")
        add("  Then re-run this script.")
        add("  Qwen3.5 (after extract):")
        add("    python -m sparse_encoding.sparse_encoding_regression \\")
        add(f"        --feature_tag {QWEN35_TAG} --lang_only \\")
        add("        --sae_col_start 0 --sae_col_end 2048")
        add("    python -m sparse_encoding.sparse_encoding_regression \\")
        add(f"        --feature_tag {QWEN35_TAG} --lang_only \\")
        add("        --sae_col_start 2048")
    else:
        add(bin_refits.round(4).to_string(index=False))
    add("")
    add("-" * 72)
    add("HOW TO READ")
    add("-" * 72)
    add("  REPLICATE = significant positive SAE gain in the same direction as the paper.")
    add("  FAIL TO REPLICATE = significant SAE gain in MFG (paper: MFG ≈ surprisal-only).")
    add("  UNDERPOWERED = n_subjects < 8; report effect size only.")
    add("  Feature Jaccard / bin occupancy use full-data Lasso supports (exploratory).")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(a) + "\n")
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_perm", type=int, default=N_PERM)
    p.add_argument("--y_cut", type=float, default=TEMPORAL_Y_CUT)
    p.add_argument(
        "--tag_suffix", default="",
        help="Appended to Gemma / Qwen3-8B / Qwen3.5 tags when reading CSVs. "
             "Use _v2 for shared-token ownership. Empty keeps v1 names.",
    )
    return p.parse_args()


def _tag_suffix(raw: str) -> str:
    suffix = str(raw or "")
    if suffix and not suffix.startswith("_"):
        return "_" + suffix
    return suffix


def _feature_csv(tag: str, *fallbacks: Path) -> Path:
    candidates = [
        ap.SAE_TABLES / f"sparse_encoding_features_{tag}_all_channels.csv",
        ap.SAE_TABLES / f"sparse_encoding_features_{tag}.csv",
        *fallbacks,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _resid_csv(kind: str, suffix: str) -> Path:
    defaults = {
        "qwen": ap.SAE_STUDY3_QWEN_RESID,
        "gemma": ap.SAE_STUDY3_GEMMA_RESID,
        "qwen35": ap.SAE_STUDY3_QWEN35_RESID,
    }
    if not suffix:
        return defaults[kind]
    return ap.SAE_TABLES / f"sparse_encoding_dense_baselines_{kind}{suffix}_resid_lang.csv"


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    global TEMPORAL_Y_CUT
    TEMPORAL_Y_CUT = float(args.y_cut)
    suffix = _tag_suffix(args.tag_suffix)
    gemma_tag = GEMMA_TAG + suffix
    qwen_tag = QWEN_TAG + suffix
    qwen35_tag = QWEN35_TAG + suffix

    gemma = load_sae_cohort(gemma_tag)
    qwen = load_sae_cohort(qwen_tag)
    qwen35 = try_load_sae_cohort(qwen35_tag)

    elec_cols = [
        "subject", "channel", "region", "region_a", "region_b",
        "hemisphere", "MNI_x", "MNI_y", "MNI_z", "is_lang",
        "lepori_froi", "anat_bucket", "paper_full_R", "paper_surprisal_R",
        "paper_sae_gain", "feature_tag",
    ]

    def _elec_export(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in elec_cols if c in df.columns]
        return df.loc[:, cols]

    elec_parts = [_elec_export(gemma), _elec_export(qwen)]
    if qwen35 is not None:
        elec_parts.append(_elec_export(qwen35))
    elec = pd.concat(elec_parts, ignore_index=True)
    elec.to_csv(ap.SAE_STUDY3_ELECTRODE_TABLE, index=False)
    print(f"Saved {len(elec)} rows -> {ap.SAE_STUDY3_ELECTRODE_TABLE}")

    cov_src = gemma
    if qwen35 is not None and len(qwen35) >= len(gemma):
        cov_src = qwen35
    cov = coverage_table(cov_src)
    summary_parts = [froi_summary(gemma, gemma_tag), froi_summary(qwen, qwen_tag)]
    stats_parts = [froi_stats(gemma, gemma_tag, args.n_perm),
                   froi_stats(qwen, qwen_tag, args.n_perm)]
    if qwen35 is not None:
        summary_parts.append(froi_summary(qwen35, qwen35_tag))
        stats_parts.append(froi_stats(qwen35, qwen35_tag, args.n_perm))
    summaries = pd.concat(summary_parts, ignore_index=True)
    summaries.to_csv(ap.SAE_STUDY3_SUMMARY_TABLE, index=False)

    stats = pd.concat(stats_parts, ignore_index=True)
    stats.to_csv(ap.SAE_STUDY3_STATS_TABLE, index=False)

    dense = dense_vs_sae(gemma)
    if not dense.empty:
        dense.to_csv(ap.SAE_STUDY3_DENSE_TABLE, index=False)

    resid_parts = [
        residual_vs_sae(qwen, _resid_csv("qwen", suffix), qwen_tag),
        residual_vs_sae(gemma, _resid_csv("gemma", suffix), gemma_tag),
    ]
    if qwen35 is not None:
        resid_parts.append(
            residual_vs_sae(qwen35, _resid_csv("qwen35", suffix), qwen35_tag))
    resid_parts = [p for p in resid_parts if p is not None and not p.empty]
    resid = (pd.concat(resid_parts, ignore_index=True)
             if resid_parts else pd.DataFrame())
    if not resid.empty:
        resid.to_csv(ap.SAE_TABLES / "lepori_study3_resid_vs_sae.csv", index=False)

    gemma_feat = _feature_csv(gemma_tag)
    qwen_feat = _feature_csv(qwen_tag, ap.SAE_FEATURES_QWEN_ALL)
    prev_g, bins_g, jac_g = feature_sharing(gemma, gemma_feat)
    prev_q, bins_q, jac_q = feature_sharing(
        qwen, qwen_feat, matryoshka=False)
    prev_q35, bins_q35, jac_q35 = pd.DataFrame(), pd.DataFrame(), {}
    q35_feat = _feature_csv(qwen35_tag, ap.SAE_FEATURES_QWEN35_ALL)
    if qwen35 is not None:
        prev_q35, bins_q35, jac_q35 = feature_sharing(
            qwen35, q35_feat, matryoshka=True, bins=QWEN35_MATRYOSHKA_BINS)
    prev_parts = []
    if not prev_g.empty:
        prev_parts.append(prev_g.assign(feature_tag=gemma_tag))
    if not prev_q.empty:
        prev_parts.append(prev_q.assign(feature_tag=qwen_tag))
    if not prev_q35.empty:
        prev_parts.append(prev_q35.assign(feature_tag=qwen35_tag))
    prev = pd.concat(prev_parts, ignore_index=True) if prev_parts else pd.DataFrame()
    if not prev.empty:
        prev.to_csv(ap.SAE_STUDY3_FEATURE_TABLE, index=False)

    bin_parts = []
    if not bins_g.empty:
        bin_parts.append(bins_g.assign(feature_tag=gemma_tag))
    if not bins_q.empty:
        bin_parts.append(bins_q.assign(feature_tag=qwen_tag))
    if not bins_q35.empty:
        bin_parts.append(bins_q35.assign(feature_tag=qwen35_tag))
    bins = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    jaccard_meta = {gemma_tag: jac_g, qwen_tag: jac_q}
    if jac_q35:
        jaccard_meta[qwen35_tag] = jac_q35

    bin_refits = load_bin_refits()
    out_bins = bins.copy()
    if not bin_refits.empty:
        out_bins = pd.concat(
            [out_bins, bin_refits.assign(kind="regression_refit")],
            ignore_index=True, sort=False)
    if not out_bins.empty:
        out_bins.to_csv(ap.SAE_STUDY3_BIN_TABLE, index=False)

    named_parts = []
    def _words_for(feat_path: Path) -> Path:
        return feat_path.with_name(
            feat_path.name.replace(
                "sparse_encoding_features", "sparse_encoding_feature_words", 1))

    named_parts.append(named_latent_table(
        gemma, gemma_feat, _words_for(gemma_feat), gemma_tag))
    named_parts.append(named_latent_table(
        qwen, qwen_feat, _words_for(qwen_feat), qwen_tag))
    if qwen35 is not None:
        named_parts.append(named_latent_table(
            qwen35, q35_feat, _words_for(q35_feat), qwen35_tag))
    named_parts = [p for p in named_parts if p is not None and not p.empty]
    named = (pd.concat(named_parts, ignore_index=True, sort=False)
             if named_parts else pd.DataFrame())
    if not named.empty:
        named.to_csv(ap.SAE_STUDY3_NAMED_TABLE, index=False)

    write_report(
        cov, summaries, stats, dense, prev, bins,
        bin_refits, jaccard_meta, ap.SAE_STUDY3_REPORT, resid=resid,
        named=named, model_tags=(gemma_tag, qwen_tag, qwen35_tag),
    )


if __name__ == "__main__":
    main()
