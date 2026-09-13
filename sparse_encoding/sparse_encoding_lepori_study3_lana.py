#!/usr/bin/env python3
"""
LanA sensitivity for the Lepori Study 3 iEEG analogue.

Restrict already-scored electrodes by overlap with Lipkin et al. LanA
(n=806 probabilistic language atlas; ``SPM/LanA_n806.nii``). Does not
re-fit encoding.

Primary mask
------------
    MITSWJN is_lang  ∩  LanA p ≥ 0.20  (2 mm sphere max)

LanA p = fraction of 806 people for whom that voxel was in the top 10%
of language > control. Discrete Fedorenko 5-fROI NIfTIs are not on disk;
AAL3 names are still used inside the restricted set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_lepori_study3 import (
    FROI_ORDER, GEMMA_TAG, QWEN_TAG, attach_froi, coverage_table,
    residual_vs_sae, load_sae_cohort,
)
from sparse_encoding.sparse_encoding_summary import language_mask
from sparse_encoding.sparse_encoding_surprisal_dominance import (
    N_PERM, RNG_SEED, MIN_GROUP_N, fdr_bh, normalize_channel,
    paper_group_summary, subject_mean_one_sample_perm,
)

PRIMARY_THRESH = 0.20
SPHERE_RADIUS_MM = 2.0
SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30)


def _load_lana(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    return data, np.asarray(img.affine, dtype=np.float64)


def _mni_to_voxel(mni: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    n = mni.shape[0]
    h = np.hstack([mni, np.ones((n, 1))])
    return (inv @ h.T).T[:, :3]


def lookup_lana_sphere(
    data: np.ndarray,
    affine: np.ndarray,
    mni: np.ndarray,
    radius_mm: float = SPHERE_RADIUS_MM,
) -> np.ndarray:
    """Max LanA value in a sphere (mm) around each MNI coordinate."""
    voxel_size = np.abs(np.diag(affine)[:3])
    shape = np.array(data.shape[:3])
    n = mni.shape[0]
    out = np.full(n, np.nan)
    radius_vox = radius_mm / voxel_size
    vox = _mni_to_voxel(mni, affine)
    for i in range(n):
        center = vox[i]
        if np.any(~np.isfinite(center)):
            continue
        lo = np.maximum(np.floor(center - radius_vox).astype(int), 0)
        hi = np.minimum(np.ceil(center + radius_vox).astype(int) + 1, shape)
        if np.any(lo >= hi):
            ijk = np.round(center).astype(int)
            if np.all(ijk >= 0) and np.all(ijk < shape):
                out[i] = data[tuple(ijk)]
            continue
        ii, jj, kk = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        cand = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
        cand_mm = cand * voxel_size[None, :]
        center_mm = center * voxel_size
        dist = np.sqrt(np.sum((cand_mm - center_mm[None, :]) ** 2, axis=1))
        keep = dist <= radius_mm
        if not np.any(keep):
            ijk = np.round(center).astype(int)
            if np.all(ijk >= 0) and np.all(ijk < shape):
                out[i] = data[tuple(ijk)]
            continue
        vals = data[cand[keep, 0], cand[keep, 1], cand[keep, 2]]
        out[i] = float(np.max(vals))
    return out


def attach_lana(df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["lana_p"] = values
    out["lana_in_primary"] = out["lana_p"] >= PRIMARY_THRESH
    return out


def _sae_gain_row(sub: pd.DataFrame, group: str, tag: str) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    s = paper_group_summary(sub, None)
    if s is None or s.empty:
        return pd.DataFrame()
    s = s.copy()
    s["group_col"] = "lana_mask"
    s["group"] = group
    s["feature_tag"] = tag
    return s


def summarize_masks(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    ok = df["paper_quality_ok"].fillna(False).astype(bool)
    lang = language_mask(df) & ok
    lana = df["lana_in_primary"].fillna(False).astype(bool) & ok
    pieces = [
        _sae_gain_row(df[lang], "is_lang", tag),
        _sae_gain_row(df[lang & lana], "is_lang_and_lana020", tag),
        _sae_gain_row(df[lang & ~lana], "is_lang_outside_lana020", tag),
        _sae_gain_row(df[lana], "lana020_any_localizer", tag),
        _sae_gain_row(df[lana & ~language_mask(df) & ok], "lana020_not_is_lang", tag),
    ]
    lang_lana = df[lang & lana]
    if not lang_lana.empty:
        mapped = lang_lana[lang_lana["lepori_froi"].isin(FROI_ORDER)]
        if not mapped.empty:
            g = paper_group_summary(mapped, "lepori_froi")
            if g is not None and not g.empty:
                g = g.copy()
                g["feature_tag"] = tag
                pieces.append(g)
        leftover = lang_lana[lang_lana["lepori_froi"].isin(
            ("lang_other_left", "lang_right"))]
        if not leftover.empty:
            g = paper_group_summary(leftover, "lepori_froi")
            if g is not None and not g.empty:
                g = g.copy()
                g["feature_tag"] = tag
                pieces.append(g)
    return pd.concat([p for p in pieces if p is not None and not p.empty],
                     ignore_index=True)


def stats_masks(df: pd.DataFrame, tag: str, n_perm: int) -> pd.DataFrame:
    ok = df["paper_quality_ok"].fillna(False).astype(bool)
    lang = language_mask(df) & ok
    lana = df["lana_in_primary"].fillna(False).astype(bool) & ok
    rows: List[Dict] = []
    seed = RNG_SEED

    def add(sub: pd.DataFrame, family: str, label: str):
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

    add(df[lang], "overall", "is_lang")
    add(df[lang & lana], "lana_primary", "is_lang_and_lana020")
    add(df[lang & ~lana], "lana_primary", "is_lang_outside_lana020")
    add(df[lana], "lana_atlas", "lana020_any_localizer")
    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    parts = []
    for fam, g in stats.groupby("family", sort=False):
        g = g.copy()
        g["q_fdr"] = fdr_bh(g["perm_p"].to_numpy())
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def threshold_sweep(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    ok = df["paper_quality_ok"].fillna(False).astype(bool)
    lang = language_mask(df) & ok
    rows = []
    for t in SWEEP:
        sub = df[lang & (df["lana_p"] >= t)]
        s = _sae_gain_row(sub, f"is_lang_and_lana{int(t * 100):03d}", tag)
        if s.empty:
            continue
        s = s.copy()
        s["lana_threshold"] = t
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bin_refits_restricted(keys: pd.DataFrame) -> pd.DataFrame:
    """Subject-mean SAE gain on Gemma bin CSVs, LanA-restricted electrodes."""
    rows = []
    keep = keys[["subject", "channel"]].drop_duplicates()
    keep["channel"] = normalize_channel(keep["channel"])
    for csv in sorted(ap.SAE_TABLES.glob("sparse_encoding_results_*lang_bin*.csv")):
        df = pd.read_csv(csv)
        if "full__R_fisher_paired" not in df.columns:
            continue
        df["channel"] = normalize_channel(df["channel"])
        m = df.merge(keep, on=["subject", "channel"], how="inner")
        if m.empty or "surprisal_only__R_fisher_paired" not in m.columns:
            continue
        m = m.copy()
        m["paper_sae_gain"] = (
            m["full__R_fisher_paired"] - m["surprisal_only__R_fisher_paired"])
        m["paper_full_R"] = m["full__R_fisher_paired"]
        m["paper_surprisal_R"] = m["surprisal_only__R_fisher_paired"]
        sf = m.groupby("subject")[["paper_full_R", "paper_surprisal_R",
                                   "paper_sae_gain"]].mean()
        rows.append({
            "file": csv.name,
            "n_electrodes": int(len(m)),
            "n_subjects": int(m["subject"].nunique()),
            "full_R_subject_mean": float(sf["paper_full_R"].mean()),
            "surprisal_R_subject_mean": float(sf["paper_surprisal_R"].mean()),
            "sae_gain_subject_mean": float(sf["paper_sae_gain"].mean()),
        })
    return pd.DataFrame(rows)


def write_report(
    cov_all: pd.DataFrame,
    cov_lana: pd.DataFrame,
    summaries: pd.DataFrame,
    stats: pd.DataFrame,
    sweep: pd.DataFrame,
    resid: pd.DataFrame,
    bins: pd.DataFrame,
    n_lookup: Dict[str, int],
    path: Path,
):
    a: List[str] = []
    add = a.append
    add("=" * 72)
    add("LANA SENSITIVITY — LEPORI STUDY 3 iEEG")
    add("LanA_n806 (Lipkin et al.)  |  2 mm sphere max  |  primary p ≥ 0.20")
    add("=" * 72)
    add("")
    add("This is a restriction of already-scored electrodes, not a new encoding.")
    add("Discrete Fedorenko 5-fROI NIfTIs were not on disk; only LanA (whole")
    add("language-network probability) is used to restrain channels.")
    add("")
    add(f"  Lookup: {n_lookup['n_mni']} electrodes with finite MNI;")
    add(f"          {n_lookup['n_nan']} missing/out-of-bounds → treated as outside.")
    add(f"  is_lang: {n_lookup['n_lang']}  |  is_lang ∩ LanA≥0.20: "
        f"{n_lookup['n_lang_lana']}  |  LanA≥0.20 any localizer: "
        f"{n_lookup['n_lana']}")
    add("")
    add("-" * 72)
    add("COVERAGE (AAL3 names inside each mask)")
    add("-" * 72)
    add("All is_lang:")
    add(cov_all.to_string(index=False))
    add("")
    add("is_lang ∩ LanA p≥0.20:")
    add(cov_lana.to_string(index=False))
    add("")
    add("-" * 72)
    add("SAE GAIN (subject-mean Fisher-z r)")
    add("-" * 72)
    for tag, g in summaries.groupby("feature_tag"):
        add(f"\n[{tag}]")
        for _, r in g.iterrows():
            nE, nS = int(r["n_electrodes"]), int(r["n_subjects"])
            mu = r["paper_sae_gain_subject_mean"]
            lo, hi = r["paper_sae_gain_ci_lo"], r["paper_sae_gain_ci_hi"]
            add(f"  {str(r['group']):28s} nE={nE:3d} nS={nS:2d}  "
                f"gain={mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    add("")
    add("-" * 72)
    add("INFERENCE (sign-flip over subjects)")
    add("-" * 72)
    if not stats.empty:
        cols = [c for c in (
            "family", "group_a", "n_a", "n_subjects_used", "mean_diff",
            "perm_p", "q_fdr", "feature_tag", "method",
        ) if c in stats.columns]
        add(stats[cols].round(4).to_string(index=False))
    add("")
    add("-" * 72)
    add("THRESHOLD SWEEP (is_lang ∩ LanA p ≥ t)")
    add("-" * 72)
    if not sweep.empty:
        show = sweep[[
            "feature_tag", "lana_threshold", "n_electrodes", "n_subjects",
            "paper_sae_gain_subject_mean", "paper_sae_gain_ci_lo",
            "paper_sae_gain_ci_hi",
        ]].copy()
        add(show.round(4).to_string(index=False))
    add("")
    if not resid.empty:
        add("-" * 72)
        add("RESIDUAL vs SAE on is_lang ∩ LanA p≥0.20")
        add("-" * 72)
        cols = [c for c in (
            "sae_tag", "group", "n_electrodes", "n_subjects",
            "resid_R_subject_mean", "sae_full_R_subject_mean",
            "surprisal_R_subject_mean", "sae_gain_subject_mean",
            "resid_minus_sae_subject_mean",
        ) if c in resid.columns]
        add(resid[cols].round(4).to_string(index=False))
        add("")
    if not bins.empty:
        add("-" * 72)
        add("GEMMA BIN REFITS on is_lang ∩ LanA p≥0.20")
        add("-" * 72)
        add(bins.round(4).to_string(index=False))
        add("")
    add("-" * 72)
    add("HOW TO READ")
    add("-" * 72)
    add("  Primary sensitivity = is_lang AND LanA p≥0.20.")
    add("  LanA is a group probability atlas, not subject-specific parcels.")
    add("  If SAE gain stays positive here, the network-level replication")
    add("  is not an artefact of AAL3 leftovers outside the atlas.")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(a) + "\n")
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_perm", type=int, default=N_PERM)
    p.add_argument("--threshold", type=float, default=PRIMARY_THRESH)
    p.add_argument("--radius_mm", type=float, default=SPHERE_RADIUS_MM)
    p.add_argument("--atlas", type=str, default=str(ap.LANA_NII))
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    global PRIMARY_THRESH, SPHERE_RADIUS_MM
    PRIMARY_THRESH = float(args.threshold)
    SPHERE_RADIUS_MM = float(args.radius_mm)

    atlas_path = Path(args.atlas)
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)
    data, affine = _load_lana(atlas_path)
    print(f"LanA {atlas_path.name} shape={data.shape} "
          f"range=[{data.min():.3f}, {data.max():.3f}]")

    gemma = load_sae_cohort(GEMMA_TAG)
    qwen = load_sae_cohort(QWEN_TAG)

    def lookup_for(df: pd.DataFrame) -> np.ndarray:
        xyz = df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
        bad = ~np.isfinite(xyz).all(axis=1)
        vals = np.full(len(df), np.nan)
        if np.any(~bad):
            vals[~bad] = lookup_lana_sphere(
                data, affine, xyz[~bad], radius_mm=SPHERE_RADIUS_MM)
        return vals

    vg = lookup_for(gemma)
    vq = lookup_for(qwen)
    gemma = attach_lana(gemma, vg)
    qwen = attach_lana(qwen, vq)

    lang_g = language_mask(gemma)
    n_mni = int(np.isfinite(vg).sum())
    n_nan = int((~np.isfinite(vg)).sum())
    n_lang = int(lang_g.sum())
    n_lang_lana = int((lang_g & gemma["lana_in_primary"].fillna(False)).sum())
    n_lana = int(gemma["lana_in_primary"].fillna(False).sum())
    counts = {
        "n_mni": n_mni, "n_nan": n_nan, "n_lang": n_lang,
        "n_lang_lana": n_lang_lana, "n_lana": n_lana,
    }
    print(f"Gemma lookup: MNI={n_mni} missing={n_nan} "
          f"is_lang={n_lang} ∩Lana={n_lang_lana} LanaAny={n_lana}")

    elec = pd.concat([
        gemma.assign(feature_tag=GEMMA_TAG),
        qwen.assign(feature_tag=QWEN_TAG),
    ], ignore_index=True)
    cols = [c for c in (
        "subject", "channel", "region", "hemisphere", "is_lang",
        "MNI_x", "MNI_y", "MNI_z", "lepori_froi", "lana_p", "lana_in_primary",
        "paper_full_R", "paper_surprisal_R", "paper_sae_gain",
        "paper_quality_ok", "feature_tag",
    ) if c in elec.columns]
    elec[cols].to_csv(ap.SAE_STUDY3_LANA_ELECTRODES, index=False)
    print(f"Saved {len(elec)} rows -> {ap.SAE_STUDY3_LANA_ELECTRODES.name}")

    cov_all = coverage_table(gemma)
    gemma_lana = gemma.copy()
    gemma_lana.loc[~(language_mask(gemma) & gemma["lana_in_primary"]), "is_lang"] = False
    cov_lana = coverage_table(attach_froi(gemma_lana))

    summaries = pd.concat([
        summarize_masks(gemma, GEMMA_TAG),
        summarize_masks(qwen, QWEN_TAG),
    ], ignore_index=True)
    summaries.to_csv(ap.SAE_STUDY3_LANA_SUMMARY, index=False)

    stats = pd.concat([
        stats_masks(gemma, GEMMA_TAG, args.n_perm),
        stats_masks(qwen, QWEN_TAG, args.n_perm),
    ], ignore_index=True)
    stats.to_csv(ap.SAE_STUDY3_LANA_STATS, index=False)

    sweep = pd.concat([
        threshold_sweep(gemma, GEMMA_TAG),
        threshold_sweep(qwen, QWEN_TAG),
    ], ignore_index=True)
    sweep.to_csv(ap.SAE_STUDY3_LANA_SWEEP, index=False)

    keys = gemma.loc[
        language_mask(gemma) & gemma["lana_in_primary"].fillna(False)
        & gemma["paper_quality_ok"].fillna(False),
        ["subject", "channel"],
    ]
    qwen_l = qwen.copy()
    qwen_l["is_lang"] = language_mask(qwen) & qwen["lana_in_primary"].fillna(False)
    gemma_l = gemma.copy()
    gemma_l["is_lang"] = language_mask(gemma) & gemma["lana_in_primary"].fillna(False)
    resid = pd.concat([
        residual_vs_sae(qwen_l, ap.SAE_STUDY3_QWEN_RESID, QWEN_TAG),
        residual_vs_sae(gemma_l, ap.SAE_STUDY3_GEMMA_RESID, GEMMA_TAG),
    ], ignore_index=True)
    if not resid.empty:
        resid.to_csv(ap.SAE_TABLES / "lepori_study3_lana_resid_vs_sae.csv", index=False)

    bins = bin_refits_restricted(keys)
    if not bins.empty:
        bins.to_csv(ap.SAE_TABLES / "lepori_study3_lana_bin_refits.csv", index=False)

    write_report(
        cov_all, cov_lana, summaries, stats, sweep, resid, bins, counts,
        ap.SAE_STUDY3_LANA_REPORT,
    )


if __name__ == "__main__":
    main()
