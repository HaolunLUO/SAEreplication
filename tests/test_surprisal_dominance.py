#!/usr/bin/env python3
"""Tests for Qwen all-channel surprisal-dominance + null helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_encoding.sparse_encoding_surprisal_dominance import (
    assign_task_class,
    classify_dominance,
    collapse_task_group,
    empirical_p,
    fdr_bh,
    merge_localizers,
    normalize_channel,
    overlap_paired_subject_value_diff,
    paper_group_summary,
    rank_candidates,
    run_paper_sae_gain_stats,
    run_task_type_pairwise_stats,
    subject_mean_one_sample_perm,
)
from sparse_encoding.sparse_encoding_deconvolution import (
    apply_null_shift,
    corrected_empirical_p,
    load_channel_allowlist,
)
from sparse_encoding.sparse_encoding_summary import bootstrap_subject_means


def _toy_encoding_df() -> pd.DataFrame:
    rows = [
        dict(
            subject="S1", channel="A1-A2",
            full__R_fisher_paired=0.05,
            content__R_fisher_paired=0.01,
            surprisal_only__R_fisher_paired=0.04,
            all_folds_paired=1,
            content__feature_mean=5.0,
        ),
        dict(
            subject="S1", channel="B1-B2",
            full__R_fisher_paired=0.06,
            content__R_fisher_paired=0.05,
            surprisal_only__R_fisher_paired=0.01,
            all_folds_paired=1,
            content__feature_mean=8.0,
        ),
        dict(
            subject="S1", channel="C1-C2",
            full__R_fisher_paired=0.04,
            content__R_fisher_paired=-0.02,
            surprisal_only__R_fisher_paired=0.03,
            all_folds_paired=1,
            content__feature_mean=0.0,
        ),
        dict(
            subject="S2", channel="D1-D2",
            full__R_fisher_paired=-0.01,
            content__R_fisher_paired=-0.02,
            surprisal_only__R_fisher_paired=-0.015,
            all_folds_paired=1,
            content__feature_mean=3.0,
        ),
        dict(
            subject="S2", channel="E1-E2",
            full__R_fisher_paired=0.05,
            content__R_fisher_paired=0.01,
            surprisal_only__R_fisher_paired=0.04,
            all_folds_paired=0,
            content__feature_mean=4.0,
        ),
    ]
    return pd.DataFrame(rows)


def test_classify_dominance_and_content_collapse():
    d = classify_dominance(_toy_encoding_df())
    by_ch = d.set_index("channel")
    assert bool(by_ch.loc["A1-A2", "surprisal_dominant"])
    assert bool(by_ch.loc["B1-B2", "sae_dominant"])
    assert bool(by_ch.loc["C1-C2", "content_collapsed"])
    assert not bool(by_ch.loc["C1-C2", "surprisal_dominant"])
    assert bool(by_ch.loc["C1-C2", "surprisal_dominant_desc"])
    assert bool(by_ch.loc["D1-D2", "low_performing"])
    assert not bool(by_ch.loc["E1-E2", "quality_ok"])
    assert abs(by_ch.loc["A1-A2", "surprisal_gain"] - 0.04) < 1e-9
    assert abs(by_ch.loc["A1-A2", "sae_gain"] - 0.01) < 1e-9
    assert abs(by_ch.loc["A1-A2", "dominance_index"] - 0.03) < 1e-9
    # Paper-aligned fields
    assert abs(by_ch.loc["A1-A2", "paper_sae_gain"] - 0.01) < 1e-9
    assert abs(by_ch.loc["A1-A2", "paper_full_R"] - 0.05) < 1e-9
    assert abs(by_ch.loc["A1-A2", "paper_surprisal_R"] - 0.04) < 1e-9
    assert bool(by_ch.loc["A1-A2", "paper_quality_ok"])
    assert not bool(by_ch.loc["E1-E2", "paper_quality_ok"])
    assert by_ch.loc["A1-A2", "exploratory_dominance_index"] == by_ch.loc[
        "A1-A2", "dominance_index"]
    print("OK test_classify_dominance_and_content_collapse")


def test_paper_sae_gain_one_sample_and_stats():
    d = classify_dominance(_dominance_task_df())
    # One-sample subject means
    res = subject_mean_one_sample_perm(
        d[d["paper_quality_ok"]], "paper_sae_gain", n_perm=100, min_group_n=3,
    )
    assert res["n_subjects_used"] >= 3
    assert np.isfinite(res["mean_diff"])
    assert np.isfinite(res["perm_p"])
    assert res["method"] == "sign_flip_subjects"

    # Small-group gate
    res_big = subject_mean_one_sample_perm(
        d, "paper_sae_gain", n_perm=20, min_group_n=100,
    )
    assert res_big["method"] == "insufficient"
    assert not np.isfinite(res_big["perm_p"])

    stats = run_paper_sae_gain_stats(d, n_perm=50, min_group_n=3)
    assert not stats.empty
    assert "family" in stats.columns
    assert "q_fdr" in stats.columns
    assert (stats["family"] == "overall").any()
    assert (stats["family"] == "exclusive_onesample").any()
    assert (stats["value_col"] == "paper_sae_gain").all()
    # Exploratory dominance columns must not be the paper test target
    assert not (stats["value_col"] == "dominance_index").any()

    summ = paper_group_summary(d, "task_group")
    assert not summ.empty
    assert "paper_sae_gain_subject_mean" in summ.columns
    print("OK test_paper_sae_gain_one_sample_and_stats")


def test_normalize_channel_strips_whitespace():
    s = pd.Series([" A1-A2 ", "B1-B2"])
    assert list(normalize_channel(s)) == ["A1-A2", "B1-B2"]
    print("OK test_normalize_channel_strips_whitespace")


def test_merge_localizers_coverage(tmp_path: Path):
    enc = classify_dominance(_toy_encoding_df())
    loc = pd.DataFrame({
        "subject": ["S1", "S1", "S1", "S2"],
        "channel": ["A1-A2", "B1-B2", "C1-C2", "D1-D2"],
        "is_pos_MITSWJNTask": [True, False, True, False],
        "is_pos_WM": [False, True, False, False],
        "is_pos_DMN": [False, False, True, False],
        "resp_MITSWJNTask_SN_diff": [0.5, 0.1, 0.4, np.nan],
        "resp_WM_SN_diff": [0.0, 0.3, 0.1, 0.0],
        "resp_DMN_SN_diff": [0.0, 0.0, 0.2, 0.0],
        "resp_MITSWJNTask_S": [1.0, 0.2, 0.8, np.nan],
        "resp_MITSWJNTask_N": [0.5, 0.1, 0.4, np.nan],
        "resp_MITSWJNTask_W": [0.7, 0.15, 0.5, np.nan],
        "resp_MITSWJNTask_J": [0.6, 0.12, 0.45, np.nan],
        "is_lang": [True, False, True, False],
        "region": ["STG", "MFG", "IFG", "Unknown"],
    })
    path = tmp_path / "cross_task.csv"
    loc.to_csv(path, index=False)
    import core.analysis_paths as ap
    old = ap.CROSS_TASK_RESPONSES
    ap.CROSS_TASK_RESPONSES = path
    try:
        merged, cov = merge_localizers(enc)
    finally:
        ap.CROSS_TASK_RESPONSES = old
    assert cov["n_encoding"] == 5
    assert cov["n_with_localizer_table"] == 4
    assert cov["n_dropped_no_localizer_row"] == 1
    assert "task_class" in merged.columns
    a = merged.set_index("channel").loc["A1-A2"]
    assert a["task_class"] == "mitswjn_only"
    c = merged.set_index("channel").loc["C1-C2"]
    assert c["task_class"] == "mitswjn_dmn"
    print("OK test_merge_localizers_coverage")


def test_assign_task_class():
    row = pd.Series({
        "is_pos_MITSWJNTask": True,
        "is_pos_WM": True,
        "is_pos_DMN": False,
    })
    assert assign_task_class(row) == "mitswjn_wm"
    assert collapse_task_group(assign_task_class(row)) == "multi_task"
    row2 = pd.Series({
        "is_pos_MITSWJNTask": False,
        "is_pos_WM": False,
        "is_pos_DMN": False,
    })
    assert assign_task_class(row2) == "none"
    print("OK test_assign_task_class")


def _dominance_task_df() -> pd.DataFrame:
    """Toy dominance table with exclusive + overlapping task groups."""
    rows = []
    # S1: mitswjn_only (high surprisal), wm_only (high SAE)
    for ch, m, w, di, sd, sae in [
        ("A1", True, False, 0.05, True, False),
        ("A2", True, False, 0.04, True, False),
        ("B1", False, True, -0.03, False, True),
        ("B2", False, True, -0.04, False, True),
        ("M1", True, True, 0.01, True, False),  # multi / both+
    ]:
        rows.append(dict(
            subject="S1", channel=ch,
            full__R_fisher_paired=0.05,
            content__R_fisher_paired=0.01 if sd else 0.04,
            surprisal_only__R_fisher_paired=0.04 if sd else 0.01,
            all_folds_paired=1, content__feature_mean=5.0,
            is_pos_MITSWJNTask=m, is_pos_WM=w, is_pos_DMN=False,
        ))
    # S2: same pattern
    for ch, m, w, sd in [
        ("A1", True, False, True),
        ("A2", True, False, True),
        ("B1", False, True, False),
        ("B2", False, True, False),
        ("M1", True, True, True),
    ]:
        rows.append(dict(
            subject="S2", channel=ch,
            full__R_fisher_paired=0.05,
            content__R_fisher_paired=0.01 if sd else 0.04,
            surprisal_only__R_fisher_paired=0.04 if sd else 0.01,
            all_folds_paired=1, content__feature_mean=5.0,
            is_pos_MITSWJNTask=m, is_pos_WM=w, is_pos_DMN=False,
        ))
    # S3 for min_group_n
    for ch, m, w, sd in [
        ("A1", True, False, True),
        ("B1", False, True, False),
        ("M1", True, True, True),
    ]:
        rows.append(dict(
            subject="S3", channel=ch,
            full__R_fisher_paired=0.05,
            content__R_fisher_paired=0.01 if sd else 0.04,
            surprisal_only__R_fisher_paired=0.04 if sd else 0.01,
            all_folds_paired=1, content__feature_mean=5.0,
            is_pos_MITSWJNTask=m, is_pos_WM=w, is_pos_DMN=False,
        ))
    d = classify_dominance(pd.DataFrame(rows))
    d["task_class"] = d.apply(assign_task_class, axis=1)
    d["task_group"] = d["task_class"].map(collapse_task_group)
    return d


def test_run_task_type_pairwise_stats():
    d = _dominance_task_df()
    stats = run_task_type_pairwise_stats(d, n_perm=100, min_group_n=3)
    assert not stats.empty
    assert "family" in stats.columns
    assert "q_fdr" in stats.columns
    # Exclusive continuous family should include mitswjn_only vs wm_only
    excl = stats[stats["family"] == "exclusive_continuous"]
    pairs = set(zip(excl["group_a"], excl["group_b"]))
    assert ("mitswjn_only", "wm_only") in pairs or ("wm_only", "mitswjn_only") in pairs
    # Binary family present
    assert (stats["family"] == "exclusive_binary").any()
    # Overlap family present
    assert (stats["family"] == "overlap_continuous").any()
    # Small-group gate
    stats_big = run_task_type_pairwise_stats(d, n_perm=20, min_group_n=100)
    assert (stats_big["method"] == "insufficient").all()
    print("OK test_run_task_type_pairwise_stats")


def test_overlap_paired_subject_means():
    d = _dominance_task_df()
    res = overlap_paired_subject_value_diff(
        d, "dominance_index", "MITSWJN+", "WM+", n_perm=100, rng_seed=19,
    )
    assert res["n_subjects_used"] >= 3
    assert np.isfinite(res["mean_diff"])
    assert np.isfinite(res["perm_p"])
    assert res["method"] == "paired_subject_sign_flip"
    print("OK test_overlap_paired_subject_means")


def test_empirical_p_corrected():
    nulls = np.array([0.0, 0.1, 0.2, 0.3])
    assert abs(empirical_p(0.5, nulls, "greater") - 0.2) < 1e-9
    assert abs(corrected_empirical_p(0.5, nulls, "greater") - 0.2) < 1e-9
    assert empirical_p(1.0, nulls, "greater") > 0
    print("OK test_empirical_p_corrected")


def test_fdr_bh_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = fdr_bh(p)
    assert np.all(np.isfinite(q))
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] == q.min()
    print("OK test_fdr_bh_monotone")


def test_true_block_permutation_shuffles_blocks():
    x = np.arange(200, dtype=float).reshape(-1, 1)
    slices = [(0, 200)]
    y = apply_null_shift(x, slices, "block", seed=123, block_len=20)
    assert np.sort(y[:, 0]).tolist() == np.sort(x[:, 0]).tolist()
    assert not np.array_equal(y, x)
    blocks_orig = [tuple(x[i:i + 20, 0]) for i in range(0, 200, 20)]
    blocks_new = [tuple(y[i:i + 20, 0]) for i in range(0, 200, 20)]
    assert sorted(blocks_orig) == sorted(blocks_new)
    print("OK test_true_block_permutation_shuffles_blocks")


def test_load_channel_allowlist(tmp_path: Path):
    p = tmp_path / "ch.csv"
    pd.DataFrame({
        "subject": ["Subject04", "Subject04", "Subject12"],
        "channel": [" T5-T4 ", "T2-T1", "Q4-Q3"],
    }).to_csv(p, index=False)
    allow = load_channel_allowlist(p)
    assert allow["Subject04"] == {"T5-T4", "T2-T1"}
    assert allow["Subject12"] == {"Q4-Q3"}
    print("OK test_load_channel_allowlist")


def test_rank_candidates_includes_nonlang():
    d = classify_dominance(pd.DataFrame([
        dict(subject="S1", channel="L1", full__R_fisher_paired=0.05,
             content__R_fisher_paired=0.01, surprisal_only__R_fisher_paired=0.04,
             all_folds_paired=1, content__feature_mean=5, is_lang=True,
             method_agree_surprisal=True, deconv_surprisal_gain=0.01,
             deconv_sae_gain=-0.01),
        dict(subject="S1", channel="N1", full__R_fisher_paired=0.06,
             content__R_fisher_paired=0.01, surprisal_only__R_fisher_paired=0.05,
             all_folds_paired=1, content__feature_mean=5, is_lang=False,
             method_agree_surprisal=False, deconv_surprisal_gain=np.nan,
             deconv_sae_gain=np.nan),
        dict(subject="S2", channel="N2", full__R_fisher_paired=0.07,
             content__R_fisher_paired=0.01, surprisal_only__R_fisher_paired=0.06,
             all_folds_paired=1, content__feature_mean=5, is_lang=False,
             method_agree_surprisal=False, deconv_surprisal_gain=np.nan,
             deconv_sae_gain=np.nan),
    ]))
    ranked = rank_candidates(d, top_n=2)
    assert set(ranked["channel"]) & {"N1", "N2"}
    assert ranked["surprisal_dominant"].all()
    print("OK test_rank_candidates_includes_nonlang")


def test_subject_aggregation_unit():
    d = pd.DataFrame({
        "subject": ["A"] * 10 + ["B"] * 2,
        "dominance_index": [0.1] * 10 + [-0.2] * 2,
    })
    subj = d.groupby("subject")["dominance_index"].mean().reset_index()
    mu, lo, hi, n = bootstrap_subject_means(
        subj, "dominance_index", n_boot=200, seed=1)
    assert n == 2
    assert abs(mu - (-0.05)) < 1e-9
    print("OK test_subject_aggregation_unit")


def main():
    import tempfile
    test_classify_dominance_and_content_collapse()
    test_normalize_channel_strips_whitespace()
    with tempfile.TemporaryDirectory() as td:
        test_merge_localizers_coverage(Path(td))
        test_load_channel_allowlist(Path(td))
    test_assign_task_class()
    test_paper_sae_gain_one_sample_and_stats()
    test_run_task_type_pairwise_stats()
    test_overlap_paired_subject_means()
    test_empirical_p_corrected()
    test_fdr_bh_monotone()
    test_true_block_permutation_shuffles_blocks()
    test_rank_candidates_includes_nonlang()
    test_subject_aggregation_unit()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
