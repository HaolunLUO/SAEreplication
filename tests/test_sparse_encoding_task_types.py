#!/usr/bin/env python3
"""Tests for SAE task-type Jaccard analysis helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_encoding.sparse_encoding_qualitative import (
    parse_signed_features,
    parse_unsigned_features,
    serialize_signed_features,
    select_electrodes,
)
from sparse_encoding.sparse_encoding_surprisal_dominance import collapse_task_group
from sparse_encoding.sparse_encoding_task_type_analysis import (
    EXCLUSIVE_GROUPS,
    attach_task_metadata,
    compute_pairwise_electrode_jaccard,
    compute_union_jaccard_table,
    jaccard,
    run_jaccard_stats,
    summarize_pairwise_by_subject,
)


def test_serialize_parse_signed_index_zero():
    s = serialize_signed_features([0, 4, 38], [1, -1, 1])
    assert "0:+" in s.split()
    assert "4:-" in s.split()
    assert "38:+" in s.split()
    signed = parse_signed_features(s)
    assert (0, 1) in signed
    assert (4, -1) in signed
    assert (38, 1) in signed
    # Negative index 0 polarity preserved
    s_neg0 = serialize_signed_features([0], [-1])
    assert parse_signed_features(s_neg0) == {(0, -1)}
    print("OK test_serialize_parse_signed_index_zero")


def test_parse_legacy_and_unsigned():
    legacy = "-4 38 0"
    signed = parse_signed_features(legacy)
    assert (4, -1) in signed
    assert (38, 1) in signed
    assert (0, 1) in signed  # bare 0 → +1
    assert parse_unsigned_features(legacy) == {0, 4, 38}
    assert parse_signed_features("") == set()
    assert parse_signed_features(np.nan) == set()
    print("OK test_parse_legacy_and_unsigned")


def test_jaccard_empty_and_self():
    assert np.isnan(jaccard(set(), set()))
    assert jaccard({1}, {1}) == 1.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard({(1, 1)}, {(1, -1)}) == 0.0
    print("OK test_jaccard_empty_and_self")


def test_collapse_task_group():
    assert collapse_task_group("mitswjn_only") == "mitswjn_only"
    assert collapse_task_group("mitswjn_wm") == "multi_task"
    assert collapse_task_group("all_three") == "multi_task"
    assert collapse_task_group("none") == "none"
    assert collapse_task_group("unknown") == "unknown"
    print("OK test_collapse_task_group")


def test_select_electrodes_all_and_allowlist():
    res = pd.DataFrame({
        "subject": ["S1", "S1", "S2"],
        "channel": ["A1-A2", "B1-B2", "C1-C2"],
        "full__R_fisher": [0.1, 0.2, 0.05],
        "category": ["both", "enc_only", "neither"],
    })
    all_df, mode = select_electrodes(
        res, "full__R_fisher", top_n=0, per_category=False,
        all_channels=True, allowlist=None,
    )
    assert mode == "all_channels"
    assert len(all_df) == 3
    allow = {"S1": {"A1-A2"}}
    al_df, mode2 = select_electrodes(
        res, "full__R_fisher", top_n=0, per_category=False,
        all_channels=True, allowlist=allow,
    )
    assert mode2 == "allowlist"
    assert list(al_df["channel"]) == ["A1-A2"]
    print("OK test_select_electrodes_all_and_allowlist")


def _toy_feature_df() -> pd.DataFrame:
    # Two subjects, exclusive groups with overlapping SAE sets
    rows = [
        dict(subject="S1", channel="A1", feature_indices="1:+ 2:+", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S1", channel="A2", feature_indices="1:+ 3:+", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S1", channel="B1", feature_indices="10:+ 11:+", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S1", channel="B2", feature_indices="10:+ 12:+", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S2", channel="A1", feature_indices="1:+ 2:-", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S2", channel="A2", feature_indices="1:+ 4:+", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S2", channel="B1", feature_indices="10:+ 11:-", feature_tag="sae_qwen3_8b_l18"),
        dict(subject="S2", channel="C1", feature_indices="20:+ 21:+", feature_tag="sae_qwen3_8b_l18"),
        # Multi-task channel (MITSWJN + WM)
        dict(subject="S2", channel="M1", feature_indices="1:+ 10:+", feature_tag="sae_qwen3_8b_l18"),
    ]
    return pd.DataFrame(rows)


def _toy_localizer(tmp_path: Path) -> Path:
    loc = pd.DataFrame({
        "subject": ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2", "S2"],
        "channel": ["A1", "A2", "B1", "B2", "A1", "A2", "B1", "C1", "M1"],
        "is_pos_MITSWJNTask": [True, True, False, False, True, True, False, False, True],
        "is_pos_WM": [False, False, True, True, False, False, True, False, True],
        "is_pos_DMN": [False, False, False, False, False, False, False, True, False],
    })
    path = tmp_path / "cross_task.csv"
    loc.to_csv(path, index=False)
    return path


def test_attach_and_union_jaccard(tmp_path: Path):
    import core.analysis_paths as ap
    old = ap.CROSS_TASK_RESPONSES
    ap.CROSS_TASK_RESPONSES = _toy_localizer(tmp_path)
    try:
        feat, cov = attach_task_metadata(_toy_feature_df())
    finally:
        ap.CROSS_TASK_RESPONSES = old
    assert cov["n_merged_localizer"] == 9
    assert set(feat["task_group"]) >= {"mitswjn_only", "wm_only", "dmn_only", "multi_task"}
    m1 = feat.set_index("channel").loc["M1"]
    # M1 may appear once; task_group should be multi_task
    if isinstance(m1, pd.DataFrame):
        assert (m1["task_group"] == "multi_task").all()
    else:
        assert m1["task_group"] == "multi_task"

    union = compute_union_jaccard_table(
        feat, "task_group", EXCLUSIVE_GROUPS + ("none",),
        "exclusive", "sae_qwen3_8b_l18",
    )
    assert not union.empty
    assert "jaccard_unsigned" in union.columns
    assert "jaccard_signed" in union.columns
    print("OK test_attach_and_union_jaccard")


def test_self_pair_exclusion_overlap(tmp_path: Path):
    import core.analysis_paths as ap
    old = ap.CROSS_TASK_RESPONSES
    ap.CROSS_TASK_RESPONSES = _toy_localizer(tmp_path)
    try:
        feat, _ = attach_task_metadata(_toy_feature_df())
    finally:
        ap.CROSS_TASK_RESPONSES = old
    # Build overlap long form
    rows = []
    for g, col in (("MITSWJN+", "is_pos_MITSWJNTask"), ("WM+", "is_pos_WM")):
        sub = feat[feat[col]].copy()
        sub["overlap_group"] = g
        rows.append(sub)
    overlap = pd.concat(rows, ignore_index=True)
    pairs = compute_pairwise_electrode_jaccard(
        overlap, "overlap_group", ("MITSWJN+", "WM+"), "overlap",
        "sae_qwen3_8b_l18", exclude_self_pairs=True,
    )
    between = pairs[pairs["pair_type"] == "between"]
    # Multi-task channel M1 should not pair with itself across groups
    selfish = between[
        (between["channel_a"] == between["channel_b"])
        & (between["subject"] == "S2")
    ]
    assert selfish.empty, f"self-pairs leaked: {selfish}"
    print("OK test_self_pair_exclusion_overlap")


def test_subject_summary_and_stats_deterministic(tmp_path: Path):
    import core.analysis_paths as ap
    old = ap.CROSS_TASK_RESPONSES
    ap.CROSS_TASK_RESPONSES = _toy_localizer(tmp_path)
    try:
        feat, _ = attach_task_metadata(_toy_feature_df())
    finally:
        ap.CROSS_TASK_RESPONSES = old
    excl = feat[feat["task_group"].isin(EXCLUSIVE_GROUPS + ("none",))].copy()
    pairs = compute_pairwise_electrode_jaccard(
        excl, "task_group", EXCLUSIVE_GROUPS, "exclusive", "sae_qwen3_8b_l18",
    )
    subj = summarize_pairwise_by_subject(pairs)
    assert not subj.empty
    assert set(subj["metric"]) <= {"jaccard_unsigned", "jaccard_signed"}

    s1 = run_jaccard_stats(
        feat, subj, n_perm=50, min_group_n=2,
        feature_tag="sae_qwen3_8b_l18", rng_seed=19,
    )
    s2 = run_jaccard_stats(
        feat, subj, n_perm=50, min_group_n=2,
        feature_tag="sae_qwen3_8b_l18", rng_seed=19,
    )
    assert len(s1) == len(s2)
    # Deterministic p-values for finite tests
    m1 = s1.dropna(subset=["perm_p"]).sort_values(
        ["family", "metric", "group_a", "group_b"])
    m2 = s2.dropna(subset=["perm_p"]).sort_values(
        ["family", "metric", "group_a", "group_b"])
    if len(m1):
        np.testing.assert_allclose(
            m1["perm_p"].to_numpy(), m2["perm_p"].to_numpy(), rtol=0, atol=0)
    # Small-group gate: with min_group_n=100 everything insufficient
    s_big = run_jaccard_stats(
        feat, subj, n_perm=20, min_group_n=100,
        feature_tag="sae_qwen3_8b_l18", rng_seed=1,
    )
    assert (s_big["method"] == "insufficient").all()
    print("OK test_subject_summary_and_stats_deterministic")


def main():
    import tempfile
    test_serialize_parse_signed_index_zero()
    test_parse_legacy_and_unsigned()
    test_jaccard_empty_and_self()
    test_collapse_task_group()
    test_select_electrodes_all_and_allowlist()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_attach_and_union_jaccard(p)
        test_self_pair_exclusion_overlap(p)
        test_subject_summary_and_stats_deterministic(p)
    print("\nALL TASK-TYPE TESTS PASSED")


if __name__ == "__main__":
    main()
