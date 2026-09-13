#!/usr/bin/env python3
"""Qwen3.5-4B Matryoshka extract preset and nested-bin helpers (no weights)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_encoding.sae_extract_features import PRESETS, parse_args
from sparse_encoding.sparse_encoding_lepori_study3 import (
    GEMMA_TAG, QWEN35_MATRYOSHKA_BINS, QWEN35_TAG,
    _bin_name, _tag_from_bin_csv,
)


def test_qwen35_preset_fills_backbone():
    args = parse_args(["--preset", "qwen35_4b_mat_l15"])
    cfg = PRESETS["qwen35_4b_mat_l15"]
    assert args.model_name == cfg["model_name"]
    assert args.sae_release == cfg["sae_release"]
    assert args.sae_id == cfg["sae_id"]
    assert args.sae_layer == 15
    assert args.feature_tag == "sae_qwen35_4b_mat_l15"
    assert args.matryoshka_widths == (2048, 16384, 65536)
    assert args.trust_remote_code is True


def test_preset_cli_override_keeps_other_fields():
    args = parse_args([
        "--preset", "qwen35_4b_mat_l15", "--feature_tag", "custom_tag",
    ])
    assert args.feature_tag == "custom_tag"
    assert args.model_name == "Qwen/Qwen3.5-4B-Base"
    assert args.sae_layer == 15


def test_qwen35_nested_bin_edges():
    bins = QWEN35_MATRYOSHKA_BINS
    assert _bin_name(0, bins) == "bin0_2048"
    assert _bin_name(2047, bins) == "bin0_2048"
    assert _bin_name(2048, bins) == "bin2048_16384"
    assert _bin_name(16383, bins) == "bin2048_16384"
    assert _bin_name(16384, bins) == "bin16384_plus"
    assert _bin_name(65535, bins) == "bin16384_plus"


def test_tag_from_bin_csv_names():
    assert _tag_from_bin_csv("sparse_encoding_results_lang_bin0-128.csv") == GEMMA_TAG
    assert (
        _tag_from_bin_csv(
            "sparse_encoding_results_sae_qwen35_4b_mat_l15_lang_bin0-2048.csv"
        ) == QWEN35_TAG
    )


if __name__ == "__main__":
    test_qwen35_preset_fills_backbone()
    test_preset_cli_override_keeps_other_fields()
    test_qwen35_nested_bin_edges()
    test_tag_from_bin_csv_names()
    print("ok")
