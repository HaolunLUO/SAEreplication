#!/usr/bin/env python3
"""Focused tests for unique token ownership and SAE-then-mean aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_encoding.sae_extract_features import (
    adjacent_duplicate_row_rate,
    aggregate_owned_token_sae,
    alignment_report,
    assign_tokens_shared_char_weighted,
    assign_tokens_to_words,
    build_section_text,
    parse_args,
)


class _FakeWT:
    """Minimal stand-in for a word_timing DataFrame column."""

    def __init__(self, words):
        self._words = words

    def astype(self, _dtype):
        return self

    def tolist(self):
        return list(self._words)


def test_unique_ownership_no_shared_tokens():
    # Text: "ABCD" segmented as A|BC|D ; tokens cover [0,2), [2,3), [3,4)
    word_spans = [(0, 1), (1, 3), (3, 4)]
    offsets = [(0, 2), (2, 3), (3, 4)]
    tpw, owner = assign_tokens_to_words(offsets, word_spans)
    # Token0 last char at 1 → word1; token1 last char 2 → word1; token2 → word2
    assert owner == [1, 1, 2]
    assert tpw[0] == []          # word0 owns nothing
    assert tpw[1] == [0, 1]
    assert tpw[2] == [2]
    # No token appears in more than one word
    seen = []
    for toks in tpw:
        seen.extend(toks)
    assert len(seen) == len(set(seen)) == 3


def test_owner_is_final_overlapping_word():
    # One coarse token spanning three words → owned by the last word only.
    word_spans = [(0, 1), (1, 2), (2, 3)]
    offsets = [(0, 3)]
    tpw, owner = assign_tokens_to_words(offsets, word_spans)
    assert owner == [2]
    assert tpw == [[], [], [0]]


def test_sae_then_mean_and_exclude_ownerless():
    # 3 tokens, 3 words; word0 owns none, word1 owns toks 0+1, word2 owns tok2
    token_lat = np.array([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 6.0, 0.0],
    ], dtype=np.float32)
    surprisal_tok = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    tpw = [[], [0, 1], [2]]
    lat, surp, valid, stats = aggregate_owned_token_sae(
        token_lat, surprisal_tok, tpw, aggregation="mean")
    assert valid.tolist() == [False, True, True]
    assert np.allclose(lat[1].toarray(), [[2.0, 0.0, 3.0]])  # mean of tok0/1
    assert np.allclose(lat[2].toarray(), [[5.0, 6.0, 0.0]])
    assert lat[0].nnz == 0
    assert np.isclose(surp[1], 2.0)   # mean NLL
    assert np.isclose(surp[2], 5.0)
    assert np.isnan(surp[0])
    assert stats["surprisal_imputation_count"] == 0
    assert stats["excluded_word_count"] == 1


def test_no_surprisal_imputation_for_nan_owned_tokens():
    token_lat = np.array([[1.0, 2.0]], dtype=np.float32)
    surprisal_tok = np.array([np.nan], dtype=np.float32)
    lat, surp, valid, stats = aggregate_owned_token_sae(
        token_lat, surprisal_tok, [[0]], aggregation="mean")
    assert valid.tolist() == [False]
    assert np.isnan(surp[0])
    assert stats["surprisal_imputation_count"] == 0


def test_adjacent_duplicate_rate_detects_shared_rows():
    a = sparse.csr_matrix([[1, 0], [1, 0], [0, 2]], dtype=np.float32)
    assert np.isclose(adjacent_duplicate_row_rate(a), 0.5)
    b = sparse.csr_matrix([[1, 0], [0, 1], [0, 2]], dtype=np.float32)
    assert np.isclose(adjacent_duplicate_row_rate(b), 0.0)


def test_tagged_surprisal_paths_do_not_collide(tmp_path, monkeypatch=None):
    """Gemma and Qwen surprisal filenames are distinct."""
    tag_g = "sae_gemma2_2b_mat_l12"
    tag_q = "sae_qwen3_8b_l18"
    g = tmp_path / f"X_word_{tag_g}_surprisal.npy"
    q = tmp_path / f"X_word_{tag_q}_surprisal.npy"
    np.save(g, np.array([[1.0]], dtype=np.float32))
    np.save(q, np.array([[9.0]], dtype=np.float32))
    assert g.name != q.name
    assert float(np.load(g).ravel()[0]) == 1.0
    assert float(np.load(q).ravel()[0]) == 9.0


def test_shared_char_weighted_de_shihou():
    """Token 的时候 over words 的 / 时候: weights 1/3 and 2/3, full coverage."""
    word_spans = [(0, 1), (1, 3)]  # 的 | 时候
    offsets = [(0, 3)]
    assigned = assign_tokens_shared_char_weighted(offsets, word_spans)
    assert assigned.tokens_per_word == [[0], [0]]
    assert np.isclose(assigned.weights_per_word[0][0], 1.0 / 3.0)
    assert np.isclose(assigned.weights_per_word[1][0], 2.0 / 3.0)
    assert assigned.shared_token == [True, True]
    assert assigned.n_shared_tokens == 1
    rep = alignment_report(
        assigned.tokens_per_word, weights_per_word=assigned.weights_per_word)
    assert rep["coverage_pct"] == 100.0
    assert rep["words_with_0_tokens"] == 0
    assert rep["ownership_mismatch_words"] == 0

    token_lat = np.array([[3.0, 0.0, 6.0]], dtype=np.float32)
    surprisal_tok = np.array([1.5], dtype=np.float32)
    lat, surp, valid, stats = aggregate_owned_token_sae(
        token_lat, surprisal_tok, assigned.tokens_per_word,
        weights_per_word=assigned.weights_per_word)
    assert valid.tolist() == [True, True]
    assert np.allclose(lat[0].toarray(), [[3.0, 0.0, 6.0]])
    assert np.allclose(lat[1].toarray(), [[3.0, 0.0, 6.0]])
    assert np.isclose(surp[0], 1.5) and np.isclose(surp[1], 1.5)
    assert stats["surprisal_imputation_count"] == 0


def test_shared_weight_normalized_mean():
    token_lat = np.array([[3.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    surprisal_tok = np.array([3.0, 9.0], dtype=np.float32)
    lat, surp, valid, _stats = aggregate_owned_token_sae(
        token_lat, surprisal_tok, [[0, 1]],
        weights_per_word=[[1.0 / 3.0, 2.0 / 3.0]])
    assert valid.tolist() == [True]
    # (1/3)*[3, 0] + (2/3)*[0, 9] = [1, 6]
    assert np.allclose(lat[0].toarray(), [[1.0, 6.0]])
    assert np.isclose(surp[0], 7.0)


def test_equal_weights_match_unweighted_mean():
    token_lat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    surprisal_tok = np.array([1.0, 5.0], dtype=np.float32)
    lat_u, surp_u, valid_u, _ = aggregate_owned_token_sae(
        token_lat, surprisal_tok, [[0, 1]])
    lat_w, surp_w, valid_w, _ = aggregate_owned_token_sae(
        token_lat, surprisal_tok, [[0, 1]],
        weights_per_word=[[1.0, 1.0]])
    assert valid_u.tolist() == valid_w.tolist() == [True]
    assert np.array_equal(lat_u.toarray(), lat_w.toarray())
    assert np.isclose(surp_u[0], surp_w[0])


def test_shared_ownership_appends_v2_tag():
    args = parse_args([
        "--preset", "qwen35_4b_mat_l15",
        "--ownership", "shared_char_weighted",
    ])
    assert args.ownership == "shared_char_weighted"
    assert args.feature_tag == "sae_qwen35_4b_mat_l15_v2"
    plain = parse_args(["--preset", "qwen35_4b_mat_l15"])
    assert plain.ownership == "unique_final_word"
    assert plain.feature_tag == "sae_qwen35_4b_mat_l15"


def test_build_section_text_spans():
    import pandas as pd
    wt = pd.DataFrame({"word": ["当", "我", "们"]})
    sect = build_section_text(wt)
    assert sect.text == "当我们"
    assert sect.word_spans == [(0, 1), (1, 2), (2, 3)]


if __name__ == "__main__":
    # Lightweight runner without pytest dependency for env smoke checks.
    test_unique_ownership_no_shared_tokens()
    test_owner_is_final_overlapping_word()
    test_sae_then_mean_and_exclude_ownerless()
    test_no_surprisal_imputation_for_nan_owned_tokens()
    test_adjacent_duplicate_rate_detects_shared_rows()
    from pathlib import Path as _P
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_tagged_surprisal_paths_do_not_collide(_P(d))
    test_shared_char_weighted_de_shihou()
    test_shared_weight_normalized_mean()
    test_equal_weights_match_unweighted_mean()
    test_shared_ownership_appends_v2_tag()
    test_build_section_text_spans()
    print("All token-ownership tests passed.")
