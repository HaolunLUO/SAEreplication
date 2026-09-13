#!/usr/bin/env python3
"""Tests for tagged surprisal / validity loading (no model weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_regression import load_sae_features


def test_tagged_surprisal_and_validity(tmp_path, monkeypatch):
    sec = tmp_path / "section_001"
    sec.mkdir()
    tag = "sae_test_tag"
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]))
    sparse.save_npz(sec / f"X_word_{tag}.npz", X)
    np.save(sec / f"X_word_{tag}_surprisal.npy", np.array([[1.0], [np.nan], [2.0]]))
    np.save(sec / f"X_word_{tag}_valid.npy", np.array([True, False, True]))
    # Poison untagged file — loader must prefer tagged.
    np.save(sec / "X_word_surprisal.npy", np.array([[9.0], [9.0], [9.0]]))

    monkeypatch.setattr(ap, "FEATURES_DIR", tmp_path)
    X2, surp, valid = load_sae_features(tag, [1])
    assert X2.shape == (3, 2)
    assert np.allclose(surp[[0, 2]], [1.0, 2.0])
    assert np.isnan(surp[1])
    assert valid.tolist() == [True, False, True]


def test_no_cross_model_surprisal_overwrite(tmp_path, monkeypatch):
    for tag, val in (("sae_gemma_x", 1.0), ("sae_qwen_x", 7.0)):
        sec = tmp_path / "section_001"
        sec.mkdir(exist_ok=True)
        sparse.save_npz(sec / f"X_word_{tag}.npz",
                          sparse.csr_matrix(np.ones((1, 2))))
        np.save(sec / f"X_word_{tag}_surprisal.npy", np.array([[val]]))
        np.save(sec / f"X_word_{tag}_valid.npy", np.array([True]))
    monkeypatch.setattr(ap, "FEATURES_DIR", tmp_path)
    _, sg, _ = load_sae_features("sae_gemma_x", [1])
    _, sq, _ = load_sae_features("sae_qwen_x", [1])
    assert float(sg[0]) == 1.0
    assert float(sq[0]) == 7.0


if __name__ == "__main__":
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # minimal manual run without pytest monkeypatch
        class MP:
            def setattr(self, obj, name, val):
                setattr(obj, name, val)
        # Use direct setattr / restore
        old = ap.FEATURES_DIR
        try:
            # inline the two tests with setattr
            sec = p / "section_001"
            sec.mkdir()
            tag = "sae_test_tag"
            X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]))
            sparse.save_npz(sec / f"X_word_{tag}.npz", X)
            np.save(sec / f"X_word_{tag}_surprisal.npy",
                    np.array([[1.0], [np.nan], [2.0]]))
            np.save(sec / f"X_word_{tag}_valid.npy", np.array([True, False, True]))
            np.save(sec / "X_word_surprisal.npy", np.array([[9.0], [9.0], [9.0]]))
            ap.FEATURES_DIR = p
            X2, surp, valid = load_sae_features(tag, [1])
            assert valid.tolist() == [True, False, True]
            assert float(surp[0]) == 1.0
            for tag, val in (("sae_gemma_x", 1.0), ("sae_qwen_x", 7.0)):
                sparse.save_npz(sec / f"X_word_{tag}.npz",
                                  sparse.csr_matrix(np.ones((1, 2))))
                np.save(sec / f"X_word_{tag}_surprisal.npy", np.array([[val]]))
                np.save(sec / f"X_word_{tag}_valid.npy", np.array([True]))
            assert float(load_sae_features("sae_gemma_x", [1])[1][0]) == 1.0
            assert float(load_sae_features("sae_qwen_x", [1])[1][0]) == 7.0
            print("load_sae_features tests passed.")
        finally:
            ap.FEATURES_DIR = old
