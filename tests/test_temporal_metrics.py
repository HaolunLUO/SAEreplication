#!/usr/bin/env python3
"""Unit tests for temporal_metrics helpers (no EEG I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparse_encoding.temporal_metrics import (
    area_under_curve,
    detect_multi_peaks,
    electrode_lag_peaks_from_long,
    fwhm_ms,
    kernel_shape_table,
    peak_lag_amplitude,
    subject_mean_sign_flip_p,
    summarize_curve,
    temporal_qc_report,
)


def test_peak_and_fwhm_triangle():
    lags = np.arange(0, 801, 50, dtype=float)
    # Triangle peaking at 300 ms with height 1.
    values = np.maximum(0.0, 1.0 - np.abs(lags - 300.0) / 200.0)
    peak_lag, peak_amp, idx = peak_lag_amplitude(lags, values)
    assert peak_lag == 300.0
    assert abs(peak_amp - 1.0) < 1e-9
    width = fwhm_ms(lags, values, idx)
    # Half-max at ±100 ms from peak → FWHM ≈ 200 ms.
    assert 150 <= width <= 250


def test_summarize_curve_and_auc():
    lags = np.array([0.0, 50.0, 100.0, 150.0])
    values = np.array([0.0, 1.0, 0.5, 0.0])
    stats = summarize_curve(lags, values, prefix="full")
    assert stats["full_peak_lag_ms"] == 50.0
    assert stats["full_peak_amp"] == 1.0
    assert np.isfinite(stats["full_auc"])
    assert area_under_curve(lags, values) > 0


def test_multi_peak_detection():
    lags = np.arange(0, 401, 50, dtype=float)
    values = np.zeros_like(lags)
    values[2] = 1.0   # 100 ms
    values[6] = 0.8   # 300 ms
    peaks = detect_multi_peaks(lags, values, min_prominence_frac=0.25)
    assert len(peaks) >= 2


def test_electrode_peaks_from_long():
    rows = []
    for lag in (0, 100, 200, 300, 400):
        for mode, amp in (("full", 0.1 if lag != 200 else 0.5),
                          ("surprisal_only", 0.1 if lag != 100 else 0.4),
                          ("content", 0.05)):
            rows.append({
                "subject": "S1", "channel": "A1",
                "lag_ms": float(lag), "mode": mode, "R_fisher": amp,
                "feature_tag": "tag",
            })
    long_df = pd.DataFrame(rows)
    peaks = electrode_lag_peaks_from_long(long_df)
    assert len(peaks) == 1
    assert peaks.iloc[0]["full_peak_lag_ms"] == 200.0
    assert peaks.iloc[0]["surprisal_peak_lag_ms"] == 100.0
    assert peaks.iloc[0]["lag_diff_full_vs_surprisal_ms"] == 100.0


def test_kernel_shape_table():
    lags = np.arange(0, 401, 50, dtype=float)
    rows = []
    for lag in lags:
        rows.append({
            "subject": "S1", "channel": "A1", "feature_tag": "t",
            "lag_ms": float(lag),
            "kernel_surprisal": float(1.0 - abs(lag - 150) / 150) if abs(lag - 150) < 150 else 0.0,
            "kernel_sae_l2": float(max(0.0, 1.0 - abs(lag - 250) / 150)),
            "kernel_nuisance_l2": 0.1,
        })
    shapes = kernel_shape_table(pd.DataFrame(rows))
    assert len(shapes) == 1
    assert shapes.iloc[0]["sae_l2_peak_lag_ms"] == 250.0


def test_sign_flip_and_qc():
    vals = np.array([10.0, 20.0, 15.0, 12.0, 18.0, 14.0, 16.0, 11.0])
    mu, p = subject_mean_sign_flip_p(vals, n_perm=2000, seed=0)
    assert mu > 0
    assert p < 0.05
    peaks = pd.DataFrame({
        "full_peak_lag_ms": [100, 200, 450, 500],
        "full_fwhm_ms": [120, 150, 180, 200],
    })
    lines = temporal_qc_report(peaks)
    text = "\n".join(lines)
    assert "TEMPORAL PRECISION QC" in text
    assert "PASS" in text or "WARN" in text


if __name__ == "__main__":
    test_peak_and_fwhm_triangle()
    test_summarize_curve_and_auc()
    test_multi_peak_detection()
    test_electrode_peaks_from_long()
    test_kernel_shape_table()
    test_sign_flip_and_qc()
    print("OK all temporal_metrics tests")
