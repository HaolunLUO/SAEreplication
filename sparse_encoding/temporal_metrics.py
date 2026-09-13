#!/usr/bin/env python3
"""
temporal_metrics.py
===================
Shared peak / FWHM / kernel-shape helpers for lag-resolved and deconvolution
analyses. Pure NumPy — no EEG I/O.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def peak_lag_amplitude(
    lags_ms: np.ndarray,
    values: np.ndarray,
    *,
    use_abs: bool = False,
) -> Tuple[float, float, int]:
    """Return (peak_lag_ms, peak_value, peak_index).

    Peak is argmax of ``values`` (or ``|values|`` if ``use_abs``).
    Empty / all-NaN input → (nan, nan, -1).
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    if lags_ms.size == 0 or values.size == 0 or lags_ms.size != values.size:
        return np.nan, np.nan, -1
    ok = np.isfinite(values)
    if not ok.any():
        return np.nan, np.nan, -1
    score = np.abs(values) if use_abs else values
    # Mask non-finite so they cannot win.
    score = np.where(ok, score, -np.inf)
    idx = int(np.argmax(score))
    if not np.isfinite(score[idx]):
        return np.nan, np.nan, -1
    return float(lags_ms[idx]), float(values[idx]), idx


def fwhm_ms(
    lags_ms: np.ndarray,
    values: np.ndarray,
    peak_idx: int,
    *,
    use_abs: bool = False,
) -> float:
    """Full-width at half-maximum around ``peak_idx`` (ms).

    Half-max is relative to the peak amplitude measured from zero baseline.
    Returns NaN if the peak is non-positive (signed) or the curve never
    drops to half-max on both sides.
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    if peak_idx < 0 or peak_idx >= values.size:
        return np.nan
    curve = np.abs(values) if use_abs else values
    peak = float(curve[peak_idx])
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    half = 0.5 * peak

    # Left crossing: last index ≤ peak where curve drops below half.
    left = peak_idx
    while left > 0 and curve[left] >= half:
        left -= 1
    if curve[left] >= half:
        left_ms = float(lags_ms[0])
    else:
        # Linear interpolate between left and left+1.
        y0, y1 = float(curve[left]), float(curve[left + 1])
        t0, t1 = float(lags_ms[left]), float(lags_ms[left + 1])
        frac = (half - y0) / (y1 - y0) if abs(y1 - y0) > 1e-15 else 0.0
        left_ms = t0 + frac * (t1 - t0)

    right = peak_idx
    n = values.size
    while right < n - 1 and curve[right] >= half:
        right += 1
    if curve[right] >= half:
        right_ms = float(lags_ms[-1])
    else:
        y0, y1 = float(curve[right - 1]), float(curve[right])
        t0, t1 = float(lags_ms[right - 1]), float(lags_ms[right])
        frac = (half - y0) / (y1 - y0) if abs(y1 - y0) > 1e-15 else 0.0
        right_ms = t0 + frac * (t1 - t0)

    width = right_ms - left_ms
    return float(width) if np.isfinite(width) and width >= 0 else np.nan


def onset_offset_ms(
    lags_ms: np.ndarray,
    values: np.ndarray,
    peak_idx: int,
    *,
    threshold_frac: float = 0.2,
    use_abs: bool = False,
) -> Tuple[float, float]:
    """Onset / offset at ``threshold_frac`` of peak amplitude (from zero)."""
    lags_ms = np.asarray(lags_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    if peak_idx < 0 or peak_idx >= values.size:
        return np.nan, np.nan
    curve = np.abs(values) if use_abs else values
    peak = float(curve[peak_idx])
    if not np.isfinite(peak) or peak <= 0:
        return np.nan, np.nan
    thr = threshold_frac * peak

    left = peak_idx
    while left > 0 and curve[left] >= thr:
        left -= 1
    onset = float(lags_ms[left]) if curve[left] < thr else float(lags_ms[0])

    right = peak_idx
    n = values.size
    while right < n - 1 and curve[right] >= thr:
        right += 1
    offset = float(lags_ms[right]) if curve[right] < thr else float(lags_ms[-1])
    return onset, offset


def area_under_curve(lags_ms: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal integral over lag (units: value × ms)."""
    lags_ms = np.asarray(lags_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(lags_ms) & np.isfinite(values)
    if ok.sum() < 2:
        return np.nan
    return float(np.trapezoid(values[ok], lags_ms[ok]))


def detect_multi_peaks(
    lags_ms: np.ndarray,
    values: np.ndarray,
    *,
    min_prominence_frac: float = 0.3,
    use_abs: bool = False,
) -> List[Dict[str, float]]:
    """Simple local-maxima peak list with prominence relative to global peak."""
    lags_ms = np.asarray(lags_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    curve = np.abs(values) if use_abs else values
    if curve.size < 3 or not np.any(np.isfinite(curve)):
        return []
    global_peak = float(np.nanmax(curve))
    if not np.isfinite(global_peak) or global_peak <= 0:
        return []
    thr = min_prominence_frac * global_peak
    peaks: List[Dict[str, float]] = []
    for i in range(1, curve.size - 1):
        if not np.isfinite(curve[i]):
            continue
        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1] and curve[i] >= thr:
            peaks.append({
                "lag_ms": float(lags_ms[i]),
                "amplitude": float(values[i]),
                "prominence": float(curve[i] / global_peak),
            })
    return peaks


def summarize_curve(
    lags_ms: np.ndarray,
    values: np.ndarray,
    *,
    prefix: str = "",
    use_abs: bool = False,
) -> Dict[str, float]:
    """Peak lag / R / FWHM / onset / offset / AUC for one lag curve."""
    peak_lag, peak_amp, idx = peak_lag_amplitude(
        lags_ms, values, use_abs=use_abs)
    fwhm = fwhm_ms(lags_ms, values, idx, use_abs=use_abs)
    onset, offset = onset_offset_ms(lags_ms, values, idx, use_abs=use_abs)
    auc = area_under_curve(lags_ms, values)
    polarity = (
        1.0 if np.isfinite(peak_amp) and peak_amp > 0
        else (-1.0 if np.isfinite(peak_amp) and peak_amp < 0 else np.nan)
    )
    n_peaks = len(detect_multi_peaks(lags_ms, values, use_abs=use_abs))
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}peak_lag_ms": peak_lag,
        f"{p}peak_amp": peak_amp,
        f"{p}fwhm_ms": fwhm,
        f"{p}onset_ms": onset,
        f"{p}offset_ms": offset,
        f"{p}auc": auc,
        f"{p}polarity": polarity,
        f"{p}n_peaks": float(n_peaks),
    }


def electrode_lag_peaks_from_long(
    long_df: pd.DataFrame,
    modes: Sequence[str] = ("full", "content", "surprisal_only"),
) -> pd.DataFrame:
    """Per-electrode peak metrics from long-form lag-screen rows.

    Expects columns: subject, channel, lag_ms, mode, R_fisher.
    """
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    keys = ["subject", "channel"]
    if "feature_tag" in long_df.columns:
        keys.append("feature_tag")
    for key, g in long_df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec: Dict[str, object] = {k: v for k, v in zip(keys, key)}
        prefix_map = {
            "full": "full",
            "content": "content",
            "surprisal_only": "surprisal",
        }
        for mode in modes:
            sub = g.loc[g["mode"] == mode].sort_values("lag_ms")
            if sub.empty:
                continue
            prefix = prefix_map.get(mode, mode.replace("_only", ""))
            lags = sub["lag_ms"].to_numpy(dtype=float)
            vals = sub["R_fisher"].to_numpy(dtype=float)
            stats = summarize_curve(lags, vals, prefix=prefix, use_abs=False)
            # Rename peak_amp → peak_R for lag-screen semantics.
            rec.update({
                k.replace("peak_amp", "peak_R"): v for k, v in stats.items()
            })
        if "full_peak_lag_ms" in rec and "surprisal_peak_lag_ms" in rec:
            fl = rec["full_peak_lag_ms"]
            sl = rec["surprisal_peak_lag_ms"]
            rec["lag_diff_full_vs_surprisal_ms"] = (
                float(fl) - float(sl)
                if np.isfinite(fl) and np.isfinite(sl) else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def kernel_shape_table(
    kernel_df: pd.DataFrame,
    value_cols: Sequence[str] = (
        "kernel_surprisal", "kernel_sae_l2", "kernel_nuisance_l2",
    ),
) -> pd.DataFrame:
    """Per-electrode kernel peak / FWHM / polarity from long kernel CSV."""
    if kernel_df.empty:
        return pd.DataFrame()
    keys = [c for c in ("subject", "channel", "feature_tag") if c in kernel_df.columns]
    rows = []
    for key, g in kernel_df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec: Dict[str, object] = {k: v for k, v in zip(keys, key)}
        g = g.sort_values("lag_ms")
        lags = g["lag_ms"].to_numpy(dtype=float)
        for col in value_cols:
            if col not in g.columns:
                continue
            # SAE / nuisance L2 are non-negative → use_abs False still works.
            # Surprisal kernel is signed → peak by |amp| for latency, keep sign.
            use_abs = col == "kernel_surprisal"
            short = col.replace("kernel_", "")
            stats = summarize_curve(
                lags, g[col].to_numpy(dtype=float),
                prefix=short, use_abs=use_abs,
            )
            rec.update(stats)
            peaks = detect_multi_peaks(
                lags, g[col].to_numpy(dtype=float), use_abs=use_abs)
            rec[f"{short}_n_local_max"] = float(len(peaks))
        rows.append(rec)
    return pd.DataFrame(rows)


def subject_mean_sign_flip_p(
    subject_values: np.ndarray,
    n_perm: int = 2000,
    seed: int = 19,
) -> Tuple[float, float]:
    """Two-sided sign-flip p for subject-mean ≠ 0. Returns (mean, p)."""
    vals = np.asarray(subject_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    obs = float(vals.mean())
    if n == 1:
        return obs, np.nan
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        null[i] = float((vals * signs).mean())
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def temporal_qc_report(
    peaks_df: pd.DataFrame,
    *,
    fwhm_col: str = "full_fwhm_ms",
    peak_lag_col: str = "full_peak_lag_ms",
    fold_std_col: Optional[str] = None,
) -> List[str]:
    """Quality checks for temporal precision analysis (text lines)."""
    lines = [
        "TEMPORAL PRECISION QC",
        "=" * 60,
        "",
    ]
    if peaks_df.empty:
        lines.append("(no peak rows)")
        return lines

    n = len(peaks_df)
    lines.append(f"Electrodes with peak metrics: {n}")

    # Check 1: clear peaks — FWHM < 400 ms
    if fwhm_col in peaks_df.columns:
        fwhm = peaks_df[fwhm_col].to_numpy(dtype=float)
        ok = np.isfinite(fwhm)
        frac = float(np.mean(fwhm[ok] < 400.0)) if ok.any() else np.nan
        median_fwhm = float(np.nanmedian(fwhm)) if ok.any() else np.nan
        lines.append(
            f"  Clear peaks (FWHM < 400 ms): {frac:.1%}  "
            f"(median FWHM={median_fwhm:.0f} ms)  "
            f"[{'PASS' if np.isfinite(frac) and frac >= 0.8 else 'WARN'}]"
        )

    # Check 2: fold consistency (optional column)
    if fold_std_col and fold_std_col in peaks_df.columns:
        std = peaks_df[fold_std_col].to_numpy(dtype=float)
        ok = np.isfinite(std)
        frac = float(np.mean(std[ok] < 100.0)) if ok.any() else np.nan
        lines.append(
            f"  Fold-stable peaks (std < 100 ms): {frac:.1%}  "
            f"[{'PASS' if np.isfinite(frac) and frac >= 0.8 else 'WARN'}]"
        )
    else:
        lines.append(
            "  Fold-stable peaks: (fold-std column not available in this table)"
        )

    # Check 3: temporal diversity — range of peak lags > 300 ms
    if peak_lag_col in peaks_df.columns:
        lags = peaks_df[peak_lag_col].to_numpy(dtype=float)
        ok = np.isfinite(lags)
        if ok.sum() >= 2:
            span = float(np.nanmax(lags) - np.nanmin(lags))
            lines.append(
                f"  Temporal diversity (peak-lag range): {span:.0f} ms  "
                f"[{'PASS' if span > 300 else 'WARN'}]"
            )
        else:
            lines.append("  Temporal diversity: insufficient peaks")

    lines.append("")
    return lines
