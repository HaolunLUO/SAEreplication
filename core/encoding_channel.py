#!/usr/bin/env python3
"""
single_subject_wordlocked_encoder.py
====================================

Word-locked encoding analysis for a single subject using:
- EEG from .mat files (HDF5 / MATLAB v7.3)
- Pre-extracted linguistic features (embeddings, etc.)

Matches the Goldstein et al. encoding methodology (full-dimensional
embeddings, plain OLS, no PCA truncation, no ridge shrinkage):

1) Normalization
   - Y (neural):
       a) Z-score continuous EEG per channel WITHIN EACH SECTION at load time
       b) Inside CV, mean-center Y using the TRAINING-fold mean (apply to train+test)
   - X (predictors):
       a) Standardize X with StandardScaler fit on training folds, apply to test
          (fold-wise, no leakage; full embedding dimensionality retained)

2) Regression
   - Ordinary Least Squares (OLS) linear regression (no PCA, no ridge)
   - Fit separately for each lag, still evaluated with cross-validation

3) Channel Averaging
   - Option to average across ALL channels or SIGNIFICANT channels only
   - Permutation test identifies significant channels

4) Channel Selection (Optional)
   - Optionally analyze only a subset of channels
   - Selection by name, index, range, or regex pattern

5) Individual Channel Plots
   - Plot each significant channel separately

Outputs:
- Plots of performance vs lag (significant channels only by default)
- Individual plots per significant channel
- Pickle + MAT summary + per-feature CSV of channel stats
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import savemat
from scipy.fftpack import fft, ifft

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    StandardScaler = None

# -----------------------------
# Configuration
# -----------------------------

# EEG .mat files for each section (map section_id -> file path)
EEG_MAT_FILES = {
    1: "output/Naturalistic/segments/Subject02_NA_01_565sec.mat",
    2: "output/Naturalistic/segments/Subject02_NA2_01_643sec.mat",
    3: "output/Naturalistic/segments/Subject02_NA3_01_644sec.mat",
}

# Pre-extracted linguistic features directory
FEATURES_DIR = Path("extracted_linguistic_features")

# Which feature sets to analyze
FEATURE_SETS = ["glove", "gpt2cn_l05", "gpt2cn_l24", "gpt2cn_l45", "gpt2xl_l05", "gpt2xl_l24", "gpt2xl_l45", "qwen_l05", "qwen_l24", "qwen_l45"]
#FEATURE_SETS = [ "gpt2cn_l05", "gpt2cn_l24", "gpt2cn_l45", "gpt2xl_l05", "gpt2xl_l24", "gpt2xl_l45","qwen_l05", "qwen_l24", "qwen_l45"]
# Analysis parameters
FS_TARGET = 500
SECTIONS = (1, 2, 3)

# Lag parameters (Goldstein-style)
TMIN_MS = -2000
TMAX_MS = 2000
LAG_STEP_MS = 25
RESP_WIN_MS = 200

# Cross-validation
NFOLD = 10
FOLDS_WITHIN_SECTIONS = True
MIN_FOLDS_PER_SECTION = 1

# Predictor normalization (fold-wise, no leakage)
# Plain OLS on the full-dimensional embedding, matching Goldstein et al.:
# no PCA truncation, no ridge shrinkage.
STANDARDIZE_X = True

# Permutation test (required for significant channel detection)
RUN_PERMUTATION_TEST = True
PERM_KIND = "phase"
PERM_N = 1000
PERM_ALPHA = 0.05
PERM_USE_FDR = True
PERM_BATCH = 50
PERM_EPS = 1e-12

# Plotting options
PLOT_SIGNIFICANT_ONLY = True  # If True, plot only significant channels
MIN_SIGNIFICANT_CHANNELS = 1  # Minimum significant channels required to plot
PLOT_INDIVIDUAL_CHANNELS = True  # If True, plot each significant channel separately
MAX_INDIVIDUAL_PLOTS = 50  # Maximum number of individual channel plots to generate

# Channel selection - set to None for all channels
SELECTED_CHANNELS =  None

# Output
OUTDIR = Path("encoding_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42


# -----------------------------
# Utility Functions
# -----------------------------

def _require_sklearn():
    if StandardScaler is None:
        raise ImportError(
            "scikit-learn is required for fold-wise StandardScaler. "
            "Please install it (e.g., `pip install scikit-learn`)."
        )


def nansem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error of the mean, ignoring NaNs."""
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sd / np.sqrt(np.maximum(n, 1))


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, float)
    m = int(np.sum(np.isfinite(p)))
    reject = np.zeros_like(p, dtype=bool)
    p_adj = np.full_like(p, np.nan, dtype=float)

    if m == 0:
        return reject, p_adj

    idx = np.where(np.isfinite(p))[0]
    p_nonan = p[idx]
    order = np.argsort(p_nonan)
    p_sorted = p_nonan[order]

    ranks = np.arange(1, m + 1)
    thresh = (ranks / m) * alpha
    below = p_sorted <= thresh

    if np.any(below):
        kmax = np.max(np.where(below)[0])
        cutoff = p_sorted[kmax]
        reject[idx] = p_nonan <= cutoff

    p_adj_sorted = np.minimum.accumulate((m / ranks) * p_sorted[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    p_adj[idx] = p_adj_sorted[inv_order]

    return reject, p_adj


def zscore_channels_within_section(eeg_ct: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Z-score continuous EEG within a section, per channel."""
    eeg = np.asarray(eeg_ct, dtype=np.float64)
    mu = np.nanmean(eeg, axis=1, keepdims=True)
    sd = np.nanstd(eeg, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > eps), sd, 1.0)
    out = (eeg - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def corr_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pearson correlation per channel."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    yt0 = yt - yt.mean(axis=0, keepdims=True)
    yp0 = yp - yp.mean(axis=0, keepdims=True)
    num = np.sum(yt0 * yp0, axis=0)
    den = np.sqrt(np.sum(yt0 ** 2, axis=0) * np.sum(yp0 ** 2, axis=0)) + eps
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


# -----------------------------
# Phase-randomization utilities
# -----------------------------

def phase_randomize_1d(data: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Phase randomization preserving amplitude spectrum."""
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    n_samples = x.shape[0]
    if n_samples < 4:
        return x.copy()

    if n_samples % 2 == 0:
        pos_freq = np.arange(1, n_samples // 2)
        neg_freq = np.arange(n_samples - 1, n_samples // 2, -1)
    else:
        pos_freq = np.arange(1, (n_samples - 1) // 2 + 1)
        neg_freq = np.arange(n_samples - 1, (n_samples - 1) // 2, -1)

    phase_shifts = rng.random((len(pos_freq), 1)) * 2.0 * np.pi

    fft_data = fft(x, axis=0)
    fft_data[pos_freq, :] *= np.exp(1j * phase_shifts)
    fft_data[neg_freq, :] *= np.exp(-1j * phase_shifts)

    shifted_data = np.real(ifft(fft_data, axis=0))
    return np.ascontiguousarray(shifted_data)


def phase_randomized_correlations_fft(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        perm_n: int,
        rng: np.random.Generator,
        batch: int = 50,
        eps: float = 1e-12,
) -> np.ndarray:
    """Compute correlations with phase-randomized y_true (frequency-domain shortcut)."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)

    if yt.ndim == 1:
        yt = yt[:, None]
    if yp.ndim == 1:
        yp = yp[:, None]

    N, C = yt.shape
    if N < 4 or perm_n <= 0:
        return np.full((max(int(perm_n), 0), C), np.nan, dtype=float)

    yt0 = yt - yt.mean(axis=0, keepdims=True)
    yp0 = yp - yp.mean(axis=0, keepdims=True)

    den = np.sqrt(np.sum(yt0 ** 2, axis=0) * np.sum(yp0 ** 2, axis=0)) + eps
    bad = ~np.isfinite(den) | (den <= 0)
    den = np.where(bad, np.nan, den)

    Yt = fft(yt0, axis=0)
    Yp = fft(yp0, axis=0)
    cross = Yt * np.conj(Yp)

    if N % 2 == 0:
        pos = np.arange(1, N // 2)
        nyq = N // 2
    else:
        pos = np.arange(1, (N - 1) // 2 + 1)
        nyq = None

    cross0 = cross[0, :]
    cross_pos = cross[pos, :]
    const = np.real(cross0).astype(float)

    if nyq is not None:
        const = const + np.real(cross[nyq, :]).astype(float)

    K = cross_pos.shape[0]
    out = np.empty((perm_n, C), dtype=float)

    if K == 0:
        r_const = (const / float(N)) / den
        out[:] = r_const[None, :]
        return out

    two_over_N = 2.0 / float(N)

    for p0 in range(0, perm_n, max(1, int(batch))):
        p1 = min(p0 + max(1, int(batch)), perm_n)
        B = p1 - p0

        phases = rng.random((B, K)) * (2.0 * np.pi)
        exp_phi = np.exp(1j * phases)

        s = exp_phi @ cross_pos
        dot_freq = two_over_N * np.real(s) + (const / float(N))[None, :]

        out[p0:p1, :] = dot_freq / den[None, :]

    return out


# -----------------------------
# EEG Loading from .mat (HDF5)
# -----------------------------

def _read_hdf5_string(f: h5py.File, ref) -> str:
    """Dereference and read a string from HDF5 object reference."""
    try:
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            if isinstance(data, bytes):
                return data.decode("utf-8")
            elif isinstance(data, np.ndarray):
                if data.dtype.kind == "U":
                    return str(data.flat[0]) if data.size else ""
                elif data.dtype == np.uint16 or data.dtype == np.uint8:
                    return "".join(chr(c) for c in data.flatten() if c > 0)
                else:
                    return data.tobytes().decode("utf-8", errors="ignore").strip("\x00")
            else:
                return str(data)
        return str(obj)
    except Exception as e:
        return f"<error: {e}>"


def _extract_channel_labels(
        f: h5py.File,
        group: h5py.Group,
        field_name: str = "bip_ch_label",
) -> List[str]:
    """Extract channel labels from HDF5 MATLAB file."""
    labels: List[str] = []

    if field_name not in group:
        print(f"    [WARN] '{field_name}' not found in segment")
        return labels

    ch_label_data = group[field_name]

    if isinstance(ch_label_data, h5py.Dataset):
        data = ch_label_data[:]

        if data.dtype == h5py.ref_dtype or str(data.dtype).startswith("object"):
            for ref in data.flatten():
                try:
                    labels.append(_read_hdf5_string(f, ref))
                except Exception:
                    labels.append("<unknown>")
        elif data.dtype.kind in ("S", "U"):
            for s in data.flatten():
                if isinstance(s, bytes):
                    labels.append(s.decode("utf-8", errors="ignore").strip("\x00"))
                else:
                    labels.append(str(s))
        elif data.dtype.kind in ("u", "i"):
            if data.ndim == 2:
                for row in data:
                    label = "".join(chr(c) for c in row if 0 < c < 65536)
                    labels.append(label.strip())
            else:
                label = "".join(chr(c) for c in data.flatten() if 0 < c < 65536)
                labels.append(label.strip())
        else:
            print(f"    [WARN] Unexpected dtype for {field_name}: {data.dtype}")

    return labels


def load_eeg_from_mat_hdf5(
        mat_path: str | Path,
        var_name: str = "segment",
        field_name: str = "bip_elec_data",
        label_field: str = "bip_ch_label",
        convert_to_uv: bool = True,
) -> Tuple[np.ndarray, dict, List[str]]:
    """Load EEG data from MATLAB v7.3 .mat file (HDF5 format)."""
    mat_path = Path(mat_path)
    print(f"  Loading EEG (HDF5): {mat_path}")

    with h5py.File(mat_path, "r") as f:
        group = f[var_name]
        eeg_raw = group[field_name][:]

        duration = float(group["actualDuration"][0, 0]) if "actualDuration" in group else np.nan
        channel_labels = _extract_channel_labels(f, group, label_field)

    eeg = np.asarray(eeg_raw, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"Unexpected EEG array ndim={eeg.ndim}, shape={eeg.shape}")

    print(f"    Raw shape: {eeg.shape}")

    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
        print(f"    Transposed to: {eeg.shape}")

    n_channels, n_samples = eeg.shape
    sfreq = float(n_samples) / float(duration) if np.isfinite(duration) and duration > 0 else np.nan

    if len(channel_labels) != n_channels:
        print(f"    [WARN] Label count mismatch: {len(channel_labels)} labels vs {n_channels} channels")
        if len(channel_labels) < n_channels:
            channel_labels.extend([f"Ch{i}" for i in range(len(channel_labels), n_channels)])
        else:
            channel_labels = channel_labels[:n_channels]

    if convert_to_uv:
        eeg = eeg * 1e6
        print(f"    Converted to µV: range [{eeg.min():.2f}, {eeg.max():.2f}]")

    info = {
        "duration": float(duration),
        "sfreq": float(sfreq),
        "n_bip_channels": int(n_channels),
    }

    print(f"    Sample rate: {info['sfreq']:.1f} Hz, Duration: {info['duration']:.1f} s")
    if channel_labels:
        preview = channel_labels[:5]
        suffix = "..." if len(channel_labels) > 5 else ""
        print(f"    Channel names: {preview}{suffix}")

    return eeg, info, channel_labels


def resample_eeg(eeg: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Resample EEG from fs_orig to fs_target."""
    if not (np.isfinite(fs_orig) and np.isfinite(fs_target)):
        raise ValueError(f"Invalid fs_orig/fs_target: {fs_orig}, {fs_target}")
    if abs(fs_orig - fs_target) < 1e-9:
        return eeg

    ratio = float(fs_target) / float(fs_orig)
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    eeg_t = eeg.T
    eeg_resampled = signal.resample_poly(eeg_t, up, down, axis=0)

    print(f"    Resampled: {fs_orig:.1f} -> {fs_target:.1f} Hz, {eeg.shape[1]} -> {eeg_resampled.shape[0]} samples")
    return eeg_resampled.T


# -----------------------------
# Feature Loading
# -----------------------------

def load_section_features(
        features_dir: Path,
        section_id: int,
        feature_sets: List[str],
        target_sfreq: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Load pre-extracted linguistic features for a section."""
    sec_dir = features_dir / f"section_{section_id:03d}"
    if not sec_dir.exists():
        raise FileNotFoundError(f"Section directory not found: {sec_dir}")

    word_timing = pd.read_csv(sec_dir / "word_timing.csv")
    print(f"    Loaded word_timing: {len(word_timing)} words")

    if "onset_relative" in word_timing.columns:
        word_timing["onset_sample"] = np.rint(
            word_timing["onset_relative"].to_numpy(float) * float(target_sfreq)
        ).astype(int)
        word_timing["offset_sample"] = np.rint(
            word_timing["offset_relative"].to_numpy(float) * float(target_sfreq)
        ).astype(int)

    X_word: Dict[str, np.ndarray] = {}
    X_names: Dict[str, List[str]] = {}

    for feat in feature_sets:
        npy_path = sec_dir / f"X_word_{feat}.npy"
        if not npy_path.exists():
            print(f"    [WARN] Feature {feat} not found, skipping")
            continue

        X = np.load(npy_path)
        X_word[feat] = X.astype(np.float64)

        names_path = sec_dir / f"X_word_{feat}_feature_names.txt"
        if names_path.exists():
            with open(names_path, "r") as f:
                X_names[feat] = [line.strip() for line in f]
        else:
            X_names[feat] = [f"{feat}_{i}" for i in range(X.shape[1])]

        print(f"    Loaded X_word_{feat}: {X.shape}")

    return word_timing, X_word, X_names


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class SectionWordEEG:
    section_id: int
    fs: float
    eeg_cs: np.ndarray
    onsets: np.ndarray
    T: int
    W: int


@dataclass
class SubjectWordLockedData:
    subject: str
    fs: float
    sections_used: Tuple[int, ...]
    section_word_slices: List[Tuple[int, int]]
    sections: List[SectionWordEEG]
    X_by_feature: Dict[str, np.ndarray]
    n_channels: int
    channel_labels: List[str]
    good_channel_mask: np.ndarray


def load_subject_data(
        subject_name: str,
        eeg_files: Dict[int, str | Path],
        features_dir: Path,
        fs_target: float,
        sections: Tuple[int, ...],
        feature_sets: List[str],
) -> SubjectWordLockedData:
    """Load EEG and features for a single subject."""
    print(f"\n{'=' * 60}")
    print(f"Loading data for: {subject_name}")
    print(f"{'=' * 60}")

    eeg_sections_raw: List[np.ndarray] = []
    word_timing_sections: List[pd.DataFrame] = []
    X_sections: Dict[str, List[np.ndarray]] = {f: [] for f in feature_sets}
    all_channel_labels: List[str] = []

    for i, sid in enumerate(sections):
        print(f"\n[Section {sid}]")

        mat_path = eeg_files.get(sid)
        if mat_path is None or not Path(mat_path).exists():
            raise FileNotFoundError(f"EEG file not found for section {sid}: {mat_path}")

        eeg, info, ch_labels = load_eeg_from_mat_hdf5(mat_path)
        fs_orig = float(info["sfreq"])

        if i == 0:
            all_channel_labels = ch_labels

        if abs(fs_orig - fs_target) > 1e-9:
            eeg = resample_eeg(eeg, fs_orig, fs_target)

        word_timing, X_word, _ = load_section_features(
            features_dir, sid, feature_sets, fs_target
        )

        eeg_sections_raw.append(eeg)
        word_timing_sections.append(word_timing)

        for feat in feature_sets:
            if feat in X_word:
                X_sections[feat].append(X_word[feat])

    valid_features = [f for f in feature_sets if len(X_sections.get(f, [])) == len(sections)]
    if not valid_features:
        raise ValueError("No valid feature sets found across all sections")

    print(f"\nValid feature sets: {valid_features}")

    all_eeg = np.hstack(eeg_sections_raw)
    C_orig = all_eeg.shape[0]
    ch_std = np.nanstd(all_eeg, axis=1)
    good_ch = np.isfinite(ch_std) & (ch_std > 1e-10)
    n_good = int(np.sum(good_ch))
    print(f"\nChannel cleanup: {n_good}/{C_orig} channels kept")

    good_channel_labels = []
    good_idx = np.where(good_ch)[0]
    for idx in good_idx:
        if idx < len(all_channel_labels):
            good_channel_labels.append(all_channel_labels[idx])
        else:
            good_channel_labels.append(f"Ch{idx}")

    eeg_sections: List[np.ndarray] = []
    for eeg in eeg_sections_raw:
        eeg_g = eeg[good_ch, :]
        eeg_z = zscore_channels_within_section(eeg_g)
        eeg_sections.append(eeg_z)

    X_by_feature: Dict[str, np.ndarray] = {}
    for feat in valid_features:
        X = np.vstack(X_sections[feat]).astype(np.float64)
        X_by_feature[feat] = X
        print(f"Feature {feat}: {X.shape}")

    section_word_slices: List[Tuple[int, int]] = []
    start = 0
    for wt in word_timing_sections:
        end = start + len(wt)
        section_word_slices.append((start, end))
        start = end

    sections_out: List[SectionWordEEG] = []
    for sid, eeg, wt in zip(sections, eeg_sections, word_timing_sections):
        C, T = eeg.shape
        eeg_t = eeg.T.astype(np.float64)
        cs = np.vstack([np.zeros((1, C), dtype=np.float64), np.cumsum(eeg_t, axis=0)])
        onsets = wt["onset_sample"].to_numpy(dtype=int)

        sections_out.append(
            SectionWordEEG(
                section_id=int(sid),
                fs=float(fs_target),
                eeg_cs=cs,
                onsets=onsets,
                T=int(T),
                W=int(len(wt)),
            )
        )

    return SubjectWordLockedData(
        subject=subject_name,
        fs=float(fs_target),
        sections_used=tuple(sections),
        section_word_slices=section_word_slices,
        sections=sections_out,
        X_by_feature=X_by_feature,
        n_channels=int(n_good),
        channel_labels=good_channel_labels,
        good_channel_mask=good_ch,
    )


# -----------------------------
# Channel Selection
# -----------------------------

def select_channels(
        data: SubjectWordLockedData,
        selection: Optional[Union[List[str], List[int], Dict[str, any]]] = None,
) -> Tuple[SubjectWordLockedData, np.ndarray]:
    """Filter data to include only selected channels."""
    if selection is None:
        return data, np.arange(data.n_channels)

    n_channels = data.n_channels
    channel_labels = data.channel_labels

    if isinstance(selection, dict):
        if "pattern" in selection:
            pattern = re.compile(selection["pattern"], re.IGNORECASE)
            selected_idx = np.array([
                i for i, name in enumerate(channel_labels)
                if pattern.search(name)
            ], dtype=int)
            print(f"  Pattern '{selection['pattern']}' matched {len(selected_idx)} channels")
        else:
            start = selection.get("start", 0)
            end = selection.get("end", n_channels)
            selected_idx = np.arange(max(0, start), min(end, n_channels))
            print(f"  Range [{start}, {end}) selected {len(selected_idx)} channels")
    elif isinstance(selection, (list, tuple)) and len(selection) > 0:
        if isinstance(selection[0], str):
            name_to_idx = {name.lower().strip(): i for i, name in enumerate(channel_labels)}
            selected_idx = []
            not_found = []
            for name in selection:
                key = name.lower().strip()
                if key in name_to_idx:
                    selected_idx.append(name_to_idx[key])
                else:
                    not_found.append(name)
            if not_found:
                print(f"  [WARN] Channels not found: {not_found[:10]}{'...' if len(not_found) > 10 else ''}")
            selected_idx = np.array(selected_idx, dtype=int)
            print(f"  Name selection: {len(selected_idx)} channels found")
        else:
            selected_idx = np.array([
                int(i) for i in selection
                if 0 <= int(i) < n_channels
            ], dtype=int)
            invalid = [i for i in selection if not (0 <= int(i) < n_channels)]
            if invalid:
                print(f"  [WARN] Invalid indices ignored: {invalid[:10]}{'...' if len(invalid) > 10 else ''}")
            print(f"  Index selection: {len(selected_idx)} channels")
    else:
        raise ValueError(f"Invalid selection: {selection}")

    if len(selected_idx) == 0:
        raise ValueError("No valid channels in selection. Check channel names/indices.")

    _, unique_indices = np.unique(selected_idx, return_index=True)
    selected_idx = selected_idx[np.sort(unique_indices)]

    print(f"\n{'=' * 50}")
    print(f"CHANNEL SELECTION")
    print(f"{'=' * 50}")
    print(f"  Selected: {len(selected_idx)}/{n_channels} channels")

    selected_names = [channel_labels[i] for i in selected_idx]
    if len(selected_names) <= 15:
        print(f"  Channels: {selected_names}")
    else:
        print(f"  First 5:  {selected_names[:5]}")
        print(f"  Last 5:   {selected_names[-5:]}")

    new_sections = []
    for sec in data.sections:
        new_cs = sec.eeg_cs[:, selected_idx].copy()
        new_sections.append(SectionWordEEG(
            section_id=sec.section_id,
            fs=sec.fs,
            eeg_cs=new_cs,
            onsets=sec.onsets,
            T=sec.T,
            W=sec.W,
        ))

    original_indices = selected_idx.copy()

    new_data = SubjectWordLockedData(
        subject=data.subject,
        fs=data.fs,
        sections_used=data.sections_used,
        section_word_slices=data.section_word_slices,
        sections=new_sections,
        X_by_feature=data.X_by_feature,
        n_channels=len(selected_idx),
        channel_labels=selected_names,
        good_channel_mask=data.good_channel_mask,
    )

    return new_data, original_indices


# -----------------------------
# Encoding Analysis Utilities
# -----------------------------

def iter_lags_ms(fs: float, tmin_ms: float, tmax_ms: float, step_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lag values in ms and samples."""
    lags_ms = np.arange(float(tmin_ms), float(tmax_ms) + 1e-9, float(step_ms))
    lags_samp = np.rint(lags_ms / 1000.0 * float(fs)).astype(int)
    uniq_samp, idx = np.unique(lags_samp, return_index=True)
    return lags_ms[idx], uniq_samp


def make_contiguous_folds(T: int, k: int) -> List[Tuple[int, int]]:
    """Create k contiguous folds."""
    if k < 2:
        raise ValueError("k must be >= 2")
    fold_size = int(np.ceil(T / k))
    folds = []
    for i in range(k):
        a = i * fold_size
        b = min((i + 1) * fold_size, T)
        if a < b:
            folds.append((a, b))
    return folds


def allocate_folds_by_length(lengths: List[int], nfold_total: int, min_per_section: int = 1) -> List[int]:
    """Allocate folds to sections proportionally."""
    L = np.asarray(lengths, float)
    nsec = len(L)

    if nfold_total < nsec * min_per_section:
        raise ValueError("nfold_total too small for min_per_section constraint")

    weights = L / np.sum(L) if np.sum(L) > 0 else np.ones(nsec) / nsec
    k = np.full(nsec, min_per_section, dtype=int)
    rem = nfold_total - int(k.sum())

    raw = weights * rem
    flo = np.floor(raw).astype(int)
    k += flo

    rem2 = rem - int(flo.sum())
    if rem2 > 0:
        frac = raw - flo
        order = np.argsort(-frac)
        for i in range(rem2):
            k[order[i % nsec]] += 1

    return k.tolist()


def make_folds_within_sections(
        section_slices: List[Tuple[int, int]],
        nfold_total: int,
        min_per_section: int = 1,
) -> List[Tuple[int, int]]:
    """Create folds that don't span section boundaries."""
    lengths = [b - a for a, b in section_slices]
    k_per = allocate_folds_by_length(lengths, nfold_total, min_per_section)

    folds: List[Tuple[int, int]] = []
    for (s0, s1), k in zip(section_slices, k_per):
        L = s1 - s0
        if k <= 1:
            folds.append((s0, s1))
        else:
            local = make_contiguous_folds(L, k)
            folds.extend([(s0 + a, s0 + b) for a, b in local])

    return sorted(folds)


def make_grouped_folds(
        group_ids: np.ndarray,
        nfold: int,
        *,
        guard_groups: int = 0,
        random_state: int = 42,
) -> List[np.ndarray]:
    """
    Build outer CV folds that keep entire groups (sentences / boundary events)
    on one side of the split.

    Parameters
    ----------
    group_ids : array of shape (n_samples,)
        Integer group label per sample (e.g. sent_id or event_id). Samples with
        negative group ids are ignored for fold membership and left unassigned.
    nfold : int
        Target number of folds (>=2).
    guard_groups : int
        Optional number of temporally adjacent groups (by sorted group id) to
        drop from *both* train and test around each test block, acting as a
        temporal guard band.
    random_state : int
        RNG seed for stable group assignment when group count is not divisible
        by nfold.

    Returns
    -------
    list of ndarray
        For each fold, the integer sample indices assigned to the test set
        (guard-band samples are excluded from all folds' test sets and should
        also be excluded from train by the caller via ``~test & ~guard`` or
        by using :func:`grouped_fold_masks`).
    """
    group_ids = np.asarray(group_ids)
    if nfold < 2:
        raise ValueError("nfold must be >= 2")
    valid = group_ids >= 0
    uniq = np.array(sorted(set(group_ids[valid].tolist())))
    if uniq.size < nfold:
        nfold = max(2, int(uniq.size))
    # Contiguous blocks in group-id order (temporal) so guard bands are meaningful.
    # ``random_state`` reserved for API compatibility with callers.
    _ = random_state
    blocks = np.array_split(uniq, nfold)
    test_folds: List[np.ndarray] = []
    for block in blocks:
        test_groups = set(int(g) for g in block)
        te = np.flatnonzero(np.isin(group_ids, list(test_groups)))
        test_folds.append(te)
    return test_folds


def grouped_fold_masks(
        group_ids: np.ndarray,
        nfold: int,
        *,
        guard_groups: int = 0,
        random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return ``(train_idx, test_idx)`` pairs for grouped folds with optional
    temporal guard bands excluded from both train and test.
    """
    group_ids = np.asarray(group_ids)
    valid = group_ids >= 0
    uniq = np.array(sorted(set(group_ids[valid].tolist())))
    nfold_eff = min(nfold, max(2, int(uniq.size)))
    blocks = np.array_split(uniq, nfold_eff)
    pos = {int(g): i for i, g in enumerate(uniq)}
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        test_groups = set(int(g) for g in block)
        guard = set()
        if guard_groups > 0:
            i0, i1 = pos[int(block[0])], pos[int(block[-1])]
            for j in range(max(0, i0 - guard_groups), min(len(uniq), i1 + 1 + guard_groups)):
                gj = int(uniq[j])
                if gj not in test_groups:
                    guard.add(gj)
        te = np.flatnonzero(np.array([int(g) in test_groups for g in group_ids]))
        tr = np.flatnonzero(np.array([
            (int(g) >= 0) and (int(g) not in test_groups) and (int(g) not in guard)
            for g in group_ids
        ]))
        out.append((tr, te))
    return out


def compute_Y_for_lag(
        data: SubjectWordLockedData,
        lag_samp: int,
        half_win_samp: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Y (response) matrix for a given lag."""
    Wtot = data.section_word_slices[-1][1]
    C = data.n_channels
    Y = np.full((Wtot, C), np.nan, dtype=np.float64)
    valid = np.zeros(Wtot, dtype=bool)

    win_len = 2 * int(half_win_samp)
    if win_len <= 0:
        return Y, valid

    for (w0, _w1), sec in zip(data.section_word_slices, data.sections):
        T = sec.T
        on = sec.onsets
        center = on + int(lag_samp)
        start = center - int(half_win_samp)
        end = center + int(half_win_samp)

        ok = (start >= 0) & (end <= T)
        idx = np.flatnonzero(ok)

        if idx.size == 0:
            continue

        sums = sec.eeg_cs[end[idx], :] - sec.eeg_cs[start[idx], :]
        Y[w0 + idx, :] = sums / float(win_len)
        valid[w0 + idx] = True

    return Y, valid


# -----------------------------
# OLS + fold-wise Standardize (full-dimensional embedding, no PCA/ridge)
# -----------------------------

def cv_encoding_ols_with_phase_perm(
        X: np.ndarray,
        data: SubjectWordLockedData,
        folds: List[Tuple[int, int]],
        lags_samp: np.ndarray,
        lags_ms: np.ndarray,
        half_win_samp: int,
        standardize_x: bool = True,
        perm_n: int = 0,
        perm_kind: str = "phase",
        perm_batch: int = 50,
        perm_seed: int = 0,
        perm_alpha: float = 0.05,
        perm_use_fdr: bool = True,
        eps: float = 1e-12,
) -> Dict[str, object]:
    """Cross-validated encoding with plain OLS on full-dimensional embeddings
    (Goldstein et al. methodology: StandardScaler, no PCA/ridge), fold-wise
    preprocessing, optional permutation test."""
    _require_sklearn()

    X = np.asarray(X, dtype=np.float64)
    Wtot, P = X.shape
    C = data.n_channels
    F = len(folds)
    nlags = len(lags_samp)

    fold_indices = [np.arange(a, b) for a, b in folds]

    train_indices: List[np.ndarray] = []
    for o in range(F):
        parts = [fold_indices[j] for j in range(F) if j != o]
        train_indices.append(np.concatenate(parts) if parts else np.array([], dtype=int))

    X_trans_all_per_fold: List[np.ndarray] = []

    for o in range(F):
        tr = train_indices[o]
        if tr.size == 0:
            raise ValueError("Empty training set for a fold; check NFOLD vs total words.")

        X_train = X[tr, :]

        if not np.all(np.isfinite(X_train)):
            raise ValueError("Non-finite values in X_train; StandardScaler cannot handle NaNs/Infs.")

        if standardize_x:
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(X_train)
            X_all_s = scaler.transform(X)
        else:
            X_all_s = X

        X_trans_all_per_fold.append(np.asarray(X_all_s, dtype=np.float32))

    r_folds = np.full((F, nlags, C), np.nan, dtype=np.float32)
    err_folds = np.full((F, nlags, C), np.nan, dtype=np.float32)
    r_cv = np.full((nlags, C), np.nan, dtype=np.float32)
    err_cv = np.full((nlags, C), np.nan, dtype=np.float32)

    do_perm = int(perm_n) > 0
    rng = np.random.default_rng(int(perm_seed))

    if do_perm:
        if perm_kind != "phase":
            raise ValueError("Only perm_kind='phase' is implemented.")
        obs_max = np.full((C,), -np.inf, dtype=float)
        perm_max = np.full((int(perm_n), C), -np.inf, dtype=float)
        print(f"  Permutation test (phase-rand): n={perm_n}, alpha={perm_alpha}, FDR={perm_use_fdr}")

    for li, lag in enumerate(lags_samp.tolist()):
        Y_all, valid = compute_Y_for_lag(data, int(lag), int(half_win_samp))

        Y_pred_cv = np.full((Wtot, C), np.nan, dtype=np.float32)
        Y_true_cv = np.full((Wtot, C), np.nan, dtype=np.float32)

        for o in range(F):
            te = fold_indices[o]
            tr = train_indices[o]

            te_v = te[valid[te]]
            tr_v = tr[valid[tr]]

            if te_v.size <= 2 or tr_v.size <= 2:
                continue

            X_all_t = X_trans_all_per_fold[o]
            X_train = X_all_t[tr_v, :]
            X_test = X_all_t[te_v, :]

            Y_train = Y_all[tr_v, :]
            Y_test = Y_all[te_v, :]

            if not (np.all(np.isfinite(Y_train)) and np.all(np.isfinite(Y_test))):
                Y_train = np.where(np.isfinite(Y_train), Y_train, 0.0)
                Y_test = np.where(np.isfinite(Y_test), Y_test, 0.0)

            y_mean = np.mean(Y_train, axis=0, keepdims=True)
            Y_train_c = (Y_train - y_mean).astype(np.float64)
            Y_test_c = (Y_test - y_mean).astype(np.float64)

            W_hat, *_ = np.linalg.lstsq(X_train.astype(np.float64), Y_train_c, rcond=None)
            Y_pred = (X_test.astype(np.float64) @ W_hat).astype(np.float64)

            Y_pred_cv[te_v, :] = Y_pred.astype(np.float32)
            Y_true_cv[te_v, :] = Y_test_c.astype(np.float32)

            rch = corr_np(Y_test_c, Y_pred, eps=eps)
            mse = np.mean((Y_test_c - Y_pred) ** 2, axis=0)

            r_folds[o, li, :] = rch.astype(np.float32)
            err_folds[o, li, :] = mse.astype(np.float32)

        mask = np.all(np.isfinite(Y_pred_cv), axis=1) & np.all(np.isfinite(Y_true_cv), axis=1)
        if int(np.sum(mask)) > 3:
            yt = Y_true_cv[mask, :].astype(np.float64)
            yp = Y_pred_cv[mask, :].astype(np.float64)
            r_cv[li, :] = corr_np(yt, yp, eps=eps).astype(np.float32)
            err_cv[li, :] = np.mean((yt - yp) ** 2, axis=0).astype(np.float32)

            if do_perm:
                obs_max = np.maximum(obs_max, r_cv[li, :].astype(float))
                r_perm = phase_randomized_correlations_fft(
                    y_true=yt,
                    y_pred=yp,
                    perm_n=int(perm_n),
                    rng=rng,
                    batch=int(perm_batch),
                    eps=float(eps),
                )
                perm_max = np.maximum(perm_max, r_perm)

    stats: Dict[str, object] = {
        "r_folds": r_folds,
        "err_folds": err_folds,
        "r_cv": r_cv,
        "err_cv": err_cv,
        "t_ms": lags_ms,
        "lags_samp": lags_samp,
        "standardize_x": bool(standardize_x),
        "ols": True,
    }

    if do_perm:
        obs_max[~np.isfinite(obs_max)] = np.nan
        perm_max[~np.isfinite(perm_max)] = np.nan

        pvals = np.full((C,), np.nan, dtype=float)
        for ch in range(C):
            if not np.isfinite(obs_max[ch]):
                continue
            null = perm_max[:, ch]
            null = null[np.isfinite(null)]
            if null.size == 0:
                continue
            pvals[ch] = (np.sum(null >= obs_max[ch]) + 1.0) / (null.size + 1.0)

        if perm_use_fdr:
            reject, p_adj = bh_fdr(pvals, alpha=float(perm_alpha))
        else:
            reject = np.isfinite(pvals) & (pvals < float(perm_alpha))
            p_adj = pvals.copy()

        n_sig = int(np.sum(reject))
        print(f"  Permutation result: {n_sig}/{C} channels significant (alpha={perm_alpha}, FDR={perm_use_fdr})")

        stats.update({
            "perm_kind": str(perm_kind),
            "perm_n": int(perm_n),
            "perm_alpha": float(perm_alpha),
            "perm_use_fdr": bool(perm_use_fdr),
            "perm_obs_stat_ch": obs_max,
            "perm_pvals_ch": pvals,
            "perm_p_adj_ch": p_adj,
            "perm_reject_ch": reject,
            "n_sig_channels": n_sig,
        })

    return stats


# -----------------------------
# Significant Channel Reporting
# -----------------------------

def print_significant_channels(
        channel_labels: List[str],
        reject: np.ndarray,
        pvals: np.ndarray,
        p_adj: np.ndarray,
        obs_stat: np.ndarray,
        feature_name: str,
        top_n: int = 20,
):
    """Print detailed information about significant channels."""
    sig_idx = np.where(reject)[0]
    n_sig = len(sig_idx)
    n_total = len(channel_labels)

    print(f"\n{'=' * 60}")
    print(f"SIGNIFICANT CHANNELS for {feature_name}")
    print(f"{'=' * 60}")
    print(f"Total: {n_sig}/{n_total} channels significant")

    if n_sig == 0:
        print("  No significant channels found.")
        return

    sig_stats = obs_stat[sig_idx]
    sort_order = np.argsort(-sig_stats)
    sig_idx_sorted = sig_idx[sort_order]

    print(f"\n{'Rank':<6}{'Channel':<25}{'r_max':<12}{'p-value':<12}{'p_adj':<12}")
    print("-" * 67)

    n_display = min(top_n, n_sig)
    for rank, ch_idx in enumerate(sig_idx_sorted[:n_display], 1):
        ch_name = channel_labels[ch_idx] if ch_idx < len(channel_labels) else f"Ch{ch_idx}"
        r_val = float(obs_stat[ch_idx])
        p_raw = float(pvals[ch_idx]) if np.isfinite(pvals[ch_idx]) else np.nan
        p_corrected = float(p_adj[ch_idx]) if np.isfinite(p_adj[ch_idx]) else np.nan

        print(f"{rank:<6}{ch_name:<25}{r_val:<12.4f}{p_raw:<12.4f}{p_corrected:<12.4f}")

    if n_sig > top_n:
        print(f"  ... and {n_sig - top_n} more significant channels")


def create_channel_results_dataframe(
        channel_labels: List[str],
        reject: np.ndarray,
        pvals: np.ndarray,
        p_adj: np.ndarray,
        obs_stat: np.ndarray,
        mean_r_per_ch: Optional[np.ndarray] = None,
        original_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Create a DataFrame with per-channel results."""
    n_ch = len(channel_labels)

    def _pad(arr, fill):
        arr = np.asarray(arr)
        if arr.shape[0] >= n_ch:
            return arr[:n_ch]
        out = np.full((n_ch,), fill, dtype=arr.dtype if arr.dtype != object else object)
        out[: arr.shape[0]] = arr
        return out

    df = pd.DataFrame({
        "channel_idx": np.arange(n_ch),
        "channel": channel_labels[:n_ch],
        "significant": _pad(reject, False).astype(bool),
        "r_max": _pad(obs_stat, np.nan).astype(float),
        "p_value": _pad(pvals, np.nan).astype(float),
        "p_adj": _pad(p_adj, np.nan).astype(float),
    })

    if mean_r_per_ch is not None:
        df["mean_r"] = _pad(mean_r_per_ch, np.nan).astype(float)

    if original_indices is not None:
        df["original_idx"] = _pad(original_indices, -1).astype(int)

    df = df.sort_values("r_max", ascending=False).reset_index(drop=True)
    return df


# -----------------------------
# Main Analysis
# -----------------------------

def run_single_subject_analysis(
        subject_name: str,
        eeg_files: Dict[int, str | Path],
        features_dir: Path,
        feature_sets: List[str],
        sections: Tuple[int, ...],
        fs_target: float,
        lags_ms: np.ndarray,
        lags_samp: np.ndarray,
        half_win_samp: int,
        nfold: int,
        standardize_x: bool,
        perm_n: int = 0,
        perm_alpha: float = 0.05,
        perm_use_fdr: bool = True,
        perm_seed: int = 0,
        selected_channels: Optional[Union[List[str], List[int], Dict]] = None,
) -> Tuple[Dict[str, Dict], SubjectWordLockedData, Optional[np.ndarray]]:
    """Run encoding analysis for a single subject."""

    data = load_subject_data(
        subject_name=subject_name,
        eeg_files=eeg_files,
        features_dir=features_dir,
        fs_target=fs_target,
        sections=sections,
        feature_sets=feature_sets,
    )

    original_ch_indices = None
    if selected_channels is not None:
        data, original_ch_indices = select_channels(data, selected_channels)

    if FOLDS_WITHIN_SECTIONS:
        folds = make_folds_within_sections(
            data.section_word_slices, nfold, MIN_FOLDS_PER_SECTION
        )
    else:
        Wtot = data.section_word_slices[-1][1]
        folds = make_contiguous_folds(Wtot, nfold)

    print(f"\nFolds: {len(folds)} (within sections: {FOLDS_WITHIN_SECTIONS})")

    results: Dict[str, Dict] = {}

    for feat in data.X_by_feature.keys():
        print(f"\n{'─' * 50}")
        print(f"Feature: {feat}")
        print(f"{'─' * 50}")

        X = data.X_by_feature[feat]

        stats = cv_encoding_ols_with_phase_perm(
            X=X,
            data=data,
            folds=folds,
            lags_samp=lags_samp,
            lags_ms=lags_ms,
            half_win_samp=half_win_samp,
            standardize_x=standardize_x,
            perm_n=perm_n,
            perm_kind=PERM_KIND,
            perm_batch=PERM_BATCH,
            perm_seed=perm_seed + (hash(feat) % 1000),
            perm_alpha=perm_alpha,
            perm_use_fdr=perm_use_fdr,
            eps=PERM_EPS,
        )

        # Use r_cv (concatenated CV score) for lag curves
        r_cv = np.asarray(stats["r_cv"], dtype=float)  # (nlags, C)

        # AVERAGE ACROSS ALL CHANNELS (regardless of significance)
        mean_r_all_channels = np.nanmean(r_cv, axis=1)  # (nlags,)
        sem_r_all_channels = nansem(r_cv, axis=1)  # (nlags,)
        mean_r_per_ch = np.nanmean(r_cv, axis=0)  # (C,)

        # Initialize significant channel stats
        mean_r_sig_channels = np.full_like(mean_r_all_channels, np.nan)
        sem_r_sig_channels = np.full_like(sem_r_all_channels, np.nan)
        peak_r_sig = np.nan
        peak_lag_sig = float(lags_ms[0])
        n_sig = 0
        sig_ch = None
        sig_ch_indices = np.array([], dtype=int)

        # Peak stats (using all-channel average)
        valid_mask = np.isfinite(mean_r_all_channels)
        if np.any(valid_mask):
            peak_idx = int(np.nanargmax(mean_r_all_channels))
            peak_r = float(mean_r_all_channels[peak_idx])
            peak_lag = float(lags_ms[peak_idx])
        else:
            peak_idx = 0
            peak_r = np.nan
            peak_lag = float(lags_ms[0])

        print(f"  Peak r (all channels) = {peak_r:.4f} at lag = {peak_lag:.1f} ms (n={data.n_channels} ch)")

        # Significant channel processing
        ch_df = None
        if perm_n > 0 and "perm_reject_ch" in stats:
            sig_ch = np.asarray(stats["perm_reject_ch"], dtype=bool)
            n_sig = int(np.sum(sig_ch))
            sig_ch_indices = np.where(sig_ch)[0]

            print_significant_channels(
                channel_labels=data.channel_labels,
                reject=sig_ch,
                pvals=stats["perm_pvals_ch"],
                p_adj=stats["perm_p_adj_ch"],
                obs_stat=stats["perm_obs_stat_ch"],
                feature_name=feat,
                top_n=20,
            )

            ch_df = create_channel_results_dataframe(
                channel_labels=data.channel_labels,
                reject=sig_ch,
                pvals=stats["perm_pvals_ch"],
                p_adj=stats["perm_p_adj_ch"],
                obs_stat=stats["perm_obs_stat_ch"],
                mean_r_per_ch=mean_r_per_ch,
                original_indices=original_ch_indices,
            )

            # Compute significant-channel-only stats
            if n_sig > 0:
                r_cv_sig = r_cv[:, sig_ch]  # (nlags, n_sig)
                mean_r_sig_channels = np.nanmean(r_cv_sig, axis=1)  # (nlags,)
                sem_r_sig_channels = nansem(r_cv_sig, axis=1)  # (nlags,)

                # Peak for significant channels
                valid_sig_mask = np.isfinite(mean_r_sig_channels)
                if np.any(valid_sig_mask):
                    peak_idx_sig = int(np.nanargmax(mean_r_sig_channels))
                    peak_r_sig = float(mean_r_sig_channels[peak_idx_sig])
                    peak_lag_sig = float(lags_ms[peak_idx_sig])

                print(f"  Peak r (sig. channels only) = {peak_r_sig:.4f} at lag = {peak_lag_sig:.1f} ms (n={n_sig} ch)")

        results[feat] = {
            "stats": stats,
            # All channels
            "mean_r_lags": mean_r_all_channels,
            "sem_r_lags": sem_r_all_channels,
            "mean_r_per_channel": mean_r_per_ch,
            "r_cv_all": r_cv,  # Full (nlags, C) array
            "t_ms": lags_ms,
            "peak_r": peak_r,
            "peak_lag_ms": peak_lag,
            "n_channels": data.n_channels,
            # Significant channels only
            "mean_r_sig_lags": mean_r_sig_channels,
            "sem_r_sig_lags": sem_r_sig_channels,
            "peak_r_sig": peak_r_sig,
            "peak_lag_sig_ms": peak_lag_sig,
            "n_sig_channels": n_sig,
            "sig_channels": sig_ch,
            "sig_channel_indices": sig_ch_indices,
            "sig_channel_names": [data.channel_labels[i] for i in sig_ch_indices] if len(sig_ch_indices) > 0 else [],
            "channel_results_df": ch_df,
        }

    return results, data, original_ch_indices


def plot_individual_channel(
        r_cv: np.ndarray,
        lags_ms: np.ndarray,
        channel_name: str,
        channel_idx: int,
        feature_name: str,
        outdir: Path,
        subject: str,
        p_value: Optional[float] = None,
        p_adj: Optional[float] = None,
        color: str = 'blue',
):
    """Plot encoding results for a single channel."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get correlation values for this channel
    r_ch = r_cv[:, channel_idx]

    # Plot the correlation curve
    ax.plot(lags_ms, r_ch, color=color, lw=2.5, label=f'{channel_name}')

    # Add reference lines
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5, label='Word onset')
    ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)

    # Find and mark peak
    valid_mask = np.isfinite(r_ch)
    if np.any(valid_mask):
        peak_idx = int(np.nanargmax(r_ch))
        peak_r = float(r_ch[peak_idx])
        peak_lag = float(lags_ms[peak_idx])

        ax.scatter([peak_lag], [peak_r], color='red', s=100, zorder=5,
                   edgecolor='k', marker='*', label=f'Peak: r={peak_r:.4f} @ {peak_lag:.0f}ms')
    else:
        peak_r = np.nan
        peak_lag = np.nan

    # Title with statistics
    title = f"{feature_name}: {channel_name}"
    if p_value is not None and np.isfinite(p_value):
        title += f"\n(p={p_value:.4f}"
        if p_adj is not None and np.isfinite(p_adj):
            title += f", p_adj={p_adj:.4f}"
        title += ")"

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lag (ms)", fontsize=10)
    ax.set_ylabel("Correlation (r)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Set y-axis limits with some padding
    y_min = np.nanmin(r_ch) if np.any(np.isfinite(r_ch)) else -0.1
    y_max = np.nanmax(r_ch) if np.any(np.isfinite(r_ch)) else 0.1
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    # Create safe filename
    safe_channel_name = re.sub(r'[^\w\-]', '_', channel_name)
    safe_feat = re.sub(r'[^\w\-]', '_', feature_name)

    fig.savefig(outdir / f"{subject}_{safe_feat}_ch_{safe_channel_name}.png", dpi=150)
    plt.close(fig)

    return peak_r, peak_lag


def plot_all_significant_channels_grid(
        r_cv: np.ndarray,
        lags_ms: np.ndarray,
        sig_channel_indices: np.ndarray,
        sig_channel_names: List[str],
        feature_name: str,
        outdir: Path,
        subject: str,
        pvals: Optional[np.ndarray] = None,
        p_adj: Optional[np.ndarray] = None,
        max_cols: int = 4,
):
    """Plot all significant channels in a grid layout."""
    n_sig = len(sig_channel_indices)
    if n_sig == 0:
        return

    # Calculate grid dimensions
    n_cols = min(max_cols, n_sig)
    n_rows = int(np.ceil(n_sig / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    colors = plt.cm.tab20(np.linspace(0, 1, n_sig))

    for i, (ch_idx, ch_name) in enumerate(zip(sig_channel_indices, sig_channel_names)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Get correlation values for this channel
        r_ch = r_cv[:, ch_idx]

        # Plot
        ax.plot(lags_ms, r_ch, color=colors[i], lw=2)
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)

        # Find peak
        valid_mask = np.isfinite(r_ch)
        if np.any(valid_mask):
            peak_idx = int(np.nanargmax(r_ch))
            peak_r = float(r_ch[peak_idx])
            peak_lag = float(lags_ms[peak_idx])
            ax.scatter([peak_lag], [peak_r], color='red', s=60, zorder=5, edgecolor='k', marker='*')
        else:
            peak_r = np.nan
            peak_lag = np.nan

        # Title
        title = f"{ch_name}"
        if pvals is not None and ch_idx < len(pvals) and np.isfinite(pvals[ch_idx]):
            p_val = pvals[ch_idx]
            title += f"\n(r={peak_r:.3f}, p={p_val:.3f})"
        else:
            title += f"\n(r={peak_r:.3f})"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Lag (ms)", fontsize=8)
        ax.set_ylabel("r", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_sig, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"{feature_name}: All Significant Channels (n={n_sig})\n{subject}", fontsize=12)
    plt.tight_layout()

    safe_feat = re.sub(r'[^\w\-]', '_', feature_name)
    fig.savefig(outdir / f"{subject}_{safe_feat}_all_sig_channels_grid.png", dpi=150)
    fig.savefig(outdir / f"{subject}_{safe_feat}_all_sig_channels_grid.pdf")
    plt.close(fig)


def plot_results(
        results: Dict[str, Dict],
        lags_ms: np.ndarray,
        outdir: Path,
        subject: str,
        channel_labels: List[str],
        channel_selection_info: Optional[str] = None,
        plot_significant_only: bool = True,
        min_significant_channels: int = 1,
        plot_individual_channels: bool = True,
        max_individual_plots: int = 50,
):
    """Plot encoding results - significant channels only or all channels."""
    n_features = len(results)

    # Check if we have any significant channels to plot
    has_sig_channels = {}
    for feat, res in results.items():
        n_sig = res.get("n_sig_channels", 0)
        has_sig_channels[feat] = n_sig >= min_significant_channels

    if plot_significant_only and not any(has_sig_channels.values()):
        print("\n[WARNING] No feature sets have significant channels. Falling back to all-channel plot.")
        plot_significant_only = False

    # Determine which features to plot
    features_to_plot = []
    for feat, res in results.items():
        if plot_significant_only:
            if has_sig_channels[feat]:
                features_to_plot.append(feat)
            else:
                print(f"  [SKIP] {feat}: No significant channels (n_sig={res.get('n_sig_channels', 0)})")
        else:
            features_to_plot.append(feat)

    if len(features_to_plot) == 0:
        print("[ERROR] No features to plot!")
        return

    # Create main figure (summary)
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 4), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(features_to_plot)))

    for i, feat in enumerate(features_to_plot):
        res = results[feat]
        ax = axes[0, i]

        if plot_significant_only:
            mean_r = res["mean_r_sig_lags"]
            sem_r = res["sem_r_sig_lags"]
            n_ch = res["n_sig_channels"]
            peak_r = res["peak_r_sig"]
            peak_lag = res["peak_lag_sig_ms"]
            label_suffix = "sig. ch"
        else:
            mean_r = res["mean_r_lags"]
            sem_r = res["sem_r_lags"]
            n_ch = res["n_channels"]
            peak_r = res["peak_r"]
            peak_lag = res["peak_lag_ms"]
            label_suffix = "all ch"

        # Plot mean ± SEM
        ax.plot(lags_ms, mean_r, color=colors[i], lw=2.5, label=f"Mean (n={n_ch} {label_suffix})")
        ax.fill_between(
            lags_ms,
            mean_r - sem_r,
            mean_r + sem_r,
            color=colors[i],
            alpha=0.3,
            label="±SEM"
        )

        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)

        # Mark peak
        if np.isfinite(peak_r):
            peak_idx = int(np.nanargmax(mean_r))
            ax.scatter(
                [lags_ms[peak_idx]],
                [mean_r[peak_idx]],
                color=colors[i],
                s=100,
                zorder=5,
                edgecolor="k",
            )

        title = f"{feat}\n(r={peak_r:.4f} @ {peak_lag:.0f}ms)"
        title += f"\n[{n_ch} {label_suffix}]"
        ax.set_title(title)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Correlation (r)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plot_type = "Significant Channels Only" if plot_significant_only else "All Channels"
    title = f"Word-Locked Encoding (OLS): {subject}\n[{plot_type}]"
    if channel_selection_info:
        title += f"\n[{channel_selection_info}]"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    suffix = "_sig_only" if plot_significant_only else "_all_ch"
    fig.savefig(outdir / f"{subject}_encoding_results{suffix}.png", dpi=150)
    fig.savefig(outdir / f"{subject}_encoding_results{suffix}.pdf")
    plt.close(fig)

    # ============================================================
    # INDIVIDUAL CHANNEL PLOTS - NEW SECTION
    # ============================================================
    if plot_individual_channels:
        print(f"\n{'=' * 60}")
        print("GENERATING INDIVIDUAL CHANNEL PLOTS")
        print(f"{'=' * 60}")

        # Create subdirectory for individual channel plots
        individual_dir = outdir / "individual_channels"
        individual_dir.mkdir(parents=True, exist_ok=True)

        for feat in features_to_plot:
            res = results[feat]
            r_cv = res.get("r_cv_all")

            if r_cv is None:
                print(f"  [SKIP] {feat}: No r_cv data available")
                continue

            # Get significant channel info
            sig_ch_indices = res.get("sig_channel_indices", np.array([], dtype=int))
            sig_ch_names = res.get("sig_channel_names", [])
            n_sig = len(sig_ch_indices)

            # Get p-values if available
            stats = res.get("stats", {})
            pvals = stats.get("perm_pvals_ch", None)
            p_adj = stats.get("perm_p_adj_ch", None)

            if n_sig == 0:
                print(f"  [SKIP] {feat}: No significant channels to plot individually")
                continue

            print(f"\n  {feat}: Plotting {min(n_sig, max_individual_plots)} individual channels...")

            # Create feature-specific subdirectory
            feat_dir = individual_dir / feat
            feat_dir.mkdir(parents=True, exist_ok=True)

            # Get colors for channels
            channel_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_sig))

            # Limit number of individual plots
            n_to_plot = min(n_sig, max_individual_plots)

            # Sort channels by peak correlation for consistent ordering
            peak_r_values = []
            for ch_idx in sig_ch_indices:
                r_ch = r_cv[:, ch_idx]
                peak_r_values.append(np.nanmax(r_ch) if np.any(np.isfinite(r_ch)) else -np.inf)

            sort_order = np.argsort(-np.array(peak_r_values))
            sorted_indices = sig_ch_indices[sort_order]
            sorted_names = [sig_ch_names[i] for i in sort_order]

            # Plot individual channels
            for plot_num, (ch_idx, ch_name) in enumerate(zip(sorted_indices[:n_to_plot],
                                                             sorted_names[:n_to_plot])):
                p_val = pvals[ch_idx] if pvals is not None and ch_idx < len(pvals) else None
                p_adj_val = p_adj[ch_idx] if p_adj is not None and ch_idx < len(p_adj) else None

                # Find position in original sorted list for color
                color_idx = np.where(sorted_indices == ch_idx)[0][0]
                color = channel_colors[min(color_idx, len(channel_colors) - 1)]

                plot_individual_channel(
                    r_cv=r_cv,
                    lags_ms=lags_ms,
                    channel_name=ch_name,
                    channel_idx=ch_idx,
                    feature_name=feat,
                    outdir=feat_dir,
                    subject=subject,
                    p_value=p_val,
                    p_adj=p_adj_val,
                    color=color,
                )

                if (plot_num + 1) % 10 == 0:
                    print(f"    Plotted {plot_num + 1}/{n_to_plot} channels...")

            print(f"    Saved individual plots to: {feat_dir}")

            # Also create grid plot of all significant channels
            if n_sig > 0:
                plot_all_significant_channels_grid(
                    r_cv=r_cv,
                    lags_ms=lags_ms,
                    sig_channel_indices=sorted_indices,
                    sig_channel_names=sorted_names,
                    feature_name=feat,
                    outdir=outdir,
                    subject=subject,
                    pvals=pvals,
                    p_adj=p_adj,
                    max_cols=4,
                )
                print(f"    Saved grid plot to: {outdir}")

    # Per-channel heatmap for significant channels only
    if plot_significant_only:
        for feat in features_to_plot:
            res = results[feat]
            r_cv = res.get("r_cv_all")
            sig_ch = res.get("sig_channels")

            if r_cv is not None and sig_ch is not None and np.sum(sig_ch) > 0:
                sig_indices = np.where(sig_ch)[0]
                r_cv_sig = r_cv[:, sig_indices]
                sig_labels = [res["sig_channel_names"][j] for j in range(len(sig_indices))]

                if r_cv_sig.shape[1] <= 100:  # Only plot if reasonable number of channels
                    fig2, ax2 = plt.subplots(figsize=(12, max(4, 0.15 * r_cv_sig.shape[1])))

                    # Sort by mean correlation for better visualization
                    mean_per_ch = np.nanmean(r_cv_sig, axis=0)
                    sort_order = np.argsort(-mean_per_ch)
                    r_cv_sorted = r_cv_sig[:, sort_order]
                    labels_sorted = [sig_labels[j] for j in sort_order]

                    im = ax2.imshow(
                        r_cv_sorted.T,
                        aspect="auto",
                        origin="lower",
                        extent=[lags_ms[0], lags_ms[-1], 0, r_cv_sorted.shape[1]],
                        cmap="RdBu_r",
                        vmin=-0.15,
                        vmax=0.15,
                    )
                    ax2.axvline(0, color="k", ls="--", lw=1)
                    ax2.set_xlabel("Lag (ms)")
                    ax2.set_ylabel("Significant Channel (sorted by mean r)")
                    ax2.set_title(f"{feat}: Significant channels encoding (n={r_cv_sorted.shape[1]})")

                    # Add channel labels if not too many
                    if r_cv_sorted.shape[1] <= 30:
                        ax2.set_yticks(np.arange(len(labels_sorted)) + 0.5)
                        ax2.set_yticklabels(labels_sorted, fontsize=8)

                    plt.colorbar(im, ax=ax2, label="r")
                    fig2.tight_layout()
                    fig2.savefig(outdir / f"{subject}_{feat}_sig_channel_heatmap.png", dpi=150)
                    plt.close(fig2)

    print(f"\nSaved plots to: {outdir}")


def format_channel_selection_info(selection: Optional[Union[List, Dict]]) -> str:
    """Format channel selection for display."""
    if selection is None:
        return "All channels"
    elif isinstance(selection, dict):
        if "pattern" in selection:
            return f"Pattern: {selection['pattern']}"
        else:
            return f"Range: [{selection.get('start', 0)}, {selection.get('end', '?')})"
    elif isinstance(selection, (list, tuple)):
        if len(selection) <= 5:
            return f"Channels: {list(selection)}"
        else:
            return f"{len(selection)} selected channels"
    return str(selection)


def main():
    np.random.seed(RNG_SEED)

    lags_ms, lags_samp = iter_lags_ms(FS_TARGET, TMIN_MS, TMAX_MS, LAG_STEP_MS)
    half_win_samp = max(1, int(np.rint((RESP_WIN_MS / 1000.0) * FS_TARGET / 2.0)))

    print(f"{'=' * 60}")
    print("Single-Subject Word-Locked Encoding Analysis")
    print(f"{'=' * 60}")
    print("\n[Configuration]")
    print(f"  Model: OLS per lag")
    print(f"  X: StandardScaler (fold-wise) = {STANDARDIZE_X}, no PCA/ridge (plain OLS)")
    print(f"  Y: section-wise z-score + fold-wise train-mean centering")
    print(f"  Plotting: {'Significant channels only' if PLOT_SIGNIFICANT_ONLY else 'All channels'}")
    print(f"  Individual channel plots: {PLOT_INDIVIDUAL_CHANNELS} (max {MAX_INDIVIDUAL_PLOTS})")
    print(f"  Lags: {len(lags_samp)} unique ({TMIN_MS} to {TMAX_MS} ms, step {LAG_STEP_MS})")
    print(f"  Response window: ~{2 * half_win_samp / FS_TARGET * 1000:.1f} ms")
    print(f"  Sections: {SECTIONS}")
    print(f"  Feature sets: {FEATURE_SETS}")
    print(
        f"  Permutation test: run={RUN_PERMUTATION_TEST}, kind={PERM_KIND}, n={PERM_N}, alpha={PERM_ALPHA}, FDR={PERM_USE_FDR}")

    channel_selection_info = format_channel_selection_info(SELECTED_CHANNELS)
    print(f"  Channel selection: {channel_selection_info}")

    if PLOT_SIGNIFICANT_ONLY and not RUN_PERMUTATION_TEST:
        print("\n[WARNING] PLOT_SIGNIFICANT_ONLY=True but RUN_PERMUTATION_TEST=False!")
        print("  Enabling permutation test to identify significant channels.")
        run_perm = True
    else:
        run_perm = RUN_PERMUTATION_TEST

    if not FEATURES_DIR.exists():
        print(f"\n[ERROR] Features directory not found: {FEATURES_DIR}")
        print("Please run feature extraction first.")
        return

    for sid, path in EEG_MAT_FILES.items():
        if not Path(path).exists():
            print(f"\n[ERROR] EEG file not found: {path}")
            return

    subject_name = "Subject02"

    results, data, original_ch_indices = run_single_subject_analysis(
        subject_name=subject_name,
        eeg_files=EEG_MAT_FILES,
        features_dir=FEATURES_DIR,
        feature_sets=FEATURE_SETS,
        sections=SECTIONS,
        fs_target=FS_TARGET,
        lags_ms=lags_ms,
        lags_samp=lags_samp,
        half_win_samp=half_win_samp,
        nfold=NFOLD,
        standardize_x=STANDARDIZE_X,
        perm_n=PERM_N if run_perm else 0,
        perm_alpha=PERM_ALPHA,
        perm_use_fdr=PERM_USE_FDR,
        perm_seed=RNG_SEED,
        selected_channels=SELECTED_CHANNELS,
    )

    plot_results(
        results,
        lags_ms,
        OUTDIR,
        subject_name,
        channel_labels=data.channel_labels,
        channel_selection_info=channel_selection_info,
        plot_significant_only=PLOT_SIGNIFICANT_ONLY,
        min_significant_channels=MIN_SIGNIFICANT_CHANNELS,
        plot_individual_channels=PLOT_INDIVIDUAL_CHANNELS,
        max_individual_plots=MAX_INDIVIDUAL_PLOTS,
    )

    # Save results
    save_dict = {
        "subject": subject_name,
        "lags_ms": lags_ms,
        "lags_samp": lags_samp,
        "sections": SECTIONS,
        "feature_sets": list(results.keys()),
        "channel_labels": data.channel_labels,
        "n_channels": data.n_channels,
        "original_channel_indices": original_ch_indices,
        "channel_selection": SELECTED_CHANNELS,
        "channel_selection_info": channel_selection_info,
        "plot_significant_only": PLOT_SIGNIFICANT_ONLY,
        "plot_individual_channels": PLOT_INDIVIDUAL_CHANNELS,
        "results": results,
        "params": {
            "fs_target": FS_TARGET,
            "tmin_ms": TMIN_MS,
            "tmax_ms": TMAX_MS,
            "lag_step_ms": LAG_STEP_MS,
            "resp_win_ms": RESP_WIN_MS,
            "nfold": NFOLD,
            "standardize_x": STANDARDIZE_X,
            "perm_kind": PERM_KIND,
            "perm_n": PERM_N if run_perm else 0,
            "perm_alpha": PERM_ALPHA,
            "perm_use_fdr": PERM_USE_FDR,
            "selected_channels": SELECTED_CHANNELS,
            "plot_significant_only": PLOT_SIGNIFICANT_ONLY,
            "min_significant_channels": MIN_SIGNIFICANT_CHANNELS,
            "plot_individual_channels": PLOT_INDIVIDUAL_CHANNELS,
            "max_individual_plots": MAX_INDIVIDUAL_PLOTS,
        },
    }

    with open(OUTDIR / f"{subject_name}_encoding_results.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    for feat, res in results.items():
        if res.get("channel_results_df") is not None:
            csv_path = OUTDIR / f"{subject_name}_{feat}_channel_results.csv"
            res["channel_results_df"].to_csv(csv_path, index=False)
            print(f"Saved channel results: {csv_path}")

    # MATLAB summary
    mat_dict = {
        "lags_ms": lags_ms,
        "lags_samp": lags_samp,
        "channel_labels": np.array(data.channel_labels, dtype=object),
        "n_channels": data.n_channels,
        "plot_significant_only": float(PLOT_SIGNIFICANT_ONLY),
        "plot_individual_channels": float(PLOT_INDIVIDUAL_CHANNELS),
    }

    if original_ch_indices is not None:
        mat_dict["original_channel_indices"] = original_ch_indices

    for feat, res in results.items():
        safe_feat = feat.replace("-", "_")
        # All channels
        mat_dict[f"mean_r_{safe_feat}"] = np.asarray(res["mean_r_lags"], float)
        mat_dict[f"sem_r_{safe_feat}"] = np.asarray(res["sem_r_lags"], float)
        mat_dict[f"peak_r_{safe_feat}"] = float(res["peak_r"])
        mat_dict[f"peak_lag_{safe_feat}"] = float(res["peak_lag_ms"])
        mat_dict[f"n_channels_{safe_feat}"] = int(res["n_channels"])
        mat_dict[f"r_cv_{safe_feat}"] = np.asarray(res["r_cv_all"], float)

        # Significant channels
        mat_dict[f"mean_r_sig_{safe_feat}"] = np.asarray(res["mean_r_sig_lags"], float)
        mat_dict[f"sem_r_sig_{safe_feat}"] = np.asarray(res["sem_r_sig_lags"], float)
        mat_dict[f"peak_r_sig_{safe_feat}"] = float(res["peak_r_sig"]) if np.isfinite(res["peak_r_sig"]) else 0.0
        mat_dict[f"peak_lag_sig_{safe_feat}"] = float(res["peak_lag_sig_ms"])
        mat_dict[f"n_sig_ch_{safe_feat}"] = int(res["n_sig_channels"])

        if res["sig_channels"] is not None:
            mat_dict[f"sig_ch_{safe_feat}"] = np.asarray(res["sig_channels"], bool).astype(float)
            mat_dict[f"sig_ch_indices_{safe_feat}"] = np.asarray(res["sig_channel_indices"], int)

        stats = res.get("stats", {})
        if "perm_pvals_ch" in stats:
            mat_dict[f"pvals_{safe_feat}"] = np.asarray(stats["perm_pvals_ch"], float)
        if "perm_obs_stat_ch" in stats:
            mat_dict[f"obs_stat_{safe_feat}"] = np.asarray(stats["perm_obs_stat_ch"], float)

    savemat(str(OUTDIR / f"{subject_name}_encoding_results.mat"), mat_dict)

    # Final summary
    print(f"\n{'=' * 60}")
    print("Analysis Complete!")
    print(f"{'=' * 60}")
    print("\n[SUMMARY]")
    print("-" * 60)
    print(f"Channel selection: {channel_selection_info}")
    print(f"Total channels analyzed: {data.n_channels}")
    print(f"Plotting mode: {'Significant channels only' if PLOT_SIGNIFICANT_ONLY else 'All channels'}")
    print(f"Individual channel plots: {PLOT_INDIVIDUAL_CHANNELS}")
    for feat, res in results.items():
        print(f"\n{feat}:")
        print(f"  All channels ({res['n_channels']}):")
        print(f"    Peak r = {res['peak_r']:.4f} at {res['peak_lag_ms']:.0f} ms")
        if res["n_sig_channels"] > 0:
            print(f"  Significant channels ({res['n_sig_channels']}):")
            print(f"    Peak r = {res['peak_r_sig']:.4f} at {res['peak_lag_sig_ms']:.0f} ms")
            if len(res["sig_channel_names"]) <= 10:
                print(f"    Channels: {res['sig_channel_names']}")
            else:
                print(f"    Top 5: {res['sig_channel_names'][:5]}")
        else:
            print(f"  No significant channels found")
    print("-" * 60)
    print(f"\nResults saved to: {OUTDIR.resolve()}")
    if PLOT_INDIVIDUAL_CHANNELS:
        print(f"Individual channel plots saved to: {(OUTDIR / 'individual_channels').resolve()}")


if __name__ == "__main__":
    main()