#!/usr/bin/env python3
"""Recompute the GloVe channel test after the v3 partial timing control.

The original ``sig_glove`` column is the within-subject BH reject flag from
``encoding_channel.cv_encoding_ols_with_pca_and_phase_perm`` (10-fold OLS,
PCA 50, max over lags of the concatenated out-of-fold Pearson r, 1000
phase-randomized draws, p = (#{null >= observed} + 1) / 1001, BH at 0.05).
This script repeats that test on the same 1210 taxonomy channels and the
same GloVe-300 matrix, but residualizes neural Y with the partial-mode
word-boundary nuisance inside every CV fold and inside every permutation.

The nuisance has the eight partial-mode columns. The six word-boundary
columns are rebuilt to match the finished cache. ``env_mean`` and
``env_rise`` are linearly interpolated from that cache onto the 25 ms lag
centers, including section 3, and are not recomputed from audio. The
nuisance is scaled on the training rows and fit with the same RidgeCV alphas.
The Ridge is refit on the phase-randomized training Y of that draw. Y is
not residualized once and then permuted, and the GloVe OLS/PCA map is not
refit inside the null (the original null keeps y_pred fixed).

Writes ``tables/sig_glove_timing.csv`` only. It does not edit the taxonomy
table or the existing ``sig_glove`` column.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import encoding_channel as ec
import timing_control as tc

DATA_ROOT = Path(os.environ.get("ANALYSISEV_DATA_ROOT", "/orcd/pool/005/haolun52/analysisEV"))
TC_ROOT = Path(__file__).resolve().parent
TABLES = TC_ROOT / "tables"
CACHE = TC_ROOT / "cache"
OUT_CSV = TABLES / "sig_glove_timing.csv"
NUIS_CACHE = CACHE / "nuisance_sig_glove_25ms.npz"
TAX = DATA_ROOT / "group_encoding_results" / "functional_taxonomy" / "tables" / "functional_taxonomy_electrode_table.csv"

FS = 500.0
TMIN_MS = -2000.0
TMAX_MS = 2000.0
LAG_STEP_MS = 25.0
RESP_WIN_MS = 200.0
NFOLD = 10
PCA_K = 50
PCA_WHITEN = True
PERM_N = 1000
PERM_ALPHA = 0.05
PERM_SEED = 42  # encoding_group.RNG_SEED; hash("glove") is process-salted
PERM_BATCH = 32
PERM_EPS = 1e-12
EXPECTED_CHANNELS = 1210

# encoding_group.SUBJECTS eeg paths. Section 1 labels are the BH family.
EEG_FILES = {
    "Subject01": {
        1: "output/Naturalistic/segments/Subject01_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject01_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject01_NA3_01_644sec_HG.mat",
    },
    "Subject03": {
        1: "output/Naturalistic/segments/Subject03_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject03_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject03_NA3_01_644sec_HG.mat",
    },
    "Subject04": {
        1: "output/Naturalistic/segments/Subject04_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject04_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject04_NA3_01_644sec_HG.mat",
    },
    "Subject06": {
        1: "output/Naturalistic/segments/Subject06_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject06_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject06_NA3_01_644sec_HG.mat",
    },
    "Subject07": {
        1: "output/Naturalistic/segments/Subject07_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject07_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject07_NA3_01_644sec_HG.mat",
    },
    "Subject08": {
        1: "output/Naturalistic/segments/Subject08_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject08_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject08_NA3_01_644sec_HG.mat",
    },
    "Subject09": {
        1: "output/Naturalistic/segments/Subject09_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject09_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject09_NA3_01_632sec_HG.mat",
    },
    "Subject10": {
        1: "output/Naturalistic/segments/Subject10_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject10_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject10_NA3_01_644sec_HG.mat",
    },
    "Subject11": {
        1: "output/Naturalistic/segments/Subject11_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject11_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject11_NA3_01_638sec_HG.mat",
    },
    "Subject12": {
        1: "output/Naturalistic/segments/Subject12_NA_01_565sec_HG.mat",
        2: "output/Naturalistic/segments/Subject12_NA2_01_643sec_HG.mat",
        3: "output/Naturalistic/segments/Subject12_NA3_01_644sec_HG.mat",
    },
}


def _stamp() -> str:
    return time.strftime("%F %T")


def log(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def prepare_ridge(Ztr: np.ndarray) -> dict:
    """Centered design and X'X spectrum. Ztr is already StandardScaler output."""
    X_offset = Ztr.mean(axis=0)
    Xc = Ztr - X_offset
    eigvals, V = np.linalg.eigh(Xc.T @ Xc)
    return {
        "Xc": Xc,
        "X_offset": X_offset,
        "eigvals": eigvals,
        "V": V,
        "XT_ones": Xc.T @ np.ones(Ztr.shape[0]),
        "n": Ztr.shape[0],
    }


def ridge_coef_intercept(prep: dict, Ytr: np.ndarray, alphas: np.ndarray):
    """Match sklearn RidgeCV (fit_intercept, shared alpha, LOO MSE).

    Ytr is (n, C, B). Returns coef (p, C, B) and intercept (C, B), in the
    StandardScaler feature space RidgeCV.predict uses.
    """
    n, C, B = Ytr.shape
    Xc = prep["Xc"]
    y_offset = Ytr.mean(axis=0)
    yc = Ytr - y_offset
    XT_y = np.tensordot(Xc, yc, axes=(0, 0))
    Syc2 = np.sum(yc * yc, axis=1)
    best_score = None
    best_coef = None
    ones = np.ones(n)
    XT_ones = prep["XT_ones"]
    eigvals = prep["eigvals"]
    V = prep["V"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for alpha in alphas:
            D = 1.0 / (eigvals + float(alpha))
            Hinv = (V * D) @ V.T
            coef = np.tensordot(Hinv, XT_y, axes=(1, 0))
            XA = Xc @ Hinv
            XAX = np.sum(XA * Xc, axis=1)
            X_Hinv_ones = Xc @ (Hinv @ XT_ones)
            alpha_d = 1.0 - XAX - (ones - X_Hinv_ones) / n
            w = 1.0 / (alpha_d * alpha_d)
            term1 = w @ Syc2
            Xw = Xc * w[:, None]
            M = np.tensordot(Xw, yc, axes=(0, 0))
            term2 = np.sum(coef * M, axis=(0, 1))
            Gram = np.einsum("kcb,lcb->klb", coef, coef, optimize=True)
            Cw = Xc.T @ Xw
            term3 = np.einsum("klb,kl->b", Gram, Cw, optimize=True)
            score = -(term1 - 2.0 * term2 + term3) / (n * C)
            # First alpha is kept on ties, matching RidgeCV's strict > update.
            if best_score is None:
                best_score = np.array(score, dtype=np.float64, copy=True)
                best_coef = np.array(coef, dtype=np.float64, copy=True)
            else:
                upd = score > best_score
                best_score[upd] = score[upd]
                best_coef[:, :, upd] = coef[:, :, upd]
    intercept = y_offset - np.tensordot(prep["X_offset"], best_coef, axes=(0, 0))
    return best_coef, intercept


def centered_residuals(Ztr, Zte, Ytr, Yte, prep, alphas):
    """Train-mean-centered nuisance residuals. Y arrays are (n, C, B)."""
    coef, intercept = ridge_coef_intercept(prep, Ytr, alphas)
    pred_tr = np.tensordot(Ztr, coef, axes=(1, 0)) + intercept
    pred_te = np.tensordot(Zte, coef, axes=(1, 0)) + intercept
    resid_tr = Ytr - pred_tr
    mu = resid_tr.mean(axis=0, keepdims=True)
    return resid_tr - mu, Yte - pred_te - mu


def sklearn_centered_residuals(T_tr, T_te, Ytr, Yte, alphas):
    """Same residual, via RidgeCV, for the self-check and rare small folds."""
    sc = StandardScaler().fit(T_tr)
    Ztr = sc.transform(T_tr)
    Zte = sc.transform(T_te)
    n, C, B = Ytr.shape
    resid_tr = np.empty((n, C, B), dtype=np.float64)
    resid_te = np.empty((T_te.shape[0], C, B), dtype=np.float64)
    for b in range(B):
        model = RidgeCV(alphas=alphas).fit(Ztr, Ytr[:, :, b])
        resid_tr[:, :, b] = Ytr[:, :, b] - model.predict(Ztr)
        resid_te[:, :, b] = Yte[:, :, b] - model.predict(Zte)
    mu = resid_tr.mean(axis=0, keepdims=True)
    return resid_tr - mu, resid_te - mu


def phase_axes(n_samples: int):
    if n_samples % 2 == 0:
        pos = np.arange(1, n_samples // 2)
        neg = np.arange(n_samples - 1, n_samples // 2, -1)
    else:
        pos = np.arange(1, (n_samples - 1) // 2 + 1)
        neg = np.arange(n_samples - 1, (n_samples - 1) // 2, -1)
    return pos, neg


def apply_phase(spec: np.ndarray, pos: np.ndarray, neg: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """shift is (B, K) complex. Returns (B, N, C) real, shared phase across channels."""
    scrambled = np.empty((shift.shape[0], spec.shape[0], spec.shape[1]), dtype=np.complex128)
    scrambled[:] = spec
    scrambled[:, pos, :] *= shift[:, :, None]
    scrambled[:, neg, :] *= np.conjugate(shift[:, :, None])
    return np.real(ifft(scrambled, axis=1))


def corr_batch(y_true: np.ndarray, y_pred: np.ndarray, eps: float = PERM_EPS) -> np.ndarray:
    """Pearson r per channel. y_true (N, C, B), y_pred (N, C) -> (B, C)."""
    yt0 = y_true - y_true.mean(axis=0, keepdims=True)
    yp0 = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = np.sum(yt0 * yp0[:, :, None], axis=0)
    den = np.sqrt(np.sum(yt0 * yt0, axis=0) * np.sum(yp0 * yp0, axis=0)[:, None]) + eps
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.transpose(r)


def _fold_valid(valid: np.ndarray, fold_indices, train_indices):
    usable = []
    for o, te in enumerate(fold_indices):
        te_v = te[valid[te]]
        tr_v = train_indices[o][valid[train_indices[o]]]
        if te_v.size <= 2 or tr_v.size <= 2:
            continue
        usable.append((o, tr_v, te_v))
    return usable


def lag_statistic(Y, valid, nuisance, designs, fold_indices, train_indices, alphas, rng, perm_n, batch):
    """One lag: observed max-r piece and the (perm_n, C) null correlations.

    Phase-randomizes the valid-word Y, then refits the nuisance inside each
    fold of each draw. y_pred is the observed out-of-fold prediction.
    """
    Y = np.asarray(Y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 8:
        C = Y.shape[1]
        return np.full(C, np.nan), np.full((perm_n, C), np.nan)

    Yv = Y[valid_idx]
    Yv = np.where(np.isfinite(Yv), Yv, 0.0)
    Tv = np.asarray(nuisance[valid_idx], dtype=np.float64)
    usable = _fold_valid(valid, fold_indices, train_indices)
    if not usable:
        C = Y.shape[1]
        return np.full(C, np.nan), np.full((perm_n, C), np.nan)

    pos_of = np.empty(Y.shape[0], dtype=np.int64)
    pos_of[valid_idx] = np.arange(valid_idx.size)

    C = Yv.shape[1]
    oof_true = []
    oof_pred = []
    fold_pack = []
    for o, tr_w, te_w in usable:
        tr = pos_of[tr_w]
        te = pos_of[te_w]
        Xtr = np.asarray(designs[o][tr_w], dtype=np.float64)
        Xte = np.asarray(designs[o][te_w], dtype=np.float64)
        Ttr = Tv[tr]
        Tte = Tv[te]
        Ytr = Yv[tr][:, :, None]
        Yte = Yv[te][:, :, None]
        if tr.size <= Ttr.shape[1]:
            resid_tr, resid_te = sklearn_centered_residuals(Ttr, Tte, Ytr, Yte, alphas)
            prep = None
            Ztr = Zte = None
        else:
            sc = StandardScaler().fit(Ttr)
            Ztr = sc.transform(Ttr)
            Zte = sc.transform(Tte)
            prep = prepare_ridge(Ztr)
            resid_tr, resid_te = centered_residuals(Ztr, Zte, Ytr, Yte, prep, alphas)
        resid_tr = resid_tr[:, :, 0]
        resid_te = resid_te[:, :, 0]
        weight, *_ = np.linalg.lstsq(Xtr, resid_tr, rcond=None)
        pred = Xte @ weight
        oof_true.append(resid_te)
        oof_pred.append(pred)
        fold_pack.append((tr, te, Ttr, Tte, Ztr, Zte, prep, pred))

    y_true = np.concatenate(oof_true, axis=0)
    y_pred = np.concatenate(oof_pred, axis=0)
    if y_true.shape[0] <= 3:
        return np.full(C, np.nan), np.full((perm_n, C), np.nan)
    observed = ec.corr_np(y_true, y_pred, eps=PERM_EPS)

    N = Yv.shape[0]
    pos, neg = phase_axes(N)
    spec = fft(Yv, axis=0)
    phases = rng.random((perm_n, pos.size)) * (2.0 * np.pi)
    null = np.empty((perm_n, C), dtype=np.float64)
    for b0 in range(0, perm_n, batch):
        b1 = min(b0 + batch, perm_n)
        shift = np.exp(1j * phases[b0:b1])
        Yb = np.moveaxis(apply_phase(spec, pos, neg, shift), 0, 2)  # (N, C, B)
        parts = []
        for tr, te, Ttr, Tte, Ztr, Zte, prep, _pred in fold_pack:
            Ytr = Yb[tr]
            Yte = Yb[te]
            if prep is None:
                _rtr, rte = sklearn_centered_residuals(Ttr, Tte, Ytr, Yte, alphas)
            else:
                _rtr, rte = centered_residuals(Ztr, Zte, Ytr, Yte, prep, alphas)
            parts.append(rte)
        null[b0:b1] = corr_batch(np.concatenate(parts, axis=0), y_pred, eps=PERM_EPS)
    return observed, null


def pvals_from_max(obs_max: np.ndarray, perm_max: np.ndarray) -> np.ndarray:
    finite = np.isfinite(perm_max)
    ge = finite & (perm_max >= obs_max[None, :])
    n_finite = finite.sum(axis=0)
    n_ge = ge.sum(axis=0)
    p = np.full(obs_max.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(obs_max) & (n_finite > 0)
    p[ok] = (n_ge[ok] + 1.0) / (n_finite[ok] + 1.0)
    return p


def verify_against_sklearn(alphas: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    n, p, C, B = 360, 8, 5, 4
    T = rng.normal(size=(n + 40, p)) * rng.uniform(0.3, 2.0, size=p)
    beta = rng.normal(size=(p, C))
    signal = T[:n] @ beta
    Y = np.empty((n, C, B))
    for b in range(B):
        Y[:, :, b] = signal + (0.4 + 0.2 * b) * rng.normal(size=(n, C))
    tr, te = np.arange(40, n), np.arange(40)
    sc = StandardScaler().fit(T[tr])
    Ztr, Zte = sc.transform(T[tr]), sc.transform(T[te])
    fast_tr, fast_te = centered_residuals(
        Ztr, Zte, Y[tr], Y[te], prepare_ridge(Ztr), alphas,
    )
    sk_tr, sk_te = sklearn_centered_residuals(T[tr], T[te], Y[tr], Y[te], alphas)
    err = max(np.max(np.abs(fast_tr - sk_tr)), np.max(np.abs(fast_te - sk_te)))
    if err > 1e-6:
        raise RuntimeError(f"batched nuisance residual disagrees with RidgeCV (max abs {err})")
    x = rng.normal(size=(128, 3))
    got = apply_phase(fft(x, axis=0), *phase_axes(x.shape[0]), np.exp(1j * rng.random((1, 63)) * 0))
    # independent stream check against encoding_channel.phase_randomize_1d
    x2 = rng.normal(size=(64, 2))
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    ref = ec.phase_randomize_1d(x2, rng_a)
    spec = fft(x2, axis=0)
    pos, neg = phase_axes(x2.shape[0])
    phases = rng_b.random((1, pos.size)) * (2.0 * np.pi)
    hat = apply_phase(spec, pos, neg, np.exp(1j * phases))[0]
    if np.max(np.abs(ref - hat)) > 1e-8:
        raise RuntimeError("phase randomization does not match encoding_channel.phase_randomize_1d")
    if not np.allclose(got[0], x, atol=1e-8):
        raise RuntimeError("zero phase shift did not reproduce the series")
    log(f"self-check passed (ridge max abs {err:.2e})")


def _scaled(Ttr, Tte):
    sc = StandardScaler().fit(Ttr)
    return sc.transform(Ttr), sc.transform(Tte)


def verify_lag_refits(alphas: np.ndarray) -> None:
    """A draw must refit the nuisance; residualizing first would not."""
    rng = np.random.default_rng(1)
    W, C, P = 80, 2, 6
    Y = rng.normal(size=(W, C))
    valid = np.ones(W, dtype=bool)
    nuisance = rng.normal(size=(W, 8))
    X = rng.normal(size=(W, P))
    folds = [(0, 40), (40, 80)]
    idx = [np.arange(a, b) for a, b in folds]
    trains = [idx[1], idx[0]]
    designs = []
    for tr in trains:
        sc = StandardScaler().fit(X[tr])
        designs.append(sc.transform(X))
    obs, null = lag_statistic(
        Y, valid, nuisance, designs, idx, trains, alphas,
        np.random.default_rng(2), perm_n=5, batch=5,
    )
    if obs.shape != (C,) or null.shape != (5, C):
        raise RuntimeError(f"lag statistic shape {obs.shape} {null.shape}")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(null)):
        raise RuntimeError("synthetic lag statistic was not finite")
    # Different training targets must change the test residual.
    Ttr, Tte = nuisance[:60], nuisance[60:]
    Ytr = Y[:60, :, None]
    Yte = Y[60:, :, None]
    Ytr2 = Ytr + rng.normal(size=Ytr.shape)
    a, b = sklearn_centered_residuals(Ttr, Tte, Ytr, Yte, alphas)
    c, d = sklearn_centered_residuals(Ttr, Tte, Ytr2, Yte, alphas)
    if np.allclose(b, d):
        raise RuntimeError("nuisance residual did not change when training Y changed")
    del a, c
    r_ref = ec.corr_np(y_true := rng.normal(size=(50, C)), y_pred := rng.normal(size=(50, C)))
    r_hat = corr_batch(y_true[:, :, None], y_pred)[0]
    if np.max(np.abs(r_ref - r_hat)) > 1e-10:
        raise RuntimeError("batched correlation does not match encoding_channel.corr_np")


# Same column order as timing_control.NUIS_NAMES.
WORD_BOUNDARY_NAMES = tc.NUIS_NAMES[:6]
EXT_GRID_PATH = CACHE / "nuisance_ext_grid.npy"


def _word_boundary_block(on_s, off_s, lags_ms) -> np.ndarray:
    """Columns 0:6 of ``build_covariates``, with no envelope."""
    on = np.rint(on_s * tc.FS).astype(int)
    off = np.rint(off_s * tc.FS).astype(int)
    T = int(off.max() + 4 * tc.FS)
    speech = np.zeros(T)
    for a, b in zip(on, off):
        speech[a:max(b, a + 1)] = 1.0
    onset_imp = np.zeros(T)
    np.add.at(onset_imp, on, 1.0)
    cs_speech = np.r_[0.0, np.cumsum(speech)]
    cs_onset = np.r_[0.0, np.cumsum(onset_imp)]
    sil = np.r_[1.0, np.clip(on_s[1:] - off_s[:-1], 0, None)]
    ioi_p = np.log(np.r_[1.0, np.diff(on_s)] + 0.01)
    ioi_n = np.log(np.r_[np.diff(on_s), 1.0] + 0.01)
    dur = np.log(off_s - on_s + 0.01)
    cols = []
    for L in lags_ms:
        c = on + int(round(float(L) / 1000.0 * tc.FS))
        s = np.clip(c - tc.HALF, 0, T)
        e = np.clip(c + tc.HALF, 0, T)
        w = np.maximum(e - s, 1)
        speech_cov = (cs_speech[e] - cs_speech[s]) / w
        n_onsets = cs_onset[e] - cs_onset[s]
        cols.append(np.column_stack([speech_cov, n_onsets, sil, ioi_p, ioi_n, dur]))
    return np.stack(cols, axis=1)


def build_word_boundary_covariates(lags_ms: np.ndarray) -> np.ndarray:
    """25 ms (or any) grid using the cache's word-boundary columns only.

    Does not call ``_section_envelope`` and does not read a wav.
    """
    lags_ms = np.asarray(lags_ms, dtype=np.float64)
    blocks = []
    for sid in (1, 2, 3):
        wt = pd.read_csv(
            DATA_ROOT / "extracted_linguistic_features" / f"section_{sid:03d}" / "word_timing.csv"
        )
        blocks.append(_word_boundary_block(
            wt["onset_relative"].to_numpy(float),
            wt["offset_relative"].to_numpy(float),
            lags_ms,
        ))
    return np.concatenate(blocks, axis=0).astype(np.float64)


def interpolate_envelope(lags_ms: np.ndarray) -> np.ndarray:
    """``env_mean`` and ``env_rise`` on ``lags_ms``.

    Linear interpolation of those two columns from the finished 50 ms cache,
    separately for each word. Knots are copied from the cache, so shared lag
    centers match it. This does not read a wav and does not resynthesize
    section 3.
    """
    full = np.load(EXT_GRID_PATH)
    src_lags = np.asarray(tc.EXT_LAGS, dtype=np.float64)
    if full.ndim != 3 or full.shape[1] != src_lags.size or full.shape[2] != len(tc.NUIS_NAMES):
        raise RuntimeError(f"unexpected envelope cache shape {full.shape}")
    lags_ms = np.asarray(lags_ms, dtype=np.float64)
    if lags_ms[0] < src_lags[0] - 1e-6 or lags_ms[-1] > src_lags[-1] + 1e-6:
        raise RuntimeError("25 ms lags fall outside the cached envelope grid")
    idx = np.interp(lags_ms, src_lags, np.arange(src_lags.size, dtype=np.float64))
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, src_lags.size - 1)
    weight = idx - i0
    cached_env = full[:, :, 6:8]
    left = np.take(cached_env, i0, axis=1)
    right = np.take(cached_env, i1, axis=1)
    env = left * (1.0 - weight)[None, :, None] + right * weight[None, :, None]
    nearest = np.argmin(np.abs(lags_ms[:, None] - src_lags[None, :]), axis=1)
    on_knot = np.abs(lags_ms - src_lags[nearest]) < 1e-6
    env[:, on_knot, :] = np.take(cached_env, nearest[on_knot], axis=1)
    return np.asarray(env, dtype=np.float64)


def load_nuisance(lags_ms: np.ndarray) -> np.ndarray:
    lags_ms = np.asarray(lags_ms, dtype=np.float64)
    names = tuple(tc.NUIS_NAMES)
    if NUIS_CACHE.exists():
        with np.load(NUIS_CACHE) as z:
            cached = z["lags_ms"]
            arr = z["nuisance"]
            saved = tuple(str(x) for x in z["columns"]) if "columns" in z.files else ()
        if (arr.shape[1] == lags_ms.size and arr.shape[2] == len(names)
                and saved == names and np.allclose(cached, lags_ms)):
            log(f"loaded nuisance cache {NUIS_CACHE.name} {arr.shape} columns={saved}")
            return arr
        log(f"nuisance cache is {arr.shape} columns={saved or 'unknown'}; rebuilding 8 columns")

    log("building 25 ms nuisance: word boundaries plus envelope interpolated from the 50 ms cache")
    boundary = build_word_boundary_covariates(lags_ms)
    envelope = interpolate_envelope(lags_ms)
    if boundary.shape[:2] != envelope.shape[:2]:
        raise RuntimeError(f"boundary {boundary.shape} vs envelope {envelope.shape}")
    arr = np.concatenate([boundary, envelope], axis=2)
    if arr.shape[1] != lags_ms.size or arr.shape[2] != len(names):
        raise RuntimeError(f"nuisance shape {arr.shape} does not match {lags_ms.size} lags and {names}")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = NUIS_CACHE.with_suffix(".tmp.npz")
    np.savez(tmp, nuisance=arr, lags_ms=lags_ms, columns=np.asarray(names))
    tmp.replace(NUIS_CACHE)
    log(f"wrote {NUIS_CACHE.name} {arr.shape} columns={names}")
    return arr


def taxonomy() -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "ch_idx"])
    if len(tax) != EXPECTED_CHANNELS:
        raise RuntimeError(f"taxonomy has {len(tax)} rows, expected {EXPECTED_CHANNELS}")
    missing = [s for s in tax["subject"].unique() if s not in EEG_FILES]
    if missing:
        raise RuntimeError(f"taxonomy subjects without EEG paths: {missing}")
    return tax


def finished_subjects(expected: dict, path: Path | None = None) -> set:
    path = OUT_CSV if path is None else path
    if not path.exists() or path.stat().st_size == 0:
        return set()
    prev = pd.read_csv(path)
    done = set()
    drop = set()
    for subject, n in prev.groupby("subject").size().items():
        if expected.get(subject) == int(n):
            done.add(subject)
        else:
            drop.add(subject)
    if drop:
        keep = prev[prev["subject"].isin(done)]
        tmp = path.with_suffix(".rewrite.csv")
        keep.to_csv(tmp, index=False)
        tmp.replace(path)
        log(f"dropped partial rows for {sorted(drop)} from {path.name}")
    return done


def append_subject(df: pd.DataFrame, path: Path | None = None) -> None:
    path = OUT_CSV if path is None else path
    TABLES.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{df['subject'].iloc[0]}.csv")
    df.to_csv(tmp, index=False)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a") as out, open(tmp) as inp:
        header = inp.readline()
        if write_header:
            out.write(header)
        out.writelines(inp)
    tmp.unlink()


def glove_designs(X: np.ndarray, folds):
    if X.shape[1] != 300:
        raise RuntimeError(f"expected GloVe-300, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite GloVe values")
    fold_indices = [np.arange(a, b) for a, b in folds]
    train_indices = []
    designs = []
    for o in range(len(folds)):
        tr = np.concatenate([fold_indices[j] for j in range(len(folds)) if j != o])
        train_indices.append(tr)
        scaler = StandardScaler().fit(X[tr])
        X_tr = scaler.transform(X[tr])
        X_all = scaler.transform(X)
        n_comp = int(min(PCA_K, X_tr.shape[1], X_tr.shape[0]))
        pca = PCA(n_components=n_comp, whiten=PCA_WHITEN)
        pca.fit(X_tr)
        designs.append(np.asarray(pca.transform(X_all), dtype=np.float64))
    return fold_indices, train_indices, designs


def run_subject(subject: str, channels: list, ch_idx: np.ndarray, nuisance: np.ndarray,
                lags_ms: np.ndarray, lags_samp: np.ndarray, half: int, alphas: np.ndarray,
                n_jobs: int) -> pd.DataFrame:
    eeg = {sid: str(DATA_ROOT / rel) for sid, rel in EEG_FILES[subject].items()}
    t0 = time.perf_counter()
    log(f"{subject} loading EEG ({len(channels)} channels)")
    data = ec.load_subject_data(
        subject_name=subject,
        eeg_files=eeg,
        features_dir=DATA_ROOT / "extracted_linguistic_features",
        fs_target=FS,
        sections=(1, 2, 3),
        feature_sets=["glove"],
    )
    labels = [str(c).strip() for c in data.channel_labels]
    if labels != [str(c).strip() for c in channels]:
        raise RuntimeError(
            f"{subject} loaded channels do not match the taxonomy BH family "
            f"(loaded {len(labels)}, taxonomy {len(channels)})"
        )
    if data.X_by_feature["glove"].shape[0] != nuisance.shape[0]:
        raise RuntimeError(
            f"{subject} word count {data.X_by_feature['glove'].shape[0]} "
            f"!= nuisance rows {nuisance.shape[0]}"
        )
    folds = ec.make_folds_within_sections(
        data.section_word_slices, NFOLD, ec.MIN_FOLDS_PER_SECTION,
    )
    if len(folds) < 2:
        raise RuntimeError(f"{subject} produced {len(folds)} folds")
    fold_indices, train_indices, designs = glove_designs(data.X_by_feature["glove"], folds)
    log(f"{subject} loaded in {time.perf_counter() - t0:.0f}s, {len(folds)} folds, running {lags_ms.size} lags")

    children = np.random.SeedSequence(PERM_SEED).spawn(int(lags_ms.size))
    obs_max = np.full(len(channels), -np.inf)
    perm_max = np.full((PERM_N, len(channels)), -np.inf)

    def one_lag(li: int):
        Y, valid = ec.compute_Y_for_lag(data, int(lags_samp[li]), int(half))
        rng = np.random.default_rng(children[li])
        return li, lag_statistic(
            Y, valid, nuisance[:, li, :], designs, fold_indices, train_indices,
            alphas, rng, PERM_N, PERM_BATCH,
        )

    n_done = 0
    t_lag = time.perf_counter()
    workers = max(1, min(int(n_jobs), int(lags_ms.size)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_lag, li) for li in range(lags_ms.size)]
        for fut in as_completed(futures):
            li, (obs, null) = fut.result()
            obs_max = np.maximum(obs_max, obs)
            perm_max = np.maximum(perm_max, null)
            n_done += 1
            if n_done == 1 or n_done % 20 == 0 or n_done == lags_ms.size:
                log(f"{subject} lags {n_done}/{lags_ms.size} {time.perf_counter() - t_lag:.0f}s")

    obs_max = obs_max.astype(np.float64)
    obs_max[~np.isfinite(obs_max)] = np.nan
    perm_max = perm_max.astype(np.float64)
    perm_max[~np.isfinite(perm_max)] = np.nan
    pvals = pvals_from_max(obs_max, perm_max)
    reject, padj = ec.bh_fdr(pvals, alpha=PERM_ALPHA)
    log(
        f"{subject} reject {int(np.sum(reject))}/{len(channels)} "
        f"in {time.perf_counter() - t0:.0f}s"
    )
    out = pd.DataFrame({
        "subject": subject,
        "channel": channels,
        "ch_idx": ch_idx.astype(int),
        "obs_stat": obs_max,
        "pval": pvals,
        "padj": padj,
        "sig_glove_timing": reject.astype(bool),
    })
    del data, designs
    gc.collect()
    return out


TRF_CSV = TABLES / "sig_glove_trf.csv"
V3_CACHE = DATA_ROOT / "group_encoding_results" / "sparse_encoding_v3" / "cache"


def _oof_ols(Y, valid, designs, fold_indices, train_indices):
    """10-fold OLS on one lag. No timing nuisance. Returns centered OOF (true, pred)."""
    Y = np.asarray(Y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    oof_true, oof_pred = [], []
    for o, te in enumerate(fold_indices):
        tr = train_indices[o]
        te_v = te[valid[te]]
        tr_v = tr[valid[tr]]
        if te_v.size <= 2 or tr_v.size <= 2:
            continue
        Xtr = np.asarray(designs[o][tr_v], dtype=np.float64)
        Xte = np.asarray(designs[o][te_v], dtype=np.float64)
        Ytr = np.where(np.isfinite(Y[tr_v]), Y[tr_v], 0.0)
        Yte = np.where(np.isfinite(Y[te_v]), Y[te_v], 0.0)
        y_mean = Ytr.mean(axis=0, keepdims=True)
        Ytr_c = Ytr - y_mean
        Yte_c = Yte - y_mean
        weight, *_ = np.linalg.lstsq(Xtr, Ytr_c, rcond=None)
        oof_true.append(Yte_c)
        oof_pred.append(Xte @ weight)
    if not oof_true:
        return None, None
    return np.concatenate(oof_true, axis=0), np.concatenate(oof_pred, axis=0)


def run_subject_trf(subject: str, n_jobs: int) -> pd.DataFrame:
    """Original sig_glove phase-permutation test on the n161 TRF m200 cache.

    Y is already the 200 ms mean. There is no eight-column timing residualization.
    PCA 50, 10-fold OLS, 1000 phase-randomized draws with y_pred held fixed, BH
    within the channels stored for this subject.
    """
    path = V3_CACHE / f"{subject}_n161_a-2000.0_b2000.0_sh0_m0_trfresid_m200.npz"
    if not path.exists():
        raise RuntimeError(f"missing {path.name}")
    t0 = time.perf_counter()
    with np.load(path) as z:
        Yall = np.array(z["Y"], dtype=np.float64)
        wok = np.array(z["window_ok"], dtype=bool)
        lags_ms = np.array(z["lags_ms"], dtype=np.float64)
        slices = [tuple(map(int, row)) for row in np.array(z["section_slices"])]
        channels = [str(c).strip() for c in z["channels"]]
        ch_idx = np.array(z["channel_index"], dtype=int)
    expect = np.arange(-2000.0, 2000.0 + 1e-6, 25.0)
    if Yall.shape[0] != 5472 or Yall.shape[2] != 161 or not np.allclose(lags_ms, expect):
        raise RuntimeError(f"{path.name} is not the n161 m200 grid, Y={Yall.shape}")
    if wok.shape != (Yall.shape[0], 161):
        raise RuntimeError(f"{path.name} window_ok {wok.shape}")
    parts = [
        np.load(DATA_ROOT / "extracted_linguistic_features" / f"section_{sid:03d}" / "X_word_glove.npy")
        for sid in (1, 2, 3)
    ]
    X = np.vstack(parts)
    if X.shape[0] != Yall.shape[0]:
        raise RuntimeError(f"GloVe rows {X.shape[0]} != Y rows {Yall.shape[0]}")
    folds = ec.make_folds_within_sections(slices, NFOLD, ec.MIN_FOLDS_PER_SECTION)
    fold_indices, train_indices, designs = glove_designs(X, folds)
    log(f"{subject} trf {len(channels)} channels, {lags_ms.size} lags, {len(folds)} folds")

    children = np.random.SeedSequence(PERM_SEED).spawn(int(lags_ms.size))
    obs_max = np.full(len(channels), -np.inf)
    perm_max = np.full((PERM_N, len(channels)), -np.inf)

    def one_lag(li: int):
        rng = np.random.default_rng(children[li])
        yt, yp = _oof_ols(Yall[:, :, li], wok[:, li], designs, fold_indices, train_indices)
        if yt is None or yt.shape[0] <= 3:
            return np.full(len(channels), np.nan), np.full((PERM_N, len(channels)), np.nan)
        obs = ec.corr_np(yt, yp, eps=PERM_EPS)
        null = ec.phase_randomized_correlations_fft(
            yt, yp, PERM_N, rng, batch=PERM_BATCH, eps=PERM_EPS,
        )
        return obs, null

    n_done = 0
    t_lag = time.perf_counter()
    workers = max(1, min(int(n_jobs), int(lags_ms.size)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_lag, li) for li in range(lags_ms.size)]
        for fut in as_completed(futures):
            obs, null = fut.result()
            obs_max = np.maximum(obs_max, obs)
            perm_max = np.maximum(perm_max, null)
            n_done += 1
            if n_done == 1 or n_done % 20 == 0 or n_done == lags_ms.size:
                log(f"{subject} trf lags {n_done}/{lags_ms.size} {time.perf_counter() - t_lag:.0f}s")
    obs_max = obs_max.astype(np.float64)
    obs_max[~np.isfinite(obs_max)] = np.nan
    perm_max = perm_max.astype(np.float64)
    perm_max[~np.isfinite(perm_max)] = np.nan
    pvals = pvals_from_max(obs_max, perm_max)
    reject, padj = ec.bh_fdr(pvals, alpha=PERM_ALPHA)
    log(f"{subject} trf reject {int(np.sum(reject))}/{len(channels)} in {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame({
        "subject": subject,
        "channel": channels,
        "ch_idx": ch_idx.astype(int),
        "obs_stat": obs_max,
        "pval": pvals,
        "padj": padj,
        "sig_glove_trf": reject.astype(bool),
    })


def run_trf(n_jobs: int) -> None:
    if TRF_CSV.resolve() == OUT_CSV.resolve():
        raise RuntimeError("trf output collided with sig_glove_timing.csv")
    subjects = []
    expected = {}
    for path in sorted(V3_CACHE.glob("Subject*_n161_a-2000.0_b2000.0_sh0_m0_trfresid_m200.npz")):
        subject = path.name.split("_")[0]
        with np.load(path) as z:
            expected[subject] = int(z["channels"].shape[0])
        subjects.append(subject)
    if len(subjects) < 10:
        raise RuntimeError(f"need 10 n161 trfresid caches, found {len(subjects)}: {subjects}")
    done = finished_subjects(expected, TRF_CSV)
    log(f"sig_glove_trf subjects {subjects} done {sorted(done)} -> {TRF_CSV.name}")
    for subject in subjects:
        if subject in done:
            log(f"{subject} already in {TRF_CSV.name}")
            continue
        df = run_subject_trf(subject, n_jobs)
        if len(df) != expected[subject]:
            raise RuntimeError(f"{subject} wrote {len(df)} rows, expected {expected[subject]}")
        append_subject(df, TRF_CSV)
        log(f"{subject} appended to {TRF_CSV.name}")
    final = pd.read_csv(TRF_CSV)
    log(f"finished {TRF_CSV.name} reject {int(final['sig_glove_trf'].sum())}/{len(final)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--self-check", action="store_true", help="ridge and phase checks only")
    parser.add_argument(
        "--trf-resid",
        action="store_true",
        help="Phase-permutation test on n161 trfresid_m200 caches. Writes tables/sig_glove_trf.csv.",
    )
    args = parser.parse_args()
    alphas = np.asarray(tc.NUIS_ALPHAS, dtype=np.float64)
    verify_against_sklearn(alphas)
    verify_lag_refits(alphas)
    if args.self_check:
        return
    if args.trf_resid:
        run_trf(args.n_jobs)
        return

    lags_ms, lags_samp = ec.iter_lags_ms(FS, TMIN_MS, TMAX_MS, LAG_STEP_MS)
    half = max(1, int(np.rint((RESP_WIN_MS / 1000.0) * FS / 2.0)))
    if half != int(tc.HALF):
        raise RuntimeError(f"response window {half} samples != timing_control.HALF {tc.HALF}")
    samp_tc = np.array([int(round(float(L) / 1000.0 * FS)) for L in lags_ms], dtype=int)
    if not np.array_equal(samp_tc, lags_samp):
        raise RuntimeError("lag samples disagree with timing_control window centers")
    nuisance = load_nuisance(lags_ms)
    tax = taxonomy()
    if nuisance.shape[0] != sum(
        len(pd.read_csv(DATA_ROOT / "extracted_linguistic_features" / f"section_{sid:03d}" / "word_timing.csv"))
        for sid in (1, 2, 3)
    ):
        raise RuntimeError("nuisance word count != stacked word_timing rows")
    expected = {s: int(n) for s, n in tax.groupby("subject").size().items()}
    done = finished_subjects(expected)
    log(
        f"sig_glove_timing channels {len(tax)} lags {lags_ms.size} "
        f"perm {PERM_N} pca {PCA_K} n_jobs {args.n_jobs} done {sorted(done)}"
    )
    order = list(dict.fromkeys(tax["subject"].tolist()))
    for subject in order:
        if subject in done:
            log(f"{subject} already in {OUT_CSV.name}")
            continue
        sub = tax.loc[tax["subject"] == subject].sort_values("ch_idx")
        df = run_subject(
            subject,
            sub["channel"].astype(str).str.strip().tolist(),
            sub["ch_idx"].to_numpy(),
            nuisance,
            lags_ms,
            lags_samp,
            half,
            alphas,
            args.n_jobs,
        )
        if len(df) != expected[subject]:
            raise RuntimeError(f"{subject} wrote {len(df)} rows, expected {expected[subject]}")
        append_subject(df)
        log(f"{subject} appended to {OUT_CSV}")
    final = pd.read_csv(OUT_CSV)
    if len(final) != EXPECTED_CHANNELS or set(final["subject"]) != set(expected):
        raise RuntimeError(f"output has {len(final)} rows, expected {EXPECTED_CHANNELS}")
    log(f"finished {OUT_CSV} reject {int(final['sig_glove_timing'].sum())}/{len(final)}")


if __name__ == "__main__":
    main()
