#!/usr/bin/env python3
"""
sparse_encoding_deconvolution.py
================================
**Primary** continuous-time temporal kernel analysis for Augmented Sparse
Encoding Models.

Builds a downsampled continuous high-gamma design (default 50 Hz) with
impulses at valid owned-token/word event times. Predictors are lag-expanded
over 0–800 ms for:

- selected SAE features (training-only joint temporal support),
- surprisal,
- nuisance regressors from ``word_timing.csv`` (duration, log frequency,
  character length, preceding pause, conversation onset/boundary, section
  intercepts).

SAE support is selected on training events jointly across lag bins
(top-F screen → MultiTaskLasso), then the lag-expanded continuous design is
fit with RidgeCV. Contiguous held-out time blocks use lag guard bands.

Kernel shape metrics (peak latency, FWHM, onset/offset, polarity, multi-peak
counts) are written alongside fold-averaged kernels. For discrete lag curves
see ``sparse_encoding_lag_screen.py``. Fixed single-lag
``sparse_encoding_regression.py`` is a simplified baseline only.

Nested contrasts
----------------
- nuisance only
- nuisance + surprisal
- nuisance + surprisal + SAE   ("full")

Paper-comparable content gain = full − (nuisance + surprisal), **not**
full − content.

Scores both held-out continuous prediction and held-out event-locked prediction.

Example
-------
    python -m sparse_encoding.sparse_encoding_deconvolution \\
        --feature_tag sae_qwen35_4b_mat_l15 --roi_filter lang
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import MultiTaskLassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_regression import (
    FEATURE_TAG, FS_TARGET, RANDOM_STATE, SECTIONS,
    _fmt_duration, _progress_line, get_subjects, load_sae_features,
)
from sparse_encoding.sparse_encoding_lag_screen import make_buffered_contiguous_splits
from sparse_encoding.sparse_encoding_summary import LEFT_MFG_REGION, language_mask, merge_taxonomy
from sparse_encoding.temporal_metrics import kernel_shape_table

# ======================================================================
# CONFIG
# ======================================================================
FS_DS = 50.0
LAG_START_MS = 0.0
LAG_END_MS = 800.0
LAG_STEP_MS = 50.0          # match lag-screen grid (not every 20 ms sample)
N_FOLDS = 5
LASSO_K_BEST = 512          # univariate screen width before MultiTaskLasso
N_SAE_KEEP = 32
MTL_WIDTH = 128             # features passed into MultiTaskLassoCV
ALPHAS = np.logspace(-1, 5, 8)
MAX_TRAIN_TIMEPOINTS = 15000   # subsample continuous design for Ridge speed


def lag_samples(
    fs_ds: float,
    start_ms: float,
    end_ms: float,
    step_ms: float = LAG_STEP_MS,
) -> np.ndarray:
    """Lag sample indices at the downsampled rate (default 50 ms steps)."""
    lags_ms = np.arange(start_ms, end_ms + 0.5 * step_ms, step_ms, dtype=float)
    return np.asarray(np.round(lags_ms / 1000.0 * fs_ds), dtype=int)


def downsample_eeg(eeg_t: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Downsample (T, C) EEG by block averaging (anti-alias lite)."""
    if abs(fs_in - fs_out) < 1e-9:
        return eeg_t.astype(np.float64)
    factor = int(round(fs_in / fs_out))
    if factor < 1:
        raise ValueError(f"fs_out ({fs_out}) > fs_in ({fs_in})")
    T, C = eeg_t.shape
    n = (T // factor) * factor
    if n <= 0:
        return np.zeros((0, C), dtype=np.float64)
    return eeg_t[:n].reshape(-1, factor, C).mean(axis=1)


def eeg_from_cumsum(eeg_cs: np.ndarray) -> np.ndarray:
    """Recover (T, C) from cumsum with leading zero row."""
    return np.diff(eeg_cs, axis=0)


def load_word_timing_concat(sections) -> pd.DataFrame:
    frames = []
    for sid in sections:
        wt = pd.read_csv(ap.FEATURES_DIR / f"section_{sid:03d}" / "word_timing.csv")
        wt = wt.copy()
        wt["section_id"] = sid
        frames.append(wt)
    return pd.concat(frames, ignore_index=True)


def build_nuisance(wt: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Construct per-word nuisance matrix from word_timing columns."""
    duration = (wt["offset_relative"] - wt["onset_relative"]).to_numpy(dtype=float)
    logfreq = wt["logfreq"].to_numpy(dtype=float) if "logfreq" in wt.columns else np.zeros(len(wt))
    char_len = wt["word"].astype(str).str.len().to_numpy(dtype=float)
    # Preceding pause (seconds); first word in a conversation gets 0.
    onset = wt["onset_relative"].to_numpy(dtype=float)
    offset = wt["offset_relative"].to_numpy(dtype=float)
    pause = np.zeros(len(wt), dtype=float)
    pause[1:] = np.maximum(0.0, onset[1:] - offset[:-1])
    # Conversation onset / boundary
    if "conversation_id" in wt.columns:
        conv = wt["conversation_id"].to_numpy()
        conv_onset = np.zeros(len(wt), dtype=float)
        conv_onset[0] = 1.0
        conv_onset[1:] = (conv[1:] != conv[:-1]).astype(float)
    else:
        conv_onset = np.zeros(len(wt), dtype=float)
        conv_onset[0] = 1.0
    # Section intercepts (drop first)
    sec = wt["section_id"].to_numpy()
    sec_ids = sorted(set(int(s) for s in sec))
    cols = [duration, logfreq, char_len, pause, conv_onset]
    names = ["duration", "logfreq", "char_len", "pre_pause", "conv_onset"]
    for s in sec_ids[1:]:
        cols.append((sec == s).astype(float))
        names.append(f"section_{s}")
    X = np.column_stack(cols).astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    return X, names


def fir_expand_events(
    n_time: int,
    event_samples: np.ndarray,
    values: np.ndarray,
    lags: np.ndarray,
) -> np.ndarray:
    """Build (T, n_lags * n_feat) lag-expanded impulse design.

    ``values`` is (n_events,) or (n_events, n_feat).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    n_feat = values.shape[1]
    n_lags = len(lags)
    X = np.zeros((n_time, n_lags * n_feat), dtype=np.float64)
    event_samples = np.asarray(event_samples, dtype=int)
    for li, lag in enumerate(lags):
        t = event_samples + int(lag)
        good = (t >= 0) & (t < n_time)
        if not np.any(good):
            continue
        sl = slice(li * n_feat, (li + 1) * n_feat)
        # Accumulate in case multiple events land on the same sample.
        np.add.at(X[:, sl], t[good], values[good])
    return X


def _fast_univariate_scores(X_sae: sparse.csr_matrix, Y: np.ndarray) -> np.ndarray:
    """Cheap multi-lag screen: |X' y| summed over centered lag targets."""
    n_feat = X_sae.shape[1]
    scores = np.zeros(n_feat, dtype=np.float64)
    # Use at most a few representative lags + the mean response.
    n_lags = Y.shape[1]
    lag_idx = sorted(set([0, n_lags // 4, n_lags // 2, (3 * n_lags) // 4, n_lags - 1]))
    targets = [Y[:, li] for li in lag_idx]
    targets.append(np.nanmean(Y, axis=1))
    for y in targets:
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        yc = y[ok] - y[ok].mean()
        denom = float(np.linalg.norm(yc)) + 1e-12
        # sparse matvec: (n_feat,) 
        scores += np.abs(X_sae[ok].T.dot(yc)) / denom
    return scores


def select_sae_support_joint(
    X_sae: sparse.csr_matrix,
    Y_lags: np.ndarray,
    k_best: int = LASSO_K_BEST,
    n_keep: int = N_SAE_KEEP,
    mtl_width: int = MTL_WIDTH,
) -> np.ndarray:
    """Top-score screen then MultiTaskLasso across lag bins. Returns bool mask."""
    n_feat = X_sae.shape[1]
    Y = np.asarray(Y_lags, dtype=np.float64)
    scores = _fast_univariate_scores(X_sae, Y)
    k = min(k_best, n_feat)
    keep_idx = np.argsort(scores)[::-1][:k]
    screen = np.zeros(n_feat, dtype=bool)
    screen[keep_idx] = True
    Xs = X_sae[:, keep_idx].toarray()

    # Further cap for MultiTaskLasso.
    width = min(mtl_width, Xs.shape[1])
    if Xs.shape[1] > width:
        sc = np.zeros(Xs.shape[1])
        for li in range(0, Y.shape[1], max(1, Y.shape[1] // 4)):
            y = Y[:, li]
            y = y - np.nanmean(y)
            denom = np.linalg.norm(y[np.isfinite(y)]) + 1e-12
            y2 = np.nan_to_num(y, nan=0.0)
            sc += np.abs(Xs.T @ y2) / denom
        top = np.argsort(sc)[::-1][:width]
        Xs = Xs[:, top]
        keep_idx = keep_idx[top]
        screen = np.zeros(n_feat, dtype=bool)
        screen[keep_idx] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mt = MultiTaskLassoCV(
            alphas=np.logspace(-2, 0, 5), cv=3, max_iter=1000,
            n_jobs=1, random_state=RANDOM_STATE, tol=1e-3,
        )
        mt.fit(Xs, np.nan_to_num(Y, nan=0.0))
    coef = np.asarray(mt.coef_)  # (n_targets, n_features)
    active = np.any(np.abs(coef) > 1e-8, axis=0)
    if not active.any():
        support = np.zeros(n_feat, dtype=bool)
        support[keep_idx[:n_keep]] = True
        return support
    strength = np.max(np.abs(coef), axis=0)
    active_idx = np.where(active)[0]
    order = active_idx[np.argsort(strength[active_idx])[::-1]][:n_keep]
    support = np.zeros(n_feat, dtype=bool)
    support[keep_idx[order]] = True
    return support


def event_locked_matrix(
    eeg_ds: np.ndarray,
    event_samples: np.ndarray,
    lags: np.ndarray,
) -> np.ndarray:
    """Return (n_events, n_lags) mean? No — samples at each lag for one channel.

    Caller passes single-channel eeg_ds (T,).
    """
    n_e = len(event_samples)
    n_l = len(lags)
    out = np.full((n_e, n_l), np.nan, dtype=np.float64)
    T = eeg_ds.shape[0]
    for i, t0 in enumerate(event_samples):
        for j, lag in enumerate(lags):
            t = int(t0 + lag)
            if 0 <= t < T:
                out[i, j] = eeg_ds[t]
    return out


def _score_r(y, yhat) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() < 10:
        return np.nan
    if np.std(y[ok]) < 1e-12 or np.std(yhat[ok]) < 1e-12:
        return 0.0
    r, _ = pearsonr(y[ok], yhat[ok])
    return 0.0 if not np.isfinite(r) else float(r)


def _subsample_rows(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def fit_ridge(
    Xtr, ytr, Xte, max_train: int = MAX_TRAIN_TIMEPOINTS, seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """RidgeCV on a subsampled training set for speed on long continuous designs.

    Returns ``(predictions, alpha, coefficients)``.
    """
    rng = np.random.default_rng(seed)
    idx = _subsample_rows(Xtr.shape[0], max_train, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = RidgeCV(alphas=ALPHAS, fit_intercept=True)
        cv.fit(Xtr[idx], ytr[idx])
        alpha = float(cv.alpha_)
        model = Ridge(alpha=alpha, fit_intercept=True).fit(Xtr[idx], ytr[idx])
        pred = model.predict(Xte).ravel()
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    return pred, alpha, coef


def expand_coef_to_full(coef: np.ndarray, col_ok: np.ndarray) -> np.ndarray:
    """Map coefficients on ``col_ok`` columns back into the full design width."""
    full = np.zeros(col_ok.shape[0], dtype=np.float64)
    full[col_ok] = coef
    return full


def kernels_from_full_coef(
    coef_full: np.ndarray,
    n_lags: int,
    n_nuis: int,
    n_sae: int,
) -> Dict[str, np.ndarray]:
    """Split lag-major full-model coefficients into temporal kernels."""
    n_surp = 1
    need = n_lags * (n_nuis + n_surp + n_sae)
    if coef_full.shape[0] != need:
        # Pad/truncate defensively if a column was entirely dropped upstream.
        tmp = np.zeros(need, dtype=np.float64)
        n = min(need, coef_full.shape[0])
        tmp[:n] = coef_full[:n]
        coef_full = tmp
    nuis = coef_full[: n_lags * n_nuis].reshape(n_lags, n_nuis)
    surp = coef_full[n_lags * n_nuis: n_lags * (n_nuis + n_surp)].reshape(n_lags, n_surp)
    if n_sae > 0:
        sae = coef_full[n_lags * (n_nuis + n_surp):].reshape(n_lags, n_sae)
        sae_l2 = np.linalg.norm(sae, axis=1)
    else:
        sae = np.zeros((n_lags, 0))
        sae_l2 = np.zeros(n_lags)
    return {
        "nuisance_l2": np.linalg.norm(nuis, axis=1),
        "surprisal": surp[:, 0],
        "sae_l2": sae_l2,
        "nuisance": nuis,
        "sae": sae,
    }


def apply_null_shift(
    eeg_all: np.ndarray,
    section_time_slices: List[Tuple[int, int]],
    kind: str,
    seed: int,
    block_len: int = 0,
) -> np.ndarray:
    """Return a copy of continuous EEG with autocorrelation-preserving null.

    ``circular``: random circular shift within each section.
    ``block``: true block permutation — partition each section into contiguous
    blocks of length ``block_len`` (default ≈ 10 s at the working sampling
    rate inferred from segment length, floored at 50 samples) and shuffle
    those blocks. Remainder samples stay at the end.
    """
    out = eeg_all.copy()
    rng = np.random.default_rng(seed)
    for a, b in section_time_slices:
        seg = out[a:b]
        n = int(seg.shape[0])
        if n < 2:
            continue
        if kind == "circular":
            shift = int(rng.integers(1, n))
            out[a:b] = np.roll(seg, shift, axis=0)
        elif kind == "block":
            bl = int(block_len) if int(block_len) > 0 else max(50, n // 20)
            bl = min(bl, max(1, n // 2))
            n_full = (n // bl) * bl
            if n_full < 2 * bl:
                # Too short for multiple blocks: fall back to circular.
                shift = int(rng.integers(1, n))
                out[a:b] = np.roll(seg, shift, axis=0)
                continue
            blocks = [seg[i:i + bl] for i in range(0, n_full, bl)]
            order = rng.permutation(len(blocks))
            perm = np.concatenate([blocks[i] for i in order], axis=0)
            if n_full < n:
                out[a:b] = np.concatenate([perm, seg[n_full:]], axis=0)
            else:
                out[a:b] = perm
        else:
            raise ValueError(f"Unknown null kind: {kind}")
    return out


def corrected_empirical_p(obs: float, nulls: np.ndarray,
                          alternative: str = "greater") -> float:
    """Corrected empirical p: (exceedances + 1) / (n_perm + 1)."""
    nulls = np.asarray(nulls, dtype=float)
    nulls = nulls[np.isfinite(nulls)]
    if not np.isfinite(obs) or nulls.size == 0:
        return float("nan")
    if alternative == "greater":
        exceed = int(np.sum(nulls >= obs))
    elif alternative == "less":
        exceed = int(np.sum(nulls <= obs))
    else:
        exceed = int(np.sum(np.abs(nulls) >= abs(obs)))
    return float((exceed + 1) / (nulls.size + 1))


def load_channel_allowlist(path: Path) -> Dict[str, set]:
    """Load subject→channel allowlist from a CSV with subject,channel columns."""
    df = pd.read_csv(path)
    if "subject" not in df.columns or "channel" not in df.columns:
        raise ValueError(
            f"{path} must contain 'subject' and 'channel' columns")
    out: Dict[str, set] = {}
    for subj, g in df.groupby("subject"):
        out[str(subj)] = set(g["channel"].astype(str).str.strip())
    return out


def process_subject(
    subject: str, eeg_files: Dict[int, str], args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eeg_files_abs = {sid: str(ap.DATA_ROOT / p) for sid, p in eeg_files.items()}
    missing = [p for p in eeg_files_abs.values() if not Path(p).exists()]
    if missing:
        print(f"[{subject}] missing EEG, skip")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data = ec.load_subject_data(
        subject_name=subject, eeg_files=eeg_files_abs,
        features_dir=ap.FEATURES_DIR, fs_target=args.fs_target,
        sections=tuple(args.sections), feature_sets=["glove"],
    )
    X_sae, surprisal, feat_valid = load_sae_features(
        args.feature_tag, args.sections)
    wt = load_word_timing_concat(args.sections)
    Wtot = len(wt)
    if X_sae.shape[0] != Wtot:
        raise ValueError(f"[{subject}] feature/word_timing mismatch")

    nuisance, nuis_names = build_nuisance(wt)
    lag_step = getattr(args, "lag_step_ms", LAG_STEP_MS)
    lags = lag_samples(args.fs_ds, args.lag_start_ms, args.lag_end_ms, lag_step)
    guard_w = max(1, int(np.ceil(args.lag_end_ms / max(lag_step, 1.0))))

    # Continuous EEG per section → downsample → concatenate with section offsets
    eeg_ds_sections: List[np.ndarray] = []
    event_samples_global: List[int] = []
    section_time_slices: List[Tuple[int, int]] = []
    t_cursor = 0
    word_cursor = 0
    factor = int(round(args.fs_target / args.fs_ds))

    for sec, (w0, w1) in zip(data.sections, data.section_word_slices):
        eeg = eeg_from_cumsum(sec.eeg_cs)  # (T, C)
        eeg_ds = downsample_eeg(eeg, args.fs_target, args.fs_ds)
        eeg_ds_sections.append(eeg_ds)
        # Map word onsets (in original samples) to downsampled indices.
        on = sec.onsets  # length W for this section
        assert w1 - w0 == len(on) == sec.W
        for oi in on:
            event_samples_global.append(t_cursor + int(oi // factor))
        section_time_slices.append((t_cursor, t_cursor + eeg_ds.shape[0]))
        t_cursor += eeg_ds.shape[0]
        word_cursor = w1

    eeg_all = np.vstack(eeg_ds_sections)  # (T_all, C)
    event_samples = np.asarray(event_samples_global, dtype=int)
    assert len(event_samples) == Wtot

    # Word-level validity
    word_ok = feat_valid & np.isfinite(surprisal)
    # Also require event sample in range with room for max lag
    T_all = eeg_all.shape[0]
    word_ok = word_ok & (event_samples >= 0) & (event_samples + lags[-1] < T_all)

    # Optional whole-run null of neural signal (legacy single-perm path).
    null_kind = getattr(args, "null", "none")
    null_seed = int(getattr(args, "null_seed", RANDOM_STATE))
    if null_kind in ("circular", "block") and int(getattr(args, "null_perms", 0) or 0) <= 0:
        eeg_all = apply_null_shift(
            eeg_all, section_time_slices, null_kind,
            seed=null_seed + sum(ord(c) for c in subject),
        )

    channels = list(data.channel_labels)
    allowlist = getattr(args, "_channel_allowlist", None)
    if allowlist is not None:
        keep = allowlist.get(subject, set())
        channels = [c for c in channels if str(c).strip() in keep]
        print(f"  [{subject}] channels_from_csv -> {len(channels)} channels",
              flush=True)
        if not channels:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        roi_filter = getattr(args, "roi_filter", "all")
        if getattr(args, "lang_only", False) and roi_filter == "all":
            roi_filter = "lang"
        if getattr(args, "mfg_only", False):
            roi_filter = "mfg"
        if roi_filter != "all":
            tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
            tax_s = tax[tax["subject"] == subject]
            lang_s = tax_s["is_lang"].fillna(False)
            if roi_filter == "mfg":
                keep = set(
                    tax_s.loc[
                        lang_s & tax_s["region"].astype(str).eq(LEFT_MFG_REGION),
                        "channel",
                    ].astype(str)
                )
            elif roi_filter == "temporal":
                keep = set(
                    tax_s.loc[
                        lang_s & tax_s["region"].astype(str).str.contains(
                            "Temp|STG|MTG|ITG|STS|temporal|Superior temporal|Middle temporal",
                            case=False, na=False),
                        "channel",
                    ].astype(str)
                )
            elif roi_filter == "mfg_temporal":
                mfg = lang_s & tax_s["region"].astype(str).eq(LEFT_MFG_REGION)
                temp = lang_s & tax_s["region"].astype(str).str.contains(
                    "Temp|STG|MTG|ITG|STS|temporal|Superior temporal|Middle temporal",
                    case=False, na=False)
                keep = set(tax_s.loc[mfg | temp, "channel"].astype(str))
            else:  # lang
                keep = set(tax_s.loc[lang_s, "channel"].astype(str))
            channels = [c for c in channels if c in keep]
            print(f"  [{subject}] filtered to {len(channels)} "
                  f"{roi_filter} channels", flush=True)
    if args.max_channels:
        channels = channels[: args.max_channels]

    # Contiguous folds over words (buffered)
    folds = make_buffered_contiguous_splits(
        Wtot, args.n_folds, list(data.section_word_slices), word_ok, guard_w,
    )
    if not folds:
        print(f"[{subject}] no valid folds")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows = []
    kernel_rows = []
    null_rows = []
    t0 = time.time()
    n_ch = len(channels)
    n_lags = len(lags)
    lag_ms = (lags.astype(float) / args.fs_ds) * 1000.0
    n_null = int(getattr(args, "null_perms", 0) or 0)
    print(f"  [{subject}] deconv {n_ch} ch, T={T_all}, lags={n_lags}, "
          f"valid_words={int(word_ok.sum())}, n_folds={len(folds)}, "
          f"null_perms={n_null}", flush=True)

    for ci, ch in enumerate(channels):
        # Map channel label → column in eeg_all
        try:
            eeg_ci = list(data.channel_labels).index(ch)
        except ValueError:
            eeg_ci = ci
        y_obs = eeg_all[:, eeg_ci]
        fold_scores = {
            "nuisance": [], "surprisal": [], "full": [],
            "nuisance_event": [], "surprisal_event": [], "full_event": [],
            "sae_gain": [], "surp_gain": [],
        }
        alphas = []
        n_sae_sel = []
        fold_packs = []  # frozen designs for fast null calibration
        k_surp_acc = []
        k_sae_acc = []
        k_nuis_acc = []
        t_ch = time.time()

        for fi, (tr_words, te_words) in enumerate(folds):
            t_fold = time.time()
            Y_ev_tr = event_locked_matrix(y_obs, event_samples[tr_words], lags)
            ok_ev = np.all(np.isfinite(Y_ev_tr), axis=1)
            if ok_ev.sum() < 30:
                continue
            tr_sel = tr_words[ok_ev]
            Y_sel = Y_ev_tr[ok_ev]
            support = select_sae_support_joint(
                X_sae[tr_sel], Y_sel,
                k_best=args.k_best, n_keep=args.n_sae_keep,
                mtl_width=getattr(args, "mtl_width", MTL_WIDTH),
            )
            n_sae_sel.append(int(support.sum()))
            n_sae = int(support.sum())

            sae_tr = X_sae[tr_sel][:, support].toarray() if n_sae else np.zeros((tr_sel.size, 0))
            sae_scaler = StandardScaler().fit(sae_tr) if n_sae else None
            surp_scaler = StandardScaler().fit(surprisal[tr_sel].reshape(-1, 1))
            nuis_scaler = StandardScaler().fit(nuisance[tr_sel])

            all_idx = np.flatnonzero(word_ok)
            surp_v = surp_scaler.transform(surprisal[all_idx].reshape(-1, 1)).ravel()
            nuis_v = nuis_scaler.transform(nuisance[all_idx])
            if n_sae:
                sae_v = sae_scaler.transform(X_sae[all_idx][:, support].toarray())
            else:
                sae_v = np.zeros((all_idx.size, 0))
            ev = event_samples[all_idx]
            n_nuis = nuis_v.shape[1]

            X_nuis = fir_expand_events(T_all, ev, nuis_v, lags)
            X_surp = fir_expand_events(T_all, ev, surp_v, lags)
            X_sae_fir = (fir_expand_events(T_all, ev, sae_v, lags)
                         if n_sae else np.zeros((T_all, 0)))

            guard_samp = int(lags[-1])
            tr_time = np.zeros(T_all, dtype=bool)
            te_time = np.zeros(T_all, dtype=bool)

            def cover(idxs, out_mask):
                for t0w in event_samples[idxs]:
                    a = max(0, int(t0w) - guard_samp)
                    b = min(T_all, int(t0w) + guard_samp + 1)
                    out_mask[a:b] = True

            cover(tr_words, tr_time)
            cover(te_words, te_time)
            both = tr_time & te_time
            tr_time[both] = False
            te_time[both] = False
            if tr_time.sum() < 100 or te_time.sum() < 50:
                continue

            designs = {
                "nuisance": X_nuis,
                "surprisal": np.hstack([X_nuis, X_surp]),
                "full": np.hstack([X_nuis, X_surp, X_sae_fir]) if n_sae
                        else np.hstack([X_nuis, X_surp]),
            }
            te_idx = np.flatnonzero(te_time)
            max_tp = getattr(args, "max_train_timepoints", MAX_TRAIN_TIMEPOINTS)
            if te_idx.size > max_tp:
                te_idx = np.sort(np.random.default_rng(RANDOM_STATE + fi).choice(
                    te_idx, size=max_tp, replace=False))
            preds = {}
            col_ok_full = None
            coef_full = None
            for name, Xd in designs.items():
                col_ok = np.any(Xd[tr_time] != 0, axis=0)
                if not col_ok.any():
                    preds[name] = np.zeros(te_idx.size)
                    continue
                pred, alpha, coef = fit_ridge(
                    Xd[tr_time][:, col_ok], y_obs[tr_time],
                    Xd[te_idx][:, col_ok],
                    max_train=max_tp,
                    seed=RANDOM_STATE + 17 * eeg_ci + fi,
                )
                preds[name] = pred
                if name == "full":
                    alphas.append(alpha)
                    col_ok_full = col_ok
                    coef_full = expand_coef_to_full(coef, col_ok)

            y_te = y_obs[te_idx]
            r_n = _score_r(y_te, preds["nuisance"])
            r_s = _score_r(y_te, preds["surprisal"])
            r_f = _score_r(y_te, preds["full"])
            fold_scores["nuisance"].append(r_n)
            fold_scores["surprisal"].append(r_s)
            fold_scores["full"].append(r_f)
            fold_scores["sae_gain"].append(r_f - r_s)
            fold_scores["surp_gain"].append(r_s - r_n)

            if coef_full is not None:
                kern = kernels_from_full_coef(coef_full, n_lags, n_nuis, n_sae)
                k_surp_acc.append(kern["surprisal"])
                k_sae_acc.append(kern["sae_l2"])
                k_nuis_acc.append(kern["nuisance_l2"])

            peak_lag = lags[len(lags) // 3]
            best = -np.inf
            for lag in lags:
                tt = event_samples[tr_sel] + lag
                good = (tt >= 0) & (tt < T_all)
                if good.sum() < 20:
                    continue
                r, _ = pearsonr(y_obs[tt[good]], surprisal[tr_sel][good])
                if np.isfinite(r) and r > best:
                    best = float(r)
                    peak_lag = lag

            te_ok = te_words[np.isin(te_words, all_idx)]
            for name, Xd in designs.items():
                col_ok = np.any(Xd[tr_time] != 0, axis=0)
                if not col_ok.any():
                    fold_scores[f"{name}_event"].append(0.0)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, alpha_e, _ = fit_ridge(
                        Xd[tr_time][:, col_ok], y_obs[tr_time],
                        Xd[tr_time][:1][:, col_ok],
                        max_train=max_tp,
                        seed=RANDOM_STATE + 31 * eeg_ci + fi,
                    )
                    tr_sub = _subsample_rows(
                        int(tr_time.sum()), max_tp,
                        np.random.default_rng(RANDOM_STATE + 31 * eeg_ci + fi),
                    )
                    tr_rows = np.flatnonzero(tr_time)[tr_sub]
                    model = Ridge(alpha=alpha_e, fit_intercept=True).fit(
                        Xd[tr_rows][:, col_ok], y_obs[tr_rows])
                tt = event_samples[te_ok] + peak_lag
                good = (tt >= 0) & (tt < T_all)
                if good.sum() < 5:
                    fold_scores[f"{name}_event"].append(0.0)
                    continue
                pred_e = model.predict(Xd[tt[good]][:, col_ok]).ravel()
                fold_scores[f"{name}_event"].append(
                    _score_r(y_obs[tt[good]], pred_e))

            # Cache fold design for fast frozen-support null calibration.
            fold_packs.append({
                "designs": designs,
                "tr_time": tr_time,
                "te_idx": te_idx,
                "te_ok": te_ok,
                "peak_lag": peak_lag,
            })
            if ci == 0:
                print(
                    f"      fold {fi+1}/{len(folds)} n_sae={n_sae} "
                    f"fullR={r_f:+.4f} ({time.time()-t_fold:.1f}s)",
                    flush=True,
                )

        def agg(xs):
            xs = [x for x in xs if np.isfinite(x)]
            return float(np.mean(xs)) if xs else np.nan

        row = {
            "subject": subject,
            "channel": ch,
            "channel_index": eeg_ci,
            "feature_tag": args.feature_tag,
            "null": null_kind if n_null <= 0 else "none",
            "null_seed": null_seed if n_null <= 0 and null_kind != "none" else -1,
            "n_folds": len(fold_scores["full"]),
            "n_sae_mean": float(np.mean(n_sae_sel)) if n_sae_sel else 0.0,
            "alpha_median": float(np.median(alphas)) if alphas else np.nan,
            "nuisance__R": agg(fold_scores["nuisance"]),
            "surprisal__R": agg(fold_scores["surprisal"]),
            "full__R": agg(fold_scores["full"]),
            "sae_gain": agg(fold_scores["sae_gain"]),
            "surprisal_gain": agg(fold_scores["surp_gain"]),
            "nuisance__R_event": agg(fold_scores["nuisance_event"]),
            "surprisal__R_event": agg(fold_scores["surprisal_event"]),
            "full__R_event": agg(fold_scores["full_event"]),
            "n_valid_words": int(word_ok.sum()),
        }
        rows.append(row)

        # Fold-averaged temporal kernels (main observed fit).
        if k_surp_acc:
            surp_k = np.mean(np.stack(k_surp_acc, axis=0), axis=0)
            sae_k = np.mean(np.stack(k_sae_acc, axis=0), axis=0)
            nuis_k = np.mean(np.stack(k_nuis_acc, axis=0), axis=0)
            for li in range(n_lags):
                kernel_rows.append({
                    "subject": subject,
                    "channel": ch,
                    "feature_tag": args.feature_tag,
                    "lag_ms": float(lag_ms[li]),
                    "kernel_surprisal": float(surp_k[li]),
                    "kernel_sae_l2": float(sae_k[li]),
                    "kernel_nuisance_l2": float(nuis_k[li]),
                })

        # Nulls: either fast frozen-support (exploratory) or refit-support
        # confirmatory permutations (slower; training-only SAE reselection).
        refit_support = bool(getattr(args, "null_refit_support", False))
        block_len = int(getattr(args, "null_block_len", 0) or 0)
        if n_null > 0 and fold_packs and null_kind in ("circular", "block"):
            max_tp = getattr(args, "max_train_timepoints", MAX_TRAIN_TIMEPOINTS)
            for pi in range(n_null):
                seed_i = null_seed + 10007 * pi + sum(ord(c) for c in subject)
                y_null = apply_null_shift(
                    y_obs.reshape(-1, 1), section_time_slices, null_kind,
                    seed=seed_i, block_len=block_len,
                )[:, 0]
                pg, ps, pf = [], [], []
                if refit_support:
                    # Confirmatory: rebuild SAE support on null Y each fold.
                    for fi, (tr_words, te_words) in enumerate(folds):
                        Y_ev_tr = event_locked_matrix(
                            y_null, event_samples[tr_words], lags)
                        ok_ev = np.all(np.isfinite(Y_ev_tr), axis=1)
                        if ok_ev.sum() < 30:
                            continue
                        tr_sel = tr_words[ok_ev]
                        support = select_sae_support_joint(
                            X_sae[tr_sel], Y_ev_tr[ok_ev],
                            k_best=args.k_best, n_keep=args.n_sae_keep,
                            mtl_width=getattr(args, "mtl_width", MTL_WIDTH),
                        )
                        n_sae = int(support.sum())
                        sae_tr = (X_sae[tr_sel][:, support].toarray()
                                  if n_sae else np.zeros((tr_sel.size, 0)))
                        sae_scaler = (StandardScaler().fit(sae_tr)
                                      if n_sae else None)
                        surp_scaler = StandardScaler().fit(
                            surprisal[tr_sel].reshape(-1, 1))
                        nuis_scaler = StandardScaler().fit(nuisance[tr_sel])
                        all_idx = np.flatnonzero(word_ok)
                        surp_v = surp_scaler.transform(
                            surprisal[all_idx].reshape(-1, 1)).ravel()
                        nuis_v = nuis_scaler.transform(nuisance[all_idx])
                        sae_v = (sae_scaler.transform(
                            X_sae[all_idx][:, support].toarray())
                            if n_sae else np.zeros((all_idx.size, 0)))
                        ev = event_samples[all_idx]
                        X_nuis = fir_expand_events(T_all, ev, nuis_v, lags)
                        X_surp = fir_expand_events(T_all, ev, surp_v, lags)
                        X_sae_fir = (fir_expand_events(T_all, ev, sae_v, lags)
                                     if n_sae else np.zeros((T_all, 0)))
                        guard_samp = int(lags[-1])
                        tr_time = np.zeros(T_all, dtype=bool)
                        te_time = np.zeros(T_all, dtype=bool)

                        def cover(idxs, out_mask):
                            for t0w in event_samples[idxs]:
                                aa = max(0, int(t0w) - guard_samp)
                                bb = min(T_all, int(t0w) + guard_samp + 1)
                                out_mask[aa:bb] = True

                        cover(tr_words, tr_time)
                        cover(te_words, te_time)
                        both = tr_time & te_time
                        tr_time[both] = False
                        te_time[both] = False
                        if tr_time.sum() < 100 or te_time.sum() < 50:
                            continue
                        designs = {
                            "nuisance": X_nuis,
                            "surprisal": np.hstack([X_nuis, X_surp]),
                            "full": (np.hstack([X_nuis, X_surp, X_sae_fir])
                                     if n_sae else np.hstack([X_nuis, X_surp])),
                        }
                        te_idx = np.flatnonzero(te_time)
                        if te_idx.size > max_tp:
                            te_idx = np.sort(
                                np.random.default_rng(RANDOM_STATE + fi).choice(
                                    te_idx, size=max_tp, replace=False))
                        rs = {}
                        for name, Xd in designs.items():
                            col_ok = np.any(Xd[tr_time] != 0, axis=0)
                            if not col_ok.any():
                                rs[name] = 0.0
                                continue
                            pred, _, _ = fit_ridge(
                                Xd[tr_time][:, col_ok], y_null[tr_time],
                                Xd[te_idx][:, col_ok],
                                max_train=max_tp,
                                seed=RANDOM_STATE + 41 * pi + eeg_ci,
                            )
                            rs[name] = _score_r(y_null[te_idx], pred)
                        pg.append(rs["full"] - rs["surprisal"])
                        ps.append(rs["surprisal"] - rs["nuisance"])
                        pf.append(rs["full"])
                else:
                    for pack in fold_packs:
                        designs = pack["designs"]
                        tr_time = pack["tr_time"]
                        te_idx = pack["te_idx"]
                        rs = {}
                        for name, Xd in designs.items():
                            col_ok = np.any(Xd[tr_time] != 0, axis=0)
                            if not col_ok.any():
                                rs[name] = 0.0
                                continue
                            pred, _, _ = fit_ridge(
                                Xd[tr_time][:, col_ok], y_null[tr_time],
                                Xd[te_idx][:, col_ok],
                                max_train=max_tp,
                                seed=RANDOM_STATE + 41 * pi + eeg_ci,
                            )
                            rs[name] = _score_r(y_null[te_idx], pred)
                        pg.append(rs["full"] - rs["surprisal"])
                        ps.append(rs["surprisal"] - rs["nuisance"])
                        pf.append(rs["full"])
                null_rows.append({
                    "subject": subject,
                    "channel": ch,
                    "feature_tag": args.feature_tag,
                    "null": null_kind,
                    "null_mode": "refit_support" if refit_support else "frozen_support",
                    "perm": pi,
                    "null_seed": null_seed + 10007 * pi,
                    "full__R": agg(pf),
                    "sae_gain": agg(pg),
                    "surprisal_gain": agg(ps),
                    "obs_sae_gain": row["sae_gain"],
                    "obs_surprisal_gain": row["surprisal_gain"],
                    "obs_full__R": row["full__R"],
                })

        print(
            f"    [{subject}] ch {ci+1}/{n_ch} {ch} "
            f"fullR={row['full__R']:+.4f} sae_gain={row['sae_gain']:+.4f} "
            f"in {time.time()-t_ch:.1f}s",
            flush=True,
        )
        if (ci + 1) % 5 == 0 or ci + 1 == n_ch:
            print(_progress_line(subject, ci + 1, n_ch, t0, kept=len(rows)),
                  flush=True)

    return (
        pd.DataFrame(rows),
        pd.DataFrame(kernel_rows),
        pd.DataFrame(null_rows),
    )


def write_deconv_report(df: pd.DataFrame, path: Path):
    lines = ["=" * 70, "CONTINUOUS-TIME DECONVOLUTION SUMMARY", "=" * 70, ""]
    if df.empty:
        lines.append("(empty)")
        path.write_text("\n".join(lines))
        return
    tax = merge_taxonomy(df)
    lang = language_mask(tax)
    mfg = lang & tax["region"].astype(str).eq(LEFT_MFG_REGION)

    def block(sub, title):
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)
        if sub.empty:
            lines.append("  (empty)\n")
            return
        lines.append(
            f"Electrodes={len(sub)}  subjects={sub['subject'].nunique()}")
        for col in ("nuisance__R", "surprisal__R", "full__R",
                    "sae_gain", "surprisal_gain"):
            v = sub[col].dropna()
            if len(v):
                lines.append(f"  {col:18s} mean={v.mean():+.4f}  "
                             f"median={v.median():+.4f}")
        # Subject-level
        g = sub.groupby("subject")[["full__R", "sae_gain", "surprisal_gain"]].mean()
        lines.append("Subject means:")
        lines.append(g.round(4).to_string())
        lines.append("")

    block(tax, "ALL ELECTRODES")
    block(tax.loc[lang], "LOCALIZER-POSITIVE (is_lang)")
    block(tax.loc[mfg], f"LEFT MFG ({LEFT_MFG_REGION}) — underpowered extension")
    # Temporal-ish
    if "region" in tax.columns:
        temp = tax.loc[lang & tax["region"].astype(str).str.contains(
            "Temp|STG|MTG|ITG|STS|temporal", case=False, na=False)]
        block(temp, "LOCALIZER-POSITIVE TEMPORAL-ish")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--fs_ds", type=float, default=FS_DS)
    p.add_argument("--lag_start_ms", type=float, default=LAG_START_MS)
    p.add_argument("--lag_end_ms", type=float, default=LAG_END_MS)
    p.add_argument("--lag_step_ms", type=float, default=LAG_STEP_MS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--max_channels", type=int, default=0)
    p.add_argument("--k_best", type=int, default=LASSO_K_BEST)
    p.add_argument("--n_sae_keep", type=int, default=N_SAE_KEEP)
    p.add_argument("--mtl_width", type=int, default=MTL_WIDTH)
    p.add_argument("--max_train_timepoints", type=int, default=MAX_TRAIN_TIMEPOINTS)
    p.add_argument("--lang_only", action="store_true",
                   help="Restrict to localizer-positive (is_lang) channels.")
    p.add_argument("--mfg_only", action="store_true",
                   help="Restrict to localizer-positive left MFG channels.")
    p.add_argument(
        "--roi_filter",
        choices=("all", "lang", "mfg", "temporal", "mfg_temporal"),
        default="all",
        help="Electrode ROI filter (lang_only/mfg_only are aliases).",
    )
    p.add_argument("--null", choices=("none", "circular", "block"), default="none",
                   help="Null type. With --null_perms>0, runs null permutations "
                        "after the observed fit (frozen-support by default).")
    p.add_argument("--null_seed", type=int, default=RANDOM_STATE,
                   help="Base seed for null shifts (vary across permutations).")
    p.add_argument("--null_perms", type=int, default=0,
                   help="Number of circular/block null permutations. Default 0.")
    p.add_argument("--null_block_len", type=int, default=0,
                   help="Block length (samples at fs_ds) for true block "
                        "permutation. 0 = auto (~10%% of section).")
    p.add_argument("--null_refit_support", action="store_true",
                   help="Confirmatory nulls: reselect training-only SAE support "
                        "under each permutation (slow). Frozen-support results "
                        "without this flag are exploratory.")
    p.add_argument(
        "--channels_from_csv", default="",
        help="CSV with subject,channel columns. Restricts analysis to this "
             "explicit channel list (ignores is_lang / ROI filters).",
    )
    p.add_argument("--out_suffix", default="")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    subjects_map = get_subjects()
    requested = args.subjects or list(subjects_map.keys())
    args._channel_allowlist = None
    if args.channels_from_csv:
        allow = load_channel_allowlist(Path(args.channels_from_csv))
        args._channel_allowlist = allow
        # Restrict to subjects present in the allowlist when not specified.
        if args.subjects is None:
            requested = [s for s in requested if s in allow]
        print(f"Channel allowlist: {sum(len(v) for v in allow.values())} "
              f"channels across {len(allow)} subjects")
    suffix = args.out_suffix
    if args.null != "none" and int(args.null_perms or 0) <= 0 and args.null not in suffix:
        suffix = f"{suffix}_null_{args.null}"
        if args.null_seed != RANDOM_STATE:
            suffix = f"{suffix}_seed{args.null_seed}"

    score_dfs, kern_dfs, null_dfs = [], [], []
    t0 = time.time()
    for i, sid in enumerate(requested, 1):
        if sid not in subjects_map:
            print(f"[skip] {sid}")
            continue
        print(f"\n=== [{i}] {sid} ===", flush=True)
        scores, kernels, nulls = process_subject(
            sid, subjects_map[sid]["eeg_files"], args)
        if not scores.empty:
            score_dfs.append(scores)
        if not kernels.empty:
            kern_dfs.append(kernels)
        if not nulls.empty:
            null_dfs.append(nulls)
    if not score_dfs:
        print("No deconvolution results.")
        return
    out = pd.concat(score_dfs, ignore_index=True)
    out_path = ap.SAE_TABLES / f"sparse_encoding_deconv{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows -> {out_path} ({_fmt_duration(time.time()-t0)})")
    write_deconv_report(
        out, ap.SAE_REPORTS / f"sparse_encoding_deconv{suffix}.txt")

    if kern_dfs:
        kdf = pd.concat(kern_dfs, ignore_index=True)
        k_path = ap.SAE_TABLES / f"sparse_encoding_kernels{suffix}.csv"
        kdf.to_csv(k_path, index=False)
        # Also save a compact NPZ of mean kernels by lag for quick plotting.
        npz_path = ap.SAE_TABLES / f"sparse_encoding_kernels{suffix}.npz"
        lags_u = np.sort(kdf["lag_ms"].unique())
        surp = kdf.groupby("lag_ms")["kernel_surprisal"].mean().reindex(lags_u).to_numpy()
        sae = kdf.groupby("lag_ms")["kernel_sae_l2"].mean().reindex(lags_u).to_numpy()
        nuis = kdf.groupby("lag_ms")["kernel_nuisance_l2"].mean().reindex(lags_u).to_numpy()
        np.savez_compressed(npz_path, lag_ms=lags_u, surprisal=surp, sae_l2=sae,
                            nuisance_l2=nuis)
        print(f"Saved kernels -> {k_path} and {npz_path}")

        # Per-electrode kernel shape metrics (peak / FWHM / polarity).
        shapes = kernel_shape_table(kdf)
        if not shapes.empty:
            s_path = ap.SAE_TABLES / f"sparse_encoding_kernel_shapes{suffix}.csv"
            shapes.to_csv(s_path, index=False)
            print(f"Saved kernel shapes -> {s_path}")
            # Append kernel timing block to the deconv report.
            rep_path = ap.SAE_REPORTS / f"sparse_encoding_deconv{suffix}.txt"
            extra = [
                "",
                "-" * 70,
                "TEMPORAL KERNEL SHAPE (subject-mean of electrode peaks)",
                "-" * 70,
            ]
            for col, label in (
                ("surprisal_peak_lag_ms", "surprisal peak lag"),
                ("sae_l2_peak_lag_ms", "SAE L2 peak lag"),
                ("surprisal_fwhm_ms", "surprisal FWHM"),
                ("sae_l2_fwhm_ms", "SAE L2 FWHM"),
            ):
                if col not in shapes.columns:
                    continue
                subj = shapes.groupby("subject")[col].mean()
                extra.append(
                    f"  {label:24s} median={np.nanmedian(shapes[col]):.1f}  "
                    f"subject-mean={subj.mean():+.1f}  nS={subj.notna().sum()}"
                )
            if rep_path.exists():
                rep_path.write_text(rep_path.read_text() + "\n".join(extra) + "\n")

        # Kernel visualization suite (best-effort).
        try:
            from sparse_encoding.sparse_encoding_plot_kernels import (
                fig_anterior_posterior, fig_grand_average, fig_heatmap,
                fig_surprisal_vs_sae, _merge_anat,
            )
            kplot = _merge_anat(kdf)
            print("Kernel figures:")
            fig_grand_average(kplot, ap.SAE_FIGURES, suffix)
            fig_heatmap(kplot, ap.SAE_FIGURES, suffix)
            fig_surprisal_vs_sae(kplot, ap.SAE_FIGURES, suffix)
            fig_anterior_posterior(kplot, ap.SAE_FIGURES, suffix)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] kernel plots skipped: {exc}", flush=True)

    if null_dfs:
        ndf = pd.concat(null_dfs, ignore_index=True)
        n_path = ap.SAE_TABLES / f"sparse_encoding_deconv_nulls{suffix}.csv"
        n_mode = ("refit_support" if args.null_refit_support
                  else "frozen_support")
        lines = [
            "NULL CALIBRATION",
            "=" * 60, "",
            f"null={args.null}  n_perm={args.null_perms}  "
            f"null_seed={args.null_seed}  mode={n_mode}",
            f"null_block_len={args.null_block_len or 'auto'}",
        ]
        if n_mode == "frozen_support":
            lines.append(
                "EXPLORATORY: support/designs frozen from observed folds; "
                "only Y is shifted.")
        else:
            lines.append(
                "CONFIRMATORY: training-only SAE support reselected under "
                "each permutation.")
        lines.append(
            "p-values use corrected empirical form "
            "(exceedances + 1) / (n_perm + 1).")
        lines.append("")
        p_rows = []
        for (subj, ch), g in ndf.groupby(["subject", "channel"]):
            obs_sae = float(g["obs_sae_gain"].iloc[0])
            obs_surp = float(g["obs_surprisal_gain"].iloc[0]) if "obs_surprisal_gain" in g.columns else np.nan
            null_sae = g["sae_gain"].to_numpy(dtype=float)
            null_surp = g["surprisal_gain"].to_numpy(dtype=float)
            p_sae = corrected_empirical_p(obs_sae, null_sae, "greater")
            p_surp = corrected_empirical_p(obs_surp, null_surp, "greater")
            p_rows.append({
                "subject": subj, "channel": ch,
                "obs_sae_gain": obs_sae, "p_sae_gain": p_sae,
                "obs_surprisal_gain": obs_surp, "p_surprisal_gain": p_surp,
                "null_mode": n_mode, "n_perm": int(g["perm"].nunique()),
            })
            lines.append(
                f"{subj} {ch}: sae_gain={obs_sae:+.4f} p={p_sae:.3f}  "
                f"surp_gain={obs_surp:+.4f} p={p_surp:.3f}")
        pdf = pd.DataFrame(p_rows)
        if not pdf.empty and "p_surprisal_gain" in pdf.columns:
            # FDR across tested candidate channels for surprisal gain
            def _fdr_bh(pvals):
                p = np.asarray(pvals, dtype=float)
                out = np.full_like(p, np.nan)
                ok = np.isfinite(p)
                if not ok.any():
                    return out
                pv = p[ok]
                n = len(pv)
                order = np.argsort(pv)
                ranked = pv[order]
                q = ranked * n / (np.arange(1, n + 1))
                q = np.minimum.accumulate(q[::-1])[::-1]
                q = np.clip(q, 0, 1)
                out_ok = np.empty(n)
                out_ok[order] = q
                out[ok] = out_ok
                return out
            pdf["q_surprisal_gain"] = _fdr_bh(pdf["p_surprisal_gain"].to_numpy())
            pdf["q_sae_gain"] = _fdr_bh(pdf["p_sae_gain"].to_numpy())
        p_path = ap.SAE_TABLES / f"sparse_encoding_deconv_null_pvals{suffix}.csv"
        pdf.to_csv(p_path, index=False)
        ndf.to_csv(n_path, index=False)
        # Aggregate subject-mean nulls (SAE gain)
        obs_subj = (ndf.drop_duplicates(["subject", "channel"])
                      .groupby("subject")["obs_sae_gain"].mean())
        null_subj = []
        for pi, g in ndf.groupby("perm"):
            null_subj.append(g.groupby("subject")["sae_gain"].mean().mean())
        null_subj = np.asarray(null_subj, dtype=float)
        obs_mean = float(obs_subj.mean()) if len(obs_subj) else np.nan
        p_all = corrected_empirical_p(obs_mean, null_subj, "greater")
        lines.append("")
        lines.append(
            f"Subject-mean SAE gain: obs={obs_mean:+.4f}  "
            f"null_mean={np.nanmean(null_subj):+.4f}  "
            f"p_ge_obs={p_all:.3f}  n_perm={len(null_subj)}"
        )
        if not pdf.empty and "q_surprisal_gain" in pdf.columns:
            lines.append("")
            lines.append("Channel-level FDR (surprisal_gain):")
            for _, r in pdf.sort_values("p_surprisal_gain").iterrows():
                lines.append(
                    f"  {r['subject']} {r['channel']}: "
                    f"p={r['p_surprisal_gain']:.3f} q={r['q_surprisal_gain']:.3f}")
        rep = ap.SAE_REPORTS / f"sparse_encoding_deconv_nulls{suffix}.txt"
        rep.write_text("\n".join(lines))
        print(f"Saved nulls -> {n_path}")
        print(f"Saved p-values -> {p_path}")
        print(f"Wrote {rep}")


if __name__ == "__main__":
    main()
