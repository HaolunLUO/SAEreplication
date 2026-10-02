#!/usr/bin/env python3
"""SAE replication v3: lag-resolved Lepori readout on Qwen3.5-4B-Base L15.

Primary CV is leave-one-section-out. Inside each training fold the LASSO
support is chosen once, at the lag that maximizes training-fold full-model
Fisher-z, and that support is Ridge-refit at every lag. The test fold never
chooses the lag or the support.

Outputs go only to ``group_encoding_results/sparse_encoding_v3/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_regression import (
    LASSO_K_BEST,
    RANDOM_STATE,
    _fit_predict,
    _fold_r,
    _fmt_duration,
    _to_dense,
    compute_joint_support,
    get_subjects,
    load_residual_features,
    load_sae_features,
    maybe_shuffle_features,
)

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------
FEATURE_TAG = "sae_qwen35_4b_mat_l15_v2"
FS_TARGET = 500.0
RESP_WIN_MS = 200.0
LAG_START_MS = -500.0
LAG_END_MS = 2000.0
LAG_STEP_MS = 50.0
SECTIONS = (1, 2, 3)
N_CONTIGUOUS_FOLDS = 5
N_NULL = 20
CIRC_SHIFT_FRACTIONS = (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0)
BIN_SLICES = (
    ("bin0-2048", 0, 2048),
    ("bin2048-16384", 2048, 16384),
    ("bin16384-end", 16384, 65536),
)
DENSE_RIDGE_ALPHA = 10000.0
PREFER_LAG_MS = 300.0
READOUT_MATCHED = "matched_lasso"
READOUT_DENSE = "dense_ridge_alpha10000"
SPACES_MAIN = ("sae", "residual", "surprisal")


def lag_grid_ms() -> np.ndarray:
    lags = np.arange(
        LAG_START_MS, LAG_END_MS + 0.5 * LAG_STEP_MS, LAG_STEP_MS, dtype=float,
    )
    if lags.size != 51 or lags[0] != -500.0 or lags[-1] != 2000.0:
        raise RuntimeError(f"lag grid is not the v3 protocol: {lags.size} {lags[:2]} {lags[-1]}")
    return lags


def window_sample_bounds(lag_ms: float, win_ms: float = RESP_WIN_MS, fs: float = FS_TARGET):
    """Half-open window centered on ``lag_ms``.

    At lag 300 ms and a 200 ms window this is samples [100, 200) at 500 Hz,
    i.e. [200, 400) ms after onset. Matches ``encoding_channel.compute_Y_for_lag``.
    """
    half = int(round((win_ms / 1000.0) * fs / 2.0))
    lag = int(round((lag_ms / 1000.0) * fs))
    return lag - half, lag + half, half


def v3_root() -> Path:
    root = (ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v3").resolve()
    if root.name != "sparse_encoding_v3":
        raise RuntimeError(f"refusing to write outside sparse_encoding_v3: {root}")
    banned = {"sparse_encoding", "sparse_encoding_v2"}
    if root.name in banned or any(p.name in banned for p in root.parents):
        raise RuntimeError(f"refusing to write into v1/v2: {root}")
    return root


def _dirs():
    root = v3_root()
    tables = root / "tables"
    reports = root / "reports"
    figures = root / "figures"
    logs = root / "logs"
    for d in (tables, reports, figures, logs):
        d.mkdir(parents=True, exist_ok=True)
    return root, tables, reports, figures, logs


def fisher_z_mean(rs: Sequence[float]) -> float:
    """Mean Fisher-z. Every provided fold counts; non-finite / degenerate -> 0."""
    if len(rs) == 0:
        return 0.0
    out = np.empty(len(rs), dtype=float)
    for i, r in enumerate(rs):
        if r is None or not np.isfinite(r):
            out[i] = 0.0
        else:
            out[i] = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    return float(np.mean(out))


def r_from_fisher_z(z: float) -> float:
    return float(np.tanh(z))


def argmax_lag(values: np.ndarray, lags_ms: np.ndarray, prefer_ms: float = PREFER_LAG_MS) -> int:
    """Index of the maximum. Ties break toward ``prefer_ms``, then the earlier lag."""
    values = np.asarray(values, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return int(np.argmin(np.abs(lags_ms - prefer_ms)))
    m = np.max(values[finite])
    cand = np.flatnonzero(finite & np.isclose(values, m, rtol=0.0, atol=1e-12))
    dist = np.abs(lags_ms[cand] - prefer_ms)
    order = np.lexsort((lags_ms[cand], dist))
    return int(cand[order[0]])


# --------------------------------------------------------------------------
# CV splits. Test indices are never passed into support selection.
# --------------------------------------------------------------------------

def section_id_array(n_words: int, section_slices: Sequence[Tuple[int, int]]) -> np.ndarray:
    sid = np.full(n_words, -1, dtype=int)
    for sec_i, (a, b) in enumerate(section_slices, start=1):
        sid[a:b] = sec_i
    if np.any(sid < 0):
        raise RuntimeError("section slices do not cover every word")
    return sid


def loso_outer_splits(section_id: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    splits = []
    for held in sorted(set(int(s) for s in section_id)):
        te = np.flatnonzero(section_id == held)
        tr = np.flatnonzero(section_id != held)
        splits.append((tr, te, int(held)))
    if len(splits) != 3:
        raise RuntimeError(f"LOSO expected 3 sections, got {len(splits)}")
    return splits


def loso_inner_splits(train_idx: np.ndarray, section_id: np.ndarray):
    """Leave-one-training-section-out. Both sides stay inside ``train_idx``."""
    train_idx = np.asarray(train_idx, dtype=int)
    secs = sorted(set(int(s) for s in section_id[train_idx]))
    splits = []
    for held in secs:
        te = train_idx[section_id[train_idx] == held]
        tr = train_idx[section_id[train_idx] != held]
        if te.size >= 5 and tr.size >= 20:
            splits.append((tr, te))
    return splits


def contiguous_outer_splits(section_slices, n_words: int, section_id: np.ndarray, n_folds: int = 5):
    ranges = ec.make_folds_within_sections(list(section_slices), n_folds, min_per_section=1)
    splits = []
    for fa, fb in ranges:
        te = np.arange(fa, fb, dtype=int)
        tr = np.concatenate([np.arange(0, fa, dtype=int), np.arange(fb, n_words, dtype=int)])
        held = int(section_id[fa]) if te.size else -1
        splits.append((tr, te, held))
    return splits


def contiguous_inner_splits(train_idx: np.ndarray, section_id: np.ndarray, val_frac: float = 0.30):
    """One held-out tail inside the training fold (per section, then pooled)."""
    train_idx = np.asarray(train_idx, dtype=int)
    tr_parts, te_parts = [], []
    for sid in sorted(set(int(s) for s in section_id[train_idx])):
        idx = np.sort(train_idx[section_id[train_idx] == sid])
        if idx.size < 40:
            tr_parts.append(idx)
            continue
        n_val = max(5, int(round(val_frac * idx.size)))
        n_val = min(n_val, idx.size - 20)
        if n_val < 5:
            tr_parts.append(idx)
            continue
        te_parts.append(idx[-n_val:])
        tr_parts.append(idx[:-n_val])
    if not te_parts or not tr_parts:
        return []
    tr = np.concatenate(tr_parts)
    te = np.concatenate(te_parts)
    if te.size < 5 or tr.size < 20:
        return []
    return [(tr, te)]


def _valid_rows(idx, y, window_col, base_valid) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return idx
    m = base_valid[idx] & window_col[idx] & np.isfinite(y[idx])
    return idx[m]


def _scale_surp(surp, tr, te):
    sc = StandardScaler()
    sur_tr = sc.fit_transform(np.asarray(surp[tr], dtype=np.float64).reshape(-1, 1))
    sur_te = sc.transform(np.asarray(surp[te], dtype=np.float64).reshape(-1, 1))
    return sur_tr, sur_te


def _design(X, support, sur_tr, sur_te, tr, te):
    if support is not None and np.asarray(support).any():
        left_tr = _to_dense(X[tr][:, support])
        left_te = _to_dense(X[te][:, support])
        return np.hstack([left_tr, sur_tr]), np.hstack([left_te, sur_te])
    return sur_tr, sur_te


def predict_full(X, surp, y, tr, te, dense: bool):
    """F-test + LassoCV + Ridge on the training rows. Returns r, pred, support."""
    n_feat = X.shape[1]
    empty = np.zeros(n_feat, dtype=bool)
    y = np.asarray(y, dtype=np.float64)
    if tr.size < 20 or te.size < 5 or np.std(y[tr]) < 1e-12:
        return 0.0, np.full(te.size, np.nan), empty
    sur_tr, sur_te = _scale_surp(surp, tr, te)
    try:
        support = compute_joint_support(
            X[tr], sur_tr.ravel(), y[tr], dense=dense,
        )
    except ValueError:
        return 0.0, np.full(te.size, np.nan), empty
    Xtr, Xte = _design(X, support, sur_tr, sur_te, tr, te)
    try:
        pred, _alpha = _fit_predict(Xtr, y[tr], Xte)
    except ValueError:
        return 0.0, np.full(te.size, np.nan), support
    return _fold_r(pred, y[te], True), pred, support


def predict_with_support(X, surp, y, tr, te, support):
    y = np.asarray(y, dtype=np.float64)
    if tr.size < 20 or te.size < 5 or np.std(y[tr]) < 1e-12:
        return 0.0, np.full(max(te.size, 0), np.nan)
    sur_tr, sur_te = _scale_surp(surp, tr, te)
    Xtr, Xte = _design(X, support, sur_tr, sur_te, tr, te)
    try:
        pred, _alpha = _fit_predict(Xtr, y[tr], Xte)
    except ValueError:
        return 0.0, np.full(te.size, np.nan)
    return _fold_r(pred, y[te], True), pred


def predict_surprisal(surp, y, tr, te):
    y = np.asarray(y, dtype=np.float64)
    if tr.size < 20 or te.size < 5 or np.std(y[tr]) < 1e-12:
        return 0.0, np.full(max(te.size, 0), np.nan)
    sur_tr, sur_te = _scale_surp(surp, tr, te)
    try:
        pred, _alpha = _fit_predict(sur_tr, y[tr], sur_te)
    except ValueError:
        return 0.0, np.full(te.size, np.nan)
    return _fold_r(pred, y[te], True), pred


def _inner_splits(cv_mode: str, train_idx, section_id):
    if cv_mode == "loso":
        return loso_inner_splits(train_idx, section_id)
    if cv_mode == "contiguous5":
        return contiguous_inner_splits(train_idx, section_id)
    raise ValueError(cv_mode)


def training_peak_index(
    X, surp, Y, window_ok, base_valid, train_idx, section_id, lags_ms, cv_mode: str, dense: bool,
) -> int:
    """Lag index maximizing training-fold full-model Fisher-z. Test rows are absent."""
    n_lags = Y.shape[1]
    lags_ms = np.asarray(lags_ms, dtype=float)
    inners = _inner_splits(cv_mode, train_idx, section_id)
    outer = set(np.asarray(train_idx, dtype=int).tolist())
    for tr, te in inners:
        if not set(tr.tolist()).issubset(outer) or not set(te.tolist()).issubset(outer):
            raise RuntimeError("training-fold peak search left the training fold")
        if np.intersect1d(tr, te).size:
            raise RuntimeError("inner train/test overlap")
    z = np.zeros(n_lags, dtype=float)
    if not inners:
        return argmax_lag(z, np.arange(n_lags))
    for li in range(n_lags):
        y = Y[:, li]
        rs = []
        for tr, te in inners:
            tr_i = _valid_rows(tr, y, window_ok[:, li], base_valid)
            te_i = _valid_rows(te, y, window_ok[:, li], base_valid)
            if tr_i.size < 20 or te_i.size < 5:
                rs.append(0.0)
                continue
            r, _pred, _sup = predict_full(X, surp, y, tr_i, te_i, dense=dense)
            rs.append(r)
        z[li] = fisher_z_mean(rs)
    return argmax_lag(z, lags_ms)


def _aggregate_folds(fold_rs: List[float], preds: List[np.ndarray], gts: List[np.ndarray]):
    z = fisher_z_mean(fold_rs)
    if preds and gts:
        pred = np.concatenate(preds)
        gt = np.concatenate(gts)
        finite = np.isfinite(pred) & np.isfinite(gt)
        if int(finite.sum()) >= 5:
            r_pooled = _fold_r(pred[finite], gt[finite], True)
            n_pooled = int(finite.sum())
        else:
            r_pooled, n_pooled = 0.0, int(finite.sum())
    else:
        r_pooled, n_pooled = 0.0, 0
    r_mean = float(np.mean(fold_rs)) if fold_rs else 0.0
    return z, r_from_fisher_z(z), r_mean, r_pooled, n_pooled


def _blank_folds(n_max=5):
    r = [np.nan] * n_max
    sec = [np.nan] * n_max
    peak = [np.nan] * n_max
    nsup = [np.nan] * n_max
    return r, sec, peak, nsup


def _score_row(
    subject, channel, channel_index, feature_space, readout, cv_mode, lags_ms, li,
    fold_rs, fold_sections, peak_lags, n_supports, z, r_fisher, r_mean, r_pooled, n_pooled,
    feature_tag, shift_id=-1, null_draw=-1, bin_name="",
) -> dict:
    r, sec, peak, nsup = _blank_folds()
    for i, val in enumerate(fold_rs):
        r[i] = val
    for i, val in enumerate(fold_sections):
        sec[i] = val
    for i, val in enumerate(peak_lags):
        peak[i] = val
    for i, val in enumerate(n_supports):
        nsup[i] = val
    finite_ns = [x for x in n_supports if np.isfinite(x)]
    return {
        "subject": subject,
        "channel": channel,
        "channel_index": int(channel_index),
        "feature_space": feature_space,
        "readout": readout,
        "cv": cv_mode,
        "lag_ms": float(lags_ms[li]),
        "lag_index": int(li),
        "r_fold0": r[0], "r_fold1": r[1], "r_fold2": r[2], "r_fold3": r[3], "r_fold4": r[4],
        "fold_section0": sec[0], "fold_section1": sec[1], "fold_section2": sec[2],
        "fold_section3": sec[3], "fold_section4": sec[4],
        "n_folds": int(len(fold_rs)),
        "fisher_z_mean": z,
        "r_fisher": r_fisher,
        "r_mean": r_mean,
        "r_pooled": r_pooled,
        "n_pooled": n_pooled,
        "train_peak_lag_fold0": peak[0], "train_peak_lag_fold1": peak[1],
        "train_peak_lag_fold2": peak[2], "train_peak_lag_fold3": peak[3],
        "train_peak_lag_fold4": peak[4],
        "n_support_fold0": nsup[0], "n_support_fold1": nsup[1], "n_support_fold2": nsup[2],
        "n_support_fold3": nsup[3], "n_support_fold4": nsup[4],
        "n_support_mean": float(np.mean(finite_ns)) if finite_ns else 0.0,
        "feature_tag": feature_tag,
        "shift_id": int(shift_id),
        "null_draw": int(null_draw),
        "bin_name": bin_name,
    }


def fit_feature_lags(
    X,
    surp,
    Y,
    window_ok,
    base_valid,
    section_id,
    outer_splits,
    lags_ms,
    cv_mode: str,
    dense: bool,
    fixed_lag_index: Optional[int],
    col_offset: int = 0,
) -> Tuple[List[dict], List[dict], List[float], List[int]]:
    """Fit one feature space at every lag.

    Returns per-lag score field lists, support records (feature index, fold,
    train peak lag), the per-fold training peak lags, and per-fold support counts.
    ``X`` is None for surprisal-only.
    """
    n_lags = len(lags_ms)
    fold_pack = []  # per fold: support, peak_lag_ms, held section, list of (r, pred, gt) per lag
    support_recs = []
    for fold_i, (tr, te, held) in enumerate(outer_splits):
        if np.intersect1d(tr, te).size:
            raise RuntimeError("outer train/test overlap")
        if fixed_lag_index is None:
            if X is None:
                peak_i = argmax_lag(np.zeros(n_lags), lags_ms)
            else:
                peak_i = training_peak_index(
                    X, surp, Y, window_ok, base_valid, tr, section_id, lags_ms, cv_mode, dense,
                )
        else:
            peak_i = int(fixed_lag_index)
        peak_ms = float(lags_ms[peak_i])
        support = None
        if X is not None:
            y_peak = Y[:, peak_i]
            tr_i = _valid_rows(tr, y_peak, window_ok[:, peak_i], base_valid)
            if tr_i.size >= 20 and np.std(y_peak[tr_i]) >= 1e-12:
                sur_tr, _ = _scale_surp(surp, tr_i, tr_i[:1] if tr_i.size else tr_i)
                try:
                    support = compute_joint_support(
                        X[tr_i], sur_tr.ravel(), y_peak[tr_i], dense=dense,
                    )
                except ValueError:
                    support = np.zeros(X.shape[1], dtype=bool)
            else:
                support = np.zeros(X.shape[1], dtype=bool)
            for feat_i in np.flatnonzero(support):
                support_recs.append((fold_i, int(held), peak_ms, int(feat_i) + int(col_offset)))
        n_sup = int(np.sum(support)) if support is not None else 0
        per_lag = []
        for li in range(n_lags):
            y = Y[:, li]
            tr_i = _valid_rows(tr, y, window_ok[:, li], base_valid)
            te_i = _valid_rows(te, y, window_ok[:, li], base_valid)
            if X is None:
                r, pred = predict_surprisal(surp, y, tr_i, te_i)
            else:
                r, pred = predict_with_support(X, surp, y, tr_i, te_i, support)
            if te_i.size == 0 or pred is None or len(pred) != te_i.size:
                per_lag.append((0.0, None, None))
            else:
                per_lag.append((r, pred, y[te_i]))
        fold_pack.append((peak_ms, held, n_sup, per_lag))

    rows_fields = []
    for li in range(n_lags):
        fold_rs, secs, peaks, nsups = [], [], [], []
        preds, gts = [], []
        for peak_ms, held, n_sup, per_lag in fold_pack:
            r, pred, gt = per_lag[li]
            fold_rs.append(r)
            secs.append(held)
            peaks.append(peak_ms)
            nsups.append(n_sup)
            if pred is not None and gt is not None:
                preds.append(np.asarray(pred, dtype=float))
                gts.append(np.asarray(gt, dtype=float))
        z, r_fisher, r_mean, r_pooled, n_pooled = _aggregate_folds(fold_rs, preds, gts)
        rows_fields.append((fold_rs, secs, peaks, nsups, z, r_fisher, r_mean, r_pooled, n_pooled))
    return rows_fields, support_recs, [fp[0] for fp in fold_pack], [fp[2] for fp in fold_pack]


def fit_electrode(task: dict) -> dict:
    """One language electrode. ``task`` is picklable for joblib."""
    t0 = time.time()
    subject = task["subject"]
    channel = task["channel"]
    ci = task["channel_index"]
    Y = np.asarray(task["Y"], dtype=np.float64)  # (n_words, n_lags)
    window_ok = np.asarray(task["window_ok"], dtype=bool)
    base_valid = np.asarray(task["base_valid"], dtype=bool)
    section_id = np.asarray(task["section_id"], dtype=int)
    lags_ms = np.asarray(task["lags_ms"], dtype=float)
    surp = np.asarray(task["surprisal"], dtype=np.float64)
    cv_mode = task["cv_mode"]
    spaces = tuple(task["spaces"])
    fixed_lag_ms = task.get("fixed_lag_ms")
    fixed_i = None
    if fixed_lag_ms is not None:
        fixed_i = int(np.argmin(np.abs(lags_ms - float(fixed_lag_ms))))
    if cv_mode == "loso":
        outer = loso_outer_splits(section_id)
    elif cv_mode == "contiguous5":
        outer = contiguous_outer_splits(
            task["section_slices"], Y.shape[0], section_id, N_CONTIGUOUS_FOLDS,
        )
    else:
        raise ValueError(cv_mode)

    rows = []
    supports = []
    feature_tag = task["feature_tag"]
    shift_id = int(task.get("shift_id", -1))
    null_draw = int(task.get("null_draw", -1))
    bin_name = task.get("bin_name") or ""
    col_offset = int(task.get("col_offset", 0))

    space_X = {}
    if "sae" in spaces or bin_name:
        space_X["sae"] = task["X_sae"]
    if "residual" in spaces:
        space_X["residual"] = task["X_resid"]

    def _emit(space, readout, fields, dense_flag_unused=None):
        for li, (fold_rs, secs, peaks, nsups, z, r_fisher, r_mean, r_pooled, n_pooled) in enumerate(fields):
            rows.append(_score_row(
                subject, channel, ci, space, readout, cv_mode, lags_ms, li,
                fold_rs, secs, peaks, nsups, z, r_fisher, r_mean, r_pooled, n_pooled,
                feature_tag, shift_id=shift_id, null_draw=null_draw, bin_name=bin_name,
            ))

    for space in spaces:
        if space == "surprisal":
            fields, _recs, _peaks, _ns = fit_feature_lags(
                None, surp, Y, window_ok, base_valid, section_id, outer, lags_ms,
                cv_mode, dense=False, fixed_lag_index=fixed_i,
            )
            _emit(space, READOUT_MATCHED, fields)
            continue
        X = space_X[space]
        dense = bool(X.shape[1] <= LASSO_K_BEST)
        fields, recs, _peaks, _ns = fit_feature_lags(
            X, surp, Y, window_ok, base_valid, section_id, outer, lags_ms,
            cv_mode, dense=dense, fixed_lag_index=fixed_i, col_offset=col_offset,
        )
        _emit(space, READOUT_MATCHED, fields)
        for fold_i, held, peak_ms, feat_i in recs:
            supports.append({
                "subject": subject,
                "channel": channel,
                "cv": cv_mode,
                "feature_space": space if not bin_name else bin_name,
                "fold": int(fold_i),
                "heldout_section": int(held),
                "train_peak_lag_ms": float(peak_ms),
                "feature_index": int(feat_i),
                "bin_name": bin_name,
                "shift_id": shift_id,
                "null_draw": null_draw,
                "feature_tag": feature_tag,
            })

    if task.get("dense_ridge"):
        fields = _fit_dense_ridge_lags(
            task["X_resid"], surp, Y, window_ok, base_valid, outer, lags_ms,
        )
        _emit("residual_dense_ridge", READOUT_DENSE, fields)

    elapsed = time.time() - t0
    return {
        "rows": rows,
        "supports": supports,
        "log": (
            f"  [{subject}] {channel} cv={cv_mode} spaces={','.join(spaces)} "
            f"bin={bin_name or '-'} shift={shift_id} null={null_draw} "
            f"in {_fmt_duration(elapsed)}"
        ),
        "elapsed": elapsed,
        "subject": subject,
        "channel": channel,
    }


def _fit_dense_ridge_lags(X, surp, Y, window_ok, base_valid, outer, lags_ms):
    """Fixed-alpha Ridge, no LASSO. Labeled sensitivity, not the residual comparator."""
    n_lags = len(lags_ms)
    per_fold = []
    X = np.asarray(X, dtype=np.float64)
    for tr, te, held in outer:
        per_lag = []
        for li in range(n_lags):
            y = np.asarray(Y[:, li], dtype=np.float64)
            tr_i = _valid_rows(tr, y, window_ok[:, li], base_valid)
            te_i = _valid_rows(te, y, window_ok[:, li], base_valid)
            if tr_i.size < 20 or te_i.size < 5 or np.std(y[tr_i]) < 1e-12:
                per_lag.append((0.0, None, None))
                continue
            sur_tr, sur_te = _scale_surp(surp, tr_i, te_i)
            Xtr = np.hstack([X[tr_i], sur_tr])
            Xte = np.hstack([X[te_i], sur_te])
            model = Ridge(
                alpha=DENSE_RIDGE_ALPHA, fit_intercept=True,
                solver="lsqr", max_iter=4000, tol=1e-3,
            )
            try:
                model.fit(Xtr, y[tr_i])
                pred = model.predict(Xte).ravel()
                r = _fold_r(pred, y[te_i], True)
            except ValueError:
                per_lag.append((0.0, None, None))
                continue
            per_lag.append((r, pred, y[te_i]))
        per_fold.append((held, per_lag))
    fields = []
    for li in range(n_lags):
        fold_rs, secs, preds, gts = [], [], [], []
        for held, per_lag in per_fold:
            r, pred, gt = per_lag[li]
            fold_rs.append(r)
            secs.append(held)
            if pred is not None and gt is not None:
                preds.append(pred)
                gts.append(gt)
        z, r_fisher, r_mean, r_pooled, n_pooled = _aggregate_folds(fold_rs, preds, gts)
        peaks = [np.nan] * len(fold_rs)
        nsups = [X.shape[1]] * len(fold_rs)
        fields.append((fold_rs, secs, peaks, nsups, z, r_fisher, r_mean, r_pooled, n_pooled))
    return fields


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def _append_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def complete_keys(path: Path, key_cols: Sequence[str], spaces: Sequence[str], n_each: int):
    """Return complete key tuples and drop partial groups from ``path``."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    if df.empty or any(c not in df.columns for c in list(key_cols) + ["feature_space"]):
        return set()
    ct = (
        df.groupby(list(key_cols) + ["feature_space"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    wide = ct.pivot_table(
        index=list(key_cols), columns="feature_space", values="n", fill_value=0,
    )
    for s in spaces:
        if s not in wide.columns:
            wide[s] = 0
    ok_mask = np.ones(len(wide), dtype=bool)
    for s in spaces:
        ok_mask &= wide[s].to_numpy() >= n_each
    complete_index = wide.index[ok_mask]
    if len(key_cols) == 1:
        good = set((ix,) for ix in complete_index.tolist())
    else:
        good = set(complete_index.tolist())
    # Drop partial groups so a resumed electrode is not duplicated.
    if len(good) != len(wide):
        key_frame = df[list(key_cols)].apply(tuple, axis=1)
        keep = key_frame.isin(good)
        df.loc[keep].to_csv(path, index=False)
    return good


def _key_of(row_subject, row_channel, extra: Optional[dict] = None, key_cols: Sequence[str] = ()):
    vals = []
    for c in key_cols:
        if c == "subject":
            vals.append(row_subject)
        elif c == "channel":
            vals.append(row_channel)
        else:
            vals.append(extra[c])
    return tuple(vals)


# --------------------------------------------------------------------------
# Subject loading
# --------------------------------------------------------------------------

def _lang_channels(subject: str, labels: Sequence[str]) -> List[Tuple[int, str]]:
    tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
    keep = set(
        tax.loc[
            (tax["subject"] == subject) & tax["is_lang"].fillna(False).astype(bool),
            "channel",
        ].astype(str).str.strip()
    )
    out = []
    for i, ch in enumerate(labels):
        if str(ch).strip() in keep:
            out.append((i, str(ch).strip()))
    return out


def _shift_neural(data, fraction: float):
    from dataclasses import replace
    new_secs = []
    for sec in data.sections:
        eeg = np.diff(np.asarray(sec.eeg_cs, dtype=np.float64), axis=0)
        shift = int(np.round(float(fraction) * sec.T)) % int(sec.T)
        if shift == 0:
            shift = max(1, int(sec.T) // 5)
        rolled = np.roll(eeg, shift, axis=0)
        cs = np.vstack([
            np.zeros((1, rolled.shape[1]), dtype=np.float64),
            np.cumsum(rolled, axis=0),
        ])
        new_secs.append(replace(sec, eeg_cs=cs))
    return replace(data, sections=new_secs)


def _stack_Y(data, channel_index: Sequence[int], lags_ms: np.ndarray):
    half = window_sample_bounds(0.0)[2]
    cols = []
    oks = []
    ch_idx = np.asarray(channel_index, dtype=int)
    for lag_ms in lags_ms:
        _s, _e, half_i = window_sample_bounds(float(lag_ms))
        if half_i != half:
            raise RuntimeError("window half-width changed across lags")
        lag_samp = int(round((float(lag_ms) / 1000.0) * FS_TARGET))
        Y, ok = ec.compute_Y_for_lag(data, lag_samp, half)
        cols.append(Y[:, ch_idx].astype(np.float32, copy=False))
        oks.append(ok.astype(bool, copy=False))
    # (n_words, n_chan, n_lags)
    Yall = np.stack(cols, axis=-1)
    window_ok = np.column_stack(oks)
    return Yall, window_ok


def _load_features(feature_tag: str):
    X_sae, sur_s, valid_s = load_sae_features(feature_tag, SECTIONS)
    X_res, sur_r, valid_r = load_residual_features(feature_tag, SECTIONS)
    if X_sae.shape[0] != X_res.shape[0]:
        raise RuntimeError("SAE / residual row mismatch")
    if not np.array_equal(valid_s, valid_r):
        # Intersect so both spaces see the same trials.
        both = valid_s & valid_r
        valid_s = both
    surp = np.asarray(sur_s, dtype=np.float64).ravel()
    if not np.allclose(surp, np.asarray(sur_r, dtype=np.float64).ravel(), equal_nan=True):
        raise RuntimeError("SAE / residual surprisal mismatch")
    base_valid = valid_s & np.isfinite(surp)
    X_resid = np.asarray(X_res.toarray(), dtype=np.float32)
    return X_sae.tocsr(), X_resid, surp, base_valid


def _subject_order(subjects: dict, only: Optional[Sequence[str]]) -> List[str]:
    tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
    lang = tax[tax["is_lang"].fillna(False).astype(bool)]
    counts = lang.groupby("subject").size().to_dict()
    names = list(only) if only else list(subjects)
    names = [s for s in names if s in subjects]
    names.sort(key=lambda s: (int(counts.get(s, 10**6)), s))
    return names


def _neural_cache_path(subject: str, shift_fraction: float, lags_ms: np.ndarray, max_channels: int) -> Path:
    cache = v3_root() / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    sig = (
        f"n{len(lags_ms)}_a{float(lags_ms[0]):.1f}_b{float(lags_ms[-1]):.1f}"
        f"_sh{int(round(float(shift_fraction) * 10000))}_m{int(max_channels)}"
    )
    return cache / f"{subject}_{sig}.npz"


def _pack_neural(Yall, window_ok, section_id, section_slices, lang):
    ch_idx = np.asarray([i for i, _ in lang], dtype=np.int32)
    names = np.asarray([ch for _, ch in lang], dtype="U64")
    slices = np.asarray(section_slices, dtype=np.int32)
    return {
        "Y": np.asarray(Yall, dtype=np.float32),
        "window_ok": np.asarray(window_ok, dtype=bool),
        "section_id": np.asarray(section_id, dtype=np.int32),
        "section_slices": slices,
        "channel_index": ch_idx,
        "channels": names,
    }


def _save_neural(path: Path, pack: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **pack)
    os.replace(tmp, path)


def _load_neural_file(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _try_load_neural(subject, shift_fraction, lags_ms, max_channels):
    path = _neural_cache_path(subject, shift_fraction, lags_ms, max_channels)
    if path.exists():
        print(f"[{subject}] neural cache {path.name}", flush=True)
        return _load_neural_file(path)
    full_lags = lag_grid_ms()
    full = _neural_cache_path(subject, shift_fraction, full_lags, max_channels)
    if not full.exists():
        return None
    pack = _load_neural_file(full)
    idx = []
    for lag in lags_ms:
        j = int(np.argmin(np.abs(full_lags - float(lag))))
        if not np.isclose(full_lags[j], float(lag)):
            return None
        idx.append(j)
    pack["Y"] = pack["Y"][:, :, idx]
    pack["window_ok"] = pack["window_ok"][:, idx]
    print(f"[{subject}] sliced neural cache {full.name} -> {len(lags_ms)} lags", flush=True)
    return pack


def _run_subject_electrodes(
    subject: str,
    eeg_files: dict,
    feature_tag: str,
    X_sae,
    X_resid,
    surp,
    base_valid,
    score_path: Path,
    support_path: Optional[Path],
    spaces: Sequence[str],
    cv_mode: str,
    lags_ms: np.ndarray,
    n_jobs: int,
    key_cols: Sequence[str],
    key_extra: Optional[dict],
    fixed_lag_ms: Optional[float],
    shift_fraction: float,
    shift_id: int,
    null_draw: int,
    bin_name: str,
    col_offset: int,
    dense_ridge: bool,
    max_channels: int,
) -> None:
    n_words_X = None if X_sae is None else int(X_sae.shape[0])
    cached = _try_load_neural(subject, shift_fraction, lags_ms, max_channels)
    if cached is not None:
        Yall = cached["Y"]
        window_ok = cached["window_ok"]
        section_id = cached["section_id"]
        section_slices = [tuple(int(x) for x in row) for row in cached["section_slices"]]
        lang = [(int(i), str(ch)) for i, ch in zip(cached["channel_index"], cached["channels"])]
        W = int(Yall.shape[0])
        if n_words_X is not None and n_words_X != W:
            raise RuntimeError(f"[{subject}] cached words {W} != features {n_words_X}")
    else:
        eeg_abs = {sid: str(ap.DATA_ROOT / p) for sid, p in eeg_files.items()}
        missing = [p for p in eeg_abs.values() if not Path(p).exists()]
        if missing:
            print(f"[{subject}] missing EEG, skipping: {missing}", flush=True)
            return
        data = ec.load_subject_data(
            subject_name=subject,
            eeg_files=eeg_abs,
            features_dir=ap.FEATURES_DIR,
            fs_target=FS_TARGET,
            sections=SECTIONS,
            feature_sets=["glove"],
        )
        if shift_fraction:
            data = _shift_neural(data, shift_fraction)
        W = data.section_word_slices[-1][1]
        if n_words_X is not None and n_words_X != W:
            raise RuntimeError(f"[{subject}] word count SAE {n_words_X} != EEG {W}")
        lang = _lang_channels(subject, data.channel_labels)
        if max_channels:
            lang = lang[: max_channels]
        if not lang:
            print(f"[{subject}] no language electrodes after channel match", flush=True)
            return
        section_id = section_id_array(W, data.section_word_slices)
        section_slices = [(int(a), int(b)) for a, b in data.section_word_slices]
        ch_idx = [i for i, _ in lang]
        Yall, window_ok = _stack_Y(data, ch_idx, lags_ms)
        del data
        pack = _pack_neural(Yall, window_ok, section_id, section_slices, lang)
        _save_neural(_neural_cache_path(subject, shift_fraction, lags_ms, max_channels), pack)

    n_each = int(len(lags_ms))
    done = complete_keys(score_path, key_cols, spaces if not dense_ridge else ("residual_dense_ridge",), n_each)
    tasks = []
    for local_i, (ci, ch) in enumerate(lang):
        extra = dict(key_extra or {})
        key = _key_of(subject, ch, extra, key_cols)
        if key in done:
            continue
        tasks.append({
            "subject": subject,
            "channel": ch,
            "channel_index": int(ci),
            "Y": np.asarray(Yall[:, local_i, :], dtype=np.float32),
            "window_ok": window_ok,
            "base_valid": base_valid,
            "section_id": section_id,
            "section_slices": section_slices,
            "lags_ms": lags_ms,
            "surprisal": surp,
            "X_sae": X_sae if ("sae" in spaces or bin_name) else None,
            "X_resid": X_resid if ("residual" in spaces or dense_ridge) else None,
            "cv_mode": cv_mode,
            "spaces": list(spaces),
            "feature_tag": feature_tag,
            "fixed_lag_ms": fixed_lag_ms,
            "shift_id": shift_id,
            "null_draw": null_draw,
            "bin_name": bin_name,
            "col_offset": col_offset,
            "dense_ridge": bool(dense_ridge),
        })
    print(
        f"[{subject}] {len(tasks)} electrodes to fit "
        f"({len(lang) - len(tasks)} already in {score_path.name}), n_jobs={n_jobs}",
        flush=True,
    )
    if not tasks:
        return
    _parallel_tasks(tasks, score_path, support_path, n_jobs)


def _parallel_tasks(tasks, score_path: Path, support_path: Optional[Path], n_jobs: int) -> None:
    from joblib import Parallel, delayed
    n_jobs = max(1, min(int(n_jobs), len(tasks)))
    if n_jobs == 1:
        results = (fit_electrode(t) for t in tasks)
        _consume(results, len(tasks), score_path, support_path)
        return
    kwargs = dict(n_jobs=n_jobs, backend="loky")
    try:
        results = Parallel(return_as="generator_unordered", **kwargs)(
            delayed(fit_electrode)(t) for t in tasks
        )
    except TypeError:
        results = Parallel(return_as="generator", **kwargs)(
            delayed(fit_electrode)(t) for t in tasks
        )
    _consume(results, len(tasks), score_path, support_path)


def _consume(results, n_tasks: int, score_path: Path, support_path: Optional[Path]) -> None:
    t0 = time.time()
    for done_i, res in enumerate(results, start=1):
        rows = res["rows"]
        if not rows:
            raise RuntimeError(f"empty result for {res.get('subject')} {res.get('channel')}")
        _append_df(score_path, pd.DataFrame(rows))
        if support_path is not None and res["supports"]:
            # One file per electrode so a resumed fit overwrites instead of
            # double-counting selected indices.
            part_dir = support_path.parent / f"{support_path.stem}_parts"
            part_dir.mkdir(parents=True, exist_ok=True)
            s0 = res["supports"][0]
            safe_ch = str(res["channel"]).replace("/", "_")
            bin_name = s0.get("bin_name") or "main"
            part = part_dir / (
                f"{res['subject']}__{safe_ch}__sh{s0['shift_id']}"
                f"__n{s0['null_draw']}__{bin_name}.csv"
            )
            pd.DataFrame(res["supports"]).to_csv(part, index=False)
        elapsed = time.time() - t0
        rate = done_i / max(elapsed, 1e-6)
        eta = (n_tasks - done_i) / rate if rate else float("nan")
        print(
            f"{res['log']} | subject-batch {done_i}/{n_tasks} "
            f"ETA {_fmt_duration(eta)}",
            flush=True,
        )


def _guard_tag(tag: str) -> str:
    low = tag.lower()
    if "gemma" in low or "qwen3_8b" in low or "qwen3-8b" in low or "jumprelu" in low:
        raise SystemExit(f"v3 refuses feature tag {tag}")
    return tag


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def _shared_load(tag: str):
    print(f"[v3] loading features {tag}", flush=True)
    return _load_features(tag)


def stage_loso(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = lag_grid_ms()
    path = tables / "v3_loso_lag_scores.csv"
    supports = tables / "v3_selected_indices_loso.csv"
    for subject in _subject_order(subjects, args.subjects):
        _run_subject_electrodes(
            subject, subjects[subject]["eeg_files"], tag,
            X_sae, X_resid, surp, base_valid,
            path, supports, SPACES_MAIN, "loso", lags, args.n_jobs,
            key_cols=("subject", "channel"), key_extra=None,
            fixed_lag_ms=None, shift_fraction=0.0, shift_id=-1, null_draw=-1,
            bin_name="", col_offset=0, dense_ridge=False, max_channels=args.max_channels,
        )


def stage_contiguous(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = lag_grid_ms()
    path = tables / "v3_contiguous5_lag_scores.csv"
    for subject in _subject_order(subjects, args.subjects):
        _run_subject_electrodes(
            subject, subjects[subject]["eeg_files"], tag,
            X_sae, X_resid, surp, base_valid,
            path, None, SPACES_MAIN, "contiguous5", lags, args.n_jobs,
            key_cols=("subject", "channel"), key_extra=None,
            fixed_lag_ms=None, shift_fraction=0.0, shift_id=-1, null_draw=-1,
            bin_name="", col_offset=0, dense_ridge=False, max_channels=args.max_channels,
        )


def _peak_from_loso(tables: Path) -> Tuple[float, pd.DataFrame]:
    path = tables / "v3_loso_lag_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"LOSO scores missing: {path}")
    df = pd.read_csv(path)
    curve = subject_gain_curve(df, "sae")
    i = argmax_lag(curve["mean_z"].to_numpy(), curve["lag_ms"].to_numpy())
    return float(curve["lag_ms"].iloc[i]), curve


def stage_circshift(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = lag_grid_ms()
    path = tables / "v3_circshift_lag_scores.csv"
    spaces = ("sae", "surprisal")
    for shift_id, frac in enumerate(CIRC_SHIFT_FRACTIONS):
        print(f"[v3] circular shift {shift_id} fraction={frac:.4f}", flush=True)
        for subject in _subject_order(subjects, args.subjects):
            _run_subject_electrodes(
                subject, subjects[subject]["eeg_files"], tag,
                X_sae, X_resid, surp, base_valid,
                path, None, spaces, "loso", lags, args.n_jobs,
                key_cols=("subject", "channel", "shift_id"),
                key_extra={"shift_id": shift_id},
                fixed_lag_ms=None, shift_fraction=float(frac), shift_id=shift_id,
                null_draw=-1, bin_name="", col_offset=0, dense_ridge=False,
                max_channels=args.max_channels,
            )


def stage_null(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    peak_ms, _curve = _peak_from_loso(tables)
    print(f"[v3] peak-lag label-shuffle null at {peak_ms:.0f} ms (v3 LOSO, n={args.n_null})", flush=True)
    X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = np.asarray([peak_ms], dtype=float)
    path = tables / "v3_peak_null_v3.csv"
    spaces = ("sae", "surprisal")
    for draw in range(1, int(args.n_null) + 1):
        print(f"[v3] null draw {draw}/{args.n_null}", flush=True)
        for subject in _subject_order(subjects, args.subjects):
            rng = np.random.default_rng(RANDOM_STATE + int(draw) * 10007 + sum(ord(c) for c in subject))
            X_s, sur_s = maybe_shuffle_features(
                X_sae, surp, "global", rng, valid=base_valid,
            )
            _run_subject_electrodes(
                subject, subjects[subject]["eeg_files"], tag,
                X_s, X_resid, sur_s, base_valid,
                path, None, spaces, "loso", lags, args.n_jobs,
                key_cols=("subject", "channel", "null_draw"),
                key_extra={"null_draw": draw},
                fixed_lag_ms=float(peak_ms), shift_fraction=0.0, shift_id=-1,
                null_draw=draw, bin_name="", col_offset=0, dense_ridge=False,
                max_channels=args.max_channels,
            )


def stage_bins(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    peak_ms, _curve = _peak_from_loso(tables)
    print(f"[v3] Matryoshka bin refits at {peak_ms:.0f} ms", flush=True)
    X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = np.asarray([peak_ms], dtype=float)
    path = tables / "v3_bin_refit_peak.csv"
    supports = tables / "v3_selected_indices_peak_lag.csv"
    # Also refit the full dictionary at the group peak so Fig 5A is that lag.
    jobs = [("sae_full_at_group_peak", 0, X_sae.shape[1], "")] + [
        (name, lo, hi, name) for name, lo, hi in BIN_SLICES
    ]
    for space_name, lo, hi, bin_name in jobs:
        print(f"[v3] bin/support pass {space_name} cols [{lo}:{hi}]", flush=True)
        X_slice = X_sae[:, lo:hi]
        for subject in _subject_order(subjects, args.subjects):
            _run_subject_electrodes(
                subject, subjects[subject]["eeg_files"], tag,
                X_slice, X_resid, surp, base_valid,
                path, supports, ("sae",), "loso", lags, args.n_jobs,
                key_cols=("subject", "channel", "bin_name"),
                key_extra={"bin_name": bin_name or space_name},
                fixed_lag_ms=float(peak_ms), shift_fraction=0.0, shift_id=-1,
                null_draw=-1, bin_name=bin_name or space_name, col_offset=int(lo),
                dense_ridge=False, max_channels=args.max_channels,
            )


def stage_dense_ridge(args, subjects) -> None:
    _root, tables, _r, _f, _logs = _dirs()
    tag = _guard_tag(args.feature_tag)
    peak_ms, _curve = _peak_from_loso(tables)
    print(f"[v3] dense ridge alpha={DENSE_RIDGE_ALPHA:.0f} sensitivity at {peak_ms:.0f} ms", flush=True)
    _X_sae, X_resid, surp, base_valid = _shared_load(tag)
    lags = np.asarray([peak_ms], dtype=float)
    path = tables / "v3_dense_ridge_sensitivity.csv"
    # X_sae still required by the loader alignment; pass a 1-col dummy slice only if needed.
    X_dummy = _X_sae[:, :1]
    for subject in _subject_order(subjects, args.subjects):
        _run_subject_electrodes(
            subject, subjects[subject]["eeg_files"], tag,
            X_dummy, X_resid, surp, base_valid,
            path, None, (), "loso", lags, args.n_jobs,
            key_cols=("subject", "channel"), key_extra=None,
            fixed_lag_ms=float(peak_ms), shift_fraction=0.0, shift_id=-1,
            null_draw=-1, bin_name="", col_offset=0, dense_ridge=True,
            max_channels=args.max_channels,
        )


# --------------------------------------------------------------------------
# Summaries, null/ceiling, figures, report
# --------------------------------------------------------------------------

def subject_gain_curve(df: pd.DataFrame, space: str, value: str = "fisher_z_mean") -> pd.DataFrame:
    """Subject-mean (space − surprisal) and the across-subject mean and SEM."""
    a = df[df["feature_space"] == space][
        ["subject", "channel", "lag_ms", value]
    ].rename(columns={value: "full"})
    b = df[df["feature_space"] == "surprisal"][
        ["subject", "channel", "lag_ms", value]
    ].rename(columns={value: "surp"})
    m = a.merge(b, on=["subject", "channel", "lag_ms"], how="inner")
    m["gain"] = m["full"] - m["surp"]
    subj = m.groupby(["subject", "lag_ms"], as_index=False)["gain"].mean()
    g = subj.groupby("lag_ms")["gain"].agg(mean="mean", std="std", n="count").reset_index()
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    g = g.sort_values("lag_ms").reset_index(drop=True)
    g = g.rename(columns={"mean": "mean_z" if value == "fisher_z_mean" else "mean_r"})
    # Always provide both names used by callers when value is fisher z.
    if value == "fisher_z_mean":
        pass
    return g


def _pair_values(df: pd.DataFrame, space: str, lag_ms: float, value: str) -> pd.DataFrame:
    a = df[(df["feature_space"] == space) & np.isclose(df["lag_ms"], lag_ms)][
        ["subject", "channel", value]
    ].rename(columns={value: "full"})
    b = df[(df["feature_space"] == "surprisal") & np.isclose(df["lag_ms"], lag_ms)][
        ["subject", "channel", value]
    ].rename(columns={value: "surp"})
    m = a.merge(b, on=["subject", "channel"], how="inner")
    m["gain"] = m["full"] - m["surp"]
    return m


def _subject_mean_at(df, space, lag_ms, value) -> pd.Series:
    m = _pair_values(df, space, lag_ms, value)
    return m.groupby("subject")["gain"].mean()


def signflip_p(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan")
    obs = abs(float(np.mean(x)))
    count = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        if abs(float(np.mean(signs * x))) >= obs - 1e-15:
            count += 1
    return count / float(1 << n)


def split_half_reliability(df: pd.DataFrame, lag_ms: float) -> dict:
    """Electrode-level reliability of LOSO section scores at one lag.

    Each LOSO fold holds out one section. Pairwise Pearson r of the electrode
    vectors across the three sections, then Spearman-Brown for the mean of
    three sections. Reported as a ceiling, not used as a divisor.
    """
    sub = df[np.isclose(df["lag_ms"], lag_ms)].copy()
    key = ["subject", "channel"]

    def _fold_mat(space: str) -> pd.DataFrame:
        rows = sub[sub["feature_space"] == space].sort_values(key)
        mats = []
        for col in ("r_fold0", "r_fold1", "r_fold2"):
            r = rows[col].to_numpy(dtype=float)
            z = np.arctanh(np.clip(np.nan_to_num(r, nan=0.0), -0.999999, 0.999999))
            mats.append(z)
        out = rows[key].reset_index(drop=True)
        out[["z0", "z1", "z2"]] = np.column_stack(mats)
        return out

    sae = _fold_mat("sae")
    sur = _fold_mat("surprisal")
    merged = sae.merge(sur, on=key, suffixes=("_sae", "_sur"))
    z_sae = merged[["z0_sae", "z1_sae", "z2_sae"]].to_numpy(dtype=float)
    z_sur = merged[["z0_sur", "z1_sur", "z2_sur"]].to_numpy(dtype=float)
    gain = z_sae - z_sur

    def _pw(mat: np.ndarray) -> Tuple[float, float, int]:
        rs = []
        k = mat.shape[1]
        for i in range(k):
            for j in range(i + 1, k):
                a, b = mat[:, i], mat[:, j]
                m = np.isfinite(a) & np.isfinite(b)
                if int(m.sum()) < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
                    continue
                rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
        if not rs:
            return float("nan"), float("nan"), 0
        rbar = float(np.mean(rs))
        kk = 3.0
        sb = (kk * rbar) / (1.0 + (kk - 1.0) * rbar) if np.isfinite(rbar) else float("nan")
        return rbar, sb, len(rs)

    r_g, sb_g, n_g = _pw(gain)
    r_f, sb_f, n_f = _pw(z_sae)
    return {
        "lag_ms": float(lag_ms),
        "n_electrodes": int(z_sae.shape[0]),
        "gain_mean_pairwise_r": r_g,
        "gain_spearman_brown_k3": sb_g,
        "gain_n_pairs": n_g,
        "full_mean_pairwise_r": r_f,
        "full_spearman_brown_k3": sb_f,
        "full_n_pairs": n_f,
        "note": "ceiling reported beside scores; scores are not divided by this",
    }


def _v2_lang_gain() -> dict:
    path = (
        ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v2" / "tables"
        / "sparse_encoding_results_sae_qwen35_4b_mat_l15_v2_lang.csv"
    )
    out = {"path": str(path), "available": path.exists()}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    full = "full__R_fisher_paired" if "full__R_fisher_paired" in df.columns else "full__R_fisher"
    sur = (
        "surprisal_only__R_fisher_paired"
        if "surprisal_only__R_fisher_paired" in df.columns
        else "surprisal_only__R_fisher"
    )
    df = df.copy()
    df["gain"] = df[full] - df[sur]
    sub = df.groupby("subject")["gain"].mean()
    out.update({
        "column_full": full,
        "column_surprisal": sur,
        "cv": str(df["cv"].iloc[0]) if "cv" in df.columns else "",
        "n_electrodes": int(len(df)),
        "n_subjects": int(sub.size),
        "subject_mean_gain_r_fisher": float(sub.mean()),
        "sem": float(sub.std(ddof=1) / np.sqrt(sub.size)),
        "signflip_p": float(signflip_p(sub.to_numpy())),
        "lag_ms": 300.0,
        "window": "[200, 400) ms",
    })
    return out


def _null_summary(tables: Path, peak_ms: float) -> dict:
    path = tables / "v3_peak_null_v3.csv"
    v2_dir = (
        ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v2" / "tables" / "sae_null_perm"
    )
    reason = (
        "v2 sae_null_perm_*.csv rows are cv=contiguous (5 within-section folds), "
        "not leave-one-section-out SAE gain, so they are not the v3 estimand"
    )
    out = {
        "reused_v2_50": False,
        "reason": reason,
        "v2_null_dir_exists": v2_dir.exists(),
        "label": "v3",
        "peak_lag_ms": peak_ms,
        "path": str(path),
    }
    if not path.exists():
        out["available"] = False
        return out
    df = pd.read_csv(path)
    a = df[(df["feature_space"] == "sae")][
        ["subject", "channel", "null_draw", "fisher_z_mean", "r_fisher"]
    ].rename(columns={"fisher_z_mean": "full_z", "r_fisher": "full_r"})
    b = df[(df["feature_space"] == "surprisal")][
        ["subject", "channel", "null_draw", "fisher_z_mean", "r_fisher"]
    ].rename(columns={"fisher_z_mean": "surp_z", "r_fisher": "surp_r"})
    m = a.merge(b, on=["subject", "channel", "null_draw"], how="inner")
    m["gain_z"] = m["full_z"] - m["surp_z"]
    m["gain_r"] = m["full_r"] - m["surp_r"]
    draw = m.groupby(["null_draw", "subject"], as_index=False)["gain_z"].mean()
    draw_mean = draw.groupby("null_draw")["gain_z"].mean()
    out.update({
        "available": True,
        "n_draws": int(draw_mean.size),
        "null_mean_subject_gain_z": float(draw_mean.mean()) if len(draw_mean) else float("nan"),
        "null_std_subject_gain_z": float(draw_mean.std(ddof=1)) if len(draw_mean) > 1 else float("nan"),
        "draw_means_z": [float(x) for x in draw_mean.to_numpy()],
    })
    return out


def _circular_curve(tables: Path) -> Optional[pd.DataFrame]:
    path = tables / "v3_circshift_lag_scores.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    curves = []
    for sid, sub in df.groupby("shift_id"):
        c = subject_gain_curve(sub, "sae")
        c["shift_id"] = int(sid)
        curves.append(c)
    if not curves:
        return None
    return pd.concat(curves, ignore_index=True)


def _concat_support_parts(combined: Path) -> Optional[Path]:
    part_dir = combined.parent / f"{combined.stem}_parts"
    parts = sorted(part_dir.glob("*.csv")) if part_dir.exists() else []
    if not parts and not combined.exists():
        return None
    frames = []
    if parts:
        frames.extend(pd.read_csv(p) for p in parts)
    if combined.exists() and combined.stat().st_size:
        frames.append(pd.read_csv(combined))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates()
    out.to_csv(combined, index=False)
    return combined


def _verify_png(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 100 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"not a non-empty PNG: {path} size={len(data)}")


def _copy_figures(fig_paths: Sequence[Path]) -> List[Path]:
    """Best-effort media copy. Canonical PNGs stay under figures/.

    Destination is under $HOME only. A missing or unwritable directory must
    not fail the job: stage_report still writes the markdown and JSON.
    """
    home = os.environ.get("HOME", "").strip()
    if not home:
        print("[v3] skip media copy (HOME is unset)", flush=True)
        return []
    dest_dir = (
        Path(home)
        / ".cursor"
        / "stores"
        / "bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846"
        / "media"
        / "sae-v3"
    )
    out: List[Path] = []
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in fig_paths:
            _verify_png(src)
            dest = dest_dir / src.name
            dest.write_bytes(src.read_bytes())
            _verify_png(dest)
            out.append(dest)
    except (PermissionError, OSError) as exc:
        print(f"[v3] skip media copy ({exc})", flush=True)
        return []
    return out


def stage_report(args, subjects) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _root, tables, reports, figures, _logs = _dirs()
    loso_path = tables / "v3_loso_lag_scores.csv"
    df = pd.read_csv(loso_path)
    curve_z = subject_gain_curve(df, "sae", "fisher_z_mean")
    curve_r = subject_gain_curve(df, "sae", "r_fisher")
    # subject_gain_curve names the mean column from the value.
    if "mean_z" not in curve_r.columns:
        curve_r = curve_r.rename(columns={c: "mean_r" for c in curve_r.columns if c.startswith("mean")})
    i_peak = argmax_lag(curve_z["mean_z"].to_numpy(), curve_z["lag_ms"].to_numpy())
    peak_ms = float(curve_z["lag_ms"].iloc[i_peak])
    peak_gain_z = float(curve_z["mean_z"].iloc[i_peak])
    peak_sem_z = float(curve_z["sem"].iloc[i_peak])
    peak_gain_r = _gain_r_at(curve_r, peak_ms)

    subj_z = _subject_mean_at(df, "sae", peak_ms, "fisher_z_mean")
    subj_r = _subject_mean_at(df, "sae", peak_ms, "r_fisher")
    p_flip = signflip_p(subj_z.to_numpy())

    # Residual − SAE and residual gain at the peak, on the matched readout.
    resid_full = df[(df["feature_space"] == "residual") & np.isclose(df["lag_ms"], peak_ms)][
        ["subject", "channel", "fisher_z_mean", "r_fisher"]
    ].rename(columns={"fisher_z_mean": "resid_z", "r_fisher": "resid_r"})
    sae_full = df[(df["feature_space"] == "sae") & np.isclose(df["lag_ms"], peak_ms)][
        ["subject", "channel", "fisher_z_mean", "r_fisher"]
    ].rename(columns={"fisher_z_mean": "sae_z", "r_fisher": "sae_r"})
    sur_full = df[(df["feature_space"] == "surprisal") & np.isclose(df["lag_ms"], peak_ms)][
        ["subject", "channel", "fisher_z_mean", "r_fisher"]
    ].rename(columns={"fisher_z_mean": "surp_z", "r_fisher": "surp_r"})
    paired = resid_full.merge(sae_full, on=["subject", "channel"]).merge(sur_full, on=["subject", "channel"])
    paired["resid_minus_sae_z"] = paired["resid_z"] - paired["sae_z"]
    paired["sae_gain_z"] = paired["sae_z"] - paired["surp_z"]
    paired["resid_gain_z"] = paired["resid_z"] - paired["surp_z"]
    sub_delta = paired.groupby("subject")[["resid_minus_sae_z", "sae_gain_z", "resid_gain_z", "sae_z", "surp_z", "resid_z"]].mean()
    resid_minus_sae = float(sub_delta["resid_minus_sae_z"].mean())
    resid_minus_sae_sem = float(sub_delta["resid_minus_sae_z"].std(ddof=1) / np.sqrt(len(sub_delta)))

    # v3 at 300 ms
    row300 = curve_z.loc[np.isclose(curve_z["lag_ms"], 300.0)]
    gain300 = float(row300["mean_z"].iloc[0]) if len(row300) else float("nan")
    sem300 = float(row300["sem"].iloc[0]) if len(row300) else float("nan")

    v2 = _v2_lang_gain()
    null = _null_summary(tables, peak_ms)
    if null.get("available") and null.get("draw_means_z"):
        draws = np.asarray(null["draw_means_z"], dtype=float)
        k = int(np.sum(draws >= peak_gain_z))
        null["p_one_sided_addone"] = (1 + k) / (1 + len(draws))
        null["n_draws_ge_observed"] = k
    ceiling = split_half_reliability(df, peak_ms)
    circ = _circular_curve(tables)

    # Training-fold peak distribution (selection was not on the test fold).
    sae_rows = df[(df["feature_space"] == "sae") & np.isclose(df["lag_ms"], peak_ms)]
    peak_cols = [c for c in ("train_peak_lag_fold0", "train_peak_lag_fold1", "train_peak_lag_fold2") if c in sae_rows.columns]
    train_peaks = sae_rows[peak_cols].to_numpy(dtype=float).ravel()
    train_peaks = train_peaks[np.isfinite(train_peaks)]

    # Bin refits
    bin_path = tables / "v3_bin_refit_peak.csv"
    bin_summary = []
    if bin_path.exists():
        bdf = pd.read_csv(bin_path)
        for name, _lo, _hi in BIN_SLICES:
            part = bdf[bdf["bin_name"] == name]
            if part.empty:
                continue
            # part is SAE-only rows at the peak lag; subtract LOSO surprisal.
            fake = pd.concat([
                part.assign(feature_space="sae"),
                df[(df["feature_space"] == "surprisal") & np.isclose(df["lag_ms"], peak_ms)],
            ], ignore_index=True)
            sm = _subject_mean_at(fake, "sae", peak_ms, "fisher_z_mean")
            bin_summary.append({
                "bin": name,
                "subject_mean_gain_z": float(sm.mean()),
                "sem": float(sm.std(ddof=1) / np.sqrt(sm.size)) if sm.size > 1 else float("nan"),
                "n_subjects": int(sm.size),
                "n_electrodes": int(part[["subject", "channel"]].drop_duplicates().shape[0]),
            })

    dense_path = tables / "v3_dense_ridge_sensitivity.csv"
    dense_summary = {}
    if dense_path.exists() and dense_path.stat().st_size:
        ddf = pd.read_csv(dense_path)
        dsub = ddf.groupby("subject")["fisher_z_mean"].mean()
        dense_summary = {
            "readout": READOUT_DENSE,
            "alpha": DENSE_RIDGE_ALPHA,
            "subject_mean_fisher_z": float(dsub.mean()) if len(dsub) else float("nan"),
            "sem": float(dsub.std(ddof=1) / np.sqrt(dsub.size)) if len(dsub) > 1 else float("nan"),
            "n_subjects": int(dsub.size),
            "note": "sensitivity only; residual comparator is matched LASSO→Ridge",
        }

    # ---- figures ----
    fig_paths = []
    # 1. primary lag curve
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.plot(curve_z["lag_ms"], curve_z["mean_z"], color="#1f77b4", lw=2.0, label="SAE gain")
    ax.fill_between(
        curve_z["lag_ms"],
        curve_z["mean_z"] - curve_z["sem"],
        curve_z["mean_z"] + curve_z["sem"],
        color="#1f77b4", alpha=0.2, linewidth=0,
    )
    if circ is not None and len(circ):
        g = circ.groupby("lag_ms")["mean_z"].mean()
        ax.plot(g.index, g.to_numpy(), color="#7f7f7f", lw=1.5, label="circular-shift mean")
    ax.axvline(peak_ms, color="#1f77b4", ls="--", lw=0.9)
    ax.axvline(300.0, color="#444444", ls=":", lw=0.9)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Subject-mean SAE gain (Fisher-z)")
    ax.set_title("Language electrodes, leave-one-section-out")
    ax.legend(frameon=False)
    fig.tight_layout()
    p1 = figures / "lag_gain_curve.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    fig_paths.append(p1)

    # 2. Fig 4A analogue at peak lag
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    labels = ["SAE full", "Surprisal", "Residual full"]
    means = [
        float(sub_delta["sae_z"].mean()),
        float(sub_delta["surp_z"].mean()),
        float(sub_delta["resid_z"].mean()),
    ]
    sems = [
        float(sub_delta["sae_z"].std(ddof=1) / np.sqrt(len(sub_delta))),
        float(sub_delta["surp_z"].std(ddof=1) / np.sqrt(len(sub_delta))),
        float(sub_delta["resid_z"].std(ddof=1) / np.sqrt(len(sub_delta))),
    ]
    xpos = np.arange(3)
    ax.bar(xpos, means, color=["#1f77b4", "#d62728", "#2ca02c"], width=0.72, zorder=1)
    ax.errorbar(xpos, means, yerr=sems, fmt="none", ecolor="black", capsize=3, zorder=2)
    rng = np.random.default_rng(0)
    for i, col in enumerate(("sae_z", "surp_z", "resid_z")):
        jitter = rng.uniform(-0.12, 0.12, size=len(sub_delta))
        ax.scatter(np.full(len(sub_delta), i) + jitter, sub_delta[col], s=18, c="black", zorder=3)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Subject-mean Fisher-z")
    ax.set_title(f"Fig 4A analogue at {peak_ms:.0f} ms")
    fig.tight_layout()
    p2 = figures / "fig4a_peak_lag.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    fig_paths.append(p2)

    # 3. Fig 5A histogram at the group peak (supports chosen at that fixed lag)
    idx_path = _concat_support_parts(tables / "v3_selected_indices_peak_lag.csv")
    if idx_path is None:
        idx_path = _concat_support_parts(tables / "v3_selected_indices_loso.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if idx_path is not None and idx_path.exists() and idx_path.stat().st_size:
        idx = pd.read_csv(idx_path)
        if "bin_name" in idx.columns and (idx["bin_name"] == "sae_full_at_group_peak").any():
            idx = idx[idx["bin_name"] == "sae_full_at_group_peak"]
        elif "feature_space" in idx.columns:
            idx = idx[idx["feature_space"].isin(["sae", "sae_full_at_group_peak"])]
        vals = idx["feature_index"].to_numpy(dtype=float) if len(idx) else np.array([])
    else:
        vals = np.array([])
    if vals.size:
        ax.hist(vals, bins=64, range=(0, 65536), color="#1f77b4", edgecolor="none")
    for x, lab in ((2048, "2048"), (16384, "16384")):
        ax.axvline(x, color="#444444", ls="--", lw=0.9)
        ax.text(x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1, lab, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Selected SAE index")
    ax.set_ylabel("Selections (electrode × fold)")
    ax.set_title(f"Fig 5A analogue at {peak_ms:.0f} ms")
    fig.tight_layout()
    p3 = figures / "fig5a_selected_index_hist.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    fig_paths.append(p3)

    # 4. Fig 5C bin refits
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if bin_summary:
        xs = np.arange(len(bin_summary))
        ax.bar(xs, [b["subject_mean_gain_z"] for b in bin_summary], color="#1f77b4", width=0.72)
        ax.errorbar(
            xs, [b["subject_mean_gain_z"] for b in bin_summary],
            yerr=[b["sem"] for b in bin_summary], fmt="none", ecolor="black", capsize=3,
        )
        ax.set_xticks(xs, [b["bin"] for b in bin_summary], rotation=15)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_ylabel("Subject-mean SAE gain (Fisher-z)")
    ax.set_title(f"Fig 5C analogue at {peak_ms:.0f} ms")
    fig.tight_layout()
    p4 = figures / "fig5c_bin_refit.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    fig_paths.append(p4)

    # 5. residual − SAE lag curve (matched readout)
    resid_curve = _space_minus_space(df, "residual", "sae", "fisher_z_mean")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.plot(resid_curve["lag_ms"], resid_curve["mean"], color="#2ca02c", lw=2.0)
    ax.fill_between(
        resid_curve["lag_ms"],
        resid_curve["mean"] - resid_curve["sem"],
        resid_curve["mean"] + resid_curve["sem"],
        color="#2ca02c", alpha=0.2, linewidth=0,
    )
    ax.axvline(peak_ms, color="#1f77b4", ls="--", lw=0.9)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Residual Fisher-z − SAE Fisher-z")
    ax.set_title("Matched LASSO→Ridge, language electrodes")
    fig.tight_layout()
    p5 = figures / "residual_minus_sae_lag.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    fig_paths.append(p5)

    copied = _copy_figures(fig_paths)
    curve_z.to_csv(tables / "v3_subject_mean_sae_gain_z.csv", index=False)
    pd.DataFrame([ceiling]).to_csv(tables / "v3_split_half_reliability.csv", index=False)
    if bin_summary:
        pd.DataFrame(bin_summary).to_csv(tables / "v3_bin_refit_summary.csv", index=False)

    def _fmt(x, nd=4):
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    circ_at_peak = "not run"
    if circ is not None and len(circ):
        at = circ[np.isclose(circ["lag_ms"], peak_ms)]["mean_z"]
        circ_at_peak = (
            f"mean across {circ['shift_id'].nunique()} shifts of subject-mean gain "
            f"at {peak_ms:.0f} ms = {_fmt(float(at.mean()))} "
            f"(shift means: {', '.join(_fmt(float(v)) for v in at)})"
        )

    train_peak_txt = "n/a"
    if train_peaks.size:
        train_peak_txt = (
            f"n={train_peaks.size}, median={np.median(train_peaks):.0f} ms, "
            f"mean={np.mean(train_peaks):.0f} ms, "
            f"fraction at group peak={np.mean(np.isclose(train_peaks, peak_ms)):.3f}"
        )

    bin_lines = "\n".join(
        f"- {b['bin']}: subject-mean gain {_fmt(b['subject_mean_gain_z'])} "
        f"(SEM {_fmt(b['sem'])}, {b['n_electrodes']} electrodes, {b['n_subjects']} subjects)"
        for b in bin_summary
    ) or "- bin refits not run"

    null_txt = (
        f"v3 label-shuffle at {peak_ms:.0f} ms, n={null.get('n_draws', 0)}, "
        f"null mean subject gain {_fmt(null.get('null_mean_subject_gain_z', float('nan')))}, "
        f"one-sided add-one p={_fmt(null.get('p_one_sided_addone', float('nan')))} "
        f"({null.get('n_draws_ge_observed', 'n/a')} draws ≥ observed). {null['reason']}."
        if null.get("available") else
        f"Null not run. {null['reason']}."
    )
    dense_txt = (
        f"subject-mean Fisher-z {_fmt(dense_summary.get('subject_mean_fisher_z', float('nan')))} "
        f"(SEM {_fmt(dense_summary.get('sem', float('nan')))}). {dense_summary.get('note', '')}"
        if dense_summary else
        "not run"
    )
    v2_txt = (
        f"v2 language-electrode subject-mean gain on the back-transformed Fisher-z r scale "
        f"is {_fmt(v2.get('subject_mean_gain_r_fisher', float('nan')))} "
        f"(SEM {_fmt(v2.get('sem', float('nan')))}, sign-flip p={_fmt(v2.get('signflip_p', float('nan')))}, "
        f"cv={v2.get('cv', '')}, columns {v2.get('column_full', '')} − {v2.get('column_surprisal', '')}, "
        f"single 300 ms window [200, 400) ms). "
        f"v3 LOSO raw Fisher-z gain at 300 ms is {_fmt(gain300)} (SEM {_fmt(sem300)}). "
        f"v3 back-transformed r_fisher gain at the peak ({peak_ms:.0f} ms) is {_fmt(peak_gain_r)} "
        f"and at 300 ms is {_fmt(_gain_r_at(curve_r, 300.0))}."
        if v2.get("available") else
        "v2 language table was not found."
    )

    # contiguous sensitivity one-liner at its own peak and at 300
    cont_path = tables / "v3_contiguous5_lag_scores.csv"
    if cont_path.exists() and cont_path.stat().st_size:
        cdf = pd.read_csv(cont_path)
        cc = subject_gain_curve(cdf, "sae")
        ic = argmax_lag(cc["mean_z"].to_numpy(), cc["lag_ms"].to_numpy())
        cont_txt = (
            f"5 contiguous within-section folds: peak {float(cc['lag_ms'].iloc[ic]):.0f} ms, "
            f"subject-mean SAE gain {_fmt(float(cc['mean_z'].iloc[ic]))} "
            f"(SEM {_fmt(float(cc['sem'].iloc[ic]))})."
        )
    else:
        cont_txt = "5 contiguous-fold sensitivity not run."

    n_elec = int(df[df["feature_space"] == "sae"][["subject", "channel"]].drop_duplicates().shape[0])
    n_sub = int(df["subject"].nunique())
    lines = [
        "# SAE replication v3 lag-resolved gain",
        "",
        "Qwen3.5-4B-Base Matryoshka layer 15, shared-token features "
        f"`{FEATURE_TAG}`. Language electrodes only ({n_elec} electrodes, {n_sub} subjects). "
        "Primary CV is leave-one-section-out. Inside a LOSO training fold the peak lag is the "
        "lag that maximizes full-model Fisher-z when each training section predicts the other. "
        "The 5-fold sensitivity uses one held-out tail inside the training fold for that choice. "
        "LASSO support is then fit on the whole training fold at that lag and Ridge-refit at "
        "every lag. The test fold never chooses the lag or the support. Scores are raw Fisher-z "
        "(degenerate fold = 0). They are not divided by split-half reliability or by NCSNR.",
        "",
        "## Peak",
        "",
        f"- Peak lag of the subject-mean SAE-gain curve: **{peak_ms:.0f} ms**.",
        f"- Subject-mean SAE gain at that lag (Fisher-z full − Fisher-z surprisal): "
        f"**{_fmt(peak_gain_z)}** (SEM {_fmt(peak_sem_z)} across subjects).",
        f"- Sign-flip p across subjects at that lag: {_fmt(p_flip)}.",
        f"- Back-transformed r_fisher gain at that lag: {_fmt(peak_gain_r)}.",
        f"- Residual minus SAE at that lag, matched readout, subject-mean Fisher-z: "
        f"**{_fmt(resid_minus_sae)}** (SEM {_fmt(resid_minus_sae_sem)}).",
        f"- Subject-mean Fisher-z at that lag: SAE full {_fmt(float(sub_delta['sae_z'].mean()))}, "
        f"surprisal {_fmt(float(sub_delta['surp_z'].mean()))}, "
        f"residual full {_fmt(float(sub_delta['resid_z'].mean()))}.",
        f"- Training-fold peak lags (the lags that chose LASSO support): {train_peak_txt}.",
        "",
        "## Comparison to v2 at 300 ms",
        "",
        v2_txt,
        "",
        "## Null and ceiling",
        "",
        f"- {null_txt}",
        f"- Circular shift of the neural series within each section, features fixed: {circ_at_peak}.",
        f"- Split-half reliability across the three held-out sections at {peak_ms:.0f} ms "
        f"(electrode vectors): SAE-gain mean pairwise r={_fmt(ceiling['gain_mean_pairwise_r'])}, "
        f"Spearman-Brown k=3 {_fmt(ceiling['gain_spearman_brown_k3'])}; "
        f"full-model mean pairwise r={_fmt(ceiling['full_mean_pairwise_r'])}, "
        f"Spearman-Brown k=3 {_fmt(ceiling['full_spearman_brown_k3'])}. "
        "This is a ceiling reported beside the scores, not a divisor.",
        "",
        "## Bin refits at the peak lag",
        "",
        bin_lines,
        "",
        "## Dense-ridge sensitivity",
        "",
        f"- Alpha {DENSE_RIDGE_ALPHA:.0f}, no LASSO, residual + surprisal, LOSO, peak lag only: {dense_txt}",
        "",
        "## Contiguous-fold sensitivity",
        "",
        cont_txt,
        "",
        "## Figures",
        "",
    ]
    for src in fig_paths:
        lines.append(f"- `{src}`")
    for dest in copied:
        lines.append(f"- `{dest}`")
    lines.append("")
    text = "\n".join(lines)
    out_md = reports / "v3_lag_gain.md"
    out_md.write_text(text)
    meta = {
        "peak_lag_ms": peak_ms,
        "peak_gain_z": peak_gain_z,
        "peak_sem_z": peak_sem_z,
        "peak_gain_r_fisher": peak_gain_r,
        "resid_minus_sae_z": resid_minus_sae,
        "signflip_p": p_flip,
        "gain_z_at_300": gain300,
        "null": {k: v for k, v in null.items() if k != "draw_means_z"},
        "ceiling": ceiling,
        "figures": [str(p) for p in fig_paths],
        "media": [str(p) for p in copied],
        "report": str(out_md),
    }
    (tables / "v3_peak_summary.json").write_text(json.dumps(meta, indent=2))
    print(text, flush=True)
    print(f"[v3] wrote {out_md}", flush=True)


def _gain_r_at(curve_r: pd.DataFrame, lag_ms: float) -> float:
    col = "mean_r" if "mean_r" in curve_r.columns else [c for c in curve_r.columns if c.startswith("mean")][0]
    hit = curve_r.loc[np.isclose(curve_r["lag_ms"], lag_ms), col]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _space_minus_space(df: pd.DataFrame, left: str, right: str, value: str) -> pd.DataFrame:
    a = df[df["feature_space"] == left][["subject", "channel", "lag_ms", value]].rename(columns={value: "L"})
    b = df[df["feature_space"] == right][["subject", "channel", "lag_ms", value]].rename(columns={value: "R"})
    m = a.merge(b, on=["subject", "channel", "lag_ms"], how="inner")
    m["delta"] = m["L"] - m["R"]
    sub = m.groupby(["subject", "lag_ms"], as_index=False)["delta"].mean()
    g = sub.groupby("lag_ms")["delta"].agg(mean="mean", std="std", n="count").reset_index()
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    return g.sort_values("lag_ms").reset_index(drop=True)


def stage_all(args, subjects) -> None:
    stage_loso(args, subjects)
    stage_contiguous(args, subjects)
    stage_null(args, subjects)
    stage_bins(args, subjects)
    stage_dense_ridge(args, subjects)
    stage_circshift(args, subjects)
    stage_report(args, subjects)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAE replication v3 lag-resolved encoding")
    p.add_argument(
        "--stage", required=True,
        choices=("loso", "contiguous", "circshift", "null", "bins", "dense-ridge", "report", "all"),
    )
    p.add_argument("--feature-tag", default=FEATURE_TAG)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--n-null", type=int, default=N_NULL)
    p.add_argument("--max-channels", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("SAE_RESULTS_DIRNAME", "sparse_encoding_v3")
    _guard_tag(args.feature_tag)
    _dirs()
    print(
        f"[v3] stage={args.stage} tag={args.feature_tag} n_jobs={args.n_jobs} "
        f"root={v3_root()}",
        flush=True,
    )
    subjects = get_subjects()
    dispatch = {
        "loso": stage_loso,
        "contiguous": stage_contiguous,
        "circshift": stage_circshift,
        "null": stage_null,
        "bins": stage_bins,
        "dense-ridge": stage_dense_ridge,
        "report": stage_report,
        "all": stage_all,
    }
    try:
        dispatch[args.stage](args, subjects)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
