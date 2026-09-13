#!/usr/bin/env python3
"""
sparse_encoding_regression.py
=============================
Simplified **single-lag baseline** for Augmented Sparse Encoding Models.

.. deprecated::
    Prefer ``sparse_encoding_lag_screen.py`` (lag-resolved, primary) and
    ``sparse_encoding_deconvolution.py`` (continuous-time kernels, primary)
    for temporal claims. This script still supports Matryoshka bin refits and
    fixed-lag comparisons at one lag (default 300 ms).

Port of the two-stage regression from
    https://github.com/mlepori1/Interpretable_Encoding_Models
    (src/regression_analysis.py + regression_utils.py)
adapted to predict per-word iEEG responses from **sparse SAE features**
augmented with **word surprisal**.

For every electrode:
  1. Build a per-word response Y (mean signal in a window at a chosen lag),
     reusing ``encoding_channel`` machinery.
  2. Contiguous within-section CV by default (avoids train/test leakage from
     neighboring words in continuous speech). Within each training fold:
       - standardize surprisal only (SAE features are left unscaled, per the
         reference repo),
       - F-test prescreen + LassoCV feature selection,
       - RidgeCV alpha search + Ridge refit on the selected features.
  3. Score with Pearson r (per fold), averaged via Fisher-z.
     Degenerate folds score as r=0 by default (reference behavior).

By default (``--joint_lasso_surprisal``), the ``full`` model matches the
reference ``full_features`` path: surprisal is concatenated into the same
F-test + Lasso pool as the SAE latents, then force-included for Ridge.
Use ``--no-joint_lasso_surprisal`` for the older behavior where Lasso sees
SAE columns only and surprisal is appended afterward.

Three model variants are run so you can reproduce the paper's key contrast
(e.g. frontal electrodes explained by surprisal alone vs. temporal electrodes
needing SAE features):
    full          : SAE features + surprisal
    content       : SAE features only
    surprisal_only: surprisal only

Also reports *paired* Fisher-z scores averaged only over folds where all
requested modes are valid — use these for fair full/content/surprisal
contrasts. Optional ``--shuffle_control`` permutes features as a null.

Prereqs: run ``sae_extract_features.py`` first so each
``section_00X/`` has ``X_word_<tag>.npz`` and ``X_word_surprisal.npy``.

Example
-------
    python -m sparse_encoding.sparse_encoding_regression \\
        --feature_tag sae_qwen35_4b_mat_l15 --subjects Subject04 --lag_ms 300
"""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

# Known-benign numerical warnings for non-responsive electrodes / collinear
# SAE columns. We already guard for these explicitly in the scoring loop.
from scipy.stats import ConstantInputWarning
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

import core.analysis_paths as ap
import core.encoding_channel as ec

# ======================================================================
# CONFIG
# ======================================================================
FEATURE_TAG = "sae_gemma2_2b_mat_l12"
FS_TARGET = 500.0
RESP_WIN_MS = 200.0
RESPONSE_LAG_MS = 300.0        # single lag at which per-word Y is summarized
N_FOLDS = 5
RANDOM_STATE = 19
LASSO_K_BEST = 8000            # F-test prescreen before LassoCV
MODES = ("full", "content", "surprisal_only")
SECTIONS = (1, 2, 3)


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration for progress / ETA lines."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _progress_line(
    label: str, done: int, total: int, t0: float, kept: Optional[int] = None,
) -> str:
    """Build a single progress line with % complete, rate, and ETA."""
    elapsed = max(time.time() - t0, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = remaining / rate if rate > 0 else float("nan")
    pct = 100.0 * done / total if total else 100.0
    kept_s = f", kept={kept}" if kept is not None else ""
    return (
        f"  [{label}] {done}/{total} ({pct:5.1f}%){kept_s} | "
        f"elapsed {_fmt_duration(elapsed)} | "
        f"{rate:.2f}/s | ETA {_fmt_duration(eta)}"
    )


# ======================================================================
# Feature loading
# ======================================================================

def load_sae_features(
    feature_tag: str, sections,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Concatenate SAE latents, tagged surprisal, and validity across sections.

    Prefers ``X_word_<tag>_surprisal.npy`` and ``X_word_<tag>_valid.npy`` so
    Gemma/Qwen extractions cannot overwrite each other. Falls back to the
    legacy untagged surprisal file (all-True validity) with a warning.

    Row order matches ``encoding_channel.load_subject_data`` (sections in order,
    words within a section in ``word_timing.csv`` order).
    """
    mats, surps, valids = [], [], []
    used_legacy = False
    for sid in sections:
        sec_dir = ap.FEATURES_DIR / f"section_{sid:03d}"
        npz = sec_dir / f"X_word_{feature_tag}.npz"
        surp_tagged = sec_dir / f"X_word_{feature_tag}_surprisal.npy"
        surp_legacy = sec_dir / "X_word_surprisal.npy"
        valid_path = sec_dir / f"X_word_{feature_tag}_valid.npy"
        if not npz.exists():
            raise FileNotFoundError(
                f"{npz} not found. Run sae_extract_features.py first.")
        mat = sparse.load_npz(npz).tocsr()
        mats.append(mat)
        if surp_tagged.exists():
            surps.append(np.load(surp_tagged).reshape(-1, 1))
        elif surp_legacy.exists():
            used_legacy = True
            surps.append(np.load(surp_legacy).reshape(-1, 1))
        else:
            surps.append(np.full((mat.shape[0], 1), np.nan, np.float32))
        if valid_path.exists():
            valids.append(np.load(valid_path).astype(bool).ravel())
        else:
            valids.append(np.ones(mat.shape[0], dtype=bool))
    if used_legacy:
        print(
            f"[warn] load_sae_features({feature_tag}): using legacy untagged "
            "X_word_surprisal.npy; re-extract for tagged surprisal + validity.",
            flush=True,
        )
    X = sparse.vstack(mats).tocsr()
    surprisal = np.vstack(surps).astype(np.float64).ravel()
    validity = np.concatenate(valids).astype(bool)
    if validity.shape[0] != X.shape[0]:
        raise ValueError(
            f"validity length {validity.shape[0]} != SAE rows {X.shape[0]}")
    return X, surprisal, validity


def load_residual_features(
    feature_tag: str, sections,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Load dense residual-stream matrices for the same word rows as the SAE.

    Expects ``X_word_<tag>_resid.npy`` plus the SAE tag's surprisal/validity
    files so residual and SAE encodings share the same trial mask.
    Returned as CSR so ``regress_electrode`` can run the paper LASSO→Ridge
    path without a second encoder.
    """
    mats, surps, valids = [], [], []
    for sid in sections:
        sec_dir = ap.FEATURES_DIR / f"section_{sid:03d}"
        npy = sec_dir / f"X_word_{feature_tag}_resid.npy"
        if not npy.exists():
            raise FileNotFoundError(
                f"{npy} not found. Extract residuals with sae_extract_features.py "
                "--resid_only (or the matching residual dump).")
        mats.append(np.load(npy).astype(np.float32, copy=False))
        surp_tagged = sec_dir / f"X_word_{feature_tag}_surprisal.npy"
        surp_legacy = sec_dir / "X_word_surprisal.npy"
        valid_path = sec_dir / f"X_word_{feature_tag}_valid.npy"
        n_rows = mats[-1].shape[0]
        if surp_tagged.exists():
            surps.append(np.load(surp_tagged).reshape(-1, 1))
        elif surp_legacy.exists():
            surps.append(np.load(surp_legacy).reshape(-1, 1))
        else:
            surps.append(np.full((n_rows, 1), np.nan, np.float32))
        if valid_path.exists():
            valids.append(np.load(valid_path).astype(bool).ravel())
        else:
            valids.append(np.ones(n_rows, dtype=bool))
    X = np.vstack(mats)
    surprisal = np.vstack(surps).astype(np.float64).ravel()
    validity = np.concatenate(valids).astype(bool)
    if validity.shape[0] != X.shape[0]:
        raise ValueError(
            f"validity length {validity.shape[0]} != residual rows {X.shape[0]}")
    if surprisal.shape[0] != X.shape[0]:
        raise ValueError(
            f"surprisal length {surprisal.shape[0]} != residual rows {X.shape[0]}")
    return sparse.csr_matrix(X), surprisal, validity


# ======================================================================
# Two-stage regression (ported from regression_utils.py)
# ======================================================================

def select_alpha(X, y) -> float:
    alphas = [10 ** i for i in range(-2, 6)]
    m = RidgeCV(alphas=alphas, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return float(m.alpha_)


def compute_support_features(X, y, k_best: int = LASSO_K_BEST) -> Optional[np.ndarray]:
    """L1 feature selection with an F-test prescreen. Returns a boolean mask.

    Ported from the reference repo's
    ``regression_utils.compute_support_features``.
    """
    n_feat = X.shape[1]
    if n_feat > k_best:
        selector = SelectKBest(f_regression, k=k_best)
        Xs = selector.fit_transform(X, y)
        screen_mask = selector.get_support()
    else:
        Xs = X
        screen_mask = np.ones(n_feat, dtype=bool)

    reg = LassoCV(
        alphas=np.logspace(-2, 0, 10), cv=5, tol=1e-3,
        selection="random", max_iter=2000, random_state=RANDOM_STATE,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(Xs, y)
    lasso_support = SelectFromModel(reg, prefit=True).get_support()
    if int(np.sum(lasso_support)) == 0:
        return None
    support = np.zeros(n_feat, dtype=bool)
    support[np.where(screen_mask)[0][lasso_support]] = True
    return support


def compute_joint_support(
    X, surp, y, k_best: int = LASSO_K_BEST,
) -> np.ndarray:
    """F-test + Lasso where surprisal competes in the same candidate pool as ``X``.

    Matches the reference ``full_features`` path::

        acts_train = concat([acts_train, logprobs_train])
        support = compute_support_features(acts_train, ...)
        support[logprob_index] = True   # force-include for Ridge (caller)

    Returns a boolean mask over ``X``'s columns only. Surprisal is always
    force-included by the caller for the Ridge fit. If nothing survives
    selection, returns an all-False mask so the caller can fall back to
    surprisal-only for that fold.
    """
    n_feat = X.shape[1]
    X_sp = X if sparse.issparse(X) else sparse.csr_matrix(X)
    surp_col = sparse.csr_matrix(np.asarray(surp, dtype=np.float64).reshape(-1, 1))
    X_joint = sparse.hstack([X_sp, surp_col]).tocsr()

    support_joint = compute_support_features(X_joint, y, k_best=k_best)
    if support_joint is None:
        return np.zeros(n_feat, dtype=bool)
    return support_joint[:n_feat]


def _to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _fit_predict(Xtr, y_tr, Xte):
    """Ridge with CV alpha search; returns test predictions (warnings muted)."""
    alpha = select_alpha(Xtr, y_tr)
    model = Ridge(alpha=alpha, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte).ravel()
    return pred, alpha


def _fold_r(pred: np.ndarray, gt: np.ndarray, degenerate_as_zero: bool) -> float:
    """Pearson r for one fold; NaN or 0 when the fold is degenerate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (np.all(np.isfinite(pred)) and np.all(np.isfinite(gt))
                and np.std(pred) > 1e-12 and np.std(gt) > 1e-12):
            r, _ = pearsonr(gt, pred)
            if np.isnan(r):
                return 0.0 if degenerate_as_zero else np.nan
            return float(r)
    return 0.0 if degenerate_as_zero else np.nan


def _aggregate_rs(rs: List[float]) -> Tuple[float, float, int]:
    arr = np.asarray(rs, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(arr[valid], -0.999999, 0.999999))
    return float(np.mean(arr[valid])), float(np.tanh(np.mean(z))), int(valid.sum())


def _support_stats(supports) -> Tuple[int, int, float]:
    if not supports:
        return 0, 0, 0.0
    sup = np.asarray(supports, dtype=bool)
    return (int(np.sum(np.any(sup, axis=0))),
            int(np.sum(np.all(sup, axis=0))),
            float(np.mean(np.sum(sup, axis=1))))


def make_cv_splits(
    n_words: int,
    n_folds: int,
    cv_mode: str,
    section_slices: Optional[List[Tuple[int, int]]] = None,
    ok: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) in the *full* word index space.

    Contiguous mode builds folds within sections (same idea as
    ``encoding_channel.make_folds_within_sections``), then intersects each
    fold with ``ok`` words for this electrode.
    """
    if ok is None:
        ok = np.ones(n_words, dtype=bool)
    ok_idx = np.flatnonzero(ok)
    if ok_idx.size < 50:
        return []

    if cv_mode == "shuffled":
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        # Split only among ok words so fold sizes stay balanced.
        return [(ok_idx[tr], ok_idx[te]) for tr, te in kfold.split(ok_idx)]

    if cv_mode != "contiguous":
        raise ValueError(f"Unknown cv_mode: {cv_mode}")
    if not section_slices:
        raise ValueError("contiguous CV requires section_slices")

    fold_ranges = ec.make_folds_within_sections(
        section_slices, n_folds, min_per_section=1)
    splits = []
    for fa, fb in fold_ranges:
        te = ok_idx[(ok_idx >= fa) & (ok_idx < fb)]
        tr = ok_idx[(ok_idx < fa) | (ok_idx >= fb)]
        if te.size == 0 or tr.size < 20:
            continue
        splits.append((tr, te))
    return splits


def maybe_shuffle_features(
    X_sae: sparse.csr_matrix,
    surprisal: np.ndarray,
    shuffle_control: str,
    rng: np.random.Generator,
    valid: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Permute feature rows (keep Y aligned) for a null / shuffled control.

    When ``valid`` is provided, only rows with ``valid[i]`` are permuted among
    themselves; invalid rows stay fixed so NaN/invalid tokens do not leak into
    training folds after shuffle.
    """
    if shuffle_control in (None, "", "none"):
        return X_sae, surprisal
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        idx = np.flatnonzero(valid)
        if idx.size < 2:
            return X_sae, surprisal
        perm = idx[rng.permutation(idx.size)]
        X_out = X_sae.copy()
        surp_out = surprisal.copy()
        X_out[idx] = X_sae[perm]
        surp_out[idx] = surprisal[perm]
        return X_out, surp_out
    n = X_sae.shape[0]
    perm = rng.permutation(n)
    return X_sae[perm], surprisal[perm]


def regress_electrode(
    X_sae: sparse.csr_matrix,
    surprisal: np.ndarray,
    y: np.ndarray,
    modes: Tuple[str, ...],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    joint_lasso_surprisal: bool = True,
    degenerate_as_zero: bool = True,
) -> Dict[str, float]:
    """Two-stage regression for one electrode across provided CV folds.

    ``content`` always uses SAE-only Lasso. ``full`` uses joint Lasso (surprisal
    competes in the pool, then is force-included for Ridge) when
    ``joint_lasso_surprisal`` is True — matching the reference
    ``full_features`` path. When False, ``full`` reuses the SAE-only support
    and appends surprisal after selection.

    Also reports ``*_R_fisher_paired`` averaged only over folds where every
    requested mode produced a finite score — use these for fair contrasts.
    """
    if not folds:
        return {}

    need_content = "content" in modes
    need_full = "full" in modes
    need_sae_only = need_content or (need_full and not joint_lasso_surprisal)

    fold_rs = {m: [] for m in modes}
    supports = {m: [] for m in modes}
    alphas = {m: [] for m in modes}
    n_folds_total = 0

    for tr, te in folds:
        if te.size < 5 or tr.size < 20:
            continue
        n_folds_total += 1
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler().fit(surprisal[tr].reshape(-1, 1))
        surp_tr = sc.transform(surprisal[tr].reshape(-1, 1))
        surp_te = sc.transform(surprisal[te].reshape(-1, 1))

        content_support = None
        content_tr = content_te = None
        full_support = None
        full_tr = full_te = None
        acts_tr = X_sae[tr] if (need_sae_only or need_full) else None

        if need_sae_only:
            content_support = compute_support_features(acts_tr, y_tr)
            if content_support is not None:
                content_tr = _to_dense(acts_tr[:, content_support])
                content_te = _to_dense(X_sae[te][:, content_support])

        if need_full:
            if joint_lasso_surprisal:
                full_support = compute_joint_support(acts_tr, surp_tr, y_tr)
                if full_support.any():
                    full_tr = _to_dense(acts_tr[:, full_support])
                    full_te = _to_dense(X_sae[te][:, full_support])
                else:
                    full_support = None
            else:
                full_support = content_support
                full_tr, full_te = content_tr, content_te

        for m in modes:
            if m == "surprisal_only":
                Xtr, Xte = surp_tr, surp_te
            elif m == "full":
                if full_support is None:
                    Xtr, Xte = surp_tr, surp_te
                else:
                    supports[m].append(full_support)
                    Xtr = np.hstack([full_tr, surp_tr])
                    Xte = np.hstack([full_te, surp_te])
            elif m == "content":
                if content_support is None:
                    fold_rs[m].append(
                        0.0 if degenerate_as_zero else np.nan)
                    continue
                supports[m].append(content_support)
                Xtr, Xte = content_tr, content_te
            else:
                raise ValueError(f"Unknown mode: {m}")

            pred, alpha = _fit_predict(Xtr, y_tr, Xte)
            fold_rs[m].append(_fold_r(pred, y_te, degenerate_as_zero))
            alphas[m].append(alpha)

    out: Dict[str, float] = {"n_folds_total": float(n_folds_total)}
    for m in modes:
        r_mean, r_fisher, nvalid = _aggregate_rs(fold_rs[m])
        union, inter, fmean = _support_stats(supports[m])
        out[f"{m}__R"] = r_mean
        out[f"{m}__R_fisher"] = r_fisher
        out[f"{m}__n_folds_valid"] = nvalid
        out[f"{m}__all_folds_valid"] = int(
            nvalid == n_folds_total and n_folds_total > 0)
        out[f"{m}__feature_union"] = union
        out[f"{m}__feature_intersection"] = inter
        out[f"{m}__feature_mean"] = fmean
        out[f"{m}__alpha_median"] = (
            float(np.median(alphas[m])) if alphas[m] else np.nan)

    # Paired scores: only folds where every mode is finite.
    if fold_rs and n_folds_total > 0:
        stacked = np.vstack([np.asarray(fold_rs[m], dtype=float) for m in modes])
        paired_ok = np.all(np.isfinite(stacked), axis=0)
        out["n_folds_paired"] = int(paired_ok.sum())
        out["all_folds_paired"] = int(
            paired_ok.sum() == n_folds_total and n_folds_total > 0)
        for i, m in enumerate(modes):
            _, r_f, n_p = _aggregate_rs(stacked[i, paired_ok].tolist())
            out[f"{m}__R_fisher_paired"] = r_f
            out[f"{m}__n_folds_paired"] = n_p
    return out


# ======================================================================
# Per-subject driver
# ======================================================================

def process_subject(subject: str, eeg_files: Dict[int, str], args) -> pd.DataFrame:
    eeg_files_abs = {sid: str(ap.DATA_ROOT / p) for sid, p in eeg_files.items()}
    missing = [p for p in eeg_files_abs.values() if not Path(p).exists()]
    if missing:
        print(f"[{subject}] missing EEG files, skipping: {missing}")
        return pd.DataFrame()

    # Load EEG + word timing (feature_sets=["glove"] just to satisfy the loader;
    # we ignore its X_by_feature and use the SAE latents instead).
    data = ec.load_subject_data(
        subject_name=subject,
        eeg_files=eeg_files_abs,
        features_dir=ap.FEATURES_DIR,
        fs_target=args.fs_target,
        sections=tuple(args.sections),
        feature_sets=["glove"],
    )

    feature_kind = getattr(args, "feature_kind", "sae")
    if feature_kind == "residual":
        X_sae, surprisal, feat_valid = load_residual_features(
            args.feature_tag, args.sections)
        print(
            f"  [{subject}] residual LASSO→Ridge: "
            f"{X_sae.shape[0]} words × {X_sae.shape[1]} dims",
            flush=True,
        )
    else:
        X_sae, surprisal, feat_valid = load_sae_features(
            args.feature_tag, args.sections)
    col_start = int(getattr(args, "sae_col_start", 0) or 0)
    col_end = getattr(args, "sae_col_end", None)
    col_end = int(col_end) if col_end is not None else X_sae.shape[1]
    if col_start != 0 or col_end != X_sae.shape[1]:
        if col_start < 0 or col_end > X_sae.shape[1] or col_start >= col_end:
            raise ValueError(
                f"Invalid SAE column slice [{col_start}:{col_end}] "
                f"for d={X_sae.shape[1]}")
        print(
            f"  [{subject}] SAE column slice [{col_start}:{col_end}] "
            f"(d={col_end - col_start})",
            flush=True,
        )
        X_sae = X_sae[:, col_start:col_end]
    Wtot = data.section_word_slices[-1][1]
    if X_sae.shape[0] != Wtot:
        raise ValueError(
            f"[{subject}] word count mismatch: SAE has {X_sae.shape[0]} rows, "
            f"EEG pipeline has {Wtot}. Re-extract with matching sections.")

    half_win = max(1, int(round((args.resp_win_ms / 1000.0) * args.fs_target / 2.0)))
    lag_samp = int(round((args.lag_ms / 1000.0) * args.fs_target))
    Y, valid = ec.compute_Y_for_lag(data, lag_samp, half_win)

    # Trial validity: neural windows + owned-token feature validity + finite
    # surprisal, *before* any feature shuffle so Y alignment stays meaningful.
    base_valid = valid & feat_valid & np.isfinite(surprisal)
    section_slices = list(data.section_word_slices)

    # Global shuffle: one feature permutation shared across electrodes.
    if args.shuffle_control == "global":
        off = int(getattr(args, "null_seed_offset", 0) or 0)
        subj_seed = RANDOM_STATE + off + sum(ord(c) for c in subject)
        rng = np.random.default_rng(subj_seed)
        X_sae, surprisal = maybe_shuffle_features(
            X_sae, surprisal, "global", rng, valid=feat_valid)

    channel_items = list(enumerate(data.channel_labels))
    if getattr(args, "lang_only", False):
        tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
        keep = set(
            tax.loc[
                (tax["subject"] == subject)
                & tax["is_lang"].fillna(False).astype(bool),
                "channel",
            ].astype(str).str.strip()
        )
        channel_items = [
            (ci, ch) for ci, ch in channel_items if str(ch).strip() in keep
        ]
        print(
            f"  [{subject}] lang_only: {len(channel_items)}/"
            f"{len(data.channel_labels)} channels",
            flush=True,
        )
    if args.max_channels:
        channel_items = channel_items[: args.max_channels]
    channels = [ch for _, ch in channel_items]
    modes = tuple(args.modes)

    def run_channel(ci: int, ch: str):
        y = Y[:, ci].astype(np.float64)
        ok = base_valid & np.isfinite(y)
        if int(ok.sum()) < 50:
            return None
        X_use, surp_use = X_sae, surprisal
        if args.shuffle_control == "per_electrode":
            X_use, surp_use = maybe_shuffle_features(
                X_sae, surprisal, "global",
                np.random.default_rng(RANDOM_STATE + 17 * ci + 3),
                valid=feat_valid,
            )
        folds = make_cv_splits(
            Wtot, args.n_folds, args.cv, section_slices, ok=ok)
        if not folds:
            return None
        res = regress_electrode(
            X_use, surp_use, y, modes, folds,
            joint_lasso_surprisal=args.joint_lasso_surprisal,
            degenerate_as_zero=args.degenerate_as_zero,
        )
        if not res:
            return None
        out_tag = args.feature_tag
        if getattr(args, "feature_kind", "sae") == "residual":
            out_tag = f"{args.feature_tag}_resid"
        res.update({"subject": subject, "channel": ch,
                    "channel_index": ci, "n_words": int(ok.sum()),
                    "cv": args.cv, "shuffle_control": args.shuffle_control,
                    "feature_tag": out_tag,
                    "feature_kind": getattr(args, "feature_kind", "sae"),
                    "sae_col_start": int(getattr(args, "sae_col_start", 0) or 0),
                    "sae_col_end": (
                        int(args.sae_col_end)
                        if getattr(args, "sae_col_end", None) is not None
                        else -1),
                    })
        return res

    n_ch = len(channels)
    # Print often enough to track ETA, but not every single channel.
    report_every = max(1, min(20, n_ch // 10 or 1))
    t0 = time.time()
    print(f"  [{subject}] regressing {n_ch} channels "
          f"(n_jobs={args.n_jobs})…", flush=True)

    if args.n_jobs == 1:
        rows = []
        for done_i, (ci, ch) in enumerate(channel_items):
            r = run_channel(ci, ch)
            if r is not None:
                rows.append(r)
            done = done_i + 1
            if done % report_every == 0 or done == n_ch:
                print(_progress_line(subject, done, n_ch, t0, kept=len(rows)),
                      flush=True)
    else:
        from joblib import Parallel, delayed
        # Generator mode so we can stream progress / ETA while workers run.
        results_iter = Parallel(
            n_jobs=args.n_jobs, backend="loky", return_as="generator",
        )(delayed(run_channel)(ci, ch) for ci, ch in channel_items)
        rows = []
        for done, r in enumerate(results_iter, 1):
            if r is not None:
                rows.append(r)
            if done % report_every == 0 or done == n_ch:
                print(_progress_line(subject, done, n_ch, t0, kept=len(rows)),
                      flush=True)

    print(
        f"  [{subject}] done: {len(rows)}/{n_ch} electrodes kept "
        f"in {_fmt_duration(time.time() - t0)}",
        flush=True,
    )
    return pd.DataFrame(rows)


def get_subjects() -> Dict[str, Dict]:
    """Reuse the SUBJECTS map from encoding_group when available."""
    try:
        from pipelines.encoding_group import SUBJECTS
        return SUBJECTS
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import SUBJECTS from encoding_group; pass --subjects and "
            "ensure encoding_group.py is importable.") from e


def default_out_path(args) -> Path:
    """Auto-suffix the results CSV for shuffle / CV / lang / bin variants."""
    base = ap.SAE_REGRESSION_TABLE
    stem, suf = base.stem, base.suffix
    parts = []
    tag = getattr(args, "feature_tag", None)
    if tag and tag != FEATURE_TAG:
        parts.append(tag)
    if args.cv != "contiguous":
        parts.append(f"cv{args.cv}")
    if args.shuffle_control and args.shuffle_control != "none":
        parts.append(f"shuffle_{args.shuffle_control}")
    if getattr(args, "feature_kind", "sae") == "residual":
        parts.append("resid_lasso")
    if getattr(args, "lang_only", False):
        parts.append("lang")
    col_start = int(getattr(args, "sae_col_start", 0) or 0)
    col_end = getattr(args, "sae_col_end", None)
    if col_start or col_end is not None:
        end_s = "end" if col_end is None else str(int(col_end))
        parts.append(f"bin{col_start}-{end_s}")
    if not parts:
        return base
    return base.with_name(f"{stem}_{'_'.join(parts)}{suf}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument(
        "--feature_kind", choices=("sae", "residual"), default="sae",
        help="sae: sparse latents from X_word_<tag>.npz. residual: dense "
             "X_word_<tag>_resid.npy through the same F-test+LASSO→Ridge path.",
    )
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject ids; default = all in encoding_group.SUBJECTS")
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--lag_ms", type=float, default=RESPONSE_LAG_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument(
        "--joint_lasso_surprisal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let surprisal compete in the F-test+Lasso candidate pool before "
             "being force-included in Ridge (matches the reference "
             "full_features path). Use --no-joint_lasso_surprisal for the "
             "simpler SAE-only-selection behavior.",
    )
    p.add_argument(
        "--cv", choices=("contiguous", "shuffled"), default="contiguous",
        help="Outer CV scheme. Contiguous (default) keeps test blocks within "
             "sections and avoids neighboring-word leakage. 'shuffled' matches "
             "the reference repo's KFold(shuffle=True) but is less appropriate "
             "for continuous narrative iEEG.",
    )
    p.add_argument(
        "--n_folds", type=int, default=N_FOLDS,
        help="Outer CV folds (default 5). Contiguous mode allocates folds "
             "across sections via encoding_channel.make_folds_within_sections.",
    )
    p.add_argument(
        "--shuffle_control",
        choices=("none", "global", "per_electrode"),
        default="none",
        help="Null control: permute SAE+surprisal rows vs neural Y. "
             "'global' = one shuffle per subject; 'per_electrode' = independent "
             "shuffle per electrode. Auto-suffixes the output CSV.",
    )
    p.add_argument(
        "--degenerate_as_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score empty/constant folds as r=0 (reference default). "
             "Use --no-degenerate_as_zero to treat them as NaN and drop them "
             "from Fisher averages.",
    )
    p.add_argument("--max_channels", type=int, default=0,
                   help="Limit channels per subject (0 = all; useful for tests).")
    p.add_argument(
        "--lang_only", action="store_true",
        help="Restrict to is_lang electrodes from the taxonomy table.",
    )
    p.add_argument(
        "--sae_col_start", type=int, default=0,
        help="Inclusive start index for a Matryoshka SAE column slice.",
    )
    p.add_argument(
        "--sae_col_end", type=int, default=None,
        help="Exclusive end index for a Matryoshka SAE column slice "
             "(default: all remaining columns).",
    )
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Parallel workers across electrodes (-1 = all cores, "
                        "1 = serial). joblib limits inner BLAS threads to avoid "
                        "oversubscription.")
    p.add_argument(
        "--null_seed_offset",
        type=int,
        default=0,
        help="Added to the shuffle RNG seed (for multi-draw feature-shuffle nulls).",
    )
    p.add_argument("--out", default=None, help="Output CSV (default per analysis_paths).")
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute subjects already present in the output CSV. "
             "Default is to skip them and checkpoint after each new subject "
             "so a crash does not discard finished people.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    print(
        "[deprecated as primary] sparse_encoding_regression.py is a "
        f"single-lag baseline (lag_ms={args.lag_ms}). For temporal "
        "dynamics use sparse_encoding_lag_screen.py and "
        "sparse_encoding_deconvolution.py (see README_TEMPORAL.md).",
        flush=True,
    )

    subjects_map = get_subjects()
    requested = args.subjects or list(subjects_map.keys())
    subj_ids = []
    for s in requested:
        if s in subjects_map:
            subj_ids.append(s)
        else:
            print(f"[skip] {s} not in SUBJECTS", flush=True)
    n_subj = len(subj_ids)
    print(
        f"Sparse encoding regression: tag={args.feature_tag}, "
        f"kind={args.feature_kind}, "
        f"subjects={n_subj}, modes={args.modes}, cv={args.cv}, "
        f"n_folds={args.n_folds}, shuffle_control={args.shuffle_control}, "
        f"joint_lasso_surprisal={args.joint_lasso_surprisal}, "
        f"degenerate_as_zero={args.degenerate_as_zero}",
        flush=True,
    )

    out_path = Path(args.out) if args.out else default_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    done_subjects = set()
    if out_path.exists() and not args.overwrite:
        prev = pd.read_csv(out_path)
        if not prev.empty and "subject" in prev.columns:
            done_subjects = set(prev["subject"].astype(str).unique())
            all_dfs.append(prev)
            print(
                f"[resume] {out_path.name} already has {len(done_subjects)} "
                f"subjects: {sorted(done_subjects)}",
                flush=True,
            )

    t0_all = time.time()
    n_run = 0
    for i, sid in enumerate(subj_ids, 1):
        if sid in done_subjects:
            print(f"\n=== [{i}/{n_subj}] {sid} [resume skip] ===", flush=True)
            continue
        print(f"\n=== [{i}/{n_subj}] {sid} ===", flush=True)
        t0 = time.time()
        df = process_subject(sid, subjects_map[sid]["eeg_files"], args)
        n_run += 1
        if not df.empty:
            all_dfs.append(df)
            done_subjects.add(sid)
            pd.concat(all_dfs, ignore_index=True).to_csv(out_path, index=False)
            print(
                f"  checkpoint {sid}: {len(df)} rows -> {out_path}",
                flush=True,
            )
        gc.collect()
        if i < n_subj:
            elapsed = time.time() - t0_all
            denom = max(n_run, 1)
            remaining = n_subj - i
            eta = (elapsed / denom) * remaining
            print(
                f"[subjects] {i}/{n_subj} done | last {_fmt_duration(time.time() - t0)} | "
                f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}",
                flush=True,
            )

    if not all_dfs:
        print("No results produced.")
        return

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(
        f"\nSaved {len(out)} electrode rows -> {out_path} "
        f"(total {_fmt_duration(time.time() - t0_all)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
