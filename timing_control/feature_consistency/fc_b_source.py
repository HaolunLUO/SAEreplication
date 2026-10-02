#!/usr/bin/env python3
"""One source electrode: sign-constrained ridge onto every target.

Array index is the source electrode's row in generalization/population.csv.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler

import fc_common as C
import sparse_encoding.sparse_encoding_v3 as v3
from sparse_encoding.sparse_encoding_regression import _fold_r

GEN = C.OUT / "generalization"


def fit_nn(Xnn: np.ndarray, s: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Non-negative SAE weights, free surprisal weight, free intercept.

    Returns coef of length p+1 (SAE then surprisal). Intercept is coef[-1] of
    the augmented vector and is applied separately by the caller via the
    returned (weights, intercept) pair.
    """
    n, p = Xnn.shape
    A = np.column_stack([Xnn, s, np.ones(n)])
    pen = np.sqrt(alpha) * np.eye(p + 1, p + 2)
    A_aug = np.vstack([A, pen])
    b = np.concatenate([y, np.zeros(p + 1)])
    lb = np.concatenate([np.zeros(p), [-np.inf, -np.inf]])
    ub = np.full(p + 2, np.inf)
    res = lsq_linear(A_aug, b, bounds=(lb, ub), method="bvls", tol=1e-8, max_iter=120)
    if not res.success:
        res = lsq_linear(A_aug, b, bounds=(lb, ub), method="trf", tol=1e-8, max_iter=200)
    return res.x


def predict_rows(x: np.ndarray, Xnn: np.ndarray, s: np.ndarray) -> np.ndarray:
    return Xnn @ x[:-2] + s * x[-2] + x[-1]


def score_target(F, surp, y, tr, te, section_id, alphas):
    """F is sign-flipped SAE columns, shape (n_words, p). p may be 0."""
    if tr.size < 20 or te.size < 5 or not np.isfinite(y[tr]).all() or np.std(y[tr]) < 1e-12:
        return 0.0
    sc = StandardScaler().fit(surp[tr].reshape(-1, 1))
    s_all = np.full(surp.shape[0], np.nan)
    s_all[tr] = sc.transform(surp[tr].reshape(-1, 1)).ravel()
    s_all[te] = sc.transform(surp[te].reshape(-1, 1)).ravel()
    p = F.shape[1]
    inners = v3.loso_inner_splits(tr, section_id)
    best_alpha = alphas[0]
    best_mse = np.inf
    if p == 0:
        # Unconstrained one-column ridge. Alpha from the same inner splits.
        from sklearn.linear_model import Ridge
        for alpha in alphas:
            mses = []
            for itr, ite in inners:
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(s_all[itr].reshape(-1, 1), y[itr])
                pred = model.predict(s_all[ite].reshape(-1, 1)).ravel()
                mses.append(float(np.mean((y[ite] - pred) ** 2)))
            mse = float(np.mean(mses)) if mses else np.inf
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        model = Ridge(alpha=best_alpha, fit_intercept=True)
        model.fit(s_all[tr].reshape(-1, 1), y[tr])
        pred = model.predict(s_all[te].reshape(-1, 1)).ravel()
        return _fold_r(pred, y[te], True)

    Xnn = F
    for alpha in alphas:
        mses = []
        for itr, ite in inners:
            x = fit_nn(Xnn[itr], s_all[itr], y[itr], alpha)
            pred = predict_rows(x, Xnn[ite], s_all[ite])
            mses.append(float(np.mean((y[ite] - pred) ** 2)))
        mse = float(np.mean(mses)) if mses else np.inf
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    x = fit_nn(Xnn[tr], s_all[tr], y[tr], best_alpha)
    pred = predict_rows(x, Xnn[te], s_all[te])
    return _fold_r(pred, y[te], True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=None)
    args = parser.parse_args()
    source = args.source
    if source is None:
        source = int(os.environ["SLURM_ARRAY_TASK_ID"])
    out = GEN / f"source_{source:03d}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"[B] {out.name} exists, skipping", flush=True)
        return

    pop = pd.read_csv(GEN / "population.csv")
    if source < 0 or source >= len(pop):
        raise RuntimeError(f"source {source} outside 0..{len(pop)-1}")
    signs = pd.read_csv(C.OUT / "electrode_signs.csv")
    src = pop.iloc[source]
    src_signs = signs[signs["elec_i"] == source].sort_values("feature_index")
    feat_index = np.load(GEN / "feature_index.npy")
    columns = np.load(GEN / "feature_columns.npy", mmap_mode="r")
    surp = np.load(GEN / "surprisal.npy")
    z = np.load(GEN / "residuals.npz", allow_pickle=True)
    y_resid = z["y_resid"]
    train_idx = z["train_idx"]
    test_idx = z["test_idx"]
    section_id = z["section_id"]
    surp_fresh = z["surprisal_fresh_r"]

    if len(src_signs) == 0:
        F = np.zeros((columns.shape[0], 0), dtype=np.float64)
    else:
        pos = []
        flip = []
        for rec in src_signs.itertuples(index=False):
            j = int(np.searchsorted(feat_index, int(rec.feature_index)))
            if j >= feat_index.size or int(feat_index[j]) != int(rec.feature_index):
                raise RuntimeError(f"feature {rec.feature_index} missing from precompute")
            pos.append(j)
            flip.append(float(rec.sign))
        F = np.asarray(columns[:, pos], dtype=np.float64) * np.asarray(flip, dtype=np.float64)

    n = len(pop)
    rows = []
    for t in range(n):
        rs = []
        for fold in range(3):
            if F.shape[1] == 0:
                rs.append(float(surp_fresh[t, fold]))
                continue
            tr = np.asarray(train_idx[t, fold], dtype=int)
            te = np.asarray(test_idx[t, fold], dtype=int)
            y = y_resid[t, fold]
            rs.append(score_target(F, surp, y, tr, te, section_id, C.ALPHAS))
        rows.append({
            "source_i": source,
            "target_i": t,
            "r_fold0": rs[0],
            "r_fold1": rs[1],
            "r_fold2": rs[2],
            "fisher_z": v3.fisher_z_mean(rs),
            "mean_r": float(np.mean(rs)),
        })
        if t % 25 == 0:
            print(f"[B] source {source} target {t} z {rows[-1]['fisher_z']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    tmp = out.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"[B] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
