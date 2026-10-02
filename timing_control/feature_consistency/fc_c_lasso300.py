#!/usr/bin/env python3
"""LASSO at a fixed 300 ms lag for the 151 left language electrodes.

Same partial-mode path as timing_control.fit_space (residualize on the
training rows, then F-test + LassoCV). Ridge signs are fit on that mask.
Outputs stay under feature_consistency/lasso_300/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

import fc_common as C
import sparse_encoding.sparse_encoding_v3 as v3
import timing_control as tc
from sparse_encoding.sparse_encoding_regression import _fold_r, select_alpha

LASSO = C.OUT / "lasso_300"
PARTS = LASSO / "parts"
COLS = [
    "subject", "channel", "elec_i", "fold", "feature_index", "coef",
    "lag_ms", "alpha", "fold_r",
]


def _one(payload):
    subject = payload["subject"]
    channel = payload["channel"]
    elec_i = payload["elec_i"]
    part = PARTS / f"{elec_i:03d}_{subject}__{str(channel).replace('/', '_')}.csv"
    if part.exists() and part.stat().st_size > 0:
        return str(part)
    t0 = time.time()
    X = payload["X"]
    surp = payload["surp"]
    bv = payload["bv"]
    T = payload["T"]
    Y = payload["Y"]
    OK = payload["OK"]
    sid = payload["sid"]
    li = payload["lag_i"]
    y = np.asarray(Y[:, li], dtype=np.float64)
    rows = []
    for fold, (tr, te, _held) in enumerate(v3.loso_outer_splits(sid)):
        a = tc._rows(tr, y, OK[:, li], bv)
        b = tc._rows(te, y, OK[:, li], bv)
        if a.size < 20 or b.size < 5:
            continue
        yr, _ = tc._residualize(y, T, a, np.r_[a, b])
        sup = tc._support(X, surp, yr, a, dense=False)
        sur_tr, sur_te = v3._scale_surp(surp, a, b)
        Xtr, Xte = v3._design(X, sup, sur_tr, sur_te, a, b)
        try:
            alpha = select_alpha(Xtr, yr[a])
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(Xtr, yr[a])
            pred = model.predict(Xte).ravel()
            r = _fold_r(pred, yr[b], True)
            coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        except ValueError:
            alpha, r, coef = np.nan, 0.0, np.zeros(1)
        idx = np.flatnonzero(sup)
        sae_coef = coef[:-1] if coef.size == idx.size + 1 else np.zeros(idx.size)
        for f_i, c in zip(idx, sae_coef):
            rows.append({
                "subject": subject, "channel": channel, "elec_i": elec_i,
                "fold": fold, "feature_index": int(f_i), "coef": float(c),
                "lag_ms": C.LAG_300, "alpha": alpha, "fold_r": r,
            })
    frame = pd.DataFrame(rows, columns=COLS)
    tmp = part.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(part)
    print(f"[300] {elec_i:3d} {subject} {channel} rows {len(frame)} {time.time()-t0:.1f}s", flush=True)
    return str(part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    LASSO.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    pop = C.language_electrodes()
    pop.to_csv(LASSO / "population.csv", index=False)
    print("[300] loading design", flush=True)
    X, surp, bv, T = C.load_design()
    li = C.lag_index(C.LAG_300)
    neural = {s: C.load_subject_bundle(s) for s in C.SUBJECTS}
    payloads = []
    for row in pop.itertuples(index=False):
        chans, sid = neural[row.subject]
        Y, OK = chans[str(row.channel)]
        payloads.append({
            "subject": row.subject, "channel": row.channel, "elec_i": int(row.elec_i),
            "X": X, "surp": surp, "bv": bv, "T": T[:, li, :],
            "Y": Y, "OK": OK, "sid": sid, "lag_i": li,
        })
    # T is already sliced to one lag in the payload; _one indexes [:, li] on Y
    # but T is (n_words, 8). Fix by passing the lag slice as Tlag and not indexing li.
    Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(_one)(p) for p in payloads)
    parts = sorted(PARTS.glob("*.csv"))
    frames = [pd.read_csv(p) for p in parts if p.stat().st_size]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    out.to_csv(LASSO / "coefficients.csv", index=False)
    print(f"[300] electrodes {len(parts)} coef rows {len(out)}", flush=True)


if __name__ == "__main__":
    main()
