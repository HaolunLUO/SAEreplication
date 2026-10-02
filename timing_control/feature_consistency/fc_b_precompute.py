#!/usr/bin/env python3
"""Precompute timing residuals and SAE columns for the generalization array.

The residual of a target electrode depends only on that electrode and fold.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import fc_common as C
import sparse_encoding.sparse_encoding_v3 as v3
import timing_control as tc

GEN = C.OUT / "generalization"


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    pop = C.language_electrodes()
    pop.to_csv(GEN / "population.csv", index=False)
    scores = C.load_lag_scores()
    supports = C.load_sae_supports(pop)
    print("[pre] loading design", flush=True)
    X, surp, bv, T = C.load_design()
    n_words = X.shape[0]
    feat_set = set()
    for folds in supports.values():
        for _f, _lag, idx in folds:
            feat_set.update(int(i) for i in idx)
    feat_index = np.array(sorted(feat_set), dtype=np.int32)
    print(f"[pre] unique SAE features {feat_index.size}", flush=True)
    cols = np.asarray(X[:, feat_index].toarray(), dtype=np.float64)
    np.save(GEN / "feature_index.npy", feat_index)
    np.save(GEN / "feature_columns.npy", cols)
    np.save(GEN / "surprisal.npy", surp)

    y_resid = np.full((len(pop), 3, n_words), np.nan, dtype=np.float64)
    train_idx = np.empty((len(pop), 3), dtype=object)
    test_idx = np.empty((len(pop), 3), dtype=object)
    peak = np.zeros((len(pop), 3), dtype=np.float64)
    surp_fresh = np.zeros((len(pop), 3), dtype=np.float64)
    surp_stored = np.zeros((len(pop), 3), dtype=np.float64)
    section_id = None

    check_rows = []
    neural = {}
    for row in pop.itertuples(index=False):
        t0 = time.time()
        if row.subject not in neural:
            print(f"[pre] load {row.subject}", flush=True)
            neural[row.subject] = C.load_subject_bundle(row.subject)
        chans, sid = neural[row.subject]
        if section_id is None:
            section_id = sid
            if sid.shape[0] != n_words:
                raise RuntimeError("section id length mismatch")
        Y, OK = chans[str(row.channel)]
        peaks = C.peak_lags(scores, row.subject, row.channel)
        splits = v3.loso_outer_splits(sid)
        for fold in range(3):
            lag = peaks[fold]
            peak[row.elec_i, fold] = lag
            li = C.lag_index(lag)
            y = np.asarray(Y[:, li], dtype=np.float64)
            tr, te, _held = splits[fold]
            a = tc._rows(tr, y, OK[:, li], bv)
            b = tc._rows(te, y, OK[:, li], bv)
            yr, _ = tc._residualize(y, T[:, li, :], a, np.r_[a, b])
            y_resid[row.elec_i, fold] = yr
            train_idx[row.elec_i, fold] = np.asarray(a, dtype=np.int32)
            test_idx[row.elec_i, fold] = np.asarray(b, dtype=np.int32)
            r_fresh, _ = v3.predict_surprisal(surp, yr, a, b)
            r_stored = C.stored_fold_r(scores, row.subject, row.channel, "surprisal", fold, lag)
            surp_fresh[row.elec_i, fold] = r_fresh
            surp_stored[row.elec_i, fold] = r_stored
            if row.elec_i < 6:
                check_rows.append({
                    "elec_i": int(row.elec_i), "subject": row.subject, "channel": row.channel,
                    "fold": fold, "lag_ms": lag, "fresh_r": r_fresh, "stored_r": r_stored,
                    "abs_diff": abs(r_fresh - r_stored),
                })
        print(f"[pre] {row.elec_i:3d} {row.subject} {row.channel} {time.time()-t0:.1f}s", flush=True)

    np.savez(
        GEN / "residuals.npz",
        y_resid=y_resid,
        train_idx=train_idx,
        test_idx=test_idx,
        peak_lag_ms=peak,
        surprisal_fresh_r=surp_fresh,
        surprisal_stored_r=surp_stored,
        section_id=section_id,
    )
    check = pd.DataFrame(check_rows)
    check.to_csv(GEN / "surprisal_check.csv", index=False)
    print(
        f"[pre] surprisal check n={len(check)} max abs {check['abs_diff'].max():.3e}",
        flush=True,
    )
    print("[pre] done", flush=True)


if __name__ == "__main__":
    main()
