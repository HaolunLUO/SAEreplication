#!/usr/bin/env python3
"""Joint v2 partial LASSO support counts at every lag.

The support is chosen at that lag, after the eight timing covariates are
residualized inside the LOSO fold. It is not the training-peak support.

Writes only tables/tc_selected_counts_partial_anatomy_lags.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import anatomy_bin_refit as abr  # noqa: E402
import paper_match_partial as pm  # noqa: E402
import timing_control as tc  # noqa: E402

OUT = TC / "tables" / "tc_selected_counts_partial_anatomy_lags.csv"
FORBIDDEN = {
    "tc_bin_refit_partial.csv",
    "tc_bin_refit_anatomy_partial.csv",
    "tc_bin_refit_anatomy_partial_lags.csv",
    "tc_bin_refit_anatomy_rawhg.csv",
    "tc_bin_refit_anatomy_rawhg_lags.csv",
    "tc_bin_refit_rawhg_anatomy_300ms.csv",
    "tc_selected_indices_partial.csv",
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "sig_glove_timing.csv",
}
PROTECTED = [TC / "tables" / name for name in FORBIDDEN] + [
    TC / "figures" / "paper-match" / "fig5.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial.png",
    TC / "figures" / "paper-match" / "fig5b_anatomy_bin_timecourse_partial.png",
]
_STATE: dict = {}


def _guard(path: Path) -> None:
    if path.name in FORBIDDEN or path.name != OUT.name:
        raise RuntimeError(f"refusing to write {path}")


def _snap() -> dict:
    out = {}
    for path in PROTECTED:
        if path.exists():
            out[str(path)] = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.md5(path.read_bytes()).hexdigest())
    return out


def _init_worker() -> None:
    X, _R, surp, bv = tc.feature_condition("v2", None)
    if X.shape[1] < 65536:
        raise RuntimeError(f"SAE width {X.shape[1]} is below 65536")
    _STATE["X"] = X
    _STATE["surp"] = surp
    _STATE["bv"] = bv


def _bin_counts(support: np.ndarray) -> list[int]:
    idx = np.flatnonzero(np.asarray(support))
    return [int(np.sum((idx >= lo) & (idx < hi))) for _label, _name, lo, hi in pm.BINS]


def fit_electrode(payload: dict) -> list[dict]:
    if "X" not in _STATE:
        _init_worker()
    X = _STATE["X"]
    surp = _STATE["surp"]
    bv = _STATE["bv"]
    Y = payload["Y"]
    OK = payload["OK"]
    T = payload["T"]
    sid = payload["sid"]
    rows = []
    for li, lag in enumerate(payload["lags"]):
        y = np.ascontiguousarray(Y[:, li], dtype=np.float64)
        ok = OK[:, li]
        Tl = np.ascontiguousarray(T[:, li, :])
        fold_counts = []
        for tr, _te, _held in tc.v3.loso_outer_splits(sid):
            a = tc._rows(tr, y, ok, bv)
            if a.size < 20:
                fold_counts.append([np.nan, np.nan, np.nan])
                continue
            yy = tc._residualize(y, Tl, a, a)[0]
            sup = tc._support(X, surp, yy, a, dense=False)
            fold_counts.append(_bin_counts(sup))
        fold_counts = np.asarray(fold_counts, dtype=float)
        means = np.nanmean(fold_counts, axis=0)
        if not np.any(np.isfinite(means)):
            means = np.zeros(3)
        rec = {
            "subject": payload["subject"],
            "channel": payload["channel"],
            "anatomy": payload["anatomy"],
            "mode": "partial",
            "lag_ms": float(lag),
        }
        for j, (_label, name, _lo, _hi) in enumerate(pm.BINS):
            rec[f"{name}_mean"] = float(means[j])
            for fold in range(3):
                rec[f"{name}_fold{fold}"] = float(fold_counts[fold, j])
        rows.append(rec)
    return rows


def _done() -> set:
    if not OUT.exists() or OUT.stat().st_size == 0:
        return set()
    prev = pd.read_csv(OUT)
    if len(prev) and not (prev["mode"] == "partial").all():
        raise RuntimeError(f"{OUT.name} is not partial")
    counts = prev.groupby(["subject", "channel"])["lag_ms"].nunique()
    done = set(counts[counts >= 51].index)
    if len(prev):
        keep = prev.apply(lambda r: (r.subject, r.channel) in done, axis=1)
        prev.loc[keep].to_csv(OUT, index=False)
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    _guard(OUT)
    before = _snap()
    pop, _peak = abr.anatomy_population("partial")
    if len(pop) != 127:
        raise RuntimeError(f"expected 127 electrodes, got {len(pop)}")
    lags = np.asarray(tc.MAIN_LAGS, dtype=float)
    if len(lags) != 51 or not np.isclose(lags[0], -500) or not np.isclose(lags[-1], 2000):
        raise RuntimeError(f"unexpected lag grid {lags[0]}..{lags[-1]} n={len(lags)}")
    done = _done()
    T_all = tc.load_covariates(lags)
    sid_ref = None
    payloads = []
    for subject, g in pop.groupby("subject"):
        chans, sid = tc.load_main_neural(subject)
        sid_ref = sid if sid_ref is None else sid_ref
        if not np.array_equal(sid_ref, sid):
            raise RuntimeError("section ids differ across subjects")
        for row in g.itertuples(index=False):
            if (row.subject, row.channel) in done:
                continue
            if row.channel not in chans:
                raise RuntimeError(f"{subject} {row.channel} missing from the v3 neural cache")
            Y, OK = chans[row.channel]
            payloads.append({
                "subject": row.subject,
                "channel": row.channel,
                "anatomy": row.anatomy,
                "lags": [float(lag) for lag in lags],
                "Y": np.ascontiguousarray(Y[:, : len(lags)], dtype=np.float64),
                "OK": np.ascontiguousarray(OK[:, : len(lags)]),
                "T": T_all,
                "sid": sid_ref,
            })
        del chans
    print(f"[counts] {len(payloads)} electrodes to fit ({len(done)} already done)", flush=True)
    if payloads:
        from joblib import Parallel, delayed

        OUT.parent.mkdir(parents=True, exist_ok=True)
        gen = Parallel(n_jobs=args.n_jobs, initializer=_init_worker, return_as="generator_unordered", max_nbytes="2M")(
            delayed(fit_electrode)(item) for item in payloads
        )
        n = 0
        for rows in gen:
            n += 1
            df = pd.DataFrame(rows)
            if not (df["mode"] == "partial").all() or df["lag_ms"].nunique() != 51:
                raise RuntimeError("refusing a partial electrode that is not 51 lags")
            _guard(OUT)
            write_header = not OUT.exists() or OUT.stat().st_size == 0
            df.to_csv(OUT, mode="a", header=write_header, index=False)
            print(f"  counts {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']}", flush=True)
    df = pd.read_csv(OUT)
    n_elec = df.groupby(["subject", "channel"]).ngroups
    n_lag = df.groupby(["subject", "channel"])["lag_ms"].nunique()
    if n_elec != 127 or int(n_lag.min()) < 51:
        raise RuntimeError(f"incomplete counts: elecs={n_elec} minlags={int(n_lag.min())}")
    if _snap() != before:
        raise RuntimeError("a protected file changed during the count fit")
    print(f"[counts] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
