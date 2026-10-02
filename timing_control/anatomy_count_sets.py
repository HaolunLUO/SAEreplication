#!/usr/bin/env python3
"""Per-lag joint v2 LASSO support counts for the three missing count figures.

Writes only:
  tables/tc_selected_counts_partial_glove_lags.csv
  tables/tc_selected_counts_rawhg_lang_lags.csv
  tables/tc_selected_counts_rawhg_glove_lags.csv

The support is chosen at that lag. Raw skips the timing residual. Partial
residualizes the eight timing covariates inside each LOSO fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import paper_match_partial as pm  # noqa: E402
import timing_control as tc  # noqa: E402

TAX = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/functional_taxonomy/tables/functional_taxonomy_electrode_table.csv")
OUT = {
    "glove_partial": TC / "tables" / "tc_selected_counts_partial_glove_lags.csv",
    "lang_raw": TC / "tables" / "tc_selected_counts_rawhg_lang_lags.csv",
    "glove_raw": TC / "tables" / "tc_selected_counts_rawhg_glove_lags.csv",
}
MODE = {"glove_partial": "partial", "lang_raw": "raw", "glove_raw": "raw"}
FORBIDDEN = {
    "tc_selected_counts_partial_anatomy_lags.csv",
    "tc_selected_indices_partial.csv",
    "tc_bin_refit_partial.csv",
    "tc_bin_refit_anatomy_partial.csv",
    "tc_bin_refit_anatomy_partial_lags.csv",
    "tc_bin_refit_anatomy_rawhg.csv",
    "tc_bin_refit_anatomy_rawhg_lags.csv",
    "tc_bin_refit_rawhg_anatomy_300ms.csv",
    "sig_glove_timing.csv",
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_both_lag_scores_rawhg.csv",
    "tc_wide_lag_scores_rawhg.csv",
}
_STATE: dict = {}


def _guard(path: Path) -> None:
    if path.name in FORBIDDEN or path.name not in {p.name for p in OUT.values()}:
        raise RuntimeError(f"refusing to write {path}")


def _regions(frame: pd.DataFrame) -> list[str]:
    order = (
        frame.groupby("anatomy")
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["n", "anatomy"], ascending=[False, True])
    )
    return list(order["anatomy"])


def language_population() -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "is_lang"])
    lang = tax[tax["is_lang"].fillna(False).astype(bool)].copy()
    label = lang["region"].fillna("").astype(str).str.strip()
    pop = lang.loc[label.str.startswith("Left "), ["subject", "channel"]].copy()
    pop["anatomy"] = label.loc[label.str.startswith("Left ")].to_numpy()
    keep = pop.groupby("anatomy").size()
    keep = set(keep[keep >= 4].index)
    pop = pop[pop["anatomy"].isin(keep)].reset_index(drop=True)
    if len(pop) != 127 or _regions(pop) != [
        "Left Precentral gyrus",
        "Left Inferior frontal gyrus-triangular part",
        "Left Inferior frontal gyrus-opercular part",
        "Left Middle frontal gyrus",
        "Left Superior frontal gyrus-dorsolateral",
        "Left Insula",
        "Left Hippocampus",
        "Left Superior temporal gyrus",
    ]:
        raise RuntimeError(f"language population changed: {len(pop)} {_regions(pop)}")
    return pop


def glove_population() -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "sig_glove"])
    glove = tax[tax["sig_glove"].fillna(False).astype(bool)].copy()
    label = glove["region"].fillna("").astype(str).str.strip()
    pop = glove.loc[(label != "") & (label != "Unknown"), ["subject", "channel"]].copy()
    pop["anatomy"] = label.loc[(label != "") & (label != "Unknown")].to_numpy()
    keep = pop.groupby("anatomy").size()
    keep = set(keep[keep >= 4].index)
    pop = pop[pop["anatomy"].isin(keep)].reset_index(drop=True)
    if len(pop) != 157 or len(_regions(pop)) != 17:
        raise RuntimeError(f"glove population changed: {len(pop)} regions {len(_regions(pop))}")
    return pop


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
    mode = payload["mode"]
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
            if mode == "partial":
                yy = tc._residualize(y, Tl, a, a)[0]
                sup = tc._support(X, surp, yy, a, dense=False)
            elif mode == "raw":
                sup = tc._support(X, surp, y, a, dense=False)
            else:
                raise RuntimeError(mode)
            fold_counts.append(_bin_counts(sup))
        arr = np.asarray(fold_counts, dtype=float)
        if arr.shape != (3, 3):
            raise RuntimeError(f"expected 3 folds, got {arr.shape}")
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        rec = {
            "subject": payload["subject"],
            "channel": payload["channel"],
            "anatomy": payload["anatomy"],
            "mode": mode,
            "lag_ms": float(lag),
        }
        for j, (_label, name, _lo, _hi) in enumerate(pm.BINS):
            rec[f"{name}_mean"] = float(means[j])
            for fold in range(3):
                rec[f"{name}_fold{fold}"] = float(arr[fold, j])
        rows.append(rec)
    return rows


def _done(path: Path, mode: str) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    prev = pd.read_csv(path)
    if len(prev) and not (prev["mode"] == mode).all():
        raise RuntimeError(f"{path.name} is not mode {mode}")
    counts = prev.groupby(["subject", "channel"])["lag_ms"].nunique()
    done = set(counts[counts >= 51].index)
    if len(prev):
        keep = prev.apply(lambda r: (r.subject, r.channel) in done, axis=1)
        prev.loc[keep].to_csv(path, index=False)
    return done


def _payloads(pop: pd.DataFrame, mode: str, done: set, lags: np.ndarray, T_all: np.ndarray, limit: int | None) -> list[dict]:
    payloads = []
    for subject, g in pop.groupby("subject"):
        chans, sid = tc.load_main_neural(subject)
        for row in g.itertuples(index=False):
            if (row.subject, row.channel) in done:
                continue
            if row.channel not in chans:
                raise RuntimeError(f"{subject} {row.channel} missing from the v3 neural cache")
            Y, OK = chans[row.channel]
            if Y.shape[0] != T_all.shape[0]:
                raise RuntimeError(f"{subject} neural rows {Y.shape[0]} != covariates {T_all.shape[0]}")
            payloads.append({
                "subject": row.subject,
                "channel": row.channel,
                "anatomy": row.anatomy,
                "mode": mode,
                "lags": [float(lag) for lag in lags],
                "Y": np.ascontiguousarray(Y[:, : len(lags)], dtype=np.float64),
                "OK": np.ascontiguousarray(OK[:, : len(lags)]),
                "T": T_all,
                "sid": sid,
            })
            if limit is not None and len(payloads) >= limit:
                del chans
                return payloads
        del chans
    return payloads


def fit_set(name: str, n_jobs: int, limit: int | None = None) -> None:
    from joblib import Parallel, delayed

    out = OUT[name]
    mode = MODE[name]
    _guard(out)
    pop = glove_population() if name.startswith("glove") else language_population()
    lags = np.asarray(tc.MAIN_LAGS, dtype=float)
    full_grid = len(lags) == 51 and np.isclose(lags[0], -500) and np.isclose(lags[-1], 2000)
    if limit is None and not full_grid:
        raise RuntimeError(f"unexpected lag grid {lags[0]}..{lags[-1]} n={len(lags)}")
    done = set() if limit else _done(out, mode)
    T_all = tc.load_covariates(lags)
    payloads = _payloads(pop, mode, done, lags, T_all, limit)
    print(f"[counts] {name} {len(payloads)} electrodes to fit ({len(done)} already done)", flush=True)
    if not payloads:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = Parallel(n_jobs=n_jobs, initializer=_init_worker, return_as="generator_unordered", max_nbytes="2M")(
        delayed(fit_electrode)(item) for item in payloads
    )
    n = 0
    for rows in gen:
        n += 1
        df = pd.DataFrame(rows)
        if not (df["mode"] == mode).all() or df["lag_ms"].nunique() != len(lags):
            raise RuntimeError(f"refusing a {name} electrode that is not {len(lags)} {mode} lags")
        _guard(out)
        if limit:
            print(df[["subject", "channel", "lag_ms", "bin0-2048_mean", "bin2048-16384_mean", "bin16384-65536_mean"]].head().to_string(), flush=True)
            print(f"[counts] smoke {name} ok, not writing {out.name}", flush=True)
            return
        write_header = not out.exists() or out.stat().st_size == 0
        df.to_csv(out, mode="a", header=write_header, index=False)
        print(f"  {name} {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']}", flush=True)
    if limit:
        return
    df = pd.read_csv(out)
    n_elec = df.groupby(["subject", "channel"]).ngroups
    n_lag = df.groupby(["subject", "channel"])["lag_ms"].nunique()
    expect = 157 if name.startswith("glove") else 127
    if n_elec != expect or int(n_lag.min()) < 51 or not (df["mode"] == mode).all():
        raise RuntimeError(f"incomplete {name}: elecs={n_elec} minlags={int(n_lag.min())}")
    print(f"[counts] wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--only", choices=list(OUT), default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    names = [args.only] if args.only else list(OUT)
    if args.smoke:
        # One electrode, the 300 ms lag only. Does not write the count tables.
        import timing_control as tc_mod
        tc_mod.MAIN_LAGS = np.asarray([300.0])
        fit_set(names[0], n_jobs=1, limit=1)
        return
    for name in names:
        fit_set(name, args.n_jobs)


if __name__ == "__main__":
    main()
