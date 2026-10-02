#!/usr/bin/env python3
"""Bin-restricted v2 models for the Fig 5 anatomy panels.

Writes only tables/tc_bin_refit_anatomy_partial.csv and
tables/tc_bin_refit_anatomy_rawhg.csv. Does not touch tc_bin_refit_partial.csv,
the lag-score tables, or sig_glove_timing.csv.

Partial residualizes the eight speech-timing covariates inside each LOSO fold.
Raw scores the same high-gamma without that residual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
TABLES = TC / "tables"
TAX = Path(
    "/orcd/pool/005/haolun52/analysisEV/group_encoding_results/functional_taxonomy/tables/"
    "functional_taxonomy_electrode_table.csv"
)
sys.path.insert(0, str(TC))
import paper_match_partial as pm  # noqa: E402

OUT = {
    "partial": TABLES / "tc_bin_refit_anatomy_partial.csv",
    "raw": TABLES / "tc_bin_refit_anatomy_rawhg.csv",
}
SCORES = {
    "partial": ("tc_both_lag_scores.csv", "tc_wide_lag_scores.csv"),
    "raw": ("tc_both_lag_scores_rawhg.csv", "tc_wide_lag_scores_rawhg.csv"),
}
FORBIDDEN = {
    "tc_bin_refit_partial.csv",
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_both_lag_scores_rawhg.csv",
    "tc_wide_lag_scores_rawhg.csv",
    "tc_sig_glove_lag_scores_rawhg.csv",
    "sig_glove_timing.csv",
    "tc_selected_indices_partial.csv",
}
_STATE: dict = {}


def _guard(path: Path) -> None:
    if path.name in FORBIDDEN or not path.name.startswith("tc_bin_refit_anatomy_"):
        raise RuntimeError(f"refusing to write {path}")


def anatomy_population(mode: str) -> tuple[pd.DataFrame, float]:
    """Left-hemisphere language electrodes in AAL regions with at least 4 electrodes."""
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "is_lang"])
    lang = tax[tax["is_lang"].fillna(False).astype(bool)].copy()
    label = lang["region"].fillna("").astype(str).str.strip()
    pop = lang.loc[label.str.startswith("Left "), ["subject", "channel"]].copy()
    pop["anatomy"] = label.loc[label.str.startswith("Left ")].to_numpy()
    usecols = ["subject", "channel", "condition", "mode", "feature_space", "lag_ms", "fisher_z_mean"]
    frames = []
    for name in SCORES[mode]:
        df = pd.read_csv(TABLES / name, usecols=usecols)
        df = df[(df["condition"] == "v2") & (df["mode"] == mode) & (df["feature_space"].isin(pm.SPACES))]
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["subject", "channel", "condition", "mode", "feature_space", "lag_ms"], keep="last"
    )
    scored = scores[["subject", "channel"]].drop_duplicates()
    pop = pop.merge(scored, on=["subject", "channel"], how="inner")
    keep = pop.groupby("anatomy").size()
    keep = keep[keep >= 4].index
    pop = pop[pop["anatomy"].isin(keep)].reset_index(drop=True)
    wide = scores.merge(pop[["subject", "channel"]], on=["subject", "channel"])
    wide = wide.pivot_table(
        index=["subject", "channel", "lag_ms"],
        columns="feature_space",
        values="fisher_z_mean",
        aggfunc="last",
    ).reset_index()
    wide = wide.dropna(subset=list(pm.SPACES))
    wide = wide[(wide["lag_ms"] >= -500) & (wide["lag_ms"] <= 2000)]
    wide["gain"] = wide["sae"] - wide["surprisal"]
    wide["elec"] = wide["subject"] + "\t" + wide["channel"]
    n = wide["elec"].nunique()
    n_at = wide.groupby("lag_ms")["elec"].nunique()
    lags = n_at[n_at == n].index.to_numpy(float)
    use = wide[wide["lag_ms"].isin(lags)]
    curve = use.groupby(["lag_ms", "subject"])["gain"].mean().groupby("lag_ms").mean().sort_index()
    peak = pm.argmax_lag(curve.to_numpy(), curve.index.to_numpy(float))
    return pop, float(peak)


def _init_worker() -> None:
    sys.path.insert(0, str(TC))
    sys.path.insert(0, "/orcd/pool/005/haolun52/analysisEV/sae_sparse_encoding")
    import sparse_encoding.sparse_encoding_v3 as v3
    import timing_control as tc

    X, _R, surp, bv = tc.feature_condition("v2", None)
    if X.shape[1] < 65536:
        raise RuntimeError(f"SAE width {X.shape[1]} is below 65536")
    _STATE["tc"] = tc
    _STATE["v3"] = v3
    _STATE["surp"] = surp
    _STATE["bv"] = bv
    _STATE["bins"] = {name: X[:, lo:hi] for _lab, name, lo, hi in pm.BINS}


def _fit_electrode(payload: dict) -> list[dict]:
    if "tc" not in _STATE:
        _init_worker()
    tc = _STATE["tc"]
    v3 = _STATE["v3"]
    y = payload["y"]
    ok = payload["ok"]
    T = payload["T"]
    sid = payload["sid"]
    mode = payload["mode"]
    surp = _STATE["surp"]
    bv = _STATE["bv"]
    rows = []
    for _lab, name, _lo, _hi in pm.BINS:
        Xb = _STATE["bins"][name]
        fold_r = np.zeros(3, dtype=float)
        nsup = np.zeros(3, dtype=float)
        for f, (tr, te, _held) in enumerate(v3.loso_outer_splits(sid)):
            a = tc._rows(tr, y, ok, bv)
            b = tc._rows(te, y, ok, bv)
            if a.size < 20:
                continue
            if mode == "partial":
                yy = tc._residualize(y, T, a, a)[0]
                sup = tc._support(Xb, surp, yy, a, dense=False)
            elif mode == "raw":
                sup = tc._support(Xb, surp, y, a, dense=False)
            else:
                raise RuntimeError(mode)
            nsup[f] = int(np.asarray(sup).sum())
            if b.size < 5:
                continue
            if mode == "partial":
                yr, _ = tc._residualize(y, T, a, np.r_[a, b])
                target = yr
            else:
                target = y
            r, _pred = v3.predict_with_support(Xb, surp, target, a, b, sup)
            fold_r[f] = r
        rows.append({
            "subject": payload["subject"],
            "channel": payload["channel"],
            "anatomy": payload["anatomy"],
            "mode": mode,
            "bin_name": name,
            "lag_ms": payload["lag_ms"],
            "r_fold0": fold_r[0],
            "r_fold1": fold_r[1],
            "r_fold2": fold_r[2],
            "fisher_z_mean": v3.fisher_z_mean(list(fold_r)),
            "n_support_fold0": nsup[0],
            "n_support_fold1": nsup[1],
            "n_support_fold2": nsup[2],
            "n_support_mean": float(np.mean(nsup)),
        })
    return rows


def refit_mode(mode: str, n_jobs: int) -> None:
    import timing_control as tc
    from joblib import Parallel, delayed

    out = OUT[mode]
    _guard(out)
    pop, peak = anatomy_population(mode)
    print(f"[anatomy-bin] {mode} peak {peak:.0f} ms, {len(pop)} electrodes", flush=True)
    done = set()
    if out.exists() and out.stat().st_size:
        prev = pd.read_csv(out)
        if len(prev) and not np.allclose(prev["lag_ms"].unique(), peak):
            raise RuntimeError(f"{out.name} is at a different lag than {peak:.0f}")
        counts = prev.groupby(["subject", "channel"])["bin_name"].nunique()
        done = set(counts[counts >= 3].index)
        keep = prev.apply(lambda r: (r.subject, r.channel) in done, axis=1)
        prev.loc[keep].to_csv(out, index=False)
    li = int(np.argmin(np.abs(tc.MAIN_LAGS - peak)))
    if not np.isclose(tc.MAIN_LAGS[li], peak):
        raise RuntimeError(f"peak {peak} is not on the v3 lag grid")
    T = tc.load_covariates(np.asarray([peak]))[:, 0, :]
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
                "mode": mode,
                "lag_ms": float(peak),
                "y": np.ascontiguousarray(Y[:, li], dtype=np.float64),
                "ok": np.ascontiguousarray(OK[:, li]),
                "T": T,
                "sid": sid_ref,
            })
        del chans
    print(f"[anatomy-bin] {mode}: {len(payloads)} electrodes to fit ({len(done)} already done)", flush=True)
    if not payloads:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = Parallel(n_jobs=n_jobs, initializer=_init_worker, return_as="generator_unordered", max_nbytes="2M")(
        delayed(_fit_electrode)(p) for p in payloads
    )
    n = 0
    for rows in gen:
        n += 1
        df = pd.DataFrame(rows)
        write_header = not out.exists() or out.stat().st_size == 0
        df.to_csv(out, mode="a", header=write_header, index=False)
        print(f"  {mode} {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']}", flush=True)
    print(f"[anatomy-bin] wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["partial", "raw", "both"], default="both")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    modes = ("partial", "raw") if args.mode == "both" else (args.mode,)
    for mode in modes:
        refit_mode(mode, args.n_jobs)


if __name__ == "__main__":
    main()
