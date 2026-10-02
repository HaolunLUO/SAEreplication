#!/usr/bin/env python3
"""Raw bin-restricted models at 300 ms for the Fig 5 anatomy electrodes.

Writes only tables/tc_bin_refit_rawhg_anatomy_300ms.csv.
Does not touch the 600 ms raw table or tc_bin_refit_partial.csv.
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
import timing_control as tc  # noqa: E402

OUT = TC / "tables" / "tc_bin_refit_rawhg_anatomy_300ms.csv"
LAG = 300.0
PROTECTED = [
    TC / "tables" / "tc_bin_refit_partial.csv",
    TC / "tables" / "tc_bin_refit_anatomy_rawhg.csv",
    TC / "tables" / "tc_bin_refit_anatomy_partial.csv",
    TC / "tables" / "tc_both_lag_scores.csv",
    TC / "tables" / "tc_wide_lag_scores.csv",
    TC / "tables" / "tc_both_lag_scores_rawhg.csv",
    TC / "tables" / "tc_wide_lag_scores_rawhg.csv",
    TC / "tables" / "sig_glove_timing.csv",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg.png",
    TC / "figures" / "paper-match" / "fig5.png",
]


def _snap() -> dict:
    out = {}
    for path in PROTECTED:
        if path.exists():
            out[str(path)] = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.md5(path.read_bytes()).hexdigest())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    if OUT.name != "tc_bin_refit_rawhg_anatomy_300ms.csv":
        raise RuntimeError(OUT)
    before = _snap()
    pop, peak = abr.anatomy_population("raw")
    if len(pop) != 127:
        raise RuntimeError(f"expected 127 electrodes, got {len(pop)}")
    print(f"[raw-300] SAE-gain peak is {peak:.0f} ms; fitting lag {LAG:.0f} ms, {len(pop)} electrodes", flush=True)
    li = int(np.argmin(np.abs(tc.MAIN_LAGS - LAG)))
    if not np.isclose(tc.MAIN_LAGS[li], LAG):
        raise RuntimeError(f"{LAG} is not on the v3 lag grid")
    done = set()
    if OUT.exists() and OUT.stat().st_size:
        prev = pd.read_csv(OUT)
        if len(prev) and not np.allclose(prev["lag_ms"].unique(), LAG):
            raise RuntimeError(f"{OUT.name} is not at {LAG:.0f} ms")
        if len(prev) and not (prev["mode"] == "raw").all():
            raise RuntimeError(f"{OUT.name} is not raw")
        counts = prev.groupby(["subject", "channel"])["bin_name"].nunique()
        done = set(counts[counts >= 3].index)
        keep = prev.apply(lambda r: (r.subject, r.channel) in done, axis=1)
        prev.loc[keep].to_csv(OUT, index=False)
    T = tc.load_covariates(np.asarray([LAG]))[:, 0, :]
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
                "mode": "raw",
                "lag_ms": float(LAG),
                "y": np.ascontiguousarray(Y[:, li], dtype=np.float64),
                "ok": np.ascontiguousarray(OK[:, li]),
                "T": T,
                "sid": sid_ref,
            })
        del chans
    print(f"[raw-300] {len(payloads)} electrodes to fit ({len(done)} already done)", flush=True)
    if payloads:
        from joblib import Parallel, delayed

        OUT.parent.mkdir(parents=True, exist_ok=True)
        gen = Parallel(n_jobs=args.n_jobs, initializer=abr._init_worker, return_as="generator_unordered", max_nbytes="2M")(
            delayed(abr._fit_electrode)(item) for item in payloads
        )
        n = 0
        for rows in gen:
            n += 1
            if any(r["mode"] != "raw" or not np.isclose(r["lag_ms"], LAG) for r in rows):
                raise RuntimeError("refusing a row that is not raw at 300 ms")
            df = pd.DataFrame(rows)
            write_header = not OUT.exists() or OUT.stat().st_size == 0
            df.to_csv(OUT, mode="a", header=write_header, index=False)
            print(f"  raw-300 {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']}", flush=True)
    df = pd.read_csv(OUT)
    n_elec = df.groupby(["subject", "channel"]).ngroups
    nbin = df.groupby(["subject", "channel"])["bin_name"].nunique()
    if n_elec != 127 or int(nbin.min()) < 3 or not np.allclose(df["lag_ms"].unique(), LAG):
        raise RuntimeError(f"incomplete 300 ms table: elecs={n_elec} minbins={int(nbin.min())}")
    after = _snap()
    if after != before:
        raise RuntimeError("a protected file changed during the 300 ms refit")
    print(f"[raw-300] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
