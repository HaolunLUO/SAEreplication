#!/usr/bin/env python3
"""Bin-restricted models at every v3 lag for the Fig 5 anatomy electrodes.

Writes only:
  tables/tc_bin_refit_anatomy_partial_lags.csv
  tables/tc_bin_refit_anatomy_rawhg_lags.csv
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

OUT = {
    "partial": TC / "tables" / "tc_bin_refit_anatomy_partial_lags.csv",
    "raw": TC / "tables" / "tc_bin_refit_anatomy_rawhg_lags.csv",
}
FORBIDDEN = {
    "tc_bin_refit_partial.csv",
    "tc_bin_refit_rawhg_anatomy_300ms.csv",
    "tc_bin_refit_anatomy_partial.csv",
    "tc_bin_refit_anatomy_rawhg.csv",
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_both_lag_scores_rawhg.csv",
    "tc_wide_lag_scores_rawhg.csv",
    "sig_glove_timing.csv",
    "tc_selected_indices_partial.csv",
}
PROTECTED = [TC / "tables" / name for name in FORBIDDEN] + [
    TC / "figures" / "paper-match" / "fig5.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg_300ms.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial_300ms.png",
]


def _guard(path: Path) -> None:
    if path.name in FORBIDDEN or not path.name.endswith("_lags.csv"):
        raise RuntimeError(f"refusing to write {path}")


def _snap() -> dict:
    out = {}
    for path in PROTECTED:
        if path.exists():
            out[str(path)] = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.md5(path.read_bytes()).hexdigest())
    return out


def _lag_key(subject: str, channel: str, lag) -> tuple:
    return (str(subject), str(channel), int(round(float(lag))))


def _done(path: Path, mode: str, lags: np.ndarray) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    prev = pd.read_csv(path)
    if len(prev) and not (prev["mode"] == mode).all():
        raise RuntimeError(f"{path.name} contains a mode other than {mode}")
    counts = prev.groupby(["subject", "channel", "lag_ms"])["bin_name"].nunique()
    complete = counts[counts >= 3]
    keep_keys = {_lag_key(s, c, lag) for s, c, lag in complete.index}
    if len(prev):
        keep = [
            _lag_key(s, c, lag) in keep_keys
            for s, c, lag in zip(prev["subject"], prev["channel"], prev["lag_ms"])
        ]
        prev.loc[keep].to_csv(path, index=False)
    return keep_keys


def fit_electrode_lags(payload: dict) -> list[dict]:
    if "tc" not in abr._STATE:
        abr._init_worker()
    rows = []
    Y = payload["Y"]
    OK = payload["OK"]
    T = payload["T"]
    for li, lag in enumerate(payload["lags"]):
        rows.extend(abr._fit_electrode({
            "subject": payload["subject"],
            "channel": payload["channel"],
            "anatomy": payload["anatomy"],
            "mode": payload["mode"],
            "lag_ms": float(lag),
            "y": np.ascontiguousarray(Y[:, li], dtype=np.float64),
            "ok": np.ascontiguousarray(OK[:, li]),
            "T": np.ascontiguousarray(T[:, li, :]),
            "sid": payload["sid"],
        }))
    return rows


def refit_mode(mode: str, n_jobs: int, before: dict) -> None:
    out = OUT[mode]
    _guard(out)
    pop, _peak = abr.anatomy_population(mode)
    if len(pop) != 127:
        raise RuntimeError(f"{mode}: expected 127 electrodes, got {len(pop)}")
    lags = np.asarray(tc.MAIN_LAGS, dtype=float)
    if len(lags) != 51 or not np.isclose(lags[0], -500) or not np.isclose(lags[-1], 2000):
        raise RuntimeError(f"unexpected lag grid {lags[0]}..{lags[-1]} n={len(lags)}")
    done = _done(out, mode, lags)
    T_all = tc.load_covariates(lags)
    sid_ref = None
    payloads = []
    for subject, g in pop.groupby("subject"):
        chans, sid = tc.load_main_neural(subject)
        sid_ref = sid if sid_ref is None else sid_ref
        if not np.array_equal(sid_ref, sid):
            raise RuntimeError("section ids differ across subjects")
        for row in g.itertuples(index=False):
            if row.channel not in chans:
                raise RuntimeError(f"{subject} {row.channel} missing from the v3 neural cache")
            missing = [lag for lag in lags if _lag_key(row.subject, row.channel, lag) not in done]
            if not missing:
                continue
            Y, OK = chans[row.channel]
            idx = [int(np.argmin(np.abs(lags - lag))) for lag in missing]
            if not np.allclose(lags[idx], missing):
                raise RuntimeError("a requested lag is not on the v3 grid")
            payloads.append({
                "subject": row.subject,
                "channel": row.channel,
                "anatomy": row.anatomy,
                "mode": mode,
                "lags": [float(lag) for lag in missing],
                "Y": np.ascontiguousarray(Y[:, idx], dtype=np.float64),
                "OK": np.ascontiguousarray(OK[:, idx]),
                "T": np.ascontiguousarray(T_all[:, idx, :]),
                "sid": sid_ref,
            })
        del chans
    n_lags = sum(len(p["lags"]) for p in payloads)
    print(f"[bin-lags] {mode}: {len(payloads)} electrodes, {n_lags} lags to fit", flush=True)
    if not payloads:
        return
    from joblib import Parallel, delayed

    out.parent.mkdir(parents=True, exist_ok=True)
    gen = Parallel(n_jobs=n_jobs, initializer=abr._init_worker, return_as="generator_unordered", max_nbytes="2M")(
        delayed(fit_electrode_lags)(item) for item in payloads
    )
    n = 0
    for rows in gen:
        n += 1
        if any(r["mode"] != mode for r in rows):
            raise RuntimeError(f"refusing a row that is not {mode}")
        df = pd.DataFrame(rows)
        _guard(out)
        write_header = not out.exists() or out.stat().st_size == 0
        df.to_csv(out, mode="a", header=write_header, index=False)
        print(
            f"  {mode} {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']} lags={df['lag_ms'].nunique()}",
            flush=True,
        )
    df = pd.read_csv(out)
    counts = df.groupby(["subject", "channel", "lag_ms"])["bin_name"].nunique()
    n_elec = df.groupby(["subject", "channel"]).ngroups
    if n_elec != 127 or int(counts.min()) < 3 or df["lag_ms"].nunique() != 51:
        raise RuntimeError(
            f"{mode} incomplete: elecs={n_elec} lags={df['lag_ms'].nunique()} minbins={int(counts.min())}"
        )
    if _snap() != before:
        raise RuntimeError("a protected file changed during the lag refit")
    print(f"[bin-lags] wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["partial", "raw", "both"], default="both")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    for path in OUT.values():
        _guard(path)
    before = _snap()
    modes = ("partial", "raw") if args.mode == "both" else (args.mode,)
    for mode in modes:
        refit_mode(mode, args.n_jobs, before)


if __name__ == "__main__":
    main()
