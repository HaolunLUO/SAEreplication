#!/usr/bin/env python3
"""LASSO supports for raw high-gamma on the Fig 5 anatomy electrodes.

Same v2 readout as timing_control.py --stage indices, with mode raw
(no speech-timing residual). Writes only:

  tables/tc_selected_indices_rawhg_anatomy_parts/*__v2__raw.csv
  tables/tc_selected_indices_rawhg_anatomy.csv

Does not write lag scores or tc_selected_indices_partial.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import timing_control as tc  # noqa: E402

TABLES = tc.TABLES
PARTS = TABLES / "tc_selected_indices_rawhg_anatomy_parts"
COMBINED = TABLES / "tc_selected_indices_rawhg_anatomy.csv"
TAX = Path(
    "/orcd/pool/005/haolun52/analysisEV/group_encoding_results/functional_taxonomy/tables/"
    "functional_taxonomy_electrode_table.csv"
)
EXPECT = [
    "Left Precentral gyrus",
    "Left Inferior frontal gyrus-triangular part",
    "Left Inferior frontal gyrus-opercular part",
    "Left Middle frontal gyrus",
    "Left Superior frontal gyrus-dorsolateral",
    "Left Insula",
    "Left Hippocampus",
    "Left Superior temporal gyrus",
]
FORBIDDEN = {
    "tc_selected_indices_partial.csv",
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_both_lag_scores_rawhg.csv",
    "tc_wide_lag_scores_rawhg.csv",
    "tc_sig_glove_lag_scores_rawhg.csv",
    "sig_glove_timing.csv",
    "tc_bin_refit_partial.csv",
    "tc_bin_refit_anatomy_partial.csv",
    "tc_bin_refit_anatomy_rawhg.csv",
}


def _guard(path: Path) -> None:
    text = str(path)
    if path.name in FORBIDDEN or "rawhg_anatomy" not in text:
        raise RuntimeError(f"refusing to write {path}")


def _part(subject: str, channel: str) -> Path:
    safe = str(channel).replace("/", "_")
    return PARTS / f"{subject}__{safe}__v2__raw.csv"


def _snapshot() -> dict:
    partial_parts = TABLES / "tc_selected_indices_partial_parts"
    out = {
        "partial_names": tuple(sorted(p.name for p in partial_parts.glob("*.csv"))),
    }
    for name in FORBIDDEN:
        path = TABLES / name
        if path.exists():
            out[name] = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.md5(path.read_bytes()).hexdigest())
    return out


def _check_snapshot(before: dict) -> None:
    after = _snapshot()
    if after != before:
        raise RuntimeError("a protected table changed during the raw index fit")


def anatomy_population() -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "is_lang"])
    lang = tax[tax["is_lang"].fillna(False).astype(bool)].copy()
    label = lang["region"].fillna("").astype(str).str.strip()
    pop = lang.loc[label.str.startswith("Left "), ["subject", "channel"]].copy()
    pop["anatomy"] = label.loc[label.str.startswith("Left ")].to_numpy()
    counts = pop.groupby("anatomy").size()
    pop = pop[pop["anatomy"].isin(counts[counts >= 4].index)].reset_index(drop=True)
    counts = pop.groupby("anatomy").size()
    order = sorted(counts.index, key=lambda r: (-int(counts[r]), r))
    if order != EXPECT:
        raise RuntimeError(f"regions changed: {order}")
    if len(pop) != 127:
        raise RuntimeError(f"expected 127 anatomy electrodes, got {len(pop)}")
    el = tc.electrode_table()
    pop = pop.merge(el[["subject", "channel", "both"]], on=["subject", "channel"], how="inner")
    if len(pop) != 127:
        raise RuntimeError(f"anatomy electrodes missing from the timing-control table: {len(pop)}")
    return pop


def _write_part(meta: dict, recs: list) -> None:
    if meta.get("condition") != "v2" or meta.get("mode") != "raw":
        raise RuntimeError(f"refusing index rows for {meta}")
    part = _part(meta["subject"], meta["channel"])
    _guard(part)
    frame = pd.DataFrame(recs, columns=tc.INDEX_COLS)
    if len(frame) and not ((frame["condition"] == "v2") & (frame["mode"] == "raw")).all():
        raise RuntimeError("index rows are not v2 raw")
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(part)


def raw_support_task(task: dict):
    """Training-fold peak and LASSO support, mode raw.

    This is the index half of ``timing_control.fit_space`` for raw mode.
    The later lag-scoring loop does not change the support, so it is omitted.
    """
    t0 = time.time()
    lags = np.asarray(task["lags"], dtype=float)
    search = np.array([np.any(np.isclose(tc.MAIN_LAGS, lag)) for lag in lags])
    meta = {k: task[k] for k in ("subject", "channel", "condition", "mode")}
    if meta["condition"] != "v2" or meta["mode"] != "raw":
        raise RuntimeError(meta)
    recs = []
    sid = task["sid"]
    for space, X, dense in (("sae", task["X_sae"], False), ("residual", task["X_resid"], True)):
        for fold, (tr, _te, held) in enumerate(tc.v3.loso_outer_splits(sid)):
            inners = tc.v3.loso_inner_splits(tr, sid)
            z = np.full(lags.size, -np.inf)
            for li in np.flatnonzero(search):
                y = task["Y"][:, li].astype(np.float64)
                ok = task["OK"][:, li]
                rs = []
                for itr, ite in inners:
                    a = tc._rows(itr, y, ok, task["bv"])
                    b = tc._rows(ite, y, ok, task["bv"])
                    if a.size < 20 or b.size < 5:
                        rs.append(0.0)
                        continue
                    r, _pred, _sup = tc.v3.predict_full(X, task["surp"], y, a, b, dense=dense)
                    rs.append(r)
                z[li] = tc.v3.fisher_z_mean(rs)
            chosen = tc.v3.argmax_lag(z[search], lags[search])
            peak_i = int(np.flatnonzero(search)[chosen])
            y = task["Y"][:, peak_i].astype(np.float64)
            a = tc._rows(tr, y, task["OK"][:, peak_i], task["bv"])
            sup = tc._support(X, task["surp"], y, a, dense)
            peak_ms = float(lags[peak_i])
            for feat_i in np.flatnonzero(sup):
                recs.append({
                    "subject": meta["subject"], "channel": meta["channel"], "cv": "loso",
                    "feature_space": space, "fold": int(fold), "heldout_section": int(held),
                    "train_peak_lag_ms": peak_ms, "feature_index": int(feat_i), "bin_name": "",
                    "shift_id": -1, "null_draw": -1, "feature_tag": tc.TAG,
                    "condition": "v2", "mode": "raw",
                })
    return meta, recs, time.time() - t0


def _concat() -> None:
    _guard(COMBINED)
    frames = []
    for part in sorted(PARTS.glob("*__v2__raw.csv")):
        df = pd.read_csv(part)
        if len(df):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=tc.INDEX_COLS)
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMBINED.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(COMBINED)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=16)
    args = p.parse_args()
    before = _snapshot()
    pop = anatomy_population()
    print(f"[raw-idx] {len(pop)} electrodes, both={int(pop['both'].sum())}", flush=True)
    X, R, surp, bv = tc.feature_condition("v2", None)
    T_ext, T_main = tc.load_covariates(tc.EXT_LAGS), tc.load_covariates(tc.MAIN_LAGS)
    neural, sid = {}, None
    for subj, g in pop.groupby("subject"):
        chans, s_id = tc.load_main_neural(subj)
        sid = s_id if sid is None else sid
        if not np.array_equal(sid, s_id):
            raise RuntimeError("section ids differ")
        both_ch = list(g.loc[g["both"], "channel"])
        extd = tc.extended_neural(subj, both_ch)[0] if both_ch else {}
        for row in g.itertuples(index=False):
            if row.channel not in chans:
                raise RuntimeError(f"{subj} {row.channel} missing from the v3 neural cache")
            Ym, OKm = chans[row.channel]
            if row.both:
                Ye, OKe = extd[row.channel]
                neural[(subj, row.channel)] = (
                    np.concatenate([Ye, Ym], 1),
                    np.concatenate([OKe, OKm], 1),
                    tc.EXT_LAGS,
                    T_ext,
                )
            else:
                neural[(subj, row.channel)] = (Ym, OKm, tc.MAIN_LAGS, T_main)
    tasks, n_skip = [], 0
    for (subj, ch), (Y, OK, lags, T) in neural.items():
        part = _part(subj, ch)
        if part.is_file() and part.stat().st_size > 0:
            n_skip += 1
            continue
        tasks.append({
            "subject": subj, "channel": ch, "condition": "v2", "mode": "raw",
            "Y": Y, "OK": OK, "bv": bv, "sid": sid, "T": T, "lags": lags, "surp": surp,
            "X_sae": X, "X_resid": R, "timing_only": False,
        })
    print(f"[raw-idx] {len(tasks)} refits ({n_skip} already saved) -> {PARTS}", flush=True)
    if tasks:
        from joblib import Parallel, delayed

        t0 = time.time()
        gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
            delayed(raw_support_task)(t) for t in tasks
        )
        for k, (meta, recs, el_s) in enumerate(gen, 1):
            _write_part(meta, recs)
            print(
                f"  raw-idx {k}/{len(tasks)} {meta['subject']} {meta['channel']} "
                f"n_idx={len(recs)} {el_s:.0f}s total {time.time() - t0:.0f}s",
                flush=True,
            )
    _concat()
    _check_snapshot(before)
    n_parts = len(list(PARTS.glob("*__v2__raw.csv")))
    if n_parts != 127:
        raise RuntimeError(f"expected 127 raw index files, got {n_parts}")
    print(f"[raw-idx] wrote {COMBINED}", flush=True)


if __name__ == "__main__":
    main()
