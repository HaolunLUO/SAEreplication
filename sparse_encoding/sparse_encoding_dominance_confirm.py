#!/usr/bin/env python3
"""
Merge candidate deconvolution scores into the dominance table, update
method-agreement flags, and write a short confirmatory addendum.

Optionally launches calibrated nulls on deconv-confirmed surprisal-dominant
candidates (true block permutation + corrected empirical p-values).

Example
-------
    python sparse_encoding_dominance_confirm.py
    python sparse_encoding_dominance_confirm.py --run_nulls --null_perms 50
    python sparse_encoding_dominance_confirm.py --run_nulls --null_refit_support \\
        --null_perms 20 --top_n 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core.analysis_paths as ap
from sparse_encoding.sparse_encoding_surprisal_dominance import fdr_bh, normalize_channel


FEATURE_TAG = "sae_qwen3_8b_l18"


def merge_deconv_into_dominance(
    dominance_csv: Path,
    deconv_csv: Path,
) -> pd.DataFrame:
    d = pd.read_csv(dominance_csv)
    d["channel"] = normalize_channel(d["channel"])
    dd = pd.read_csv(deconv_csv)
    dd["channel"] = normalize_channel(dd["channel"])
    cols = ["subject", "channel", "full__R", "sae_gain", "surprisal_gain"]
    if "full__R_event" in dd.columns:
        cols.append("full__R_event")
    keep = dd[cols].copy().rename(columns={
        "full__R": "deconv_full_R_cand",
        "sae_gain": "deconv_sae_gain_cand",
        "surprisal_gain": "deconv_surprisal_gain_cand",
        "full__R_event": "deconv_full_R_event",
    })
    # Drop prior candidate columns if re-merging
    drop = [
        c for c in d.columns
        if c.endswith("_cand")
        or c in (
            "deconv_confirmed_surprisal",
            "deconv_candidate_run",
            "deconv_full_R_event",
        )
        or c.startswith("deconv_full_R_event")
    ]
    d = d.drop(columns=drop, errors="ignore")
    out = d.merge(keep, on=["subject", "channel"], how="left")
    out["deconv_candidate_run"] = out["deconv_full_R_cand"].notna()
    out["deconv_confirmed_surprisal"] = (
        out["deconv_candidate_run"]
        & (out["deconv_surprisal_gain_cand"] > 0)
        & (out["deconv_surprisal_gain_cand"] > out["deconv_sae_gain_cand"])
        & (out["deconv_full_R_cand"] > 0)
    )
    # Refresh method agreement using candidate deconv when available
    prior = out.get("deconv_surprisal_gain")
    if prior is not None:
        out["method_agree_surprisal"] = (
            out["surprisal_dominant"]
            & (
                out["deconv_confirmed_surprisal"]
                | (
                    out["deconv_surprisal_gain"].notna()
                    & (out["deconv_surprisal_gain"] > 0)
                    & (out["deconv_surprisal_gain"] > out["deconv_sae_gain"])
                )
            )
        )
    else:
        out["method_agree_surprisal"] = (
            out["surprisal_dominant"] & out["deconv_confirmed_surprisal"]
        )
    return out


def write_confirm_addendum(df: pd.DataFrame, null_pvals: Path | None, path: Path):
    lines = [
        "=" * 70,
        "CONFIRMATORY DECONVOLUTION ADDENDUM (Qwen candidates)",
        "=" * 70,
        "",
    ]
    cand = df[df["deconv_candidate_run"]].copy()
    lines.append(f"Candidate channels with deconv scores: {len(cand)}")
    if cand.empty:
        lines.append("(no candidate deconv results yet)")
        path.write_text("\n".join(lines))
        return
    conf = cand[cand["deconv_confirmed_surprisal"]]
    lines.append(f"Deconv-confirmed surprisal-dominant: {len(conf)}")
    lines.append(
        f"  of which is_lang=True: "
        f"{int((conf.get('is_lang', False) == True).sum()) if 'is_lang' in conf else 'n/a'}"
    )
    lines.append(
        f"  of which is_lang=False/NA: "
        f"{int((conf.get('is_lang', pd.Series(dtype=bool)).fillna(False) == False).sum()) if 'is_lang' in conf else 'n/a'}"
    )
    lines.append("")
    cols = [c for c in (
        "subject", "channel", "region", "is_lang", "task_class",
        "surprisal_gain", "sae_gain",
        "deconv_full_R_cand", "deconv_surprisal_gain_cand", "deconv_sae_gain_cand",
        "deconv_confirmed_surprisal", "method_agree_surprisal",
    ) if c in cand.columns]
    lines.append(cand.sort_values("deconv_surprisal_gain_cand", ascending=False)[cols]
                 .round(4).to_string(index=False))
    lines.append("")
    if null_pvals and null_pvals.exists():
        pv = pd.read_csv(null_pvals)
        lines.append("-" * 70)
        lines.append("NULL p-values (corrected empirical; FDR across candidates)")
        lines.append("-" * 70)
        mode = pv["null_mode"].iloc[0] if "null_mode" in pv.columns else "unknown"
        lines.append(f"null_mode={mode}")
        if mode == "frozen_support":
            lines.append("LABEL: EXPLORATORY (frozen SAE support)")
        else:
            lines.append("LABEL: CONFIRMATORY (refit SAE support under nulls)")
        lines.append(pv.round(4).to_string(index=False))
    else:
        lines.append("Null p-values: not yet available.")
    lines.append("")
    lines.append(
        "NOTE: Channel-level significance requires confirmatory "
        "(refit-support) nulls + FDR. Frozen-support results are exploratory."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def select_null_channels(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Channels confirmed (or nearly) surprisal-dominant under deconv."""
    cand = df[df["deconv_candidate_run"]].copy()
    if cand.empty:
        return cand
    # Prefer confirmed; else positive surprisal gain
    cand["score"] = cand["deconv_surprisal_gain_cand"].fillna(-np.inf)
    cand = cand.sort_values(
        ["deconv_confirmed_surprisal", "score"], ascending=[False, False]
    )
    return cand.head(top_n)[["subject", "channel"]]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dominance_csv",
        default=str(ap.SAE_DOMINANCE_TABLE),
    )
    p.add_argument(
        "--deconv_csv",
        default=str(
            ap.SAE_TABLES / f"sparse_encoding_deconv_{FEATURE_TAG}_candidates.csv"
        ),
    )
    p.add_argument("--run_nulls", action="store_true")
    p.add_argument("--null_perms", type=int, default=50)
    p.add_argument("--null_refit_support", action="store_true")
    p.add_argument("--top_n", type=int, default=12,
                   help="Max channels for null confirmation.")
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    deconv_csv = Path(args.deconv_csv)
    if not deconv_csv.exists():
        print(f"[wait] deconv results not found: {deconv_csv}")
        sys.exit(2)
    out = merge_deconv_into_dominance(Path(args.dominance_csv), deconv_csv)
    out.to_csv(ap.SAE_DOMINANCE_TABLE, index=False)
    print(f"Updated {ap.SAE_DOMINANCE_TABLE} "
          f"(deconv candidates={int(out['deconv_candidate_run'].sum())}, "
          f"confirmed={int(out['deconv_confirmed_surprisal'].sum())})")

    null_p = None
    if args.run_nulls:
        chans = select_null_channels(out, args.top_n)
        if chans.empty:
            print("No channels selected for nulls.")
        else:
            chan_path = ap.SAE_TABLES / "sparse_encoding_dominance_null_channels.csv"
            chans.to_csv(chan_path, index=False)
            mode = "refit" if args.null_refit_support else "frozen"
            suffix = f"_{FEATURE_TAG}_cand_null_{mode}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "sparse_encoding_deconvolution.py"),
                "--feature_tag", FEATURE_TAG,
                "--channels_from_csv", str(chan_path),
                "--null", "block",
                "--null_perms", str(args.null_perms),
                "--out_suffix", suffix,
            ]
            if args.null_refit_support:
                cmd.append("--null_refit_support")
            print("Running:", " ".join(cmd), flush=True)
            subprocess.check_call(cmd)
            null_p = ap.SAE_TABLES / f"sparse_encoding_deconv_null_pvals{suffix}.csv"

    addendum = ap.SAE_REPORTS / "sparse_encoding_surprisal_dominance_confirm.txt"
    # Prefer newly written null pvals if present
    if null_p is None:
        for cand in sorted(ap.SAE_TABLES.glob(
                f"sparse_encoding_deconv_null_pvals_{FEATURE_TAG}_cand_null_*.csv")):
            null_p = cand
    write_confirm_addendum(out, null_p, addendum)


if __name__ == "__main__":
    main()
