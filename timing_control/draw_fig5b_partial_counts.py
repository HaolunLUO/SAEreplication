#!/usr/bin/env python3
"""Selected-feature counts per Matryoshka bin across lags. New png only."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import draw_fig5_anatomy as fig5  # noqa: E402
import draw_fig5_bin_lags as lags  # noqa: E402
import paper_match_partial as pm  # noqa: E402

CSV = TC / "tables" / "tc_selected_counts_partial_anatomy_lags.csv"
OUT_NAME = "fig5b_anatomy_bin_counts_partial.png"
PROTECTED = [
    TC / "figures" / "paper-match" / "fig5.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial.png",
    TC / "figures" / "paper-match" / "fig5b_anatomy_bin_timecourse_partial.png",
    TC / "tables" / "tc_bin_refit_partial.csv",
    TC / "tables" / "tc_selected_indices_partial.csv",
]


def _hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _curve(frame: pd.DataFrame, value: str) -> pd.Series:
    elec = frame.groupby(["anatomy", "lag_ms", "subject"])[value].mean()
    return elec.groupby(["anatomy", "lag_ms"]).mean()


def main() -> None:
    before = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    if not CSV.exists():
        raise RuntimeError(f"missing {CSV.name}")
    scores = fig5.load_scores("partial")
    pop = fig5.population(scores)
    df = pd.read_csv(CSV)
    df = df.drop(columns=["anatomy"], errors="ignore")
    df = df.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"], how="inner")
    n_elec = df.groupby(["subject", "channel"]).ngroups
    n_lag = df.groupby(["subject", "channel"])["lag_ms"].nunique()
    if n_elec < len(pop) or int(n_lag.min()) < 51:
        raise RuntimeError(f"counts incomplete: elecs={n_elec} minlags={int(n_lag.min())}")
    curves = {label: _curve(df, f"{name}_mean") for label, name, _lo, _hi in pm.BINS}
    counts = pop.groupby("anatomy").agg(n=("channel", "size"), ns=("subject", "nunique"))
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 8.1), sharex=True)
    axes = list(axes.ravel())
    print("region,n_elec,n_subj,peak_lag_ms,peak_count_0_2048")
    for ax, region in zip(axes, fig5.EXPECT):
        series = []
        for label, _name, _lo, _hi in pm.BINS:
            piece = curves[label].xs(region).sort_index()
            ax.plot(
                piece.index.to_numpy(float), piece.to_numpy(float),
                color=pm.BIN_COLOR[label], lw=1.6, label=label,
            )
            series.append(piece.to_numpy(float))
        lo, hi = lags._panel_ylim(series)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-500, 2000)
        ax.set_xticks([-500, 0, 500, 1000, 1500, 2000])
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        n = int(counts.loc[region, "n"])
        ns = int(counts.loc[region, "ns"])
        pretty = region
        if "-" in pretty:
            head, tail = pretty.split("-", 1)
            pretty = f"{head}\n{tail}"
        ax.set_title(f"{pretty}\n({n} elec, {ns} subj)", fontsize=8)
        ax.set_xlabel("Lag (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        narrow = curves[pm.BINS[0][0]].xs(region).sort_index()
        lag_ms = narrow.index.to_numpy(float)
        values = narrow.to_numpy(float)
        peak = float(pm.argmax_lag(values, lag_ms))
        peak_n = float(values[int(np.flatnonzero(np.isclose(lag_ms, peak))[0])])
        print(f"{region},{n},{ns},{peak:.0f},{peak_n:.4f}")
    axes[0].set_ylabel("SAE features")
    axes[4].set_ylabel("SAE features")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(
        "Timing-regressed high-gamma\n"
        "Selected SAE features per bin at that lag, subject-mean of electrode-means.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.92))
    dest = fig5.POOL / OUT_NAME
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    for folder in (fig5.HOME, fig5.STORE):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, folder / OUT_NAME)
    after = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    if after != before:
        raise RuntimeError("a protected file changed while drawing the count trajectories")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
