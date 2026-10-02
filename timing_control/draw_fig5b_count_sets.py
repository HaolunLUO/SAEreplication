#!/usr/bin/env python3
"""Count trajectories for glove partial, language raw, and glove raw.

Writes only the three new pngs under figures/paper-match/. Store copies are
made by the caller so a compute node does not need the session store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import anatomy_count_sets as counts  # noqa: E402
import draw_fig5_bin_lags as lags  # noqa: E402
import paper_match_partial as pm  # noqa: E402

POOL = TC / "figures" / "paper-match"
SPECS = [
    {
        "key": "glove_partial",
        "csv": counts.OUT["glove_partial"],
        "png": "fig5b_anatomy_bin_counts_glove_partial.png",
        "title": "Timing-regressed high-gamma\nGloVe-significant electrodes. Selected SAE features per bin at that lag, subject-mean of electrode-means.",
        "expect_elec": 157,
        "expect_regions": 17,
    },
    {
        "key": "lang_raw",
        "csv": counts.OUT["lang_raw"],
        "png": "fig5b_anatomy_bin_counts_lang_rawhg.png",
        "title": "Raw high-gamma\nLeft-hemisphere language electrodes. Selected SAE features per bin at that lag, subject-mean of electrode-means.",
        "expect_elec": 127,
        "expect_regions": 8,
    },
    {
        "key": "glove_raw",
        "csv": counts.OUT["glove_raw"],
        "png": "fig5b_anatomy_bin_counts_glove_rawhg.png",
        "title": "Raw high-gamma\nGloVe-significant electrodes. Selected SAE features per bin at that lag, subject-mean of electrode-means.",
        "expect_elec": 157,
        "expect_regions": 17,
    },
]


def _curve(frame: pd.DataFrame, value: str) -> pd.Series:
    elec = frame.groupby(["anatomy", "lag_ms", "subject"])[value].mean()
    return elec.groupby(["anatomy", "lag_ms"]).mean()


def _grid(n_regions: int) -> tuple[int, int, tuple[float, float]]:
    if n_regions <= 8:
        return 2, 4, (12.8, 8.1)
    return 4, 5, (16.5, 11.4)


def draw_one(spec: dict) -> str:
    dest = POOL / spec["png"]
    if dest.exists() and dest.stat().st_size > 0 and spec["csv"].exists():
        # Already drawn from this count table. Encoding-score figures use other names.
        print(f"skip existing count figure {dest}", flush=True)
    pop = counts.glove_population() if spec["key"].startswith("glove") else counts.language_population()
    if not spec["csv"].exists():
        raise RuntimeError(f"missing {spec['csv'].name}")
    df = pd.read_csv(spec["csv"])
    df = df.drop(columns=["anatomy"], errors="ignore")
    df = df.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"], how="inner")
    n_elec = df.groupby(["subject", "channel"]).ngroups
    n_lag = df.groupby(["subject", "channel"])["lag_ms"].nunique()
    if n_elec != spec["expect_elec"] or int(n_lag.min()) < 51:
        raise RuntimeError(f"{spec['png']} incomplete: elecs={n_elec} minlags={int(n_lag.min())}")
    regions = counts._regions(pop)
    if len(regions) != spec["expect_regions"]:
        raise RuntimeError(f"{spec['key']} regions {len(regions)}")
    curves = {label: _curve(df, f"{name}_mean") for label, name, _lo, _hi in pm.BINS}
    elec_n = pop.groupby("anatomy").agg(n=("channel", "size"), ns=("subject", "nunique"))
    nrows, ncols, figsize = _grid(len(regions))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = list(np.atleast_1d(axes).ravel())
    best = None
    for ax in axes[len(regions):]:
        ax.axis("off")
    for ax, region in zip(axes, regions):
        series = []
        for label, _name, _lo, _hi in pm.BINS:
            piece = curves[label].xs(region).sort_index()
            y = piece.to_numpy(float)
            lag_ms = piece.index.to_numpy(float)
            ax.plot(lag_ms, y, color=pm.BIN_COLOR[label], lw=1.6, label=label)
            series.append(y)
            peak = float(pm.argmax_lag(y, lag_ms))
            peak_n = float(y[int(np.flatnonzero(np.isclose(lag_ms, peak))[0])])
            if best is None or peak_n > best[0] + 1e-12:
                best = (peak_n, region, label, peak)
        lo, hi = lags._panel_ylim(series)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-500, 2000)
        ax.set_xticks([-500, 0, 500, 1000, 1500, 2000])
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        n = int(elec_n.loc[region, "n"])
        ns = int(elec_n.loc[region, "ns"])
        pretty = region
        if "-" in pretty:
            head, tail = pretty.split("-", 1)
            pretty = f"{head}\n{tail}"
        ax.set_title(f"{pretty}\n({n} elec, {ns} subj)", fontsize=7.5)
        ax.set_xlabel("Lag (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    for i in range(nrows):
        axes[i * ncols].set_ylabel("SAE features")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(spec["title"], fontsize=11)
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    if not (dest.exists() and dest.stat().st_size > 0):
        POOL.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150)
        print(f"wrote {dest}", flush=True)
    plt.close(fig)
    if best is None:
        raise RuntimeError(f"no peak for {spec['png']}")
    peak_n, region, label, peak = best
    note = f"{spec['png']}: {region}, {label}, peak count {peak_n:.2f} at {peak:.0f} ms"
    print(note, flush=True)
    return note


def main() -> None:
    for spec in SPECS:
        draw_one(spec)


if __name__ == "__main__":
    main()
