#!/usr/bin/env python3
"""Bin-model lag trajectories for the eight anatomy regions. New pngs only."""

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
import paper_match_partial as pm  # noqa: E402

BIN_CSV = {
    "partial": TC / "tables" / "tc_bin_refit_anatomy_partial_lags.csv",
    "raw": TC / "tables" / "tc_bin_refit_anatomy_rawhg_lags.csv",
}
OUT_NAME = {
    "partial": "fig5_anatomy_partial_bin_lag.png",
    "raw": "fig5_anatomy_rawhg_bin_lag.png",
}
TITLE = {
    "partial": "Timing-regressed high-gamma",
    "raw": "Raw high-gamma",
}
PROTECTED = [
    TC / "figures" / "paper-match" / name
    for name in (
        "fig5.png",
        "fig5_anatomy_partial.png",
        "fig5_anatomy_rawhg.png",
        "fig5_anatomy_rawhg_300ms.png",
        "fig5_anatomy_partial_300ms.png",
        "fig5_anatomy_rawhg_ab.png",
    )
]


def _hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _complete(path: Path, pop: pd.DataFrame) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    df = df.drop(columns=["anatomy"], errors="ignore")
    df = df.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"], how="inner")
    counts = df.groupby(["subject", "channel", "lag_ms"])["bin_name"].nunique()
    n_elec = df.groupby(["subject", "channel"]).ngroups
    if n_elec < len(pop) or int(counts.min()) < 3 or df["lag_ms"].nunique() < 51:
        print(f"{path.name} incomplete: elecs={n_elec} lags={df['lag_ms'].nunique()}", flush=True)
        return None
    return df


def _subject_mean(frame: pd.DataFrame, value: str) -> pd.Series:
    elec = frame.groupby(["anatomy", "lag_ms", "subject"])[value].mean()
    return elec.groupby(["anatomy", "lag_ms"]).mean()


def _panel_ylim(ys: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([y[np.isfinite(y)] for y in ys if np.isfinite(y).any()])
    lo = min(0.0, float(np.min(finite)))
    hi = max(0.0, float(np.max(finite)))
    span = hi - lo
    if span == 0:
        span = 0.05
    pad = 0.08 * span
    return lo - pad, hi + pad


def draw_mode(mode: str, pop: pd.DataFrame, scores: pd.DataFrame, bins: pd.DataFrame) -> None:
    surp = scores.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"])
    surp = surp[(surp["feature_space"] == "surprisal") & (surp["lag_ms"] >= -500) & (surp["lag_ms"] <= 2000)]
    surp_curve = _subject_mean(surp, "fisher_z_mean")
    name_to_label = {b[1]: b[0] for b in pm.BINS}
    bin_curves = {}
    for name, label in name_to_label.items():
        piece = bins[bins["bin_name"] == name]
        bin_curves[label] = _subject_mean(piece, "fisher_z_mean")
    counts = pop.groupby("anatomy").agg(n=("channel", "size"), ns=("subject", "nunique"))
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 7.05), sharex=True)
    axes = axes.ravel()
    colors = {"Surprisal": "dimgray", **pm.BIN_COLOR}
    for ax, region in zip(axes, fig5.EXPECT):
        series = []
        lags = surp_curve.xs(region).index.to_numpy(float)
        y = surp_curve.xs(region).to_numpy(float)
        order = np.argsort(lags)
        lags, y = lags[order], y[order]
        ax.plot(lags, y, color=colors["Surprisal"], lw=1.6, label="Surprisal")
        series.append(y)
        for label, _name, _lo, _hi in pm.BINS:
            curve = bin_curves[label].xs(region)
            bl = curve.index.to_numpy(float)
            by = curve.to_numpy(float)
            order = np.argsort(bl)
            ax.plot(bl[order], by[order], color=colors[label], lw=1.6, label=label)
            series.append(by[order])
        lo, hi = _panel_ylim(series)
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(
        f"{TITLE[mode]}. Bin-restricted models, surprisal forced in. "
        "Subject-mean of electrode-means.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    dest = fig5.POOL / OUT_NAME[mode]
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    for folder in (fig5.HOME, fig5.STORE):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, folder / OUT_NAME[mode])
    print(f"wrote {dest}", flush=True)


def main() -> None:
    before = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    wrote = []
    for mode in ("partial", "raw"):
        scores = fig5.load_scores(mode)
        pop = fig5.population(scores)
        bins = _complete(BIN_CSV[mode], pop)
        if bins is None:
            print(f"skip {mode}: lag scores are not complete", flush=True)
            continue
        draw_mode(mode, pop, scores, bins)
        wrote.append(mode)
    after = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    if after != before:
        raise RuntimeError("a protected figure changed while drawing lag trajectories")
    if not wrote:
        raise SystemExit(2)
    print("drew " + ",".join(wrote), flush=True)


if __name__ == "__main__":
    main()
