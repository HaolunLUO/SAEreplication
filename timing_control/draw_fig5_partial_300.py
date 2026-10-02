#!/usr/bin/env python3
"""Panel C only: timing-regressed high-gamma bin models at 300 ms.

Does not write fig5_anatomy_partial.png or fig5_anatomy_rawhg_300ms.png.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import draw_fig5_anatomy as fig5  # noqa: E402
import paper_match_partial as pm  # noqa: E402

CSV = TC / "tables" / "tc_bin_refit_anatomy_partial.csv"
OUT_NAME = "fig5_anatomy_partial_300ms.png"
PROTECTED = [
    TC / "figures" / "paper-match" / "fig5_anatomy_partial.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg_300ms.png",
    TC / "figures" / "paper-match" / "fig5.png",
    TC / "tables" / "tc_bin_refit_partial.csv",
    TC / "tables" / "tc_bin_refit_anatomy_partial.csv",
    TC / "tables" / "tc_bin_refit_anatomy_rawhg.csv",
    TC / "tables" / "tc_bin_refit_rawhg_anatomy_300ms.csv",
]


def _hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    before = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    scores = fig5.load_scores("partial")
    pop = fig5.population(scores)
    stats = fig5.panel_c(CSV, pop, scores, 300.0)
    if stats is None:
        raise RuntimeError("300 ms partial bin scores do not cover the anatomy electrodes")
    keys = ["Surprisal"] + [b[0] for b in pm.BINS]
    subjects = sorted(pop["subject"].unique())
    cmap = plt.get_cmap("tab10")
    scol = {s: cmap(i % 10) for i, s in enumerate(subjects)}
    fig, ax = plt.subplots(1, 1, figsize=(16.2, 6.4))
    colors = {"Surprisal": "dimgray", **pm.BIN_COLOR}
    fig5.draw_bars(ax, stats, keys, colors, scol)
    ax.set_ylabel("Fisher-z")
    ax.set_title("Bin-restricted models at 300 ms (surprisal forced in)")
    ax.text(-0.045, 1.04, "C", transform=ax.transAxes, fontsize=14, fontweight="bold", clip_on=False)
    fig.suptitle(
        "Timing-regressed high-gamma. Eight left-hemisphere language AAL regions. Fixed lag 300 ms.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.0, 0.995, 0.94))
    dest = fig5.POOL / OUT_NAME
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    for folder in (fig5.HOME, fig5.STORE):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, folder / OUT_NAME)
    after = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    if after != before:
        raise RuntimeError("a protected file changed while drawing the partial 300 ms figure")
    print("region,n_elec,n_subj,surprisal,bin0-2048,bin2048-16384,bin16384-65536")
    for region in fig5.EXPECT:
        block = stats[region]
        means = block["mean"]
        print(
            f"{region},{block['n_elec']},{block['n_subj']},"
            f"{means['Surprisal']:.6f},{means['0–2048']:.6f},"
            f"{means['2048–16384']:.6f},{means['16384–65536']:.6f}"
        )
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
