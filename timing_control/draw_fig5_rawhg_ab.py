#!/usr/bin/env python3
"""Raw Fig 5 panels A and B only. Does not write the other Fig 5 pngs."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
sys.path.insert(0, str(TC))
import draw_fig5_anatomy as fig5  # noqa: E402
import paper_match_partial as pm  # noqa: E402

OUT_NAME = "fig5_anatomy_rawhg_ab.png"
PROTECTED = [
    TC / "figures" / "paper-match" / "fig5.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_rawhg_300ms.png",
    TC / "figures" / "paper-match" / "fig5_anatomy_partial_300ms.png",
]


def _hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    before = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    scores = fig5.load_scores("raw")
    pop = fig5.population(scores)
    supports = fig5.load_raw_supports(pop)
    if supports is None or supports.empty:
        raise RuntimeError("raw selected indices are not complete")
    feat = pm.feature_table(supports, pop)
    counts = fig5.fold_bin_counts(supports, pop)
    bstats = fig5.region_stats(counts, pop, [b[0] for b in pm.BINS])
    peak = fig5.gain_peak(scores, pop)
    dest = fig5.POOL / OUT_NAME
    fig5.draw(
        dest,
        signal="Raw high-gamma.",
        peak=peak,
        feat=feat,
        bin_stats=bstats,
        c_stats=None,
        pop=pop,
        note="Left-hemisphere language electrodes by AAL3 region. Raw high-gamma (v2, no timing residual).",
    )
    for folder in (fig5.HOME, fig5.STORE):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, folder / OUT_NAME)
    after = {str(p): _hash(p) for p in PROTECTED if p.exists()}
    if after != before:
        raise RuntimeError("a protected figure changed while drawing raw panels A and B")
    print(f"raw SAE indices {len(feat)} peak {peak:.0f}", flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
