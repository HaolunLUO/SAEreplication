#!/usr/bin/env python3
"""Stack the generalization array and draw the Fig 4B heatmap."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import fc_common as C
import sparse_encoding.sparse_encoding_v3 as v3

GEN = C.OUT / "generalization"
MEDIA = C.MEDIA


def pair_mean(M, mask):
    vals = M[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan, int(vals.size)


def main():
    pop = pd.read_csv(GEN / "population.csv")
    n = len(pop)
    files = [GEN / f"source_{i:03d}.csv" for i in range(n)]
    missing = [p.name for p in files if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing {len(missing)} source files, e.g. {missing[:5]}")
    Z = np.full((n, n), np.nan)
    R = np.full((n, n), np.nan)
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        if len(df) != n:
            raise RuntimeError(f"{path.name} has {len(df)} rows")
        df = df.sort_values("target_i")
        Z[i, :] = df["fisher_z"].to_numpy()
        R[i, :] = df["mean_r"].to_numpy()
    np.save(GEN / "fisher_z.npy", Z)
    np.save(GEN / "mean_r.npy", R)

    z = np.load(GEN / "residuals.npz")
    surp_r = z["surprisal_stored_r"]
    surp_fresh = z["surprisal_fresh_r"]
    check = pd.read_csv(GEN / "surprisal_check.csv")
    max_surp_diff = float(check["abs_diff"].max())
    # Stored column, after the fresh one-column check on the first electrodes.
    surp_z = np.array([v3.fisher_z_mean(surp_r[t].tolist()) for t in range(n)])
    surp_mean_r = surp_r.mean(axis=1)
    if max_surp_diff > 1e-6:
        print(f"[agg] surprisal check max abs {max_surp_diff:.3e}; using fresh scores", flush=True)
        surp_z = np.array([v3.fisher_z_mean(surp_fresh[t].tolist()) for t in range(n)])
        surp_mean_r = surp_fresh.mean(axis=1)
        surp_used = "fresh"
    else:
        surp_used = "stored"

    # Diagonal sanity against the unconstrained peak-lag SAE score from part A.
    sanity = pd.read_csv(C.OUT / "ridge_sanity.csv")
    stored_z = []
    for t, row in pop.iterrows():
        sub = sanity[(sanity.subject == row.subject) & (sanity.channel == row.channel)]
        sub = sub.sort_values("fold")
        stored_z.append(v3.fisher_z_mean(sub["stored_r"].tolist()))
    stored_z = np.asarray(stored_z)
    diag = np.diag(Z)
    diag_diff = diag - stored_z

    subj = pop["subject"].to_numpy()
    region = pop["region"].to_numpy()
    labeled = pop["labeled_region"].to_numpy().astype(bool)
    eye = np.eye(n, dtype=bool)
    same_subj = subj[:, None] == subj[None, :]
    same_reg = (region[:, None] == region[None, :]) & labeled[:, None] & labeled[None, :]
    both_labeled = labeled[:, None] & labeled[None, :]
    off = ~eye

    def pack(name, M, base, mask):
        mean, k = pair_mean(M, mask)
        gain_mat = M - base[None, :]
        gain, _ = pair_mean(gain_mat, mask)
        return {"contrast": name, "mean": mean, "gain_over_surprisal": gain, "n_pairs": k}

    rows = []
    for metric, M, base in (
        ("fisher_z", Z, surp_z),
        ("mean_r", R, surp_mean_r),
    ):
        specs = [
            ("within_subject_offdiag", same_subj & off),
            ("within_subject_with_diag", same_subj),
            ("cross_subject", ~same_subj),
            ("within_region_offdiag", same_reg & off),
            ("cross_region", both_labeled & ~same_reg),
            ("diagonal", eye),
        ]
        for name, mask in specs:
            rec = pack(name, M, base, mask)
            rec["metric"] = metric
            rows.append(rec)
        rows.append({
            "contrast": "surprisal_baseline_electrodes",
            "mean": float(np.mean(base)),
            "gain_over_surprisal": 0.0,
            "n_pairs": n,
            "metric": metric,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(MEDIA / "generalization_summary.csv", index=False)
    summary.to_csv(GEN / "generalization_summary.csv", index=False)

    diag_df = pd.DataFrame({
        "elec_i": np.arange(n),
        "subject": pop["subject"],
        "channel": pop["channel"],
        "diag_fisher_z": diag,
        "stored_peak_fisher_z": stored_z,
        "diff": diag_diff,
    })
    diag_df.to_csv(GEN / "diagonal_sanity.csv", index=False)
    meta = pd.DataFrame([{
        "surprisal_column": surp_used,
        "surprisal_check_max_abs": max_surp_diff,
        "diag_minus_stored_mean": float(np.mean(diag_diff)),
        "diag_minus_stored_max_abs": float(np.max(np.abs(diag_diff))),
        "diag_pearson_with_stored": float(np.corrcoef(diag, stored_z)[0, 1]),
        "within_subject_diagonal": "excluded",
        "paper_metric": "noise-ceiling-normalized Fisher-z",
    }])
    meta.to_csv(GEN / "aggregate_meta.csv", index=False)
    print(summary[summary.metric == "fisher_z"].to_string(index=False), flush=True)
    print(meta.to_string(index=False), flush=True)

    # Heatmap order is already subject, then region.
    boundaries = np.cumsum([int((pop.subject == s).sum()) for s in C.SUBJECTS])[:-1]
    for M, tag, label in (
        (Z, "fisherz", "Fisher z"),
        (R, "pearson", "Mean Pearson r"),
    ):
        lim = float(np.nanpercentile(np.abs(M), 98))
        lim = max(lim, 0.05)
        fig, ax = plt.subplots(figsize=(8.2, 7.2))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest", origin="upper")
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.6)
            ax.axvline(b - 0.5, color="black", lw=0.6)
        ticks = []
        labels = []
        start = 0
        for s in C.SUBJECTS:
            k = int((pop.subject == s).sum())
            ticks.append(start + k / 2 - 0.5)
            labels.append(s.replace("Subject", "S"))
            start += k
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.set_xlabel("Target electrode")
        ax.set_ylabel("Source electrode")
        ax.set_title(f"Signed-feature generalization ({label})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        fig.tight_layout()
        fig.savefig(MEDIA / f"fig4b_{tag}.png", dpi=160)
        fig.savefig(MEDIA / f"fig4b_{tag}.pdf")
        plt.close(fig)
    print("[agg] wrote heatmaps", flush=True)


if __name__ == "__main__":
    main()
