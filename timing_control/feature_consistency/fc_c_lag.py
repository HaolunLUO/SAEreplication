#!/usr/bin/env python3
"""Signed-support overlap at each electrode's peak lag versus a fixed 300 ms lag."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_a
import fc_common as C

MEDIA = C.MEDIA
LASSO = C.OUT / "lasso_300"


def supports_from_coefs(coef: pd.DataFrame, pop: pd.DataFrame):
    """Electrode-level (feature, sign) from the mean coefficient across folds."""
    id_list = []
    rows = []
    for row in pop.itertuples(index=False):
        sub = coef[(coef.subject == row.subject) & (coef.channel == row.channel)]
        pairs = []
        if len(sub):
            for feat, g in sub.groupby("feature_index"):
                mean_c = float(g["coef"].mean())
                if mean_c == 0.0:
                    continue
                pairs.append((int(feat), int(np.sign(mean_c)), mean_c, int(len(g))))
        id_list.append(fc_a.signed_ids(pairs))
        for feat, sgn, mean_c, n_folds in pairs:
            rows.append({
                "elec_i": int(row.elec_i), "subject": row.subject, "channel": row.channel,
                "feature_index": feat, "sign": sgn, "mean_coef": mean_c, "n_folds": n_folds,
            })
    return id_list, pd.DataFrame(rows)


def main():
    pop = pd.read_csv(C.OUT / "population.csv")
    peak_signs = pd.read_csv(C.OUT / "electrode_signs.csv")
    peak_ids = []
    for row in pop.itertuples(index=False):
        sub = peak_signs[peak_signs.elec_i == row.elec_i]
        pairs = [
            (int(r.feature_index), int(r.sign), float(r.mean_coef), int(r.n_folds))
            for r in sub.itertuples(index=False)
        ]
        peak_ids.append(fc_a.signed_ids(pairs))
    coef300 = pd.read_csv(LASSO / "coefficients.csv")
    ids300, signs300 = supports_from_coefs(coef300, pop)
    signs300.to_csv(LASSO / "electrode_signs.csv", index=False)

    # Per-electrode overlap of that electrode's peak support with its 300 ms support.
    elec_rows = []
    for i, row in pop.iterrows():
        a = set(map(int, peak_ids[i]))
        b = set(map(int, ids300[i]))
        if not a and not b:
            jac = np.nan
        else:
            inter = len(a & b)
            jac = inter / (len(a) + len(b) - inter)
        elec_rows.append({
            "elec_i": int(row.elec_i), "subject": row.subject, "channel": row.channel,
            "region": row.region, "jaccard_peak_vs_300": jac,
            "n_peak": len(a), "n_300": len(b),
        })
    elec = pd.DataFrame(elec_rows)
    elec.to_csv(MEDIA / "lag_overlap_electrodes.csv", index=False)

    J300 = fc_a.jaccard_matrix(ids300)
    np.save(LASSO / "jaccard_signed.npy", J300)
    subj = pop["subject"].to_numpy()
    obs_s, p_s, w_s, b_s, _ = fc_a.permutation_p(J300, subj, seed=0)
    labeled = pop["labeled_region"].to_numpy()
    obs_r, p_r, w_r, b_r, _ = fc_a.permutation_p(
        J300[np.ix_(labeled, labeled)], pop.loc[labeled, "region"].to_numpy(), seed=1,
    )
    peak_sum = pd.read_csv(C.OUT / "jaccard_summary.csv")
    rows = []
    for label, mean, n, diff, p, lag in (
        ("Within subject", float(w_s.mean()), int(w_s.size), obs_s, p_s, "300 ms"),
        ("Between subject", float(b_s.mean()), int(b_s.size), obs_s, p_s, "300 ms"),
        ("Within region", float(w_r.mean()), int(w_r.size), obs_r, p_r, "300 ms"),
        ("Between region", float(b_r.mean()), int(b_r.size), obs_r, p_r, "300 ms"),
    ):
        rows.append({"lag": lag, "label": label, "mean": mean, "n": n,
                     "diff_within_minus_between": diff, "permutation_p": p})
    for rec in peak_sum.to_dict(orient="records"):
        rows.append({
            "lag": "peak", "label": rec["label"], "mean": rec["mean"], "n": rec["n"],
            "diff_within_minus_between": rec["diff_within_minus_between"],
            "permutation_p": rec["permutation_p"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(MEDIA / "jaccard_peak_vs_300.csv", index=False)

    # Grouped bars: peak vs 300 for the four contrasts.
    labels = ["Within subject", "Between subject", "Within region", "Between region"]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(labels))
    peak_m = [float(peak_sum.loc[peak_sum.label == lab, "mean"].iloc[0]) for lab in labels]
    m300 = [float(summary[(summary.lag == "300 ms") & (summary.label == lab)]["mean"].iloc[0]) for lab in labels]
    ax.bar(x - 0.18, peak_m, width=0.36, color="#1f4e79", label="Own peak lag")
    ax.bar(x + 0.18, m300, width=0.36, color="#e67e22", label="300 ms")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Mean signed Jaccard")
    ax.set_title("Signed-support overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MEDIA / "overlap_jaccard_peak_vs_300.png", dpi=160)
    fig.savefig(MEDIA / "overlap_jaccard_peak_vs_300.pdf")
    plt.close(fig)

    finite = elec["jaccard_peak_vs_300"].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(finite, bins=20, color="#4c4c4c", range=(0, 1))
    ax.axvline(finite.mean(), color="#e67e22", lw=1.2, label=f"mean {finite.mean():.2f}")
    ax.set_xlabel("Signed Jaccard, peak lag vs 300 ms")
    ax.set_ylabel("Electrodes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MEDIA / "lag_overlap_hist.png", dpi=160)
    fig.savefig(MEDIA / "lag_overlap_hist.pdf")
    plt.close(fig)
    print(
        f"[lag] electrode Jaccard mean {finite.mean():.3f} "
        f"n {len(finite)} subject diff 300ms {obs_s:.3f} p {p_s:.3f} "
        f"region diff {obs_r:.3f} p {p_r:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
