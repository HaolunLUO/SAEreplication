#!/usr/bin/env python3
"""Part A: ridge signs on saved LASSO masks, signed Jaccard, Fig 4C.

Does not rerun LASSO and does not write into timing_control/tables.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import fc_common as C
import sparse_encoding.sparse_encoding_v3 as v3
import timing_control as tc
from sparse_encoding.sparse_encoding_regression import _fold_r, select_alpha

OUT = C.OUT
MEDIA = C.MEDIA


def _fit_fold(X, surp, yr, tr, te, support):
    """Same design and Ridge path as v3.predict_with_support, plus coefficients."""
    y = np.asarray(yr, dtype=np.float64)
    if tr.size < 20 or te.size < 5 or np.std(y[tr]) < 1e-12:
        return 0.0, np.nan, np.zeros(0, dtype=int), np.zeros(0)
    sur_tr, sur_te = v3._scale_surp(surp, tr, te)
    Xtr, Xte = v3._design(X, support, sur_tr, sur_te, tr, te)
    alpha = select_alpha(Xtr, y[tr])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, y[tr])
    pred = model.predict(Xte).ravel()
    r = _fold_r(pred, y[te], True)
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    idx = np.flatnonzero(support) if support is not None and np.asarray(support).any() else np.zeros(0, dtype=int)
    if coef.size != idx.size + 1:
        raise RuntimeError(f"coef length {coef.size} != {idx.size}+1")
    return r, alpha, idx, coef[:-1]


def refit(pop, scores, supports, X, surp, bv, T, neural):
    coef_rows = []
    check = []
    signed = {}
    for row in pop.itertuples(index=False):
        t0 = time.time()
        chans, sid = neural[row.subject]
        Y, OK = chans[str(row.channel)]
        Y = np.asarray(Y)
        OK = np.asarray(OK)
        if sid.shape[0] != Y.shape[0] or Y.shape[0] != X.shape[0]:
            raise RuntimeError(f"row mismatch {row.subject} {row.channel}")
        splits = v3.loso_outer_splits(sid)
        folds = supports[(row.subject, row.channel)]
        peaks = C.peak_lags(scores, row.subject, row.channel)
        acc = {}
        for fold, lag_from_idx, feat in folds:
            lag = peaks[fold]
            if np.isfinite(lag_from_idx) and not np.isclose(lag_from_idx, lag):
                raise RuntimeError(
                    f"peak lag mismatch {row.subject} {row.channel} fold {fold}: "
                    f"index {lag_from_idx} vs scores {lag}"
                )
            tr, te, held = splits[fold]
            if int(held) != fold + 1:
                raise RuntimeError(f"fold {fold} holds out section {held}")
            li = C.lag_index(lag)
            y = Y[:, li].astype(np.float64)
            a = tc._rows(tr, y, OK[:, li], bv)
            b = tc._rows(te, y, OK[:, li], bv)
            yr, _ = tc._residualize(y, T[:, li, :], a, np.r_[a, b])
            support = np.zeros(X.shape[1], dtype=bool)
            support[feat] = True
            r, alpha, idx, coef = _fit_fold(X, surp, yr, a, b, support)
            stored = C.stored_fold_r(scores, row.subject, row.channel, "sae", fold, lag)
            check.append({
                "subject": row.subject, "channel": row.channel, "fold": fold,
                "lag_ms": lag, "r": r, "stored_r": stored, "abs_diff": abs(r - stored),
                "n_feat": int(idx.size), "alpha": alpha,
            })
            for f_i, c in zip(idx, coef):
                coef_rows.append({
                    "subject": row.subject, "channel": row.channel, "elec_i": int(row.elec_i),
                    "fold": fold, "feature_index": int(f_i), "coef": float(c),
                    "lag_ms": lag, "alpha": alpha,
                })
                acc.setdefault(int(f_i), []).append(float(c))
        pairs = []
        for f_i, cs in acc.items():
            mean_c = float(np.mean(cs))
            if mean_c == 0.0:
                raise RuntimeError(f"zero mean coef {row.subject} {row.channel} {f_i}")
            pairs.append((f_i, int(np.sign(mean_c)), mean_c, len(cs)))
        signed[(row.subject, row.channel)] = pairs
        diffs = [c["abs_diff"] for c in check if c["channel"] == row.channel and c["subject"] == row.subject]
        worst = max(diffs) if diffs else float("nan")
        if worst > 1e-8:
            raise RuntimeError(
                f"peak-lag r mismatch {row.subject} {row.channel} max abs {worst}"
            )
        print(
            f"[A] {row.elec_i:3d} {row.subject} {row.channel} "
            f"feats {len(pairs)} max|dr| {worst:.2e} {time.time()-t0:.1f}s",
            flush=True,
        )
    return pd.DataFrame(coef_rows), pd.DataFrame(check), signed


def signed_ids(pairs) -> np.ndarray:
    """Encode (feature, sign) as feature*2 + (sign>0). Empty -> length 0."""
    if not pairs:
        return np.zeros(0, dtype=np.int32)
    ids = [f * 2 + (1 if s > 0 else 0) for f, s, _m, _n in pairs]
    return np.unique(np.asarray(ids, dtype=np.int32))


def jaccard_matrix(id_list: list[np.ndarray]) -> np.ndarray:
    n = len(id_list)
    # Sparse indicator via a python set gram is fine at n=151 and ~1k nnz.
    sets = [set(map(int, ids)) for ids in id_list]
    sizes = np.array([len(s) for s in sets], dtype=float)
    J = np.full((n, n), np.nan)
    for i in range(n):
        J[i, i] = 1.0 if sizes[i] else np.nan
        si = sets[i]
        for j in range(i + 1, n):
            if sizes[i] == 0 and sizes[j] == 0:
                continue
            inter = len(si & sets[j])
            union = sizes[i] + sizes[j] - inter
            val = inter / union if union else np.nan
            J[i, j] = J[j, i] = val
    return J


def contrast_means(J: np.ndarray, labels: np.ndarray):
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    vals = J[iu]
    same = labels[iu[0]] == labels[iu[1]]
    finite = np.isfinite(vals)
    within = vals[same & finite]
    between = vals[~same & finite]
    return within, between


def permutation_p(J, labels, n_perm=1000, seed=0):
    within, between = contrast_means(J, labels)
    obs = float(within.mean() - between.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        lab = rng.permutation(labels)
        w, b = contrast_means(J, lab)
        null[i] = w.mean() - b.mean()
    # One-sided: within minus between, add-one smoothing.
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, float(p), within, between, null


def fig4c(pop, supports, path_raw, path_wt):
    subjects = C.SUBJECTS
    n_by = {s: int((pop.subject == s).sum()) for s in subjects}
    # feature -> per-subject electrode counts
    feat_counts = {}
    for (subj, _ch), folds in supports.items():
        union = set()
        for _f, _lag, idx in folds:
            union.update(int(i) for i in idx)
        for feat in union:
            feat_counts.setdefault(feat, {s: 0 for s in subjects})
            feat_counts[feat][subj] += 1
    rows = []
    for feat, counts in feat_counts.items():
        raw = np.array([counts[s] for s in subjects], dtype=float)
        rates = raw / np.array([n_by[s] for s in subjects], dtype=float)
        rows.append({
            "feature_index": int(feat),
            "prevalence": int(raw.sum()),
            "n_subjects": int(np.sum(raw > 0)),
            "entropy_raw": C.entropy_bits(raw),
            "weight": float(rates.sum()),
            "entropy_weighted": C.entropy_bits(rates),
            **{f"n_{s}": int(counts[s]) for s in subjects},
        })
    feat = pd.DataFrame(rows).sort_values(
        ["prevalence", "feature_index"], ascending=[False, True]
    )
    h_cap = C.count_entropy_cap()
    h_eq = float(np.log2(len(subjects)))

    def _scatter(x, y, cap_x, cap_y, greens, xlabel, title, path, hline=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(x, y, s=14, c="#4c4c4c", linewidths=0, alpha=0.85, zorder=2, label="SAE feature")
        ax.scatter(
            [cap_x], [cap_y], s=64, marker="*", c="#e67e22", zorder=4,
            label=f"every electrode ({cap_y:.3f} bits)",
        )
        if greens:
            gx, gy = zip(*greens)
            ax.scatter(
                gx, gy, s=36, marker="v", c="#2ca02c", zorder=3,
                label="one subject, all of its electrodes",
            )
        if hline is not None:
            ax.axhline(hline, color="#e67e22", ls="--", lw=0.8, zorder=1)
        # Label the seven Table 1 features.
        lookup = feat.set_index("feature_index")
        for f_i in C.TABLE1:
            if f_i not in lookup.index:
                continue
            ax.annotate(
                str(f_i),
                (float(lookup.loc[f_i, x.name if False else "prevalence"] if xlabel.startswith("Electrodes") else lookup.loc[f_i, "weight"]),
                 float(lookup.loc[f_i, "entropy_raw" if xlabel.startswith("Electrodes") else "entropy_weighted"])),
                textcoords="offset points", xytext=(3, 3), fontsize=7, color="#1f4e79",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Participant entropy (bits)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(-0.08, max(h_eq, float(np.nanmax(y))) + 0.15)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    # The annotate block above is awkward. Draw explicitly.
    def draw(kind, path):
        if kind == "raw":
            x = feat["prevalence"].to_numpy()
            y = feat["entropy_raw"].to_numpy()
            cap = (151, h_cap)
            greens = [(n_by[s], 0.0) for s in subjects]
            xlabel = "Electrodes whose fold union contains the feature"
            title = "Feature prevalence, electrode counts"
            hline = None
            xcol, ycol = "prevalence", "entropy_raw"
        else:
            x = feat["weight"].to_numpy()
            y = feat["entropy_weighted"].to_numpy()
            cap = (float(len(subjects)), h_eq)
            greens = [(1.0, 0.0)]
            xlabel = "Sum of within-subject selection rates"
            title = "Feature prevalence, subject-weighted"
            hline = h_eq
            xcol, ycol = "weight", "entropy_weighted"
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        ax.scatter(x, y, s=16, c="#4c4c4c", linewidths=0, alpha=0.85, zorder=2, label="SAE feature")
        ax.scatter([cap[0]], [cap[1]], s=80, marker="*", c="#e67e22", zorder=4,
                   label=f"every electrode ({cap[1]:.3f} bits)")
        gx, gy = zip(*greens)
        ax.scatter(gx, gy, s=42, marker="v", c="#2ca02c", zorder=3,
                   label="one subject, all of its electrodes")
        if hline is not None:
            ax.axhline(hline, color="#e67e22", ls="--", lw=0.8, zorder=1)
        for f_i in C.TABLE1:
            hit = feat[feat.feature_index == f_i]
            if hit.empty:
                continue
            ax.annotate(
                str(f_i),
                (float(hit[xcol].iloc[0]), float(hit[ycol].iloc[0])),
                textcoords="offset points", xytext=(4, 3), fontsize=7, color="#1f4e79",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Participant entropy (bits)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ymax = max(h_eq, float(np.nanmax(y)), cap[1]) + 0.18
        ax.set_ylim(-0.1, ymax)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    draw("raw", path_raw)
    draw("wt", path_wt)
    return feat, h_cap, h_eq


def overlap_bars(summary_rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [r["label"] for r in summary_rows]
    means = [r["mean"] for r in summary_rows]
    colors = ["#1f4e79", "#7aa0c4", "#b85c38", "#e0b090"]
    ax.bar(np.arange(len(means)), means, color=colors[: len(means)], width=0.72)
    ax.set_xticks(np.arange(len(means)), labels, rotation=15, ha="right")
    ax.set_ylabel("Mean signed Jaccard")
    ax.set_title("Signed-support overlap")
    for i, r in enumerate(summary_rows):
        ax.text(i, means[i] + 0.008, f"n={r['n']}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(means) * 1.25 + 0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    pop = C.language_electrodes()
    pop.to_csv(OUT / "population.csv", index=False)
    print(f"[A] {len(pop)} electrodes", flush=True)
    scores = C.load_lag_scores()
    supports = C.load_sae_supports(pop)
    # Fill missing fold lags from the lag-score table (empty supports).
    for key, folds in list(supports.items()):
        peaks = C.peak_lags(scores, key[0], key[1])
        supports[key] = [
            (fold, peaks[fold] if not np.isfinite(lag) else lag, idx)
            for (fold, lag, idx), pk in zip(folds, peaks)
        ]
    print("[A] loading SAE, surprisal, nuisance", flush=True)
    X, surp, bv, T = C.load_design()
    neural = {}
    for subject in C.SUBJECTS:
        print(f"[A] neural {subject}", flush=True)
        neural[subject] = C.load_subject_bundle(subject)
        chans, sid = neural[subject]
        if sid.shape[0] != X.shape[0]:
            raise RuntimeError(f"{subject} neural rows {sid.shape[0]} != features {X.shape[0]}")
        if not np.array_equal(sid, np.repeat([1, 2, 3], [1753, 1800, 1919])):
            # Still acceptable if the three sections match the feature concatenation.
            print(f"[A] section id counts {subject} {np.bincount(sid)}", flush=True)
    coefs, check, signed = refit(pop, scores, supports, X, surp, bv, T, neural)
    coefs.to_csv(OUT / "ridge_coefficients.csv", index=False)
    check.to_csv(OUT / "ridge_sanity.csv", index=False)
    max_diff = float(check["abs_diff"].max())
    print(f"[A] sanity folds {len(check)} max abs diff {max_diff:.3e}", flush=True)
    if max_diff > 1e-8:
        raise RuntimeError(f"peak-lag Pearson r does not match stored scores ({max_diff})")

    sign_rows = []
    id_list = []
    n_flip = 0
    for row in pop.itertuples(index=False):
        pairs = signed[(row.subject, row.channel)]
        id_list.append(signed_ids(pairs))
        for f_i, sgn, mean_c, n_folds in pairs:
            # Sign flip across folds: the saved coefs for this pair.
            cs = coefs[
                (coefs.subject == row.subject)
                & (coefs.channel == row.channel)
                & (coefs.feature_index == f_i)
            ]["coef"].to_numpy()
            if np.any(cs > 0) and np.any(cs < 0):
                n_flip += 1
            sign_rows.append({
                "elec_i": int(row.elec_i),
                "subject": row.subject,
                "channel": row.channel,
                "region": row.region,
                "feature_index": int(f_i),
                "mean_coef": mean_c,
                "sign": int(sgn),
                "n_folds": int(n_folds),
            })
    signs = pd.DataFrame(sign_rows)
    signs.to_csv(OUT / "electrode_signs.csv", index=False)
    print(f"[A] selections {len(signs)} sign-changes {n_flip} empty {(np.array([len(s) for s in id_list])==0).sum()}", flush=True)

    J = jaccard_matrix(id_list)
    np.save(OUT / "jaccard_signed.npy", J)
    subj = pop["subject"].to_numpy()
    obs_s, p_s, w_s, b_s, null_s = permutation_p(J, subj, seed=0)
    labeled = pop["labeled_region"].to_numpy()
    J_r = J[np.ix_(labeled, labeled)]
    reg = pop.loc[labeled, "region"].to_numpy()
    obs_r, p_r, w_r, b_r, null_r = permutation_p(J_r, reg, seed=1)
    np.save(OUT / "jaccard_perm_subject.npy", null_s)
    np.save(OUT / "jaccard_perm_region.npy", null_r)

    summary = [
        {"contrast": "within_subject", "label": "Within subject", "mean": float(w_s.mean()),
         "n": int(w_s.size), "diff_within_minus_between": obs_s, "permutation_p": p_s},
        {"contrast": "between_subject", "label": "Between subject", "mean": float(b_s.mean()),
         "n": int(b_s.size), "diff_within_minus_between": obs_s, "permutation_p": p_s},
        {"contrast": "within_region", "label": "Within region", "mean": float(w_r.mean()),
         "n": int(w_r.size), "diff_within_minus_between": obs_r, "permutation_p": p_r},
        {"contrast": "between_region", "label": "Between region", "mean": float(b_r.mean()),
         "n": int(b_r.size), "diff_within_minus_between": obs_r, "permutation_p": p_r},
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(MEDIA / "jaccard_summary.csv", index=False)
    summary_df.to_csv(OUT / "jaccard_summary.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)
    overlap_bars(summary, MEDIA / "overlap_jaccard_peak.png")

    feat, h_cap, h_eq = fig4c(
        pop, supports, MEDIA / "fig4c_raw.png", MEDIA / "fig4c_weighted.png",
    )
    feat.to_csv(MEDIA / "fig4c_features.csv", index=False)
    feat.to_csv(OUT / "fig4c_features.csv", index=False)
    meta = pd.DataFrame([{
        "n_electrodes": 151,
        "n_empty_support": int((np.array([len(s) for s in id_list]) == 0).sum()),
        "n_selections": int(len(signs)),
        "n_sign_changes": int(n_flip),
        "sanity_max_abs_diff": max_diff,
        "sanity_n_folds": int(len(check)),
        "entropy_cap_raw": h_cap,
        "entropy_cap_weighted": h_eq,
        "n_features": int(len(feat)),
        "subject_diff": obs_s,
        "subject_p": p_s,
        "region_diff": obs_r,
        "region_p": p_r,
        "n_region_electrodes": int(labeled.sum()),
    }])
    meta.to_csv(OUT / "part_a_meta.csv", index=False)
    print(f"[A] entropy cap raw {h_cap:.4f} weighted {h_eq:.4f} features {len(feat)}", flush=True)
    print("[A] done", flush=True)


if __name__ == "__main__":
    main()
