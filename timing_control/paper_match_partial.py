#!/usr/bin/env python3
"""Study 3 paper-match figures on the v2 / partial speech-timing signal.

Reads the existing lag scores and partial supports. The only table this script
writes is tables/tc_bin_refit_partial.csv, and only in --mode refit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

os.environ.setdefault("ANALYSISEV_DATA_ROOT", "/orcd/pool/005/haolun52/analysisEV")

TC = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control")
TABLES = TC / "tables"
TAX = Path(
    "/orcd/pool/005/haolun52/analysisEV/group_encoding_results/functional_taxonomy/tables/"
    "functional_taxonomy_electrode_table.csv"
)
BIN_CSV = TABLES / "tc_bin_refit_partial.csv"
PARTS = TABLES / "tc_selected_indices_partial_parts"
COMBINED = TABLES / "tc_selected_indices_partial.csv"

POOL_FIG = TC / "figures" / "paper-match"
SYNC_ROOT = Path(
    "/home/haolun52/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/files"
)
SYNC_FIG = SYNC_ROOT / "media" / "sae-v3" / "timing-control" / "paper-match"
SYNC_DOC = SYNC_ROOT / "docs" / "sae-v3-timing-control-paper.md"
STORE_FIG = Path(
    "/run/user/245046/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/"
    "files/media/sae-v3/timing-control/paper-match"
)
STORE_DOC = Path(
    "/run/user/245046/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/"
    "files/docs/sae-v3-timing-control-paper.md"
)

Y_CUT = -20.0  # AntTemp if MNI y >= -20, else PostTemp
FROI_ORDER = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp"]
POP_FROIS = ["IFG", "MFG", "PostTemp"]  # the raw paper-match population
BINS = (
    ("0–2048", "bin0-2048", 0, 2048),
    ("2048–16384", "bin2048-16384", 2048, 16384),
    ("16384–65536", "bin16384-65536", 16384, 65536),
)
FOLDS = (0, 1, 2)
SPACES = ("surprisal", "sae", "residual")
SPACE_LABEL = {"surprisal": "Surprisal", "sae": "Matryoshka SAE", "residual": "Residual"}
SPACE_COLOR = {"surprisal": "dimgray", "sae": "skyblue", "residual": "firebrick"}
BIN_COLOR = {"0–2048": "#2171b5", "2048–16384": "#6baed6", "16384–65536": "#bdd7e7"}
REF_MS = 550.0
PROTECTED = {
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_forced_lag_scores.csv",
    "tc_glove_both_lag_scores.csv",
    "sig_glove_timing.csv",
    "tc_selected_indices_partial.csv",
    "tc_both_lag_scores_trf.csv",
    "tc_wide_lag_scores_trf.csv",
    "tc_both_lag_scores_trf_partial.csv",
    "tc_wide_lag_scores_trf_partial.csv",
    "sig_glove_trf.csv",
}

_STATE: dict = {}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def argmax_lag(values: np.ndarray, lags: np.ndarray) -> float:
    """Max of the gain curve. Ties break toward 300 ms, then the earlier lag."""
    values = np.asarray(values, dtype=float)
    lags = np.asarray(lags, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError("SAE-gain curve has no finite lag")
    m = np.max(values[finite])
    cand = np.flatnonzero(finite & np.isclose(values, m, rtol=0.0, atol=1e-12))
    dist = np.abs(lags[cand] - 300.0)
    order = np.lexsort((lags[cand], dist))
    return float(lags[cand[order[0]]])


def froi_label(region: str, mni_y: float, *, left_only: bool = True) -> str | None:
    """AAL3 name to a Study 3 fROI.

    The raw paper-match counts (IFG 49/4, MFG 16/4, PostTemp 7/1) are the
    left-hemisphere map. AntTemp is MNI y >= -20 on superior, middle, or
    inferior temporal gyrus; PostTemp is the same labels posterior to that cut.
    """
    r = str(region)
    if left_only and not r.startswith("Left "):
        return None
    if "Inferior frontal gyrus" in r and "orbital" in r:
        return "IFGorb"
    if "Inferior frontal gyrus" in r and ("triangular" in r or "opercular" in r):
        return "IFG"
    if r.endswith("Middle frontal gyrus") or " Middle frontal gyrus" in r:
        return "MFG"
    temporal = any(
        s in r
        for s in (
            "Superior temporal gyrus",
            "Middle temporal gyrus",
            "Inferior temporal gyrus",
        )
    )
    if temporal and "pole" not in r.lower():
        if not np.isfinite(mni_y):
            return None
        return "AntTemp" if float(mni_y) >= Y_CUT else "PostTemp"
    return None


def load_taxonomy() -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "MNI_y", "is_lang", "hemisphere"])
    tax = tax[tax["is_lang"].fillna(False).astype(bool)].copy()
    tax["froi"] = [
        froi_label(r, y, left_only=True) for r, y in zip(tax["region"], tax["MNI_y"])
    ]
    tax["froi_bilateral"] = [
        froi_label(r, y, left_only=False) for r, y in zip(tax["region"], tax["MNI_y"])
    ]
    return tax


def load_scores() -> pd.DataFrame:
    usecols = [
        "subject", "channel", "condition", "mode", "feature_space", "lag_ms", "fisher_z_mean",
    ]
    frames = []
    for name in ("tc_both_lag_scores.csv", "tc_wide_lag_scores.csv"):
        df = pd.read_csv(TABLES / name, usecols=usecols)
        df = df[
            (df["condition"] == "v2")
            & (df["mode"] == "partial")
            & (df["feature_space"].isin(SPACES))
        ]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(
        ["subject", "channel", "condition", "mode", "feature_space", "lag_ms"], keep="last"
    )
    keys = out[["subject", "channel"]].drop_duplicates()
    if len(keys) != out.groupby(["subject", "channel"]).ngroups:
        raise RuntimeError("electrode union is not unique")
    return out


def population(tax: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    scored = scores[["subject", "channel"]].drop_duplicates()
    pop = tax[tax["froi"].isin(POP_FROIS)].merge(scored, on=["subject", "channel"], how="inner")
    missing = tax[tax["froi"].isin(POP_FROIS)].merge(scored, on=["subject", "channel"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    if len(missing):
        print(f"[paper-match] {len(missing)} fROI electrodes have no partial lag score", flush=True)
    return pop.reset_index(drop=True)


def wide_scores(scores: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    sub = scores.merge(pop[["subject", "channel", "froi"]], on=["subject", "channel"])
    wide = sub.pivot_table(
        index=["subject", "channel", "froi", "lag_ms"],
        columns="feature_space",
        values="fisher_z_mean",
        aggfunc="last",
    ).reset_index()
    need = list(SPACES)
    wide = wide.dropna(subset=need)
    wide["gain"] = wide["sae"] - wide["surprisal"]
    return wide


def gain_curve(wide: pd.DataFrame) -> tuple[pd.Series, float, list[float]]:
    marked = wide.copy()
    marked["elec"] = marked["subject"] + "\t" + marked["channel"]
    n_elec = marked["elec"].nunique()
    n_at = marked.groupby("lag_ms")["elec"].nunique()
    lags = n_at[n_at == n_elec].index.to_numpy(float)
    use = wide[wide["lag_ms"].isin(lags)]
    subj = use.groupby(["lag_ms", "subject"])["gain"].mean()
    curve = subj.groupby("lag_ms").mean().sort_index()
    peak = argmax_lag(curve.to_numpy(), curve.index.to_numpy(float))
    return curve, peak, sorted(float(x) for x in lags)


def _subj_values(wide: pd.DataFrame, lag: float, froi: str, space: str) -> pd.Series:
    sl = wide[(np.isclose(wide["lag_ms"], lag)) & (wide["froi"] == froi)]
    if sl.empty:
        return pd.Series(dtype=float)
    return sl.groupby("subject")[space].mean()


def bar_stats(wide: pd.DataFrame, lag: float) -> dict:
    """Per fROI: subject-mean of electrode-mean Fisher-z, SEM, paired t vs surprisal."""
    out = {}
    for froi in FROI_ORDER:
        block = {"n_elec": int((wide.loc[np.isclose(wide["lag_ms"], lag), "froi"] == froi).sum())}
        # n_elec above counts rows at this lag; recompute from unique electrodes
        el = wide[np.isclose(wide["lag_ms"], lag) & (wide["froi"] == froi)]
        block["n_elec"] = int(el.groupby(["subject", "channel"]).ngroups)
        block["n_subj"] = int(el["subject"].nunique())
        means = {}
        sems = {}
        values = {}
        for space in SPACES:
            vals = _subj_values(wide, lag, froi, space)
            values[space] = {str(k): float(v) for k, v in vals.items()}
            means[space] = float(vals.mean()) if len(vals) else float("nan")
            sems[space] = float(vals.sem(ddof=1)) if len(vals) >= 2 else float("nan")
        block["mean"] = means
        block["sem"] = sems
        block["subject"] = values
        tests = {}
        sur = _subj_values(wide, lag, froi, "surprisal")
        for space in ("sae", "residual"):
            other = _subj_values(wide, lag, froi, space)
            paired = pd.concat([other.rename("a"), sur.rename("b")], axis=1).dropna()
            if len(paired) < 2:
                tests[space] = {"n": int(len(paired)), "t": None, "p": None, "p_bonferroni": None, "stars": ""}
                continue
            t, p = ttest_rel(paired["a"], paired["b"])
            p_b = float(min(1.0, float(p) * 2.0))
            tests[space] = {
                "n": int(len(paired)),
                "t": float(t),
                "p": float(p),
                "p_bonferroni": p_b,
                "stars": stars(p_b),
            }
        block["tests"] = tests
        if np.isfinite(means["sae"]) and np.isfinite(means["residual"]) and np.isfinite(means["surprisal"]):
            block["delta"] = float(0.5 * (means["sae"] + means["residual"]) - means["surprisal"])
        else:
            block["delta"] = None
        out[froi] = block
    return out


def _part_path(subject: str, channel: str) -> Path:
    safe = str(channel).replace("/", "_")
    return PARTS / f"{subject}__{safe}__v2__partial.csv"


def load_supports(pop: pd.DataFrame) -> pd.DataFrame:
    """Union the combined CSV and the part files. A part file wins for its electrode."""
    cols = ["subject", "channel", "feature_space", "fold", "feature_index"]
    frames = []
    have = set()
    for row in pop.itertuples(index=False):
        part = _part_path(row.subject, row.channel)
        if not part.exists():
            continue
        df = pd.read_csv(part)
        have.add((row.subject, row.channel))
        if df.empty:
            continue
        df = df[(df["feature_space"] == "sae") & (df["condition"] == "v2") & (df["mode"] == "partial")]
        frames.append(df[cols])
    combined = pd.read_csv(COMBINED)
    combined = combined[
        (combined["feature_space"] == "sae")
        & (combined["condition"] == "v2")
        & (combined["mode"] == "partial")
    ]
    keys = list(zip(combined["subject"], combined["channel"]))
    pop_keys = set(zip(pop["subject"], pop["channel"]))
    keep = [k in pop_keys and k not in have for k in keys]
    extra = combined.loc[keep, cols]
    if len(extra):
        frames.append(extra)
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(cols)


def feature_table(supports: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """One row per SAE index: prevalence and entropy of per-subject selection rates."""
    elec = pop[["subject", "channel"]].drop_duplicates()
    n_by = elec.groupby("subject").size().to_dict()
    subjects = sorted(n_by)
    if supports.empty:
        return pd.DataFrame(columns=["feature_index", "prevalence", "entropy"])
    union = supports.groupby(["subject", "channel", "feature_index"]).size().reset_index(name="n")
    rows = []
    for feat, g in union.groupby("feature_index"):
        selected = set(zip(g["subject"], g["channel"]))
        prev = len(selected)
        rates = []
        for subj in subjects:
            n = n_by[subj]
            hit = sum(1 for s, _c in selected if s == subj)
            rates.append(hit / n if n else 0.0)
        rates = np.asarray(rates, dtype=float)
        total = rates.sum()
        if total <= 0:
            ent = 0.0
        else:
            p = rates[rates > 0] / total
            ent = float(-(p * np.log2(p)).sum())
        rows.append({"feature_index": int(feat), "prevalence": int(prev), "entropy": ent, "rates": rates})
    return pd.DataFrame(rows)


def fold_bin_counts(supports: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Per electrode, mean across the three LOSO folds of the SAE count in each bin."""
    rows = []
    grouped = {
        (s, c, f): g["feature_index"].to_numpy(int)
        for (s, c, f), g in supports.groupby(["subject", "channel", "fold"])
    } if len(supports) else {}
    for row in pop.itertuples(index=False):
        rec = {"subject": row.subject, "channel": row.channel, "froi": row.froi}
        for label, _name, lo, hi in BINS:
            per = []
            for fold in FOLDS:
                idx = grouped.get((row.subject, row.channel, fold), np.array([], dtype=int))
                per.append(int(np.sum((idx >= lo) & (idx < hi))))
            rec[label] = float(np.mean(per))
        rows.append(rec)
    return pd.DataFrame(rows)


def bin_bar_stats(counts: pd.DataFrame) -> dict:
    out = {}
    for froi in FROI_ORDER:
        sub = counts[counts["froi"] == froi]
        block = {"n_elec": int(len(sub)), "n_subj": int(sub["subject"].nunique()) if len(sub) else 0}
        means, sems, dots = {}, {}, {}
        for label, _n, _lo, _hi in BINS:
            if sub.empty:
                means[label] = float("nan")
                sems[label] = float("nan")
                dots[label] = {}
                continue
            vals = sub.groupby("subject")[label].mean()
            means[label] = float(vals.mean())
            sems[label] = float(vals.sem(ddof=1)) if len(vals) >= 2 else float("nan")
            dots[label] = {str(k): float(v) for k, v in vals.items()}
        block["mean"] = means
        block["sem"] = sems
        block["subject"] = dots
        out[froi] = block
    return out


def _fmt(x, nd=3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def _pm(mean, sem) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(sem):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sem:.4f}"


def coverage_note(tax: pd.DataFrame) -> dict:
    left = tax[tax["froi"].notna()].groupby("froi").agg(n=("channel", "size"), n_subj=("subject", "nunique"))
    right = tax[tax["froi_bilateral"].notna() & tax["froi"].isna()]
    extra = (
        right.groupby("froi_bilateral")
        .agg(n=("channel", "size"), n_subj=("subject", "nunique"))
        .to_dict(orient="index")
        if len(right)
        else {}
    )
    return {
        "left": {k: {"electrodes": int(left.loc[k, "n"]), "subjects": int(left.loc[k, "n_subj"])} for k in left.index},
        "right_excluded": {
            k: {"electrodes": int(v["n"]), "subjects": int(v["n_subj"])} for k, v in extra.items()
        },
    }


def draw_fig4(stats_peak: dict, peak: float, ref_stats: dict, feat: pd.DataFrame, pop: pd.DataFrame, path: Path) -> None:
    subjects = sorted(pop["subject"].unique())
    cmap = plt.get_cmap("tab10")
    scol = {s: cmap(i % 10) for i, s in enumerate(subjects)}
    n_elec = len(pop)
    n_subj = pop["subject"].nunique()
    n_feat = int(feat["feature_index"].nunique()) if len(feat) else 0
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.35), gridspec_kw={"width_ratios": [1.15, 1.05]})
    _bars(axes[0], stats_peak, SPACES, SPACE_LABEL, SPACE_COLOR, scol, show_delta=True, show_stars=True)
    axes[0].set_ylabel("Raw Fisher-z")
    axes[0].set_title("Predictivity by language fROI")
    axes[0].text(-0.12, 1.06, "A", transform=axes[0].transAxes, fontsize=14, fontweight="bold")

    ax = axes[1]
    if len(feat):
        ax.scatter(feat["prevalence"], feat["entropy"], s=18, c="0.25", alpha=0.55, linewidths=0, zorder=2)
    n_by = pop.groupby("subject").size().to_dict()
    orange = (n_elec, float(np.log2(n_subj)) if n_subj else 0.0)
    ax.scatter([orange[0]], [orange[1]], s=70, c="darkorange", marker="D", zorder=4, label="every electrode")
    green_x = [int(n_by[s]) for s in subjects]
    ax.scatter(green_x, np.zeros(len(green_x)), s=46, c="green", marker="s", zorder=4, label="one subject, all electrodes")
    ax.set_xlabel("Electrodes whose fold union contains the feature")
    ax.set_ylabel("Entropy of per-subject selection rate (bits)")
    ax.set_title(f"Feature prevalence ({n_feat} features, {n_elec} electrodes / {n_subj} subjects)")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=-0.08)
    ax.text(-0.12, 1.06, "C", transform=ax.transAxes, fontsize=14, fontweight="bold")
    if len(feat):
        tail = feat[feat["entropy"] <= 1e-9]
        axins = ax.inset_axes([0.58, 0.45, 0.40, 0.38])
        src = tail if len(tail) else feat.nsmallest(min(8, len(feat)), "entropy")
        axins.scatter(src["prevalence"], src["entropy"], s=16, c="0.25", alpha=0.7, linewidths=0)
        axins.scatter(green_x, np.zeros(len(green_x)), s=28, c="green", marker="s", zorder=4)
        axins.set_title("entropy = 0", fontsize=8)
        axins.tick_params(labelsize=7)
        axins.set_ylim(-0.02, 0.08)
    ref_bits = []
    for froi in POP_FROIS:
        b = ref_stats[froi]
        ref_bits.append(
            f"{froi} {_fmt(b['mean']['surprisal'])}/{_fmt(b['mean']['sae'])}/{_fmt(b['mean']['residual'])}"
        )
    fig.suptitle(
        f"Raw Fisher-z at {peak:.0f} ms (timing-controlled SAE-gain peak). "
        "No shuffle bar, no JumpReLU.",
        fontsize=11,
    )
    fig.text(
        0.01, 0.01,
        "550 ms, same subject-mean of electrode-means (surprisal / SAE / residual): " + "; ".join(ref_bits),
        fontsize=8, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bars(ax, stats, keys, labels, colors, scol, show_delta, show_stars) -> None:
    x = np.arange(len(FROI_ORDER))
    n = len(keys)
    w = 0.8 / n
    mid = (n - 1) / 2.0
    tops = []
    group_top = np.full(len(FROI_ORDER), -np.inf)
    for i, key in enumerate(keys):
        means, sems = [], []
        for froi in FROI_ORDER:
            means.append(stats[froi]["mean"][key])
            sems.append(stats[froi]["sem"][key])
        means = np.asarray(means, dtype=float)
        sems = np.asarray(sems, dtype=float)
        for j, froi in enumerate(FROI_ORDER):
            if stats[froi]["n_elec"] == 0:
                means[j] = np.nan
                sems[j] = np.nan
        yerr = np.where(np.isfinite(sems), sems, 0.0)
        xpos = x + (i - mid) * w
        ax.bar(
            xpos, means, w,
            yerr=yerr, capsize=3, color=colors[key], label=labels[key],
            error_kw={"elinewidth": 0.8}, zorder=1,
        )
        for j, froi in enumerate(FROI_ORDER):
            dots = stats[froi]["subject"].get(key, {})
            subs = sorted(dots)
            for k, subj in enumerate(subs):
                jitter = (k - (len(subs) - 1) / 2.0) * min(0.04, w * 0.35)
                ax.scatter(
                    xpos[j] + jitter, dots[subj], s=22, color=scol[subj],
                    edgecolor="white", linewidths=0.4, zorder=3,
                )
            if np.isfinite(means[j]):
                tip = means[j] + (sems[j] if np.isfinite(sems[j]) else 0.0)
                group_top[j] = max(group_top[j], tip)
                tops.append(tip)
            if show_stars and key in stats[froi]["tests"]:
                lab = stats[froi]["tests"][key]["stars"]
                if lab and np.isfinite(means[j]):
                    y = means[j] + (sems[j] if np.isfinite(sems[j]) else 0.0)
                    ax.text(xpos[j], y + 0.001, lab, ha="center", va="bottom", fontsize=8)
                    group_top[j] = max(group_top[j], y + 0.012)
                    tops.append(y + 0.012)
    if show_delta:
        for j, froi in enumerate(FROI_ORDER):
            delta = stats[froi]["delta"]
            if delta is None or stats[froi]["n_elec"] == 0 or not np.isfinite(group_top[j]):
                continue
            y = group_top[j] + 0.004
            x0, x1 = x[j] - mid * w, x[j] + mid * w
            ax.plot([x0, x0, x1, x1], [y, y + 0.003, y + 0.003, y], color="k", lw=0.7, clip_on=False)
            ax.text(x[j], y + 0.003, f"Δ={delta:+.3f}", ha="center", va="bottom", fontsize=8, color="0.15")
            tops.append(y + 0.014)
    labels_x = []
    for froi in FROI_ORDER:
        labels_x.append(f"{froi}\n({stats[froi]['n_elec']} elec, {stats[froi]['n_subj']} subj)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    if tops:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, min(0, min(tops)) - 0.01), max(hi, max(tops) + 0.012))


def draw_fig5(feat: pd.DataFrame, bin_stats: dict, scol: dict, path: Path, c_stats: dict | None, peak: float) -> None:
    n_rows = 3 if c_stats is not None else 2
    fig_h = 8.0 if n_rows == 3 else 6.2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10.74, fig_h))
    ax = axes[0]
    if len(feat):
        order = np.argsort(feat["feature_index"].to_numpy())
        xs = feat["feature_index"].to_numpy()[order]
        ys = feat["prevalence"].to_numpy()[order]
        ax.vlines(xs, 0, ys, color="slategray", lw=0.6)
    for cut in (2048, 16384):
        ax.axvline(cut, color="red", lw=1.0)
    ax.set_xlim(0, 65536)
    ax.set_ylabel("Electrodes")
    ax.set_xlabel("SAE index")
    ax.set_title("Selected Matryoshka indices")
    ax.text(-0.08, 1.06, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    _bars(
        axes[1], bin_stats,
        [b[0] for b in BINS],
        {b[0]: b[0] for b in BINS},
        BIN_COLOR, scol, show_delta=False, show_stars=False,
    )
    axes[1].set_ylabel("SAE features (mean across folds)")
    axes[1].set_title("Features per Matryoshka bin")
    axes[1].text(-0.08, 1.06, "B", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    title = f"Speech-timing partial signal. Panels A–B use the fROI supports. Peak lag {peak:.0f} ms."
    if c_stats is not None:
        keys = ["Surprisal"] + [b[0] for b in BINS]
        labels = {k: k for k in keys}
        colors = {"Surprisal": "dimgray", **BIN_COLOR}
        _bars(axes[2], c_stats, keys, labels, colors, scol, show_delta=False, show_stars=False)
        axes[2].set_ylabel("Raw Fisher-z")
        axes[2].set_title(f"Bin-restricted models at {peak:.0f} ms (surprisal forced in)")
        axes[2].text(-0.08, 1.06, "C", transform=axes[2].transAxes, fontsize=14, fontweight="bold")
    else:
        title += " Fig 5C is waiting on the bin refit."
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bin_refit_stats(path: Path, pop: pd.DataFrame, wide: pd.DataFrame, peak: float) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    df = df.merge(pop[["subject", "channel", "froi"]], on=["subject", "channel"], how="inner", suffixes=("", "_pop"))
    if "froi_pop" in df.columns:
        df["froi"] = df["froi_pop"]
    if not np.allclose(df["lag_ms"].unique(), peak):
        return None
    # complete when every population electrode has all three bins
    have = df.groupby(["subject", "channel"])["bin_name"].nunique()
    if len(have) < len(pop) or (have < 3).any():
        return None
    name_to_label = {b[1]: b[0] for b in BINS}
    out = {}
    for froi in FROI_ORDER:
        el = pop[pop["froi"] == froi]
        block = {"n_elec": int(len(el)), "n_subj": int(el["subject"].nunique()) if len(el) else 0}
        means, sems, dots = {}, {}, {}
        sur = _subj_values(wide, peak, froi, "surprisal")
        means["Surprisal"] = float(sur.mean()) if len(sur) else float("nan")
        sems["Surprisal"] = float(sur.sem(ddof=1)) if len(sur) >= 2 else float("nan")
        dots["Surprisal"] = {str(k): float(v) for k, v in sur.items()}
        sub = df[df["froi"] == froi]
        for label, name, _lo, _hi in BINS:
            piece = sub[sub["bin_name"] == name]
            if piece.empty:
                means[label] = float("nan")
                sems[label] = float("nan")
                dots[label] = {}
                continue
            vals = piece.groupby("subject")["fisher_z_mean"].mean()
            means[label] = float(vals.mean())
            sems[label] = float(vals.sem(ddof=1)) if len(vals) >= 2 else float("nan")
            dots[label] = {str(k): float(v) for k, v in vals.items()}
        block["mean"] = means
        block["sem"] = sems
        block["subject"] = dots
        out[froi] = block
    return out


def _count_line(pop: pd.DataFrame) -> str:
    bits = []
    for froi in FROI_ORDER:
        sub = pop[pop["froi"] == froi]
        bits.append(f"{froi} {len(sub)}/{sub['subject'].nunique()}")
    return ", ".join(bits)


def write_notes(summary: dict, destinations: list[Path]) -> None:
    peak = summary["peak_lag_ms"]
    lines = [
        "---",
        'cursor:',
        '  subagentId: "bc-9bfa695f-e82d-5243-b377-5a389f78dbb8"',
        "---",
        "",
        "# Timing-controlled paper-match (Study 3)",
        "",
        "Neural signal: existing v2 / partial readout. Eight speech-timing covariates are residualized inside each leave-one-section-out fold. This is not the Zou TRF residual and not raw high-gamma.",
        "",
        "Lag scores: union of `tc_both_lag_scores.csv` and `tc_wide_lag_scores.csv`, condition v2, mode partial, one row per electrode. Supports: part files under `tc_selected_indices_partial_parts/` when present, otherwise `tc_selected_indices_partial.csv`.",
        "",
        "Population: language electrodes, left-hemisphere AAL3. IFG is opercular or triangular inferior frontal gyrus. MFG is middle frontal gyrus. PostTemp is superior, middle, or inferior temporal gyrus with MNI y < −20. AntTemp is the same temporal labels at MNI y ≥ −20. IFGorb is orbital inferior frontal gyrus. This is the labeling that matches the raw paper-match counts.",
        "",
        f"Counts (electrodes/subjects): {summary['counts_text']}. Unique subjects: {summary['n_subjects']}. Total electrodes: {summary['n_electrodes']}.",
        "",
        f"Right-hemisphere language electrodes that a bilateral map would add, and that the raw figure did not include: {summary['right_excluded_text']}.",
        "",
        f"SAE gain is Matryoshka SAE minus surprisal on each electrode. The curve is the subject-mean of those electrode means. Primary lag is the argmax on the lags shared by every fROI electrode: **{peak:.0f} ms** (gain {summary['peak_gain']:.4f}). Ties would break toward 300 ms, then the earlier lag.",
        "",
        "## Fig 4A",
        "",
        f"Bars at {peak:.0f} ms. Subject means of electrode-mean raw Fisher-z. SEM across subjects. Subject dots. Paired t versus surprisal, Bonferroni across SAE and residual (two tests). Δ = mean(SAE, residual) − surprisal. IFGorb and AntTemp have no electrodes, so they have no bars and no test. PostTemp has one subject, so no SEM and no t.",
        "",
    ]
    for froi in FROI_ORDER:
        b = summary["bars_peak"][froi]
        lines.append(
            f"- {froi} ({b['n_elec']} elec, {b['n_subj']} subj): "
            f"surprisal {_pm(b['mean']['surprisal'], b['sem']['surprisal'])}, "
            f"SAE {_pm(b['mean']['sae'], b['sem']['sae'])} {b['tests']['sae']['stars'] or 'no test'}, "
            f"residual {_pm(b['mean']['residual'], b['sem']['residual'])} {b['tests']['residual']['stars'] or 'no test'}, "
            f"Δ={_fmt(b['delta'], 4)}"
        )
        for space in ("sae", "residual"):
            t = b["tests"][space]
            if t["p"] is not None:
                lines.append(
                    f"  - {space} vs surprisal: t={t['t']:.3f}, p={t['p']:.4g}, Bonferroni p={t['p_bonferroni']:.4g}, n={t['n']}"
                )
    lines += ["", "At 550 ms, same aggregation:", ""]
    for froi in POP_FROIS:
        b = summary["bars_550"][froi]
        lines.append(
            f"- {froi}: surprisal {_pm(b['mean']['surprisal'], b['sem']['surprisal'])}, "
            f"SAE {_pm(b['mean']['sae'], b['sem']['sae'])}, "
            f"residual {_pm(b['mean']['residual'], b['sem']['residual'])}, "
            f"Δ={_fmt(b['delta'], 4)}"
        )
    lines += [
        "",
        "## Fig 4C",
        "",
        f"{summary['n_features']} SAE features on {summary['n_electrodes']} electrodes / {summary['n_subjects']} subjects. Prevalence is electrodes whose fold union contains the feature. Entropy is the entropy, in bits, of the per-subject selection rates after those rates are normalized to sum to 1. Orange marks every electrode (prevalence {summary['n_electrodes']}, entropy log2({summary['n_subjects']}) = {summary['orange_entropy']:.3f}). Green marks one subject and all of its electrodes (entropy 0) at each subject's electrode count: {summary['green_counts']}. The inset is the entropy-0 tail ({summary['n_entropy0']} features).",
        "",
        "## Fig 5A and 5B",
        "",
        "5A: one mark per selected SAE index, height = electrodes whose fold union contains it. Red lines at 2048 and 16384. Axis 0–65536.",
        "",
        "5B: subject-mean of the electrode-mean of per-fold SAE counts. SEM across subjects.",
        "",
    ]
    for froi in FROI_ORDER:
        b = summary["bins_5b"][froi]
        bits = ", ".join(f"{lab} {_pm(b['mean'][lab], b['sem'][lab])}" for lab, *_ in BINS)
        lines.append(f"- {froi}: {bits}")
    lines += ["", "## Fig 5C", ""]
    if summary.get("fig5c"):
        lines.append(
            f"Bin-restricted partial models at {peak:.0f} ms only, surprisal forced in, fROI electrodes only. "
            "Surprisal-only is the surprisal column already in the lag tables at that lag. "
            f"Table: `tables/tc_bin_refit_partial.csv`."
        )
        lines.append("")
        for froi in FROI_ORDER:
            b = summary["fig5c"][froi]
            keys = ["Surprisal"] + [lab for lab, *_ in BINS]
            bits = ", ".join(f"{k} {_pm(b['mean'][k], b['sem'][k])}" for k in keys)
            lines.append(f"- {froi}: {bits}")
    else:
        lines.append(
            f"Not drawn. Bin refit job {summary.get('job_id') or '(not submitted)'} is still running or queued. "
            "It writes only `tables/tc_bin_refit_partial.csv`."
        )
    text = "\n".join(lines) + "\n"
    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)


def write_doc(summary: dict) -> None:
    peak = summary["peak_lag_ms"]
    fig4 = "/cursor/stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/media/sae-v3/timing-control/paper-match/fig4.png"
    fig5 = "/cursor/stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/media/sae-v3/timing-control/paper-match/fig5.png"
    lines = [
        "---",
        "cursor:",
        '  subagentId: "bc-9bfa695f-e82d-5243-b377-5a389f78dbb8"',
        "---",
        "",
        "# Paper figures on the speech-timing partial signal",
        "",
        "Study 3 panels for Lepori et al. (arXiv:2606.06857v2), redrawn on the existing v2 / partial readout. Eight timing covariates are residualized inside each leave-one-section-out fold. This is not the Zou TRF residual and not raw high-gamma.",
        "",
        f"Figures: [fig4]({fig4}), [fig5]({fig5}). Notes sit beside the pngs. The pool copies are `timing_control/figures/paper-match/`.",
        "",
        "## Population",
        "",
        "Language electrodes whose left-hemisphere AAL3 labels map to IFG (opercular or triangular), MFG, or PostTemp (superior, middle, or inferior temporal gyrus, MNI y < −20). IFGorb and AntTemp stay empty. Anterior versus posterior temporal uses MNI y ≥ −20.",
        "",
        f"Counts, electrodes/subjects: {summary['counts_text']}. {summary['n_electrodes']} electrodes, {summary['n_subjects']} subjects.",
        "",
        f"A bilateral map would also pick up right-hemisphere temporal language electrodes ({summary['right_excluded_text']}). Those are outside the raw paper-match labeling, so they are not in these panels.",
        "",
        "## Peak lag",
        "",
        f"SAE gain = Matryoshka SAE − surprisal on each electrode. Subject-mean of electrode-means. Argmax on these fROIs is **{peak:.0f} ms** (gain {summary['peak_gain']:.4f}). The raw paper-match was locked at 550 ms.",
        "",
        "### Bars at the timing-controlled peak",
        "",
    ]
    for froi in POP_FROIS:
        b = summary["bars_peak"][froi]
        lines.append(
            f"- {froi}: surprisal {_pm(b['mean']['surprisal'], b['sem']['surprisal'])}, "
            f"SAE {_pm(b['mean']['sae'], b['sem']['sae'])} ({b['tests']['sae']['stars'] or 'no test'}), "
            f"residual {_pm(b['mean']['residual'], b['sem']['residual'])} ({b['tests']['residual']['stars'] or 'no test'}), "
            f"Δ={_fmt(b['delta'], 4)}"
        )
    lines += ["", "### Same bars at 550 ms", ""]
    for froi in POP_FROIS:
        b = summary["bars_550"][froi]
        lines.append(
            f"- {froi}: surprisal {_pm(b['mean']['surprisal'], b['sem']['surprisal'])}, "
            f"SAE {_pm(b['mean']['sae'], b['sem']['sae'])}, "
            f"residual {_pm(b['mean']['residual'], b['sem']['residual'])}, "
            f"Δ={_fmt(b['delta'], 4)}"
        )
    lines += [
        "",
        "## Feature panels",
        "",
        f"Fig 4C: {summary['n_features']} SAE features. Orange entropy {summary['orange_entropy']:.3f} at prevalence {summary['n_electrodes']}. Entropy-0 features: {summary['n_entropy0']}.",
        "",
        "Fig 5B, subject-mean of electrode-mean fold counts:",
        "",
    ]
    for froi in POP_FROIS:
        b = summary["bins_5b"][froi]
        bits = ", ".join(f"{lab} {_pm(b['mean'][lab], b['sem'][lab])}" for lab, *_ in BINS)
        lines.append(f"- {froi}: {bits}")
    lines += ["", "## Fig 5C", ""]
    if summary.get("fig5c"):
        lines.append(f"Fit at {peak:.0f} ms only. Surprisal-only is the lag-table column, not a new fit.")
        lines.append("")
        for froi in POP_FROIS:
            b = summary["fig5c"][froi]
            keys = ["Surprisal"] + [lab for lab, *_ in BINS]
            bits = ", ".join(f"{k} {_pm(b['mean'][k], b['sem'][k])}" for k in keys)
            lines.append(f"- {froi}: {bits}")
    else:
        lines.append(
            f"Pending. Slurm job {summary.get('job_id') or 'not submitted'}. "
            "The job writes only `timing_control/tables/tc_bin_refit_partial.csv`."
        )
    text = "\n".join(lines) + "\n"
    for dest in (STORE_DOC, SYNC_DOC):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)


def build_summary(job_id: str | None) -> dict:
    tax = load_taxonomy()
    scores = load_scores()
    pop = population(tax, scores)
    wide = wide_scores(scores, pop)
    curve, peak, _lags = gain_curve(wide)
    supports = load_supports(pop)
    feat = feature_table(supports, pop)
    counts = fold_bin_counts(supports, pop)
    cov = coverage_note(tax)
    right = cov["right_excluded"]
    if right:
        right_txt = ", ".join(f"{k} {v['electrodes']}/{v['subjects']}" for k, v in sorted(right.items()))
    else:
        right_txt = "none"
    n_by = pop.groupby("subject").size().to_dict()
    c5 = bin_refit_stats(BIN_CSV, pop, wide, peak)
    summary = {
        "signal": "v2 partial",
        "n_electrodes": int(len(pop)),
        "n_subjects": int(pop["subject"].nunique()),
        "counts_text": _count_line(pop),
        "counts": {
            froi: {
                "electrodes": int((pop["froi"] == froi).sum()),
                "subjects": int(pop.loc[pop["froi"] == froi, "subject"].nunique()),
            }
            for froi in FROI_ORDER
        },
        "right_excluded_text": right_txt,
        "peak_lag_ms": peak,
        "peak_gain": float(curve.loc[peak] if peak in curve.index else curve.iloc[np.argmin(np.abs(curve.index - peak))]),
        "gain_at_550": float(curve.loc[REF_MS]) if REF_MS in curve.index or any(np.isclose(curve.index, REF_MS)) else None,
        "bars_peak": bar_stats(wide, peak),
        "bars_550": bar_stats(wide, REF_MS),
        "n_features": int(feat["feature_index"].nunique()) if len(feat) else 0,
        "n_entropy0": int((feat["entropy"] <= 1e-9).sum()) if len(feat) else 0,
        "orange_entropy": float(np.log2(pop["subject"].nunique())),
        "green_counts": {str(k): int(v) for k, v in sorted(n_by.items())},
        "bins_5b": bin_bar_stats(counts),
        "fig5c": c5,
        "job_id": job_id,
        "subjects": sorted(pop["subject"].unique()),
    }
    # stash frames for plotting via summary side channel
    summary["_pop"] = pop
    summary["_feat"] = feat
    summary["_bin_counts"] = counts
    summary["_wide"] = wide
    return summary


def public_summary(summary: dict) -> dict:
    skip = {"_pop", "_feat", "_bin_counts", "_wide"}
    out = {k: v for k, v in summary.items() if k not in skip}
    return out


def plot_all(job_id: str | None) -> dict:
    summary = build_summary(job_id)
    pop = summary["_pop"]
    subjects = sorted(pop["subject"].unique())
    cmap = plt.get_cmap("tab10")
    scol = {s: cmap(i % 10) for i, s in enumerate(subjects)}
    for folder in (POOL_FIG, SYNC_FIG, STORE_FIG):
        folder.mkdir(parents=True, exist_ok=True)
    # draw once into pool, then copy, so the three copies match
    fig4 = POOL_FIG / "fig4.png"
    fig5 = POOL_FIG / "fig5.png"
    draw_fig4(summary["bars_peak"], summary["peak_lag_ms"], summary["bars_550"], summary["_feat"], pop, fig4)
    draw_fig5(
        summary["_feat"], summary["bins_5b"], scol, fig5,
        summary["fig5c"], summary["peak_lag_ms"],
    )
    import shutil
    for folder in (SYNC_FIG, STORE_FIG):
        shutil.copy2(fig4, folder / "fig4.png")
        shutil.copy2(fig5, folder / "fig5.png")
    note_paths = [POOL_FIG / "NOTES.md", SYNC_FIG / "NOTES.md", STORE_FIG / "NOTES.md"]
    write_notes(summary, note_paths)
    write_doc(summary)
    return public_summary(summary)


# ---------------------------------------------------------------- refit

def _init_worker() -> None:
    sys.path.insert(0, str(TC))
    sys.path.insert(0, "/orcd/pool/005/haolun52/analysisEV/sae_sparse_encoding")
    import sparse_encoding.sparse_encoding_v3 as v3
    import timing_control as tc

    X, _R, surp, bv = tc.feature_condition("v2", None)
    if X.shape[1] < 65536:
        raise RuntimeError(f"SAE width {X.shape[1]} is below 65536")
    _STATE["tc"] = tc
    _STATE["v3"] = v3
    _STATE["surp"] = surp
    _STATE["bv"] = bv
    _STATE["bins"] = {name: X[:, lo:hi] for _lab, name, lo, hi in BINS}


def _fit_electrode(payload: dict) -> list[dict]:
    if "tc" not in _STATE:
        _init_worker()
    tc = _STATE["tc"]
    v3 = _STATE["v3"]
    y = payload["y"]
    ok = payload["ok"]
    T = payload["T"]
    sid = payload["sid"]
    surp = _STATE["surp"]
    bv = _STATE["bv"]
    if y.shape[0] != surp.shape[0] or T.shape[0] != y.shape[0]:
        raise RuntimeError(f"row mismatch {payload['subject']} {payload['channel']}")
    rows = []
    for _lab, name, _lo, _hi in BINS:
        Xb = _STATE["bins"][name]
        fold_r = np.zeros(3, dtype=float)
        nsup = np.zeros(3, dtype=float)
        for f, (tr, te, _held) in enumerate(v3.loso_outer_splits(sid)):
            a = tc._rows(tr, y, ok, bv)
            b = tc._rows(te, y, ok, bv)
            if a.size < 20:
                continue
            yy = tc._residualize(y, T, a, a)[0]
            sup = tc._support(Xb, surp, yy, a, dense=False)
            nsup[f] = int(np.asarray(sup).sum())
            if b.size < 5:
                continue
            yr, _ = tc._residualize(y, T, a, np.r_[a, b])
            r, _pred = v3.predict_with_support(Xb, surp, yr, a, b, sup)
            fold_r[f] = r
        rows.append({
            "subject": payload["subject"],
            "channel": payload["channel"],
            "froi": payload["froi"],
            "bin_name": name,
            "lag_ms": payload["lag_ms"],
            "r_fold0": fold_r[0],
            "r_fold1": fold_r[1],
            "r_fold2": fold_r[2],
            "fisher_z_mean": v3.fisher_z_mean(list(fold_r)),
            "n_support_fold0": nsup[0],
            "n_support_fold1": nsup[1],
            "n_support_fold2": nsup[2],
            "n_support_mean": float(np.mean(nsup)),
        })
    return rows


def refit(n_jobs: int) -> None:
    if BIN_CSV.name in PROTECTED or BIN_CSV.name != "tc_bin_refit_partial.csv":
        raise RuntimeError(f"refusing to write {BIN_CSV}")
    sys.path.insert(0, str(TC))
    sys.path.insert(0, "/orcd/pool/005/haolun52/analysisEV/sae_sparse_encoding")
    import timing_control as tc
    from joblib import Parallel, delayed

    tax = load_taxonomy()
    scores = load_scores()
    pop = population(tax, scores)
    wide = wide_scores(scores, pop)
    _curve, peak, _lags = gain_curve(wide)
    print(f"[paper-match] bin refit at {peak:.0f} ms, {len(pop)} electrodes", flush=True)
    done = set()
    if BIN_CSV.exists() and BIN_CSV.stat().st_size:
        prev = pd.read_csv(BIN_CSV)
        if len(prev) and not np.allclose(prev["lag_ms"].unique(), peak):
            raise RuntimeError(f"{BIN_CSV.name} is at a different lag than {peak:.0f}")
        counts = prev.groupby(["subject", "channel"])["bin_name"].nunique()
        done = set(counts[counts >= 3].index)
        keep = prev.apply(lambda r: (r.subject, r.channel) in done, axis=1)
        prev.loc[keep].to_csv(BIN_CSV, index=False)
    li = int(np.argmin(np.abs(tc.MAIN_LAGS - peak)))
    if not np.isclose(tc.MAIN_LAGS[li], peak):
        raise RuntimeError(f"peak {peak} is not on the v3 lag grid")
    T = tc.load_covariates(np.asarray([peak]))[:, 0, :]
    sid_ref = None
    payloads = []
    for subject, g in pop.groupby("subject"):
        chans, sid = tc.load_main_neural(subject)
        sid_ref = sid if sid_ref is None else sid_ref
        if not np.array_equal(sid_ref, sid):
            raise RuntimeError("section ids differ across subjects")
        for row in g.itertuples(index=False):
            if (row.subject, row.channel) in done:
                continue
            if row.channel not in chans:
                raise RuntimeError(f"{subject} {row.channel} missing from the v3 neural cache")
            Y, OK = chans[row.channel]
            payloads.append({
                "subject": row.subject,
                "channel": row.channel,
                "froi": row.froi,
                "lag_ms": float(peak),
                "y": np.ascontiguousarray(Y[:, li], dtype=np.float64),
                "ok": np.ascontiguousarray(OK[:, li]),
                "T": T,
                "sid": sid_ref,
            })
        del chans
    print(f"[paper-match] {len(payloads)} electrodes to fit ({len(done)} already done)", flush=True)
    if not payloads:
        return
    BIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    gen = Parallel(n_jobs=n_jobs, initializer=_init_worker, return_as="generator_unordered", max_nbytes="2M")(
        delayed(_fit_electrode)(p) for p in payloads
    )
    n = 0
    for rows in gen:
        n += 1
        df = pd.DataFrame(rows)
        write_header = not BIN_CSV.exists() or BIN_CSV.stat().st_size == 0
        df.to_csv(BIN_CSV, mode="a", header=write_header, index=False)
        print(f"  refit {n}/{len(payloads)} {rows[0]['subject']} {rows[0]['channel']}", flush=True)
    print(f"[paper-match] wrote {BIN_CSV}", flush=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plot", "refit"], required=True)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--job-id", default="")
    args = p.parse_args()
    if args.mode == "refit":
        refit(args.n_jobs)
        return
    summary = plot_all(args.job_id or None)
    text = json.dumps(_jsonable(summary), indent=2, allow_nan=False)
    print(text)
    out = Path("/tmp/tc_paper_match_summary.json")
    out.write_text(text)


if __name__ == "__main__":
    main()
