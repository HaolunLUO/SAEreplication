#!/usr/bin/env python3
"""Fig 5 on AAL3 regions. Does not write fig5.png."""

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
import paper_match_partial as pm  # noqa: E402

POOL = TC / "figures" / "paper-match"
HOME = Path(
    "/home/haolun52/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/files/media/sae-v3/timing-control/paper-match"
)
STORE = Path(
    "/run/user/245046/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/"
    "files/media/sae-v3/timing-control/paper-match"
)
TAX = pm.TAX
BIN_CSV = {
    "partial": pm.TABLES / "tc_bin_refit_anatomy_partial.csv",
    "raw": pm.TABLES / "tc_bin_refit_anatomy_rawhg.csv",
}
RAW_PARTS = pm.TABLES / "tc_selected_indices_rawhg_anatomy_parts"
RAW_COMBINED = pm.TABLES / "tc_selected_indices_rawhg_anatomy.csv"
EXPECT = [
    "Left Precentral gyrus",
    "Left Inferior frontal gyrus-triangular part",
    "Left Inferior frontal gyrus-opercular part",
    "Left Middle frontal gyrus",
    "Left Superior frontal gyrus-dorsolateral",
    "Left Insula",
    "Left Hippocampus",
    "Left Superior temporal gyrus",
]
EXPECT_N = {
    "Left Precentral gyrus": 34,
    "Left Inferior frontal gyrus-triangular part": 32,
    "Left Inferior frontal gyrus-opercular part": 17,
    "Left Middle frontal gyrus": 16,
    "Left Superior frontal gyrus-dorsolateral": 14,
    "Left Insula": 6,
    "Left Hippocampus": 4,
    "Left Superior temporal gyrus": 4,
}


def load_scores(mode: str) -> pd.DataFrame:
    names = {
        "partial": ("tc_both_lag_scores.csv", "tc_wide_lag_scores.csv"),
        "raw": ("tc_both_lag_scores_rawhg.csv", "tc_wide_lag_scores_rawhg.csv"),
    }[mode]
    usecols = ["subject", "channel", "condition", "mode", "feature_space", "lag_ms", "fisher_z_mean"]
    frames = []
    for name in names:
        df = pd.read_csv(pm.TABLES / name, usecols=usecols)
        df = df[(df["condition"] == "v2") & (df["mode"] == mode) & (df["feature_space"].isin(pm.SPACES))]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["subject", "channel", "condition", "mode", "feature_space", "lag_ms"], keep="last"
    )


def population(scores: pd.DataFrame) -> pd.DataFrame:
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "region", "is_lang"])
    lang = tax[tax["is_lang"].fillna(False).astype(bool)].copy()
    label = lang["region"].fillna("").astype(str).str.strip()
    pop = lang.loc[label.str.startswith("Left "), ["subject", "channel"]].copy()
    pop["anatomy"] = label.loc[label.str.startswith("Left ")].to_numpy()
    pop = pop.merge(scores[["subject", "channel"]].drop_duplicates(), on=["subject", "channel"])
    counts = pop.groupby("anatomy").size()
    keep = set(counts[counts >= 4].index)
    pop = pop[pop["anatomy"].isin(keep)].reset_index(drop=True)
    order = (
        pop.groupby("anatomy")
        .agg(n=("channel", "size"), ns=("subject", "nunique"))
        .reset_index()
        .sort_values(["n", "anatomy"], ascending=[False, True])
    )
    if list(order["anatomy"]) != EXPECT:
        raise RuntimeError(f"regions changed: {list(order['anatomy'])}")
    got = {r.anatomy: int(r.n) for r in order.itertuples(index=False)}
    if got != EXPECT_N:
        raise RuntimeError(f"counts changed: {got}")
    return pop


def gain_peak(scores: pd.DataFrame, pop: pd.DataFrame) -> float:
    sub = scores.merge(pop[["subject", "channel"]], on=["subject", "channel"])
    wide = sub.pivot_table(
        index=["subject", "channel", "lag_ms"],
        columns="feature_space",
        values="fisher_z_mean",
        aggfunc="last",
    ).reset_index().dropna(subset=list(pm.SPACES))
    wide = wide[(wide["lag_ms"] >= -500) & (wide["lag_ms"] <= 2000)].copy()
    wide["gain"] = wide["sae"] - wide["surprisal"]
    wide["elec"] = wide["subject"] + "\t" + wide["channel"]
    n = wide["elec"].nunique()
    n_at = wide.groupby("lag_ms")["elec"].nunique()
    lags = n_at[n_at == n].index.to_numpy(float)
    use = wide[wide["lag_ms"].isin(lags)]
    curve = use.groupby(["lag_ms", "subject"])["gain"].mean().groupby("lag_ms").mean().sort_index()
    return float(pm.argmax_lag(curve.to_numpy(), curve.index.to_numpy(float)))


def _raw_part(subject: str, channel: str) -> Path:
    safe = str(channel).replace("/", "_")
    return RAW_PARTS / f"{subject}__{safe}__v2__raw.csv"


def load_raw_supports(pop: pd.DataFrame) -> pd.DataFrame | None:
    """SAE indices from the raw anatomy fit. None until every electrode file exists."""
    missing = [f"{r.subject} {r.channel}" for r in pop.itertuples(index=False) if not _raw_part(r.subject, r.channel).is_file()]
    if missing:
        print(f"raw supports missing {len(missing)}/{len(pop)}", flush=True)
        return None
    cols = ["subject", "channel", "feature_space", "fold", "feature_index"]
    frames = []
    for row in pop.itertuples(index=False):
        df = pd.read_csv(_raw_part(row.subject, row.channel))
        if df.empty:
            continue
        df = df[(df["feature_space"] == "sae") & (df["condition"] == "v2") & (df["mode"] == "raw")]
        if len(df):
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True).drop_duplicates(cols)


def fold_bin_counts(supports: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        (s, c, f): g["feature_index"].to_numpy(int)
        for (s, c, f), g in supports.groupby(["subject", "channel", "fold"])
    } if len(supports) else {}
    rows = []
    for row in pop.itertuples(index=False):
        rec = {"subject": row.subject, "channel": row.channel, "anatomy": row.anatomy}
        for label, _name, lo, hi in pm.BINS:
            per = []
            for fold in pm.FOLDS:
                idx = grouped.get((row.subject, row.channel, fold), np.array([], dtype=int))
                per.append(int(np.sum((idx >= lo) & (idx < hi))))
            rec[label] = float(np.mean(per))
        rows.append(rec)
    return pd.DataFrame(rows)


def region_stats(frame: pd.DataFrame, pop: pd.DataFrame, keys: list[str]) -> dict:
    """Subject-mean of electrode-means. One subject: that electrode mean, no SEM."""
    out = {}
    for region in EXPECT:
        el = pop[pop["anatomy"] == region]
        sub = frame[frame["anatomy"] == region] if "anatomy" in frame.columns else frame
        block = {
            "n_elec": int(len(el)),
            "n_subj": int(el["subject"].nunique()),
            "mean": {},
            "sem": {},
            "subject": {},
        }
        for key in keys:
            if sub.empty or key not in sub.columns:
                block["mean"][key] = float("nan")
                block["sem"][key] = float("nan")
                block["subject"][key] = {}
                continue
            vals = sub.groupby("subject")[key].mean()
            block["mean"][key] = float(vals.mean()) if len(vals) else float("nan")
            block["sem"][key] = float(vals.sem(ddof=1)) if len(vals) >= 2 else float("nan")
            block["subject"][key] = {str(k): float(v) for k, v in vals.items()}
        out[region] = block
    return out


def panel_c(path: Path, pop: pd.DataFrame, scores: pd.DataFrame, peak: float) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if "lag_ms" not in df.columns or not np.allclose(df["lag_ms"].unique(), peak):
        return None
    df = df.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"], how="inner")
    have = df.groupby(["subject", "channel"])["bin_name"].nunique()
    if len(have) < len(pop) or (have < 3).any():
        return None
    name_to_label = {b[1]: b[0] for b in pm.BINS}
    at = scores.merge(pop[["subject", "channel", "anatomy"]], on=["subject", "channel"])
    at = at[(at["feature_space"] == "surprisal") & np.isclose(at["lag_ms"], peak)]
    rows = []
    for (subj, ch, region), g in at.groupby(["subject", "channel", "anatomy"]):
        rows.append({
            "subject": subj, "channel": ch, "anatomy": region,
            "Surprisal": float(g["fisher_z_mean"].iloc[-1]),
        })
    base = pd.DataFrame(rows)
    for label, name, _lo, _hi in pm.BINS:
        piece = df[df["bin_name"] == name][["subject", "channel", "fisher_z_mean"]].rename(
            columns={"fisher_z_mean": label}
        )
        base = base.merge(piece, on=["subject", "channel"], how="left")
    keys = ["Surprisal"] + [b[0] for b in pm.BINS]
    return region_stats(base, pop, keys)


def tick_label(name: str, n_elec: int, n_subj: int) -> str:
    pretty = name
    if "-" in pretty:
        head, tail = pretty.split("-", 1)
        pretty = f"{head}\n{tail}"
    return f"{pretty}\n({n_elec} elec, {n_subj} subj)"


def draw_bars(ax, stats, keys, colors, scol) -> None:
    x = np.arange(len(EXPECT))
    n = len(keys)
    w = 0.82 / n
    mid = (n - 1) / 2.0
    for i, key in enumerate(keys):
        means, sems = [], []
        for region in EXPECT:
            means.append(stats[region]["mean"][key])
            sems.append(stats[region]["sem"][key])
        means = np.asarray(means, dtype=float)
        sems = np.asarray(sems, dtype=float)
        yerr = np.where(np.isfinite(sems), sems, 0.0)
        xpos = x + (i - mid) * w
        ax.bar(xpos, means, w * 0.92, yerr=yerr, capsize=2, color=colors[key], label=key, error_kw={"elinewidth": 0.7}, zorder=1)
        for j, region in enumerate(EXPECT):
            dots = stats[region]["subject"].get(key, {})
            subs = sorted(dots)
            for k, subj in enumerate(subs):
                jitter = (k - (len(subs) - 1) / 2.0) * min(0.035, w * 0.28)
                ax.scatter(xpos[j] + jitter, dots[subj], s=16, color=scol[subj], edgecolor="white", linewidths=0.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [tick_label(r, stats[r]["n_elec"], stats[r]["n_subj"]) for r in EXPECT],
        fontsize=6.5,
    )
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=min(4, len(keys)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw(path: Path, *, signal: str, peak: float, feat: pd.DataFrame | None, bin_stats: dict | None, c_stats: dict | None, pop: pd.DataFrame, note: str) -> None:
    subjects = sorted(pop["subject"].unique())
    cmap = plt.get_cmap("tab10")
    scol = {s: cmap(i % 10) for i, s in enumerate(subjects)}
    n_rows = 3 if c_stats is not None else 2
    if feat is None:
        n_rows = 1 if c_stats is None else 1
    fig_h = 4.2 if feat is None and c_stats is None else (9.6 if c_stats is not None and feat is not None else 7.2)
    if feat is None and c_stats is not None:
        fig_h = 5.4
        n_rows = 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(16.2, fig_h), squeeze=False)
    axes = list(axes.ravel())
    row_i = 0
    if feat is not None and bin_stats is not None:
        ax = axes[row_i]
        row_i += 1
        if len(feat):
            order = np.argsort(feat["feature_index"].to_numpy())
            ax.vlines(
                feat["feature_index"].to_numpy()[order], 0,
                feat["prevalence"].to_numpy()[order], color="slategray", lw=0.6,
            )
        for cut in (2048, 16384):
            ax.axvline(cut, color="red", lw=1.0)
        ax.set_xlim(0, 65536)
        ax.set_ylabel("Electrodes")
        ax.set_xlabel("SAE index")
        ax.set_title("Selected Matryoshka indices")
        ax.text(-0.045, 1.04, "A", transform=ax.transAxes, fontsize=14, fontweight="bold", clip_on=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = axes[row_i]
        row_i += 1
        draw_bars(ax, bin_stats, [b[0] for b in pm.BINS], pm.BIN_COLOR, scol)
        ax.set_ylabel("SAE features (mean across folds)")
        ax.set_title("Features per Matryoshka bin")
        ax.text(-0.045, 1.04, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", clip_on=False)
    if c_stats is not None:
        ax = axes[row_i]
        keys = ["Surprisal"] + [b[0] for b in pm.BINS]
        colors = {"Surprisal": "dimgray", **pm.BIN_COLOR}
        draw_bars(ax, c_stats, keys, colors, scol)
        ax.set_ylabel("Fisher-z")
        ax.set_title(f"Bin-restricted models at {peak:.0f} ms (surprisal forced in)")
        ax.text(-0.045, 1.04, "C", transform=ax.transAxes, fontsize=14, fontweight="bold", clip_on=False)
    elif feat is None:
        ax = axes[0]
        ax.axis("off")
        ax.text(0.02, 0.6, note, fontsize=11, va="center", ha="left", wrap=True)
    fig.suptitle(f"{signal}  SAE-gain peak {peak:.0f} ms. {note}", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.0, 0.995, 0.955))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    fig5 = POOL / "fig5.png"
    before = hashlib.md5(fig5.read_bytes()).hexdigest()
    partial_scores = load_scores("partial")
    raw_scores = load_scores("raw")
    pop = population(partial_scores)
    pop_raw = population(raw_scores)
    if not pop[["subject", "channel"]].sort_values(["subject", "channel"]).reset_index(drop=True).equals(
        pop_raw[["subject", "channel"]].sort_values(["subject", "channel"]).reset_index(drop=True)
    ):
        raise RuntimeError("partial and raw anatomy electrodes differ")
    peak_p = gain_peak(partial_scores, pop)
    peak_r = gain_peak(raw_scores, pop)
    print(f"partial peak {peak_p:.0f} raw peak {peak_r:.0f}", flush=True)

    supports = pm.load_supports(pop)
    if supports.empty:
        raise RuntimeError("partial supports are empty")
    feat = pm.feature_table(supports, pop)
    counts = fold_bin_counts(supports, pop)
    bstats = region_stats(counts, pop, [b[0] for b in pm.BINS])
    c_partial = panel_c(BIN_CSV["partial"], pop, partial_scores, peak_p)
    # Panel C stays with job 24504294. Draw it only when that job's CSV is complete.
    c_raw = panel_c(BIN_CSV["raw"], pop, raw_scores, peak_r)
    raw_supports = load_raw_supports(pop)
    print(
        f"panel C partial {'yes' if c_partial else 'no'} raw {'yes' if c_raw else 'no'}; "
        f"raw supports {'yes' if raw_supports is not None else 'no'}",
        flush=True,
    )

    note_p = "Left-hemisphere language electrodes by AAL3 region. Timing-regressed high-gamma (v2 partial)."
    if c_partial is None:
        note_p += " Panel C is not in this file yet."
    draw(
        POOL / "fig5_anatomy_partial.png",
        signal="Timing-regressed high-gamma.",
        peak=peak_p,
        feat=feat,
        bin_stats=bstats,
        c_stats=c_partial,
        pop=pop,
        note=note_p,
    )
    if raw_supports is None:
        note_r = (
            "Raw high-gamma selected indices are not saved yet, so panels A and B are not drawn."
        )
        feat_r, bstats_r = None, None
    else:
        feat_r = pm.feature_table(raw_supports, pop)
        bstats_r = region_stats(fold_bin_counts(raw_supports, pop), pop, [b[0] for b in pm.BINS])
        note_r = "Left-hemisphere language electrodes by AAL3 region. Raw high-gamma (v2, no timing residual)."
        print(f"raw SAE indices {len(feat_r)}", flush=True)
    if c_raw is None:
        note_r += " Panel C is not in this file yet."
    draw(
        POOL / "fig5_anatomy_rawhg.png",
        signal="Raw high-gamma.",
        peak=peak_r,
        feat=feat_r,
        bin_stats=bstats_r,
        c_stats=c_raw,
        pop=pop,
        note=note_r,
    )
    for name in ("fig5_anatomy_partial.png", "fig5_anatomy_rawhg.png"):
        for dest in (HOME, STORE):
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(POOL / name, dest / name)
    after = hashlib.md5(fig5.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("fig5.png changed")
    print("wrote figures", flush=True)


if __name__ == "__main__":
    main()
