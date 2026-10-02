#!/usr/bin/env python3
"""Tables and figures for the v3 timing control."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing_control as tc  # noqa: E402
import sparse_encoding.sparse_encoding_v3 as v3  # noqa: E402

TABLES = tc.TABLES
FIGS = HERE / "figures"
MEDIA = Path("/home/ryan/.cursor/stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/media/sae-v3/timing-control")
V3_TABLE = HERE.parent / "tables" / "v3_loso_lag_scores.csv"
POST = (100.0, 1000.0)
PRE = (-500.0, -100.0)
KEY = ["subject", "channel"]


def load_all():
    el = tc.electrode_table()
    parts = [pd.read_csv(p) for p in (TABLES / "tc_both_lag_scores.csv", TABLES / "tc_wide_lag_scores.csv",
                                      TABLES / "tc_forced_lag_scores.csv") if p.exists()]
    d = pd.concat(parts, ignore_index=True).drop_duplicates(
        KEY + ["condition", "mode", "feature_space", "lag_ms"], keep="last")
    raw = pd.read_csv(V3_TABLE).drop_duplicates(KEY + ["feature_space", "lag_ms"], keep="last")
    raw = raw[raw.readout == "matched_lasso"][KEY + ["feature_space", "lag_ms", "fisher_z_mean"]]
    raw["condition"], raw["mode"] = "v2", "raw_v3"
    g = TABLES / "tc_glove_both_lag_scores.csv"
    glove = pd.read_csv(g) if g.exists() else None
    return el, d, raw, glove


def curve(df, space, sel=None):
    x = df[df.feature_space == space]
    if sel is not None:
        x = x.merge(sel, on=KEY)
    g = x.groupby("lag_ms")["fisher_z_mean"].agg(["mean", "sem", "count"]).reset_index()
    return g.sort_values("lag_ms")


def gain_frame(df, sel, left="sae", right="surprisal"):
    a = df[df.feature_space == left][KEY + ["lag_ms", "fisher_z_mean"]].rename(columns={"fisher_z_mean": "l"})
    b = df[df.feature_space == right][KEY + ["lag_ms", "fisher_z_mean"]].rename(columns={"fisher_z_mean": "r"})
    m = a.merge(b, on=KEY + ["lag_ms"]).merge(sel, on=KEY)
    m["gain"] = m.l - m.r
    return m


def gain_curve(m):
    ch = m.groupby("lag_ms")["gain"].agg(["mean", "sem", "count"]).reset_index()
    subj = m.groupby(["subject", "lag_ms"])["gain"].mean().reset_index()
    sg = subj.groupby("lag_ms")["gain"].agg(subj_mean="mean", subj_std="std", n_subj="count").reset_index()
    sg["subj_sem"] = sg.subj_std / np.sqrt(sg.n_subj)
    return ch.merge(sg, on="lag_ms").sort_values("lag_ms")


def ratios(c):
    c = c.set_index("lag_ms")["mean"]
    post = c[(c.index >= POST[0]) & (c.index <= POST[1])]
    pre = c[(c.index >= PRE[0]) & (c.index <= PRE[1])]
    pk = float(post.idxmax())
    return {"z_-250": float(c.get(-250.0, np.nan)), "z_-500": float(c.get(-500.0, np.nan)),
            "z_0": float(c.get(0.0, np.nan)), "post_peak_lag_ms": pk, "z_post_peak": float(post.max()),
            "ratio_-250_over_post_peak": float(c.get(-250.0, np.nan) / post.max()),
            "ratio_max_pre_over_max_post": float(pre.max() / post.max())}


def style(ax, title, ylab="Fisher-z (LOSO)"):
    ax.axvline(0, color="k", lw=0.8, ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title(title, fontsize=9); ax.set_xlabel("lag re word onset (ms)"); ax.set_ylabel(ylab)


def band(ax, c, label, color, y="mean", s="sem", **kw):
    c = c.sort_values("lag_ms")
    ax.plot(c.lag_ms, c[y], color=color, label=label, **kw)
    ax.fill_between(c.lag_ms, c[y] - c[s], c[y] + c[s], color=color, alpha=0.15)


def main():
    FIGS.mkdir(exist_ok=True)
    el, d, raw, glove = load_all()
    sets = {"both_45": el[el.both][KEY], "sig_glove_217": el[el.sig_glove][KEY], "is_lang_187": el[el.is_lang][KEY]}
    have = d[(d.condition == "v2") & (d["mode"] == "partial")][KEY].drop_duplicates()
    summary, curves, figs = [], [], []

    main_d = d[d.condition == "v2"]
    timing = d[(d.condition == "v2") & (d["mode"] == "timing_only")]
    for sname, sel in sets.items():
        sel_have = sel.merge(have, on=KEY)
        if len(sel_have) < len(sel):
            print(f"[report] {sname}: {len(sel_have)}/{len(sel)} electrodes fitted", flush=True)
        if sel_have.empty:
            continue
        on_grid = lambda x: x[x.lag_ms.isin(tc.MAIN_LAGS)]
        for mode, src in (("raw_v3", raw), ("partial", main_d[main_d["mode"] == "partial"]),
                          ("forced", main_d[main_d["mode"] == "forced"])):
            src = on_grid(src)
            for sp in ("sae", "residual", "surprisal"):
                c = curve(src, sp, sel_have)
                c["set"], c["mode"], c["series"] = sname, mode, sp
                curves.append(c)
                summary.append({"set": sname, "mode": mode, "series": sp, "n_electrodes": int(c["count"].max()),
                                **ratios(c)})
            gc = gain_curve(gain_frame(src, sel_have))
            gc["set"], gc["mode"], gc["series"] = sname, mode, "sae_gain"
            curves.append(gc)
            r = ratios(gc)
            grid = gc[(gc.lag_ms >= 0)]
            pk = grid.loc[grid["mean"].idxmax()]
            spk = grid.loc[grid["subj_mean"].idxmax()]
            m = gain_frame(src, sel_have)
            subj_at = m[np.isclose(m.lag_ms, spk.lag_ms)].groupby("subject")["gain"].mean().to_numpy()
            r.update({"gain_peak_lag_ms": float(pk.lag_ms), "gain_peak_z": float(pk["mean"]),
                      "gain_peak_sem_channel": float(pk["sem"]),
                      "gain_peak_ci_excludes_zero": bool(pk["mean"] - 1.96 * pk["sem"] > 0),
                      "gain_subj_peak_lag_ms": float(spk.lag_ms), "gain_subj_peak_z": float(spk.subj_mean),
                      "gain_subj_peak_sem": float(spk.subj_sem), "n_subjects": int(spk.n_subj),
                      "gain_subj_signflip_p": float(v3.signflip_p(subj_at)),
                      "gain_z_at_550": float(gc.set_index("lag_ms").loc[550.0, "mean"]),
                      "gain_sem_at_550": float(gc.set_index("lag_ms").loc[550.0, "sem"])})
            summary.append({"set": sname, "mode": mode, "series": "sae_gain",
                            "n_electrodes": int(gc["count"].max()), **r})
        c = curve(on_grid(timing), "timing_only", sel_have)
        c["set"], c["mode"], c["series"] = sname, "timing_only", "timing_only"
        curves.append(c)
        summary.append({"set": sname, "mode": "timing_only", "series": "timing_only",
                        "n_electrodes": int(c["count"].max()), **ratios(c)})
        # forced increment over timing-only (comparable to the partial curve)
        for sp in ("sae", "residual", "surprisal"):
            f = main_d[main_d["mode"] == "forced"][KEY + ["feature_space", "lag_ms", "fisher_z_mean"]]
            f = pd.concat([f, timing[KEY + ["feature_space", "lag_ms", "fisher_z_mean"]]])
            gi = gain_curve(gain_frame(on_grid(f), sel_have, left=sp, right="timing_only"))
            gi["set"], gi["mode"], gi["series"] = sname, "forced", f"{sp}_minus_timing_only"
            curves.append(gi)

    # sanity: shuffled and previous-word projection, 45 both electrodes
    both = sets["both_45"]
    for cond in ("v2_shuffle", "v2_prevproj"):
        for mode in ("partial", "forced"):
            x = d[(d.condition == cond) & (d["mode"] == mode)]
            if x.empty:
                continue
            for sp in ("sae", "residual", "surprisal"):
                c = curve(x, sp, both); c["set"], c["mode"], c["series"] = "both_45", f"{cond}:{mode}", sp
                curves.append(c)
                summary.append({"set": "both_45", "mode": f"{cond}:{mode}", "series": sp,
                                "n_electrodes": int(c["count"].max()), **ratios(c),
                                "max_abs_mean_z": float(c["mean"].abs().max()),
                                "frac_lags_abs_mean_gt_2sem": float((c["mean"].abs() > 2 * c["sem"]).mean())})
            gc = gain_curve(gain_frame(x, both))
            gc["set"], gc["mode"], gc["series"] = "both_45", f"{cond}:{mode}", "sae_gain"
            curves.append(gc)
            summary.append({"set": "both_45", "mode": f"{cond}:{mode}", "series": "sae_gain",
                            "n_electrodes": int(gc["count"].max()), **ratios(gc),
                            "max_abs_mean_z": float(gc["mean"].abs().max()),
                            "frac_lags_abs_mean_gt_2sem": float((gc["mean"].abs() > 2 * gc["sem"]).mean())})
    if glove is not None:
        for mode in ("raw", "partial", "forced"):
            c = curve(glove[glove["mode"] == mode], "glove", both)
            c["set"], c["mode"], c["series"] = "both_45", f"glove:{mode}", "glove300"
            curves.append(c)
            summary.append({"set": "both_45", "mode": f"glove:{mode}", "series": "glove300",
                            "n_electrodes": int(c["count"].max()), **ratios(c)})

    S = pd.DataFrame(summary)
    C = pd.concat(curves, ignore_index=True)
    S.to_csv(TABLES / "tc_summary.csv", index=False)
    C.to_csv(TABLES / "tc_curves.csv", index=False)

    def get(set_, mode, series):
        return C[(C.set == set_) & (C["mode"] == mode) & (C.series == series)]

    # Figure 1: 45 both electrodes, extended grid, raw vs controlled
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, sp in zip(axs[0], ("sae", "residual", "surprisal")):
        band(ax, get("both_45", "raw_v3", sp), "raw v3", "k")
        x = d[(d.condition == "v2") & (d["mode"] == "partial")]
        band(ax, curve(x, sp, both), "timing partialled", "tab:red")
        f = d[(d.condition == "v2") & (d["mode"].isin(["forced", "timing_only"]))]
        fi = gain_curve(gain_frame(f.assign(feature_space=np.where(f["mode"] == "timing_only", "timing_only", f.feature_space)),
                                   both, left=sp, right="timing_only"))
        band(ax, fi, "forced-in minus timing-only", "tab:blue")
        style(ax, f"{sp} (45 lang & sig_glove)")
        ax.legend(fontsize=7)
    if glove is not None:
        for mode, col in (("raw", "k"), ("partial", "tab:red")):
            band(axs[1, 0], curve(glove[glove["mode"] == mode], "glove", both), f"GloVe-300 {mode}", col)
        axs[1, 0].legend(fontsize=7)
    style(axs[1, 0], "GloVe-300 dense Ridge (Goldstein Fig 4a analogue)")
    band(axs[1, 1], curve(d[(d["mode"] == "timing_only")], "timing_only", both), "timing + envelope only", "tab:purple")
    style(axs[1, 1], "timing-only model"); axs[1, 1].legend(fontsize=7)
    ax = axs[1, 2]
    band(ax, get("both_45", "raw_v3", "sae_gain"), "raw v3", "k")
    band(ax, gain_curve(gain_frame(d[(d.condition == "v2") & (d["mode"] == "partial")], both)), "partial", "tab:red")
    band(ax, gain_curve(gain_frame(d[(d.condition == "v2") & (d["mode"] == "forced")], both)), "forced-in", "tab:blue")
    style(ax, "SAE gain (SAE - surprisal)", "gain (Fisher-z)"); ax.legend(fontsize=7)
    fig.tight_layout(); p = FIGS / "tc_both45_raw_vs_controlled.png"; fig.savefig(p, dpi=140); plt.close(fig); figs.append(p)

    # Figure 2: wide sets
    for sname in ("sig_glove_217", "is_lang_187"):
        if get(sname, "partial", "sae").empty:
            continue
        fig, axs = plt.subplots(1, 5, figsize=(22, 4.2))
        for ax, sp in zip(axs[:3], ("sae", "residual", "surprisal")):
            band(ax, get(sname, "raw_v3", sp), "raw v3", "k")
            band(ax, get(sname, "partial", sp), "timing partialled", "tab:red")
            band(ax, get(sname, "forced", f"{sp}_minus_timing_only"), "forced-in minus timing-only", "tab:blue")
            style(ax, f"{sp} ({sname})"); ax.legend(fontsize=7)
        band(axs[3], get(sname, "timing_only", "timing_only"), "timing + envelope only", "tab:purple")
        style(axs[3], f"timing-only ({sname})"); axs[3].legend(fontsize=7)
        for mode, col in (("raw_v3", "k"), ("partial", "tab:red"), ("forced", "tab:blue")):
            band(axs[4], get(sname, mode, "sae_gain"), mode, col)
        style(axs[4], f"SAE gain ({sname}), channel SEM", "gain (Fisher-z)"); axs[4].legend(fontsize=7)
        fig.tight_layout(); p = FIGS / f"tc_{sname}_raw_vs_controlled.png"; fig.savefig(p, dpi=140); plt.close(fig); figs.append(p)

    # Figure 3: sanity
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, sp in zip(axs, ("sae", "residual", "surprisal")):
        for (mode, col) in (("v2_shuffle:partial", "0.5"), ("v2_shuffle:forced", "0.75"), ("v2_prevproj:partial", "tab:orange")):
            c = get("both_45", mode, sp)
            if mode == "v2_shuffle:forced":
                continue
            if not c.empty:
                band(ax, c, mode, col)
        band(ax, gain_curve(gain_frame(d[(d.condition == "v2_shuffle") & (d["mode"] == "forced")], both)),
             "shuffle forced: SAE gain", "tab:green") if sp == "sae" else None
        band(ax, curve(d[(d.condition == "v2") & (d["mode"] == "partial")], sp, both), "v2 partial", "tab:red")
        style(ax, f"sanity: {sp} (45)"); ax.legend(fontsize=7)
    fig.tight_layout(); p = FIGS / "tc_sanity_shuffle_prevproj.png"; fig.savefig(p, dpi=140); plt.close(fig); figs.append(p)

    MEDIA.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in figs:
        q = MEDIA / p.name
        shutil.copy2(p, q)
        if q.stat().st_size < 10_000:
            raise RuntimeError(f"figure too small: {q}")
        copied.append(str(q))
    js = {"figures": [str(p) for p in figs], "media": copied}
    (TABLES / "tc_figures.json").write_text(json.dumps(js, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    cols = ["set", "mode", "series", "n_electrodes", "z_-500", "z_-250", "z_0", "post_peak_lag_ms", "z_post_peak",
            "ratio_-250_over_post_peak", "ratio_max_pre_over_max_post", "gain_peak_lag_ms", "gain_peak_z",
            "gain_peak_sem_channel", "gain_peak_ci_excludes_zero", "gain_subj_peak_lag_ms", "gain_subj_peak_z",
            "gain_subj_peak_sem", "gain_subj_signflip_p", "max_abs_mean_z", "frac_lags_abs_mean_gt_2sem"]
    print(S[[c for c in cols if c in S.columns]].round(4).to_string())
    print(json.dumps(js, indent=1))


if __name__ == "__main__":
    main()
