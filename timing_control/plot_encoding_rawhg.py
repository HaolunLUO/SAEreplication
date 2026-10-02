#!/usr/bin/env python3
"""Overlay raw high-gamma SAE lag curves on the three encoding figures.

Reads finished timing-control tables. Writes pngs, encoding_summary.json, and
the Encoding tables in the syncing-store copy of the timing-vs-TRF note.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/tables")
TAX = Path("/orcd/pool/005/haolun52/analysisEV/group_encoding_results/functional_taxonomy/tables/functional_taxonomy_electrode_table.csv")
STORE = Path("/home/haolun52/.local/state/cursor/agent-stores/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846/files")
MEDIA = STORE / "media/sae-v3/timing-vs-trf"
DOC = STORE / "docs/sae-v3-timing-vs-trf.md"

COLORS = {
    "existing partial": "#222222",
    "trf": "#c45c26",
    "trf+partial": "#2a6f97",
    "raw HG": "#1b7a4a",
}


def keys_for():
    tax = pd.read_csv(TAX, usecols=["subject", "channel", "is_lang", "sig_glove"])
    tax["key"] = tax.subject.astype(str) + "|" + tax.channel.astype(str)
    both = set(tax.loc[tax.is_lang & tax.sig_glove, "key"])
    lang = set(tax.loc[tax.is_lang, "key"])
    glove = set(tax.loc[tax.sig_glove, "key"])
    lang_only = lang - both
    glove_only = glove - both
    return {"both": both, "lang": lang, "glove": glove, "lang_only": lang_only, "glove_only": glove_only}


def load(name, mode):
    df = pd.read_csv(
        TABLES / name,
        usecols=["subject", "channel", "condition", "mode", "feature_space", "lag_ms", "fisher_z_mean"],
    )
    df = df[(df.condition == "v2") & (df["mode"] == mode) & (df.feature_space == "sae")].copy()
    df["key"] = df.subject.astype(str) + "|" + df.channel.astype(str)
    return df


def series(parts, expect_n, expect_min):
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["subject", "channel", "lag_ms"], keep="last")
    g = df.groupby("lag_ms", as_index=True).agg(z=("fisher_z_mean", "mean"), n=("fisher_z_mean", "size"))
    g = g.sort_index()
    full = g[g.n == g.n.max()]
    if int(full.n.iloc[0]) != expect_n:
        raise RuntimeError(f"n={int(full.n.iloc[0])} expected {expect_n}")
    if not np.isclose(full.index.min(), expect_min):
        raise RuntimeError(f"min lag {full.index.min()} expected {expect_min}")
    if not np.any(np.isclose(full.index.to_numpy(), -250.0)):
        raise RuntimeError("missing -250 ms")
    return full


def stats(g):
    lag = float(g.z.idxmax())
    z = float(g.z.loc[lag])
    at = float(g.z.loc[np.isclose(g.index, -250.0)].iloc[0])
    return {"n": int(g.n.iloc[0]), "n_lags": int(len(g)), "peak_lag_ms": lag, "peak_z": z, "at_m250": at}


def plot_one(path, title, curves, legend_loc):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    for label, g in curves:
        ax.plot(g.index.to_numpy(), g.z.to_numpy(), color=COLORS[label], lw=1.6, label=label)
    ax.axvline(-250, color="0.55", ls="--", lw=0.9, zorder=0)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("mean Fisher z")
    ax.set_title(title)
    ax.legend(loc=legend_loc, frameon=True, fontsize=9)
    ax.set_xlim(min(g.index.min() for _, g in curves), 2000)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt_row(label, st):
    if st is None:
        return f"| {label} | — | not on disk | — |"
    return f"| {label} | {st['peak_lag_ms']:.0f} ms | {st['peak_z']:.4f} | {st['at_m250']:.4f} |"


def main():
    k = keys_for()
    assert len(k["both"]) == 45 and len(k["lang"]) == 187 and len(k["glove"]) == 217

    both_partial = load("tc_both_lag_scores.csv", "partial")
    wide_partial = load("tc_wide_lag_scores.csv", "partial")
    both_trf = load("tc_both_lag_scores_trf.csv", "raw")
    wide_trf = load("tc_wide_lag_scores_trf.csv", "raw")
    both_trfp = load("tc_both_lag_scores_trf_partial.csv", "partial")
    wide_trfp = load("tc_wide_lag_scores_trf_partial.csv", "partial")
    glove_trf = load("tc_sig_glove_lag_scores_trf.csv", "raw")
    glove_trfp = load("tc_sig_glove_lag_scores_trf_partial.csv", "partial")
    both_raw = load("tc_both_lag_scores_rawhg.csv", "raw")
    wide_raw = load("tc_wide_lag_scores_rawhg.csv", "raw")
    glove_raw = load("tc_sig_glove_lag_scores_rawhg.csv", "raw")

    curves = {
        "45": [
            ("existing partial", series([both_partial[both_partial.key.isin(k["both"])]], 45, -2000)),
            ("trf", series([both_trf[both_trf.key.isin(k["both"])]], 45, -500)),
            ("trf+partial", series([both_trfp[both_trfp.key.isin(k["both"])]], 45, -500)),
            ("raw HG", series([both_raw[both_raw.key.isin(k["both"])]], 45, -2000)),
        ],
        "187": [
            ("existing partial", series([
                both_partial[both_partial.key.isin(k["both"])],
                wide_partial[wide_partial.key.isin(k["lang_only"])],
            ], 187, -500)),
            ("trf", series([
                both_trf[both_trf.key.isin(k["both"])],
                wide_trf[wide_trf.key.isin(k["lang_only"])],
            ], 187, -500)),
            ("trf+partial", series([
                both_trfp[both_trfp.key.isin(k["both"])],
                wide_trfp[wide_trfp.key.isin(k["lang_only"])],
            ], 187, -500)),
            ("raw HG", series([
                both_raw[both_raw.key.isin(k["both"])],
                wide_raw[wide_raw.key.isin(k["lang_only"])],
            ], 187, -500)),
        ],
        "217": [
            ("existing partial", series([
                both_partial[both_partial.key.isin(k["both"])],
                wide_partial[wide_partial.key.isin(k["glove_only"])],
            ], 217, -500)),
            ("trf", series([glove_trf[glove_trf.key.isin(k["glove"])]], 217, -500)),
            ("trf+partial", series([glove_trfp[glove_trfp.key.isin(k["glove"])]], 217, -500)),
            ("raw HG", series([glove_raw[glove_raw.key.isin(k["glove"])]], 217, -500)),
        ],
    }
    summary = {name: {label: stats(g) for label, g in items} for name, items in curves.items()}
    # raw HG on the 45 must actually use the extended grid
    if summary["45"]["raw HG"]["n_lags"] != 81:
        raise RuntimeError(f"45 raw HG n_lags={summary['45']['raw HG']['n_lags']}")
    if summary["187"]["raw HG"]["n_lags"] != 51 or summary["217"]["raw HG"]["n_lags"] != 51:
        raise RuntimeError("187/217 raw HG are not on the main grid")

    MEDIA.mkdir(parents=True, exist_ok=True)
    plot_one(MEDIA / "encoding_45_both.png", "45 both, v2 SAE", curves["45"], "lower right")
    plot_one(MEDIA / "encoding_187_language.png", "187 language, v2 SAE", curves["187"], "upper right")
    plot_one(MEDIA / "encoding_217_glove.png", "217 GloVe, v2 SAE", curves["217"], "lower right")

    js_path = MEDIA / "encoding_summary.json"
    data = json.loads(js_path.read_text())
    label_key = {
        "existing partial": "existing_partial",
        "trf": "trf",
        "trf+partial": "trf_partial",
        "raw HG": "raw_hg",
    }
    set_key = {"45": "45_both", "187": "187_language", "217": "217_glove"}
    for name, items in summary.items():
        block = data["sets"][set_key[name]]
        for label, st in items.items():
            block[label_key[label]] = {
                "n": st["n"],
                "n_lags": st["n_lags"],
                "plotted": True,
                "peak": {"lag_ms": st["peak_lag_ms"], "fisher_z": st["peak_z"]},
                "at_m250": st["at_m250"],
            }
    data["raw_hg"] = {
        "plotted": True,
        "definition": "v2 SAE on original high-gamma: no timing partial and no TRF residual",
        "metric": "channel mean of fisher_z_mean",
        "job": {"id": 24431261, "name": "v3_tc_rawhg"},
        "tables": [
            "tables/tc_both_lag_scores_rawhg.csv",
            "tables/tc_wide_lag_scores_rawhg.csv",
            "tables/tc_sig_glove_lag_scores_rawhg.csv",
        ],
        "grids": {
            "45_both": "-2000 to 2000 step 50",
            "187_language": "-500 to 2000 step 50",
            "217_sig_glove": "-500 to 2000 step 50",
        },
        "sets": {set_key[n]: summary[n]["raw HG"] for n in summary},
    }
    data["existing_raw_mode"] = True
    data["note_existing_raw"] = (
        "Raw HG is v2 SAE on the original caches (no cache suffix), tables "
        "tc_both_lag_scores_rawhg.csv, tc_wide_lag_scores_rawhg.csv, and "
        "tc_sig_glove_lag_scores_rawhg.csv. The finished partial and TRF tables were not overwritten."
    )
    data["sig_glove_217_job"] = {
        "id": 24406326,
        "state": "COMPLETED",
        "elapsed": "01:47:06",
        "not_cancelled": True,
        "figure": "trf and trf+partial from tc_sig_glove_lag_scores_trf.csv and tc_sig_glove_lag_scores_trf_partial.csv are on encoding_217_glove.png",
    }
    js_path.write_text(json.dumps(data, indent=2) + "\n")

    def table(name, labels):
        lines = [
            "| Series | Peak lag | Peak mean Fisher z | At −250 ms |",
            "|---|---:|---:|---:|",
        ]
        for label in labels:
            lines.append(fmt_row(label, summary[name][label]))
        return "\n".join(lines)

    labels = ["existing partial", "trf", "trf+partial", "raw HG"]
    doc = DOC.read_text()
    start = doc.find("### 45 both")
    end = doc.find("### GloVe overlap")
    if start < 0 or end < 0:
        raise RuntimeError("Encoding headings not found")
    r45, r187, r217 = (summary[n]["raw HG"] for n in ("45", "187", "217"))
    t45, t187, t217 = (summary[n]["trf"] for n in ("45", "187", "217"))
    p45, p187, p217 = (summary[n]["trf+partial"] for n in ("45", "187", "217"))
    block = f"""### 45 both

{table("45", labels)}

GloVe lag scores exist for these 45 electrodes on the original high-gamma (`tc_glove_both_lag_scores.csv`), not on the TRF residual. GloVe raw peaks at 300 ms (0.0489; −250 ms = 0.0309). GloVe partial peaks at 250 ms (0.0360; −250 ms = 0.0008). The matched SAE−GloVe gain is existing partial SAE minus that GloVe partial: **+0.0124 at 300 ms** and **+0.0221 at −250 ms**. No GloVe model was refit on `trfresid`, so trf and trf+partial have no matched gain. A v2 shuffle on the TRF residual peaks at −150 ms with mean Fisher z 0.0057. Raw HG on these 45 electrodes peaks at {r45['peak_lag_ms']:.0f} ms ({r45['peak_z']:.4f}; −250 ms = {r45['at_m250']:.4f}).

### 187 language

{table("187", labels)}

The GloVe lag table covers 45 of these 187 electrodes, so there is no set-level SAE−GloVe gain. Raw HG peaks at {r187['peak_lag_ms']:.0f} ms ({r187['peak_z']:.4f}; −250 ms = {r187['at_m250']:.4f}). The plotted curve starts at −500 ms, where all 187 electrodes are scored.

### 217 GloVe

Job `24406326` finished (01:47:06). trf and trf+partial are the 217-electrode residual fits (`tc_sig_glove_lag_scores_trf.csv`, `tc_sig_glove_lag_scores_trf_partial.csv`). Raw HG is the same 217 electrodes on the original high-gamma (`tc_sig_glove_lag_scores_rawhg.csv`), on the −500…2000 ms grid.

{table("217", labels)}

"""
    doc = doc[:start] + block + doc[end:]
    old = "No fit was started. The figures keep the curves they already have and do not add a raw HG series."
    new = (
        "Job `24431261` (`v3_tc_rawhg`) then fit v2 SAE raw on the original caches "
        "(no cache suffix) into `tc_both_lag_scores_rawhg.csv` (45 electrodes, −2000…2000 ms), "
        "`tc_wide_lag_scores_rawhg.csv` (the 314 complement, −500…2000 ms), and "
        "`tc_sig_glove_lag_scores_rawhg.csv` (217 electrodes, −500…2000 ms). "
        "Those curves are the raw HG series on the figures."
    )
    if old not in doc:
        raise RuntimeError("raw-HG sentence not found")
    doc = doc.replace(old, new, 1)
    DOC.write_text(doc)
    print(json.dumps(summary, indent=2))
    print("wrote", MEDIA)


if __name__ == "__main__":
    main()
