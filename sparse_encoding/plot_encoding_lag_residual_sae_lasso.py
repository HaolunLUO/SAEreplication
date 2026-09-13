#!/usr/bin/env python3
"""Overlay residual vs SAE lag curves when both use LASSO→Ridge.

Reads:
  sparse_encoding_lag_screen_long_{tag}_lang.csv
  sparse_encoding_lag_screen_long_{tag}_resid_lasso_lang.csv

Subject-mean Fisher-z R ± SEM on language-localizer channels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core.analysis_paths as ap


def _curves(long_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = long_df.loc[long_df["mode"] == mode]
    if sub.empty:
        return pd.DataFrame(columns=["lag_ms", "mean", "sem", "n_subjects"])
    s = (sub.groupby(["subject", "lag_ms"])["R_fisher"].mean()
           .reset_index())
    g = s.groupby("lag_ms")["R_fisher"]
    return pd.DataFrame({
        "lag_ms": g.mean().index.astype(float),
        "mean": g.mean().to_numpy(),
        "sem": g.sem().to_numpy(),
        "n_subjects": g.count().to_numpy(),
    })


def _peak(cur: pd.DataFrame) -> tuple[float, float]:
    if cur.empty:
        return np.nan, np.nan
    i = int(np.nanargmax(cur["mean"].to_numpy()))
    return float(cur["lag_ms"].iloc[i]), float(cur["mean"].iloc[i])


def main() -> None:
    tables = getattr(ap, "SAE_TABLES", None) or ap.SAE_TABLES
    figs = getattr(ap, "SAE_FIGURES", None) or ap.SAE_FIGURES
    tag = "sae_qwen35_4b_mat_l15"
    sae_path = tables / f"sparse_encoding_lag_screen_long_{tag}_lang.csv"
    resid_path = tables / f"sparse_encoding_lag_screen_long_{tag}_resid_lasso_lang.csv"
    sae = pd.read_csv(sae_path)
    resid = pd.read_csv(resid_path)

    series = {
        "Residual full": _curves(resid, "full"),
        "Residual content": _curves(resid, "content"),
        "SAE full": _curves(sae, "full"),
        "SAE content": _curves(sae, "content"),
        "Surprisal": _curves(sae, "surprisal_only"),
    }
    colors = {
        "Residual full": "#9467bd",
        "Residual content": "#c5b0d5",
        "SAE full": "#1f77b4",
        "SAE content": "#2ca02c",
        "Surprisal": "#d62728",
    }
    styles = {
        "Residual full": "-",
        "Residual content": "--",
        "SAE full": "-",
        "SAE content": "--",
        "Surprisal": ":",
    }

    rows = []
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for name, cur in series.items():
        if cur.empty:
            continue
        pk_lag, pk_r = _peak(cur)
        rows.append({
            "model": name, "peak_lag_ms": pk_lag, "peak_R_fisher": pk_r,
            "n_subjects": int(cur["n_subjects"].max()) if len(cur) else 0,
        })
        ax.plot(cur["lag_ms"], cur["mean"], label=name,
                color=colors[name], lw=2.2, ls=styles[name])
        if cur["sem"].notna().any():
            ax.fill_between(
                cur["lag_ms"],
                cur["mean"] - cur["sem"],
                cur["mean"] + cur["sem"],
                color=colors[name], alpha=0.15,
            )
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Lag after word onset (ms)")
    ax.set_ylabel("Encoding R (Fisher-z), subject-mean ± SEM")
    ax.set_title("Language channels · LASSO→Ridge residual vs SAE")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig_path = figs / f"encoding_lag_residual_sae_lasso_{tag}_lang.png"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    out_csv = tables / f"encoding_lag_residual_sae_lasso_summary_{tag}_lang.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {fig_path}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
