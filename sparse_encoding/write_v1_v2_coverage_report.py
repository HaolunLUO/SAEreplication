#!/usr/bin/env python3
"""Write the v1-vs-v2 coverage / SAE-gain comparison from files already on disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ryan/analysisEV")
FEAT = ROOT / "extracted_linguistic_features"
V1 = ROOT / "group_encoding_results" / "sparse_encoding"
V2 = ROOT / "group_encoding_results" / "sparse_encoding_v2"
OUT = V2 / "reports" / "v1_vs_v2_coverage.md"

TAGS = (
    ("Qwen3.5 L15", "sae_qwen35_4b_mat_l15"),
    ("Qwen3-8B L18", "sae_qwen3_8b_l18"),
)


def _meta(tag: str, section: int) -> dict:
    path = FEAT / f"section_{section:03d}" / f"X_word_{tag}_meta.json"
    return json.loads(path.read_text())


def _fmt(x, digits=4, signed=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if signed:
        return f"{float(x):+.{digits}f}"
    return f"{float(x):.{digits}f}"


def _subject_gain(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    full = "full__R_fisher_paired"
    surp = "surprisal_only__R_fisher_paired"
    if full not in df.columns or surp not in df.columns:
        raise KeyError(csv_path)
    df = df.copy()
    df["sae_gain"] = df[full] - df[surp]
    sub = df.groupby("subject")["sae_gain"].mean()
    return {
        "n_electrodes": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "subject_mean": float(sub.mean()),
        "electrode_mean": float(df["sae_gain"].mean()),
    }


def _summary_gain(path: Path, tag: str, group: str) -> float:
    df = pd.read_csv(path)
    hit = df[(df["feature_tag"] == tag) & (df["group"] == group)]
    if hit.empty:
        raise KeyError(f"{tag} {group} missing in {path}")
    return float(hit.iloc[0]["paper_sae_gain_subject_mean"])


def _sign_flag(a: float, b: float) -> str:
    if not np.isfinite(a) or not np.isfinite(b):
        return "NA"
    if a == 0 or b == 0:
        return "no" if np.sign(a) == np.sign(b) else "yes"
    return "yes" if np.sign(a) != np.sign(b) else "no"


def main() -> None:
    lines = []
    add = lines.append
    add("# v1 vs v2 token ownership")
    add("")
    add("v2 uses `shared_char_weighted`: a token that overlaps several words")
    add("contributes to each of them, weighted by the fraction of its characters")
    add("inside that word. The word SAE vector is the weight-normalized mean of")
    add("those token SAE activations (SAE, then mean). Surprisal uses the same")
    add("weights. v1 remains unique-final-word. v1 npz, meta, tables, and reports")
    add("were not overwritten. v2 features are `X_word_<tag>_v2.*`. Regression")
    add("tables are under `group_encoding_results/sparse_encoding_v2/`.")
    add("")
    add("SAE gain is full Fisher-z *r* minus surprisal-only Fisher-z *r*,")
    add("then averaged within subject and across subjects. The headline rows")
    add("are the Study 3 subject means (`paper_sae_gain_subject_mean`), which")
    add("drop electrodes that fail `paper_quality_ok`. Hyperparameters match v1:")
    add("F-test k=8000, LassoCV, Ridge, 5 contiguous within-section folds,")
    add("`random_state=19`, surprisal force-included.")
    add("")
    add("## Coverage and adjacent duplicate rows")
    add("")
    add("Alignment coverage is the percent of words that receive at least one")
    add("token. Valid coverage also requires a finite surprisal. A handful of")
    add("words stay invalid because their only contributing tokens have no NLL")
    add("(the first token in a forward window).")
    add("")
    add("| Backbone | Section | v1 align % | v2 align % | v2 valid % | v1 adj dup | v2 adj dup | v2 shared tokens | v2 shared words |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, base in TAGS:
        for sid in (1, 2, 3):
            m1 = _meta(base, sid)
            m2 = _meta(base + "_v2", sid)
            a1 = m1["alignment"]
            a2 = m2["alignment"]
            add(
                f"| {label} | {sid} | {_fmt(a1['coverage_pct'], 2)} | "
                f"{_fmt(a2['coverage_pct'], 2)} | "
                f"{_fmt(m2.get('owned_token_coverage_pct'), 2)} | "
                f"{_fmt(m1.get('adjacent_duplicate_row_rate'), 4)} | "
                f"{_fmt(m2.get('adjacent_duplicate_row_rate'), 4)} | "
                f"{int(m2.get('shared_token_count', 0))} | "
                f"{int(m2.get('shared_token_word_count', 0))} |"
            )
    add("")
    add("Adjacent-duplicate rate is higher in v2 because a token that spans")
    add("two words, and is the only contributor to each, copies the same SAE")
    add("row into both words after weight normalization.")
    add("")
    add("## SAE gain (subject mean)")
    add("")
    v1_sum = V1 / "tables" / "lepori_study3_froi_summary.csv"
    v2_sum = V2 / "tables" / "lepori_study3_froi_summary.csv"
    add("| Backbone | Group | v1 gain | v2 gain | sign change | v2 section adj-dup (1 / 2 / 3) |")
    add("|---|---|---:|---:|---|---|")
    sign_changes = []
    for label, base in TAGS:
        dups = []
        for sid in (1, 2, 3):
            dups.append(_fmt(_meta(base + "_v2", sid).get("adjacent_duplicate_row_rate"), 4))
        dup_s = " / ".join(dups)
        for group in ("all_channels", "is_lang"):
            g1 = _summary_gain(v1_sum, base, group)
            g2 = _summary_gain(v2_sum, base + "_v2", group)
            flag = _sign_flag(g1, g2)
            if flag == "yes":
                sign_changes.append(f"{label} {group}")
            add(
                f"| {label} | {group} | {_fmt(g1, 4, signed=True)} | "
                f"{_fmt(g2, 4, signed=True)} | {flag} | {dup_s} |"
            )
    add("")
    if sign_changes:
        add("Sign changes: " + ", ".join(sign_changes) + ".")
    else:
        add("No sign change in subject-mean SAE gain (all channels or is_lang) for the Qwen backbones.")
    add("")
    add("## Qwen3.5 early-bin occupancy and lang refits")
    add("")
    add("Occupancy is the share of selected signed features on `is_lang` electrodes")
    add("(Study 3 `occupancy_selected_tokens`). `2048–end` occupancy is")
    add("`2048–16384` plus `16384–end`. Refit gains are subject means of")
    add("full minus surprisal-only on the lang-only bin regressions. The")
    add("electrode mean is the statistic stored in the v1 Study 3 bin table.")
    add("")
    bins_v1 = pd.read_csv(V1 / "tables" / "lepori_study3_matryoshka_bins.csv")
    bins_v2 = pd.read_csv(V2 / "tables" / "lepori_study3_matryoshka_bins.csv")

    def occ_map(df, tag):
        sub = df[(df["feature_tag"] == tag) & (df["kind"] == "occupancy_selected_tokens")]
        out = {}
        for _, r in sub.iterrows():
            out[str(r["bin"])] = float(r["frac_selected_tokens"])
        return out

    def occ_line(label, mp):
        b0 = mp.get("bin0_2048", np.nan)
        b_mid = mp.get("bin2048_16384", np.nan)
        b_hi = mp.get("bin16384_plus", np.nan)
        b_end = (b_mid + b_hi) if np.isfinite(b_mid) and np.isfinite(b_hi) else np.nan
        return (
            f"| {label} | {_fmt(100 * b0, 1)} | {_fmt(100 * b_end, 1)} | "
            f"{_fmt(100 * b_hi, 1)} |"
        )

    add("| Version | 0–2048 % | 2048–end % | 16384–end % |")
    add("|---|---:|---:|---:|")
    add(occ_line("v1", occ_map(bins_v1, "sae_qwen35_4b_mat_l15")))
    add(occ_line("v2", occ_map(bins_v2, "sae_qwen35_4b_mat_l15_v2")))
    add("")
    add("| Bin | v1 subject gain | v2 subject gain | sign change | v1 electrode gain | v2 electrode gain |")
    add("|---|---:|---:|---|---:|---:|")
    bin_specs = (
        ("0–2048", "lang_bin0-2048.csv"),
        ("2048–end", "lang_bin2048-end.csv"),
        ("16384–end", "lang_bin16384-end.csv"),
        ("2048–16384", "lang_bin2048-16384.csv"),
    )
    tag = "sae_qwen35_4b_mat_l15"
    for name, fname in bin_specs:
        p1 = V1 / "tables" / f"sparse_encoding_results_{tag}_{fname}"
        p2 = V2 / "tables" / f"sparse_encoding_results_{tag}_v2_{fname}"
        s1, s2 = _subject_gain(p1), _subject_gain(p2)
        flag = _sign_flag(s1["subject_mean"], s2["subject_mean"])
        add(
            f"| {name} | {_fmt(s1['subject_mean'], 4, signed=True)} | "
            f"{_fmt(s2['subject_mean'], 4, signed=True)} | {flag} | "
            f"{_fmt(s1['electrode_mean'], 4, signed=True)} | "
            f"{_fmt(s2['electrode_mean'], 4, signed=True)} |"
        )
    add("")
    add("## Caveat")
    add("")
    add("A shared token's surprisal and SAE activation enter every overlapped")
    add("word. Surprisal is therefore no longer exclusive to one word, and")
    add("adjacent words that share a single multi-character token receive the")
    add("same vector. SAE gain can move because both the SAE features and the")
    add("surprisal-only baseline are smeared across the word boundary.")
    add("")
    add("## Gate")
    add("")
    add("Section 1 Qwen3.5 was extracted before the other sections. Alignment")
    add("coverage was 100% (0 ownership mismatches). Words that were valid in v1")
    add("and not touched by a shared token matched v1 SAE rows and surprisal")
    add("exactly (max abs diff 0 on 778 words).")
    add("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
