#!/usr/bin/env python3
"""Word-class dependence of signed feature contributions.

Uses the stored story-word SAE matrix and the existing constituency tree.
The UD file lppCN_word_information.csv is not on this machine; content and
function words are the CTB tags already in lppCN_tree.txt. Saying verbs and
possessive 的 follow the contrasts in the Table 1 label evidence. Speech
spans are not invented.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fc_common as C

TREE = Path("/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/lppCN_tree.txt")
CONLL = Path(
    "/run/user/245046/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846"
    "/files/internal/table1-parses/hanlp_raw.conll"
)
FEATURES = Path("/orcd/pool/005/haolun52/analysisEV/extracted_linguistic_features")
SAYING = ["说", "道", "说道", "回答", "问", "喊"]
# Content words in the Chinese Treebank tagset. Function words are the rest.
CONTENT_TAGS = {"NN", "NR", "NT", "VV", "VA", "VC", "VE", "JJ", "AD", "CD"}
VERB_TAGS = {"VV", "VA", "VC", "VE"}
MEDIA = C.MEDIA


def word_table() -> pd.DataFrame:
    frames = []
    for sid in (1, 2, 3):
        wt = pd.read_csv(FEATURES / f"section_{sid:03d}" / "word_timing.csv")
        wt = wt.copy()
        wt["section_id"] = sid
        wt["row_in_section"] = np.arange(len(wt))
        frames.append(wt[["word", "section_id", "row_in_section"]])
    out = pd.concat(frames, ignore_index=True)
    out["word_i"] = np.arange(len(out))
    return out


def tree_leaves(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"\(([A-Za-z0-9]+)\s+([^()\s]+)\)", text)


def conll_tokens(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        rows.append((parts[1], parts[7]))
    return rows


def align(words: pd.DataFrame) -> pd.DataFrame:
    leaves = tree_leaves(TREE)
    # The tree is the whole book. Sections 1–3 are a prefix of its leaves.
    if len(leaves) < len(words):
        raise RuntimeError(f"tree leaves {len(leaves)} < words {len(words)}")
    leaves = leaves[: len(words)]
    mismatch = sum(w != leaf for w, (_tag, leaf) in zip(words["word"], leaves))
    if mismatch:
        raise RuntimeError(f"tree tokens differ from word_timing on {mismatch} rows")
    words = words.copy()
    words["ctb"] = [tag for tag, _w in leaves]
    words["content"] = words["ctb"].isin(CONTENT_TAGS)
    dep = conll_tokens(CONLL)
    if len(dep) != len(words):
        raise RuntimeError(f"conll tokens {len(dep)} != words {len(words)}")
    mismatch = sum(w != tok for w, (tok, _d) in zip(words["word"], dep))
    if mismatch:
        raise RuntimeError(f"conll tokens differ from word_timing on {mismatch} rows")
    words["deprel"] = [d for _t, d in dep]
    prev = words["ctb"].shift(1)
    # Pronouns in this tree are PRP (84 的 follow a PRP, matching the label evidence).
    words["de_after_pronoun"] = (words["word"] == "的") & (prev == "PRP")
    words["de_assm"] = (words["word"] == "的") & (words["deprel"] == "assm")
    words["de_other"] = (words["word"] == "的") & ~words["de_assm"]
    words["saying"] = words["word"].isin(SAYING) & words["ctb"].isin(VERB_TAGS)
    words["other_verb"] = words["ctb"].isin(VERB_TAGS) & ~words["saying"]
    quote_chars = set("\"“”「」『』")
    n_quote = int(sum(any(ch in quote_chars for ch in str(w)) for w in words["word"]))
    n_quote += TREE.read_text(encoding="utf-8").count("“") + TREE.read_text(encoding="utf-8").count("”")
    words.attrs["n_quote"] = n_quote
    return words


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    signs = pd.read_csv(C.OUT / "electrode_signs.csv")
    pop = pd.read_csv(C.OUT / "population.csv")
    feat = pd.read_csv(C.OUT / "fig4c_features.csv")
    words = align(word_table())
    print(
        f"[words] n={len(words)} content {int(words.content.sum())} "
        f"function {int((~words.content).sum())} quotes {words.attrs['n_quote']}",
        flush=True,
    )
    print(
        f"[words] saying {int(words.saying.sum())} other_verb {int(words.other_verb.sum())} "
        f"的-pron {int(words.de_after_pronoun.sum())} assm {int(words.de_assm.sum())} "
        f"的 {int((words.word=='的').sum())}",
        flush=True,
    )
    X, surp, bv, _T = C.load_design()
    # Activation rows follow the same section concatenation as word_timing.
    if X.shape[0] != len(words):
        raise RuntimeError(f"SAE rows {X.shape[0]} != words {len(words)}")

    keep = feat[feat["prevalence"] > 10]["feature_index"].astype(int).tolist()
    for f_i in C.TABLE1:
        if f_i not in keep:
            keep.append(f_i)
    keep = sorted(set(keep))

    pair_rows = []
    class_rows = []
    act_rows = []
    subj_vectors = {}
    for f_i in keep:
        col = np.asarray(X[:, f_i].toarray(), dtype=np.float64).ravel()
        sel = signs[signs.feature_index == f_i]
        by_subj = {}
        for subject, sub in sel.groupby("subject"):
            # Mean of (electrode coefficient × word activation).
            coefs = sub["mean_coef"].to_numpy(float)
            contrib = coefs.mean() * col
            by_subj[subject] = contrib
            rho, _p = spearmanr(col, contrib)
            act_rows.append({
                "feature_index": f_i,
                "subject": subject,
                "n_electrodes": int(len(sub)),
                "mean_coef": float(coefs.mean()),
                "spearman_activation_vs_contribution": float(rho),
            })
            class_rows.append({
                "feature_index": f_i,
                "subject": subject,
                "content_mean": float(contrib[words["content"].to_numpy()].mean()),
                "function_mean": float(contrib[~words["content"].to_numpy()].mean()),
                "saying_mean": float(contrib[words["saying"].to_numpy()].mean()) if words["saying"].any() else np.nan,
                "other_verb_mean": float(contrib[words["other_verb"].to_numpy()].mean()) if words["other_verb"].any() else np.nan,
                "de_pronoun_mean": float(contrib[words["de_after_pronoun"].to_numpy()].mean()) if words["de_after_pronoun"].any() else np.nan,
                "de_other_mean": float(contrib[(words.word == "的") & ~words["de_after_pronoun"]].mean()),
                "assm_mean": float(contrib[words["de_assm"].to_numpy()].mean()) if words["de_assm"].any() else np.nan,
                "assm_other_de_mean": float(contrib[words["de_other"].to_numpy()].mean()) if words["de_other"].any() else np.nan,
            })
        subj_vectors[f_i] = by_subj
        subjects = sorted(by_subj)
        rhos = []
        for i, a in enumerate(subjects):
            for b in subjects[i + 1:]:
                rho, p = spearmanr(by_subj[a], by_subj[b])
                pair_rows.append({
                    "feature_index": f_i,
                    "subject_a": a,
                    "subject_b": b,
                    "spearman": float(rho),
                    "p": float(p),
                })
                rhos.append(rho)
        print(f"[words] feat {f_i} subjects {len(subjects)} mean rho {np.mean(rhos) if rhos else np.nan:.3f}", flush=True)

    pairs = pd.DataFrame(pair_rows)
    classes = pd.DataFrame(class_rows)
    acts = pd.DataFrame(act_rows)
    pairs.to_csv(C.OUT / "word_spearman_pairs.csv", index=False)
    classes.to_csv(C.OUT / "word_class_means.csv", index=False)
    acts.to_csv(C.OUT / "word_activation_spearman.csv", index=False)

    # Prevalence > 10 table, including the seven.
    agg = pairs.groupby("feature_index")["spearman"].agg(["mean", "min", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "mean_cross_subject_spearman", "min": "min_spearman", "count": "n_subject_pairs"})
    act_mean = acts.groupby("feature_index")["spearman_activation_vs_contribution"].mean().rename("mean_activation_spearman")
    cls = classes.groupby("feature_index")[["content_mean", "function_mean"]].mean().reset_index()
    cls["content_minus_function"] = cls["content_mean"] - cls["function_mean"]
    table = feat.merge(agg, on="feature_index", how="left").merge(act_mean, on="feature_index", how="left")
    table = table.merge(cls[["feature_index", "content_mean", "function_mean", "content_minus_function"]], on="feature_index", how="left")
    table = table[table["prevalence"] > 10].sort_values(["prevalence", "feature_index"], ascending=[False, True])
    # Attach the Table 1 sign from the signed-feature list when the electrode mean agrees.
    sign_mode = signs.groupby("feature_index")["sign"].agg(lambda s: int(np.sign(np.mean(s)))).rename("electrode_sign_mean")
    table = table.merge(sign_mode, on="feature_index", how="left")
    table.to_csv(MEDIA / "prevalence_gt10_words.csv", index=False)

    # Figure: seven features, cross-subject Spearman and content vs function.
    seven = [f for f in C.TABLE1 if f in subj_vectors]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    means = []
    for f_i in seven:
        sub = pairs[pairs.feature_index == f_i]["spearman"]
        means.append(float(sub.mean()) if len(sub) else np.nan)
    axes[0].bar(np.arange(len(seven)), means, color="#1f4e79")
    axes[0].set_xticks(np.arange(len(seven)), [str(f) for f in seven])
    axes[0].set_ylabel("Mean cross-subject Spearman")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color="black", lw=0.4)
    axes[0].set_title("Word-contribution agreement")

    x = np.arange(len(seven))
    content, function = [], []
    for f_i in seven:
        sub = classes[classes.feature_index == f_i]
        content.append(float(sub["content_mean"].mean()))
        function.append(float(sub["function_mean"].mean()))
    axes[1].bar(x - 0.18, content, width=0.36, color="#b85c38", label="Content")
    axes[1].bar(x + 0.18, function, width=0.36, color="#7aa0c4", label="Function")
    axes[1].set_xticks(x, [str(f) for f in seven])
    axes[1].set_ylabel("Mean word contribution")
    axes[1].set_xlabel("Feature")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", lw=0.4)
    axes[1].set_title("Content vs function")
    fig.tight_layout()
    fig.savefig(MEDIA / "word_contribution_table1.png", dpi=160)
    fig.savefig(MEDIA / "word_contribution_table1.pdf")
    plt.close(fig)

    # Specific contrasts for 352 and 869, shown for all seven so the split is visible.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.18
    series = [
        ("saying_mean", "Saying verbs", "#b85c38"),
        ("other_verb_mean", "Other verbs", "#e0b090"),
        ("de_pronoun_mean", "的 after pronoun", "#1f4e79"),
        ("de_other_mean", "Other 的", "#7aa0c4"),
    ]
    for i, (col, lab, color) in enumerate(series):
        vals = [float(classes[classes.feature_index == f_i][col].mean()) for f_i in seven]
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=color, label=lab)
    ax.set_xticks(x, [str(f) for f in seven])
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("Mean word contribution")
    ax.set_title("Saying verbs and possessive 的")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(MEDIA / "word_class_contrasts_table1.png", dpi=160)
    fig.savefig(MEDIA / "word_class_contrasts_table1.pdf")
    plt.close(fig)

    # Compact prevalence>10 summary: mean cross-subject Spearman.
    show = table.sort_values("prevalence", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(np.arange(len(show)), show["mean_cross_subject_spearman"], color="#4c4c4c")
    ax.set_xticks(np.arange(len(show)), [str(int(i)) for i in show["feature_index"]], rotation=90, fontsize=7)
    ax.set_ylabel("Mean cross-subject Spearman")
    ax.set_xlabel("Feature (prevalence > 10)")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", lw=0.4)
    fig.tight_layout()
    fig.savefig(MEDIA / "word_spearman_prevalence_gt10.png", dpi=160)
    fig.savefig(MEDIA / "word_spearman_prevalence_gt10.pdf")
    plt.close(fig)
    print(f"[words] prevalence>10 features {len(show)}", flush=True)
    print("[words] done", flush=True)


if __name__ == "__main__":
    main()
