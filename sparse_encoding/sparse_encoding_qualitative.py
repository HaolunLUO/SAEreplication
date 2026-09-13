#!/usr/bin/env python3
"""
sparse_encoding_qualitative.py
==============================
Qualitative / feature-based analysis for Augmented Sparse Encoding Models,
adapted from
    https://github.com/mlepori1/Interpretable_Encoding_Models
    (src/qualitative_analysis.py)

Goal: interpret *what* each electrode encodes by identifying which SAE features
predict its response, then grounding those features in the stimulus (the words
that most strongly activate them).

Steps
-----
1. Read the per-electrode regression results (from sparse_encoding_regression.py)
   and pick the best-predicted electrodes (globally or per channel category).
2. For each selected electrode, refit Lasso feature selection on the FULL data
   (matching the reference repo's ``select_features``), recording the signed
   SAE feature indices (sign = Ridge coefficient direction).
3. Build a feature-usage table across electrodes, optionally split by the
   functional-taxonomy channel category (both / enc_only / loc_only / neither).
4. For each frequently-selected feature, list the top-activating stimulus words
   (data-driven interpretation, no external service required).

Outputs (under group_encoding_results/sparse_encoding/tables/):
    sparse_encoding_features[_suffix].csv
        per-electrode selected feature indices (+ source_results_csv)
    sparse_encoding_feature_words[_suffix].csv
        per-feature top-activating words (+ source_results_csv)

The suffix is inferred from the results CSV stem (e.g. shuffle_global) so a
shuffle qualitative run cannot overwrite the main tables.

Example
-------
    python sparse_encoding_qualitative.py --feature_tag sae_gemma2_2b_mat_l12 --top_n 30
    python sparse_encoding_qualitative.py \\
        --results_csv .../sparse_encoding_results_shuffle_global.csv
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

import core.analysis_paths as ap
import core.encoding_channel as ec
from sparse_encoding.sparse_encoding_regression import (
    FEATURE_TAG, FS_TARGET, RESP_WIN_MS, RESPONSE_LAG_MS,
    compute_support_features, compute_joint_support, select_alpha,
    load_sae_features, get_subjects,
)
from sparse_encoding.sparse_encoding_summary import output_stem_suffix

# Signed feature serialization: "idx:+" / "idx:-" (unambiguous for index 0).
# Legacy space-separated signed ints (e.g. "-4 38") are still accepted by the
# parser, except that bare "0" is treated as (0, +1).
_SIGNED_TOKEN_RE = re.compile(r"^(\d+):([+-])$")
_LEGACY_TOKEN_RE = re.compile(r"^([+-]?\d+)$")


# ======================================================================
# Stimulus words (for grounding features)
# ======================================================================

def load_all_words(sections) -> List[str]:
    words: List[str] = []
    for sid in sections:
        wt = pd.read_csv(ap.FEATURES_DIR / f"section_{sid:03d}" / "word_timing.csv")
        words.extend(wt["word"].astype(str).tolist())
    return words


def top_words_for_feature(
    X_sae: sparse.csr_matrix, words: List[str], feat_idx: int, top_n: int = 15
) -> List[str]:
    col = X_sae[:, feat_idx].toarray().ravel()
    order = np.argsort(col)[::-1]
    out = []
    for i in order[:top_n]:
        if col[i] <= 0:
            break
        out.append(f"{words[i]}({col[i]:.2f})")
    return out


# ======================================================================
# Signed feature serialization / parsing
# ======================================================================

def serialize_signed_features(
    indices: Sequence[int],
    signs: Optional[Sequence[int]] = None,
) -> str:
    """Serialize SAE features as ``idx:+`` / ``idx:-`` tokens.

    If ``signs`` is omitted, ``indices`` may already be signed legacy ints
    (``sign * idx``). Index 0 with a negative sign requires an explicit
    ``signs`` array — legacy ``sign * 0`` cannot encode polarity.
    """
    if signs is None:
        parts: List[str] = []
        for v in indices:
            v = int(v)
            if v < 0:
                parts.append(f"{abs(v)}:-")
            else:
                parts.append(f"{abs(v)}:+")
        return " ".join(parts)
    if len(indices) != len(signs):
        raise ValueError("indices and signs must have the same length")
    parts = []
    for idx, sgn in zip(indices, signs):
        idx = int(abs(int(idx)))
        sgn = int(np.sign(sgn)) if int(sgn) != 0 else 1
        parts.append(f"{idx}:{'+' if sgn >= 0 else '-'}")
    return " ".join(parts)


def parse_signed_features(s) -> Set[Tuple[int, int]]:
    """Parse feature string → set of ``(index, sign)`` with sign in ``{+1, -1}``.

    Accepts the unambiguous ``idx:+`` / ``idx:-`` format and the legacy
    space-separated signed-integer format. Bare ``0`` maps to ``(0, +1)``.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return set()
    text = str(s).strip()
    if not text or text.lower() in ("nan", "none"):
        return set()
    out: Set[Tuple[int, int]] = set()
    for tok in text.split():
        m = _SIGNED_TOKEN_RE.match(tok)
        if m:
            out.add((int(m.group(1)), 1 if m.group(2) == "+" else -1))
            continue
        m = _LEGACY_TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"Unrecognized feature token: {tok!r}")
        v = int(m.group(1))
        if v < 0:
            out.add((abs(v), -1))
        else:
            # Legacy "+0" / "0" / "-0" all become 0 → treat as +1.
            out.add((v, 1))
    return out


def parse_unsigned_features(s) -> Set[int]:
    """Parse feature string → set of absolute SAE feature indices."""
    return {idx for idx, _ in parse_signed_features(s)}


def load_channel_allowlist(path: Path) -> Dict[str, Set[str]]:
    """Load ``subject,channel`` allowlist CSV → ``{subject: {channels}}``."""
    df = pd.read_csv(path)
    if "subject" not in df.columns or "channel" not in df.columns:
        raise ValueError(f"{path} must contain subject and channel columns")
    out: Dict[str, Set[str]] = {}
    for subj, g in df.groupby("subject"):
        out[str(subj)] = {str(c).strip() for c in g["channel"].tolist()}
    return out


# ======================================================================
# Per-electrode feature selection on full data
# ======================================================================

def select_features_full(
    X_sae, surprisal, y, joint_lasso_surprisal: bool = True,
) -> Dict:
    """Refit selection + Ridge on ALL words; return signed feature indices.

    When ``joint_lasso_surprisal`` is True (default), matches the reference
    qualitative path with ``--use_logprobs``: surprisal joins the Lasso pool,
    then is force-included for Ridge.

    Returns absolute ``feature_indices`` plus parallel ``feature_signs`` so
    index 0 retains polarity under serialization.
    """
    ok = np.isfinite(y) & np.isfinite(surprisal)
    Xc = X_sae[ok]
    yc = y[ok].astype(np.float64)

    from sklearn.preprocessing import StandardScaler
    surp = StandardScaler().fit_transform(surprisal[ok].reshape(-1, 1))

    if joint_lasso_surprisal:
        support = compute_joint_support(Xc, surp, yc)
        if not support.any():
            # Surprisal-only Ridge still runs so we can report the coef.
            alpha = select_alpha(surp, yc)
            model = Ridge(alpha=alpha, fit_intercept=True).fit(surp, yc)
            return {
                "feature_indices": np.array([], dtype=int),
                "feature_signs": np.array([], dtype=int),
                "surprisal_coef": float(model.coef_[0]),
            }
    else:
        support = compute_support_features(Xc, yc)
        if support is None:
            return {
                "feature_indices": np.array([], dtype=int),
                "feature_signs": np.array([], dtype=int),
            }

    sel = Xc[:, support].toarray()
    X_full = np.hstack([sel, surp])

    alpha = select_alpha(X_full, yc)
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X_full, yc)
    coef = model.coef_[:-1]                       # drop surprisal coef
    sign = np.where(coef >= 0, 1, -1).astype(int)
    idx = np.where(support)[0].astype(int)
    return {
        "feature_indices": idx,
        "feature_signs": sign,
        "surprisal_coef": float(model.coef_[-1]),
    }


def maybe_merge_category(df: pd.DataFrame) -> pd.DataFrame:
    """Attach functional-taxonomy channel category if the table is available."""
    tax = ap.TAX_ELECTRODE_TABLE
    if not tax.exists():
        return df
    t = pd.read_csv(tax)
    # try to find sensible key columns
    subj_col = next((c for c in t.columns if c.lower() in
                     ("subject", "sid", "subject_id")), None)
    chan_col = next((c for c in t.columns if c.lower() in
                     ("channel", "channel_label", "elec", "electrode", "name")), None)
    cat_col = next((c for c in t.columns if "categor" in c.lower()
                    or c.lower() in ("channel_type", "type", "group")), None)
    if not (subj_col and chan_col and cat_col):
        return df
    keep = t[[subj_col, chan_col, cat_col]].rename(
        columns={subj_col: "subject", chan_col: "channel", cat_col: "category"})
    return df.merge(keep, on=["subject", "channel"], how="left")


# ======================================================================
# Main
# ======================================================================

def select_electrodes(
    res: pd.DataFrame,
    score_col: str,
    top_n: int,
    per_category: bool,
    all_channels: bool,
    allowlist: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Choose electrodes to interpret; return (subset, selection_mode)."""
    work = res.copy()
    work["channel"] = work["channel"].astype(str).str.strip()
    if allowlist is not None:
        keep = []
        for _, r in work.iterrows():
            chans = allowlist.get(str(r["subject"]), set())
            keep.append(str(r["channel"]).strip() in chans)
        work = work.loc[keep].copy()
        mode = "allowlist"
        if all_channels or top_n <= 0:
            return work, mode
        # Allowlist + top_n: keep best within allowlist
        return (
            work.sort_values(score_col, ascending=False).head(top_n),
            f"{mode}_top{top_n}",
        )

    if all_channels or top_n <= 0:
        return work, "all_channels"

    if per_category and "category" in work.columns:
        best = (work.sort_values(score_col, ascending=False)
                    .groupby("category", group_keys=False)
                    .head(top_n))
        return best, f"per_category_top{top_n}"
    best = work.sort_values(score_col, ascending=False).head(top_n)
    return best, f"global_top{top_n}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument("--sections", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--lag_ms", type=float, default=RESPONSE_LAG_MS)
    p.add_argument("--resp_win_ms", type=float, default=RESP_WIN_MS)
    p.add_argument("--fs_target", type=float, default=FS_TARGET)
    p.add_argument("--results_csv", default=None,
                   help="regression results CSV (default: analysis_paths).")
    p.add_argument("--top_n", type=int, default=30,
                   help="Best electrodes (by full__R_fisher) to interpret. "
                        "Use 0 with --all_channels for every eligible channel.")
    p.add_argument("--all_channels", action="store_true",
                   help="Interpret every electrode in the results CSV "
                        "(full-data Lasso refit; exploratory).")
    p.add_argument("--channels_from_csv", default=None,
                   help="Optional subject,channel allowlist CSV.")
    p.add_argument("--per_category", action="store_true",
                   help="Select top_n electrodes within each channel category.")
    p.add_argument("--top_words", type=int, default=15)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument(
        "--joint_lasso_surprisal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match sparse_encoding_regression joint Lasso path "
             "(default on; use --no-joint_lasso_surprisal to disable).",
    )
    p.add_argument(
        "--suffix", default=None,
        help="Override output filename suffix. Default: inferred from "
             "results CSV stem so shuffle runs do not overwrite main tables.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute electrodes already present in the output CSV.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()

    results_csv = Path(args.results_csv) if args.results_csv else ap.SAE_REGRESSION_TABLE
    results_csv = results_csv.resolve()
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run sparse_encoding_regression.py first.")
    res = pd.read_csv(results_csv)
    res = maybe_merge_category(res)

    suffix = args.suffix if args.suffix is not None else output_stem_suffix(results_csv)
    score_col = "full__R_fisher" if "full__R_fisher" in res.columns else "full__R"

    allowlist = None
    if args.channels_from_csv:
        allowlist = load_channel_allowlist(Path(args.channels_from_csv))

    best, selection_mode = select_electrodes(
        res, score_col, args.top_n, args.per_category,
        all_channels=args.all_channels, allowlist=allowlist,
    )
    print(
        f"Qualitative selection: mode={selection_mode}, "
        f"n_electrodes={len(best)} (full-data Lasso refit; exploratory)",
        flush=True,
    )

    subjects_map = get_subjects()
    loaded = load_sae_features(args.feature_tag, args.sections)
    if len(loaded) == 3:
        X_sae, surprisal, validity = loaded
    else:
        X_sae, surprisal = loaded
        validity = np.ones(X_sae.shape[0], dtype=bool)
    words = load_all_words(args.sections)
    if len(words) != X_sae.shape[0]:
        raise ValueError(
            f"Word list length {len(words)} != SAE rows {X_sae.shape[0]}")

    # cache per-subject EEG-derived Y so we don't reload per electrode
    subj_cache: Dict[str, tuple] = {}

    def subject_Y(subject: str):
        if subject in subj_cache:
            return subj_cache[subject]
        eeg_files = {sid: str(ap.DATA_ROOT / p)
                     for sid, p in subjects_map[subject]["eeg_files"].items()}
        data = ec.load_subject_data(
            subject_name=subject, eeg_files=eeg_files,
            features_dir=ap.FEATURES_DIR, fs_target=args.fs_target,
            sections=tuple(args.sections), feature_sets=["glove"])
        half_win = max(1, int(round((args.resp_win_ms / 1000.0) * args.fs_target / 2.0)))
        lag_samp = int(round((args.lag_ms / 1000.0) * args.fs_target))
        Y, valid = ec.compute_Y_for_lag(data, lag_samp, half_win)
        subj_cache[subject] = (Y, valid, data.channel_labels)
        return subj_cache[subject]

    feat_out = ap.SAE_TABLES / f"sparse_encoding_features{suffix}.csv"
    feat_rows: List[dict] = []
    done: Set[Tuple[str, str]] = set()
    if feat_out.exists() and not args.overwrite:
        prev = pd.read_csv(feat_out)
        feat_rows = prev.to_dict("records")
        done = {
            (str(r["subject"]), str(r["channel"]).strip())
            for r in feat_rows
        }
        print(f"Resume: {len(done)} electrodes already in {feat_out.name}",
              flush=True)

    usage: Dict[int, int] = {}
    last_subject = None
    n_todo = int((~best.apply(
        lambda r: (str(r["subject"]), str(r["channel"]).strip()) in done,
        axis=1)).sum()) if len(best) else 0
    n_done_new = 0
    for _, r in best.iterrows():
        subject = r["subject"]
        channel = str(r["channel"]).strip()
        if args.subjects and subject not in args.subjects:
            continue
        if subject not in subjects_map:
            continue
        if (str(subject), channel) in done:
            continue
        if last_subject is not None and subject != last_subject:
            subj_cache.pop(last_subject, None)
            gc.collect()
            pd.DataFrame(feat_rows).to_csv(feat_out, index=False)
            print(f"  checkpoint {last_subject} -> {feat_out.name} "
                  f"({len(feat_rows)} rows)", flush=True)
        last_subject = subject
        Y, valid, labels = subject_Y(subject)
        ci = int(r["channel_index"])
        if ci >= Y.shape[1]:
            continue
        y = np.where(valid & validity, Y[:, ci], np.nan)
        sel = select_features_full(
            X_sae, surprisal, y,
            joint_lasso_surprisal=args.joint_lasso_surprisal,
        )
        idxs = sel["feature_indices"]
        signs = sel.get("feature_signs", np.ones(len(idxs), dtype=int))
        feat_rows.append({
            "subject": subject,
            "channel": channel,
            "category": r.get("category", np.nan),
            score_col: r[score_col],
            "surprisal_coef": sel.get("surprisal_coef", np.nan),
            "n_features": int(len(idxs)),
            "feature_indices": serialize_signed_features(idxs, signs),
            "selection_mode": selection_mode,
            "selection_note": "full_data_lasso_refit_exploratory",
            "source_results_csv": str(results_csv),
            "feature_tag": args.feature_tag,
        })
        n_done_new += 1
        if n_done_new == 1 or n_done_new % 5 == 0:
            print(f"  qualitative {n_done_new}/{n_todo} "
                  f"{subject} {channel} n_feat={len(idxs)}", flush=True)

    if last_subject is not None:
        subj_cache.pop(last_subject, None)
        gc.collect()

    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(feat_out, index=False)
    print(f"Saved per-electrode features -> {feat_out}")
    print(f"  source_results_csv={results_csv}")
    print(f"  selection_mode={selection_mode}")

    # Rebuild usage from the full table so resume runs still write words.
    for s in feat_df.get("feature_indices", pd.Series(dtype=str)).fillna(""):
        for idx, _ in parse_signed_features(s):
            usage[int(idx)] = usage.get(int(idx), 0) + 1

    # per-feature interpretation via top-activating words
    word_rows = []
    for fi, count in sorted(usage.items(), key=lambda kv: kv[1], reverse=True):
        word_rows.append({
            "feature_index": fi,
            "n_electrodes_selected": count,
            "top_words": " ".join(top_words_for_feature(
                X_sae, words, fi, args.top_words)),
            "source_results_csv": str(results_csv),
            "feature_tag": args.feature_tag,
            "selection_mode": selection_mode,
        })
    word_df = pd.DataFrame(word_rows)
    word_out = ap.SAE_TABLES / f"sparse_encoding_feature_words{suffix}.csv"
    word_df.to_csv(word_out, index=False)
    print(f"Saved feature interpretations -> {word_out}")


if __name__ == "__main__":
    main()
