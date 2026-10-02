"""Shared loaders for the timing-controlled feature-consistency analysis.

Reads existing timing-control tables. Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TC_DIR = Path(__file__).resolve().parents[1]
SAE_ROOT = Path("/orcd/pool/005/haolun52/analysisEV/sae_sparse_encoding")
for p in (str(TC_DIR), str(SAE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import core.analysis_paths as ap  # noqa: E402
import sparse_encoding.sparse_encoding_v3 as v3  # noqa: E402
import timing_control as tc  # noqa: E402

OUT = TC_DIR / "feature_consistency"
TABLES = tc.TABLES
MEDIA = Path(
    "/run/user/245046/cursor_agent_stores/bc-0b12d8aa-9dbe-4229-9928-00e7b34e7846"
    "/files/media/sae-v3/timing-control/consistency"
)
SUBJECTS = [
    "Subject01", "Subject03", "Subject04", "Subject06", "Subject07", "Subject12",
]
SUBJECT_N = {
    "Subject01": 12, "Subject03": 3, "Subject04": 29,
    "Subject06": 41, "Subject07": 29, "Subject12": 37,
}
TABLE1 = [1236, 17, 1133, 671, 869, 1088, 352]
ALPHAS = [10 ** i for i in range(-2, 6)]
LAGS = v3.lag_grid_ms()
LAG_300 = 300.0


def language_electrodes() -> pd.DataFrame:
    tax = pd.read_csv(
        ap.TAX_ELECTRODE_TABLE,
        usecols=["subject", "channel", "region", "hemisphere", "is_lang", "MNI_y"],
    )
    lang = tax[tax["is_lang"].fillna(False).astype(bool) & (tax["hemisphere"] == "Left")].copy()
    if len(lang) != 151:
        raise RuntimeError(f"expected 151 left language electrodes, got {len(lang)}")
    counts = lang.groupby("subject").size().to_dict()
    if counts != SUBJECT_N:
        raise RuntimeError(f"subject counts {counts} != {SUBJECT_N}")
    lang["subject_ord"] = lang["subject"].map({s: i for i, s in enumerate(SUBJECTS)})
    lang = lang.sort_values(
        ["subject_ord", "region", "channel"], kind="mergesort"
    ).reset_index(drop=True)
    lang["elec_i"] = np.arange(len(lang))
    lang["labeled_region"] = lang["region"].ne("Unknown")
    return lang


def load_lag_scores() -> pd.DataFrame:
    usecols = [
        "subject", "channel", "condition", "mode", "feature_space", "lag_ms",
        "r_fold0", "r_fold1", "r_fold2", "fisher_z_mean",
        "train_peak_lag_fold0", "train_peak_lag_fold1", "train_peak_lag_fold2",
    ]
    frames = []
    for name in ("tc_both_lag_scores.csv", "tc_wide_lag_scores.csv"):
        df = pd.read_csv(TABLES / name, usecols=usecols)
        df = df[(df["condition"] == "v2") & (df["mode"] == "partial")]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(
        ["subject", "channel", "feature_space", "lag_ms"], keep="last"
    )


def peak_lags(scores: pd.DataFrame, subject: str, channel: str) -> list[float]:
    sub = scores[
        (scores["subject"] == subject)
        & (scores["channel"] == channel)
        & (scores["feature_space"] == "sae")
    ]
    if sub.empty:
        raise RuntimeError(f"no partial SAE lag score for {subject} {channel}")
    row = sub.iloc[-1]
    return [float(row[f"train_peak_lag_fold{i}"]) for i in range(3)]


def stored_fold_r(scores: pd.DataFrame, subject: str, channel: str, space: str, fold: int, lag: float) -> float:
    sub = scores[
        (scores["subject"] == subject)
        & (scores["channel"] == channel)
        & (scores["feature_space"] == space)
        & np.isclose(scores["lag_ms"], lag)
    ]
    if sub.empty:
        raise RuntimeError(f"no {space} score at {lag} ms for {subject} {channel}")
    return float(sub.iloc[-1][f"r_fold{fold}"])


def load_sae_supports(pop: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, float, np.ndarray]]]:
    """Per electrode, three folds: (fold, peak lag ms, ascending feature indices)."""
    parts = TABLES / "tc_selected_indices_partial_parts"
    combined = None
    out = {}
    for row in pop.itertuples(index=False):
        path = parts / f"{row.subject}__{str(row.channel).replace('/', '_')}__v2__partial.csv"
        if path.exists() and path.stat().st_size:
            df = pd.read_csv(path)
        else:
            if combined is None:
                combined = pd.read_csv(TABLES / "tc_selected_indices_partial.csv")
                combined = combined[
                    (combined["condition"] == "v2")
                    & (combined["mode"] == "partial")
                    & (combined["feature_space"] == "sae")
                ]
            df = combined[(combined["subject"] == row.subject) & (combined["channel"] == row.channel)]
        df = df[df["feature_space"] == "sae"] if "feature_space" in df.columns else df
        folds = []
        for fold in range(3):
            sub = df[df["fold"] == fold] if len(df) else df
            if len(sub):
                lag = float(sub["train_peak_lag_ms"].iloc[0])
                if not np.allclose(sub["train_peak_lag_ms"], lag):
                    raise RuntimeError(f"mixed peak lags {row.subject} {row.channel} fold {fold}")
                idx = np.unique(sub["feature_index"].to_numpy(int))
                idx.sort()
            else:
                lag = np.nan
                idx = np.zeros(0, dtype=int)
            folds.append((fold, lag, idx))
        out[(row.subject, row.channel)] = folds
    return out


def lag_index(lag_ms: float) -> int:
    j = int(np.argmin(np.abs(LAGS - lag_ms)))
    if not np.isclose(LAGS[j], lag_ms):
        raise RuntimeError(f"lag {lag_ms} is not on the v3 grid")
    return j


def load_subject_bundle(subject: str):
    chans, sid = tc.load_main_neural(subject)
    if sid is None:
        raise RuntimeError(f"no section ids for {subject}")
    return chans, np.asarray(sid, dtype=int)


def load_design():
    """SAE matrix, surprisal, validity, nuisance on the v3 lag grid, section ids from features."""
    # Section ids come from a neural cache; covariates and features share word order.
    T = tc.load_covariates(LAGS)
    # Feature rows follow sections 1–3. Neural section_id must match.
    X, _R, surp, bv = tc.feature_condition("v2", None)
    if X.shape[0] != T.shape[0]:
        raise RuntimeError(f"SAE rows {X.shape[0]} != nuisance rows {T.shape[0]}")
    return X.tocsr(), np.asarray(surp, dtype=np.float64), np.asarray(bv, dtype=bool), T


def entropy_bits(mass: np.ndarray) -> float:
    m = np.asarray(mass, dtype=float)
    m = m[m > 0]
    if m.size == 0:
        return np.nan
    p = m / m.sum()
    return float(-(p * np.log2(p)).sum())


def count_entropy_cap() -> float:
    return entropy_bits(np.array([SUBJECT_N[s] for s in SUBJECTS], dtype=float))
