#!/usr/bin/env python3
"""v3 lag curves with a speech-timing nuisance (plan: docs/sae-v3-timing-control-plan.md).

Nuisance per word and lag (fixed before any fit): speech coverage in the
200 ms window, word onsets in the window, silence before the word, log IOI to
the previous and next word, log word duration, and the stimulus audio envelope
(mean rectified envelope and mean positive envelope slope in the window).

Two ways of applying it, LOSO only:
  partial : nuisance Ridge fit on training rows, Y residualized, v3 readout
            (training-fold peak search, F-test + LassoCV support, Ridge refit
            at every lag) on the residual. Inner peak-search splits refit the
            nuisance on their own training rows.
  forced  : Y unchanged, v3 support selection unchanged, nuisance columns
            forced beside surprisal in every Ridge stage.

Writes only under group_encoding_results/sparse_encoding_v3/timing_control/.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import core.analysis_paths as ap
import sparse_encoding.sparse_encoding_v3 as v3
from sparse_encoding.sparse_encoding_regression import (
    _fit_predict, _fold_r, _to_dense, compute_joint_support,
)

warnings.filterwarnings("ignore")

ROOT = (ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v3" / "timing_control").resolve()
TABLES = ROOT / "tables"
CACHE = ROOT / "cache"
V3_CACHE = ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v3" / "cache"
AUDIO = ap.DATA_ROOT / "syntax_tree_probe" / "data" / "audio"
FS = v3.FS_TARGET
HALF = 50  # samples, 100 ms at 500 Hz
MAIN_LAGS = v3.lag_grid_ms()                          # -500..2000, the v3 grid
EXT_LAGS = np.arange(-2000.0, 2000.0 + 25.0, 50.0)    # figure-only extension
TAG = v3.FEATURE_TAG
NUIS_NAMES = ("speech_cov", "n_onsets", "silence_before", "log_ioi_prev",
              "log_ioi_next", "log_dur", "env_mean", "env_rise")
NUIS_ALPHAS = np.logspace(-1, 5, 13)
SEED = 0

if ROOT.name != "timing_control" or ROOT.parent.name != "sparse_encoding_v3":
    raise RuntimeError(f"refusing to write outside timing_control: {ROOT}")


# ---------------------------------------------------------------- covariates

def _section_envelope(sid: int, T: int) -> np.ndarray:
    fs, x = wavfile.read(AUDIO / f"task-lppCN_section_{sid}.wav")
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    env = np.abs(x)
    b, a = signal.butter(4, 30.0 / (fs / 2.0))
    env = signal.filtfilt(b, a, env)
    env = signal.resample_poly(env, 5, 441)  # 44100 -> 500 Hz
    env = np.clip(env, 0, None)
    out = np.zeros(T)
    n = min(T, env.size)
    out[:n] = env[:n]
    return out


def build_covariates(lags_ms: np.ndarray) -> np.ndarray:
    """(n_words, n_lags, 8). Windows are the Y windows: [c-50, c+50) samples."""
    per = []
    for sid in v3.SECTIONS:
        wt = pd.read_csv(ap.FEATURES_DIR / f"section_{sid:03d}" / "word_timing.csv")
        on_s = wt["onset_relative"].to_numpy(float)
        off_s = wt["offset_relative"].to_numpy(float)
        on = np.rint(on_s * FS).astype(int)
        off = np.rint(off_s * FS).astype(int)
        T = int(off.max() + 4 * FS)
        speech = np.zeros(T)
        for a, b in zip(on, off):
            speech[a:max(b, a + 1)] = 1.0
        onset_imp = np.zeros(T)
        np.add.at(onset_imp, on, 1.0)
        env = _section_envelope(sid, T)
        env = (env - env.mean()) / (env.std() + 1e-12)
        rise = np.clip(np.gradient(env) * FS, 0, None)
        rise = rise / (rise.std() + 1e-12)
        cs = {k: np.r_[0.0, np.cumsum(v)] for k, v in
              (("speech", speech), ("onset", onset_imp), ("env", env), ("rise", rise))}
        sil = np.r_[1.0, np.clip(on_s[1:] - off_s[:-1], 0, None)]
        ioi_p = np.log(np.r_[1.0, np.diff(on_s)] + 0.01)
        ioi_n = np.log(np.r_[np.diff(on_s), 1.0] + 0.01)
        dur = np.log(off_s - on_s + 0.01)
        cols = []
        for L in lags_ms:
            c = on + int(round(L / 1000.0 * FS))
            s = np.clip(c - HALF, 0, T)
            e = np.clip(c + HALF, 0, T)
            w = np.maximum(e - s, 1)
            m = {k: (v[e] - v[s]) / w for k, v in cs.items()}
            cols.append(np.column_stack([
                m["speech"], m["onset"] * w, sil, ioi_p, ioi_n, dur, m["env"], m["rise"],
            ]))
        per.append(np.stack(cols, axis=1))
    return np.concatenate(per, axis=0).astype(np.float64)


def load_covariates(lags_ms: np.ndarray) -> np.ndarray:
    path = CACHE / "nuisance_ext_grid.npy"
    if path.exists():
        full = np.load(path)
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        full = build_covariates(EXT_LAGS)
        np.save(path, full)
    idx = [int(np.argmin(np.abs(EXT_LAGS - l))) for l in lags_ms]
    if not np.allclose(EXT_LAGS[idx], lags_ms):
        raise RuntimeError("lag not on the extended grid")
    return full[:, idx, :]


# ---------------------------------------------------------------- electrodes / neural

def electrode_table() -> pd.DataFrame:
    tax = pd.read_csv(ap.TAX_ELECTRODE_TABLE)
    lang = tax["is_lang"].fillna(False).astype(bool)
    glove = tax["sig_glove"].fillna(False).astype(bool)
    out = tax.loc[lang | glove, ["subject", "channel"]].copy()
    out["is_lang"] = lang[lang | glove].to_numpy()
    out["sig_glove"] = glove[lang | glove].to_numpy()
    out["both"] = out["is_lang"] & out["sig_glove"]
    return out.reset_index(drop=True)


def load_main_neural(subject: str, cache_suffix: str | None = None):
    """Channel -> (Y, window_ok) on MAIN_LAGS from the v3 caches.

    The default reads the raw pair ``""`` and ``_sig_glove``. ``cache_suffix``
    replaces that pair with one token, for example ``_trfresid_m200``.
    """
    chans = {}
    sid = None
    suffixes = ("", "_sig_glove") if cache_suffix is None else (cache_suffix,)
    for suffix in suffixes:
        p = V3_CACHE / f"{subject}_n51_a-500.0_b2000.0_sh0_m0{suffix}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            if "lags_ms" in z.files and not np.allclose(z["lags_ms"], MAIN_LAGS):
                raise RuntimeError(f"{p.name} lags are not the v3 grid")
            if sid is None:
                sid = z["section_id"].astype(int)
            elif not np.array_equal(sid, z["section_id"]):
                raise RuntimeError("section ids differ")
            wok = np.array(z["window_ok"])
            for j, ch in enumerate(z["channels"]):
                chans.setdefault(str(ch), (z["Y"][:, j, :].astype(np.float32), wok))
    if cache_suffix is not None and not chans:
        raise RuntimeError(f"no neural cache for {subject} suffix {cache_suffix}")
    return chans, sid


def extended_neural(subject: str, channels):
    """Y on the lags in EXT_LAGS below -500 ms, computed from EEG, cached."""
    path = CACHE / f"{subject}_ext_neg.npz"
    lags = EXT_LAGS[EXT_LAGS < MAIN_LAGS[0] - 1e-9]
    if path.exists():
        with np.load(path) as z:
            names = [str(c) for c in z["channels"]]
            if set(channels) <= set(names):
                return {c: (z["Y"][:, names.index(c), :], z["window_ok"]) for c in channels}, lags
    import core.encoding_channel as ec
    from sparse_encoding.sparse_encoding_regression import get_subjects
    eeg = {sid: str(ap.DATA_ROOT / p) for sid, p in get_subjects()[subject]["eeg_files"].items()} \
        if "eeg_files" in get_subjects()[subject] else \
        {sid: str(ap.DATA_ROOT / p) for sid, p in get_subjects()[subject].items()}
    data = ec.load_subject_data(subject_name=subject, eeg_files=eeg, features_dir=ap.FEATURES_DIR,
                                fs_target=FS, sections=v3.SECTIONS, feature_sets=["glove"])
    labels = [str(c).strip() for c in data.channel_labels]
    idx = [labels.index(c) for c in channels]
    Y, ok = v3._stack_Y(data, idx, lags)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Y=Y, window_ok=ok, channels=np.asarray(channels, dtype="U64"))
    return {c: (Y[:, i, :], ok) for i, c in enumerate(channels)}, lags


# ---------------------------------------------------------------- features

def prev_index(section_id):
    prev = np.arange(section_id.size) - 1
    prev[0] = -1
    prev[1:][section_id[1:] != section_id[:-1]] = -1
    return prev


def feature_condition(name: str, section_id):
    X, R, s, bv = v3._load_features(TAG)
    if name == "v2":
        return X, R, s, bv
    if name == "v2_shuffle":
        rng = np.random.default_rng(SEED)
        perm = np.arange(X.shape[0])
        for sec in np.unique(section_id):
            idx = np.flatnonzero((section_id == sec) & bv)
            perm[idx] = idx[rng.permutation(idx.size)]
        return X[perm], R[perm], s[perm], bv
    if name == "v2_prevproj":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preonset_refit", ROOT.parent / "diagnostics" / "preonset" / "preonset_refit.py")
        pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
        prev = prev_index(section_id)
        return (pr.project_out_prev_sparse(X, prev), pr.project_out_prev_dense(R, prev, bv),
                pr.regress_out_prev_scalar(s, prev, bv), bv)
    raise ValueError(name)


# ---------------------------------------------------------------- readout

def _nuis_model(Tl, y, rows):
    sc = StandardScaler().fit(Tl[rows])
    m = RidgeCV(alphas=NUIS_ALPHAS).fit(sc.transform(Tl[rows]), y[rows])
    return lambda ii: m.predict(sc.transform(Tl[ii]))


def _residualize(y, Tl, fit_rows, rows):
    out = np.full(y.shape, np.nan)
    f = _nuis_model(Tl, y, fit_rows)
    out[rows] = y[rows] - f(rows)
    return out, f


def _forced_design(X, sup, surp, Tl, tr, te):
    sc = StandardScaler().fit(surp[tr].reshape(-1, 1))
    st = StandardScaler().fit(Tl[tr])
    def D(ii):
        parts = [sc.transform(surp[ii].reshape(-1, 1)), st.transform(Tl[ii])]
        if X is not None and sup is not None and sup.any():
            parts.insert(0, _to_dense(X[ii][:, sup]))
        return np.hstack(parts)
    return D(tr), D(te)


def _support(X, surp, y, tr, dense):
    sc = StandardScaler().fit(surp[tr].reshape(-1, 1))
    try:
        return compute_joint_support(X[tr], sc.transform(surp[tr].reshape(-1, 1)).ravel(), y[tr], dense=dense)
    except ValueError:
        return np.zeros(X.shape[1], dtype=bool)


def _score_forced(X, sup, surp, Tl, y, tr, te):
    A, B = _forced_design(X, sup, surp, Tl, tr, te)
    try:
        pred, _ = _fit_predict(A, y[tr], B)
    except ValueError:
        return 0.0
    return _fold_r(pred, y[te], True)


def _rows(idx, y, ok, bv):
    idx = np.asarray(idx, dtype=int)
    return idx[bv[idx] & ok[idx] & np.isfinite(y[idx])]


def fit_space(X, surp, Y, OK, bv, sid, T, lags, search, mode, dense):
    """One feature space (X None = surprisal-only). Returns per-lag fold r, peaks, n_sup, indices.

    Index records are the LASSO support chosen at the training-fold peak. Scoring is unchanged.
    """
    n_lags = len(lags)
    fold_r = np.zeros((3, n_lags))
    peaks, nsups, support_recs = [], [], []
    for f, (tr, te, held) in enumerate(v3.loso_outer_splits(sid)):
        if X is None:
            peak_i = int(np.flatnonzero(search)[v3.argmax_lag(np.zeros(search.sum()), lags[search])])
        else:
            inners = v3.loso_inner_splits(tr, sid)
            z = np.full(n_lags, -np.inf)
            for li in np.flatnonzero(search):
                y = Y[:, li].astype(np.float64)
                rs = []
                for itr, ite in inners:
                    a, b = _rows(itr, y, OK[:, li], bv), _rows(ite, y, OK[:, li], bv)
                    if a.size < 20 or b.size < 5:
                        rs.append(0.0); continue
                    if mode == "partial":
                        yr, _ = _residualize(y, T[:, li, :], a, np.r_[a, b])
                        r, _p, _s = v3.predict_full(X, surp, yr, a, b, dense=dense)
                    elif mode == "raw":
                        r, _p, _s = v3.predict_full(X, surp, y, a, b, dense=dense)
                    else:
                        sup = _support(X, surp, y, a, dense)
                        r = _score_forced(X, sup, surp, T[:, li, :], y, a, b)
                    rs.append(r)
                z[li] = v3.fisher_z_mean(rs)
            zs = z[search]
            peak_i = int(np.flatnonzero(search)[v3.argmax_lag(zs, lags[search])])
        peaks.append(float(lags[peak_i]))
        sup = None
        if X is not None:
            y = Y[:, peak_i].astype(np.float64)
            a = _rows(tr, y, OK[:, peak_i], bv)
            yy = _residualize(y, T[:, peak_i, :], a, a)[0] if mode == "partial" else y
            sup = _support(X, surp, yy, a, dense)
            for feat_i in np.flatnonzero(sup):
                support_recs.append((int(f), int(held), float(lags[peak_i]), int(feat_i)))
        nsups.append(int(sup.sum()) if sup is not None else 0)
        for li in range(n_lags):
            y = Y[:, li].astype(np.float64)
            a, b = _rows(tr, y, OK[:, li], bv), _rows(te, y, OK[:, li], bv)
            if a.size < 20 or b.size < 5:
                continue
            if mode == "partial":
                yr, _ = _residualize(y, T[:, li, :], a, np.r_[a, b])
                if X is None:
                    r, _p = v3.predict_surprisal(surp, yr, a, b)
                else:
                    r, _p = v3.predict_with_support(X, surp, yr, a, b, sup)
            elif mode == "raw":
                if X is None:
                    r, _p = v3.predict_surprisal(surp, y, a, b)
                else:
                    r, _p = v3.predict_with_support(X, surp, y, a, b, sup)
            else:
                r = _score_forced(X, sup, surp, T[:, li, :], y, a, b)
            fold_r[f, li] = r
    return fold_r, peaks, nsups, support_recs


def timing_only(Y, OK, bv, sid, T):
    fold_r = np.zeros((3, Y.shape[1]))
    for f, (tr, te, _h) in enumerate(v3.loso_outer_splits(sid)):
        for li in range(Y.shape[1]):
            y = Y[:, li].astype(np.float64)
            a, b = _rows(tr, y, OK[:, li], bv), _rows(te, y, OK[:, li], bv)
            if a.size < 20 or b.size < 5:
                continue
            fold_r[f, li] = _fold_r(_nuis_model(T[:, li, :], y, a)(b), y[b], True)
    return fold_r


def _emit(rows, meta, space, lags, fold_r, peaks=None, nsups=None):
    for li, lag in enumerate(lags):
        rs = fold_r[:, li]
        d = dict(meta)
        d.update({"feature_space": space, "lag_ms": float(lag),
                  "in_v3_grid": bool(np.any(np.isclose(MAIN_LAGS, lag))),
                  "r_fold0": rs[0], "r_fold1": rs[1], "r_fold2": rs[2],
                  "fisher_z_mean": v3.fisher_z_mean(list(rs))})
        if peaks is not None:
            d.update({f"train_peak_lag_fold{i}": p for i, p in enumerate(peaks)})
            d["n_support_mean"] = float(np.mean(nsups))
        rows.append(d)


INDEX_COLS = [
    "subject", "channel", "cv", "feature_space", "fold", "heldout_section",
    "train_peak_lag_ms", "feature_index", "bin_name", "shift_id", "null_draw",
    "feature_tag", "condition", "mode",
]
INDEX_PARTS = TABLES / "tc_selected_indices_partial_parts"
INDEX_COMBINED = TABLES / "tc_selected_indices_partial.csv"


def _index_part_path(subject, channel, condition, mode) -> Path:
    safe = str(channel).replace("/", "_")
    return INDEX_PARTS / f"{subject}__{safe}__{condition}__{mode}.csv"


def _write_index_part(meta, recs) -> Path:
    """One file per electrode. A finished file is replaced, never appended."""
    INDEX_PARTS.mkdir(parents=True, exist_ok=True)
    part = _index_part_path(meta["subject"], meta["channel"], meta["condition"], meta["mode"])
    frame = pd.DataFrame(recs, columns=INDEX_COLS)
    tmp = part.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(part)
    return part


def _concat_index_parts() -> None:
    parts = sorted(INDEX_PARTS.glob("*__v2__partial.csv"))
    frames = [pd.read_csv(p) for p in parts if p.stat().st_size]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INDEX_COLS)
    INDEX_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_COMBINED.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(INDEX_COMBINED)


def fit_task(task):
    t0 = time.time()
    lags = task["lags"]; search = np.array([np.any(np.isclose(MAIN_LAGS, l)) for l in lags])
    Y, OK, bv, sid, T = task["Y"], task["OK"], task["bv"], task["sid"], task["T"]
    meta = {k: task[k] for k in ("subject", "channel", "condition", "mode")}
    rows, recs = [], []
    for space, X, dense in (("sae", task["X_sae"], False), ("residual", task["X_resid"], True),
                            ("surprisal", None, False)):
        fr, pk, ns, sup_recs = fit_space(X, task["surp"], Y, OK, bv, sid, T, lags, search, task["mode"], dense)
        _emit(rows, meta, space, lags, fr, pk, ns)
        for fold, held, peak_ms, feat_i in sup_recs:
            recs.append({
                "subject": meta["subject"], "channel": meta["channel"], "cv": "loso",
                "feature_space": space, "fold": fold, "heldout_section": held,
                "train_peak_lag_ms": peak_ms, "feature_index": feat_i, "bin_name": "",
                "shift_id": -1, "null_draw": -1, "feature_tag": TAG,
                "condition": meta["condition"], "mode": meta["mode"],
            })
    if task.get("timing_only"):
        _emit(rows, dict(meta, mode="timing_only"), "timing_only", lags, timing_only(Y, OK, bv, sid, T))
    return pd.DataFrame(rows), time.time() - t0, meta, recs


V3_SUPPORTS = ap.DATA_ROOT / "group_encoding_results" / "sparse_encoding_v3" / "tables" / "v3_selected_indices_loso_parts"


def v3_supports(subject, channel, n_feat_sae, n_feat_res, v3_nsup):
    """Per held-out section, the v3 LOSO supports (chosen at the v3 training-fold peak lag)."""
    p = V3_SUPPORTS / f"{subject}__{channel}__sh-1__n-1__main.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["feature_space", "heldout_section", "feature_index", "train_peak_lag_ms"])
    out = {}
    for space, n in (("sae", n_feat_sae), ("residual", n_feat_res)):
        for held in v3.SECTIONS:
            m = np.zeros(n, dtype=bool)
            idx = df[(df.feature_space == space) & (df.heldout_section == held)].feature_index.to_numpy(int)
            m[idx] = True
            want = v3_nsup.get((space, held))
            if want is not None and int(m.sum()) != int(want):
                raise RuntimeError(f"{subject} {channel} {space} held {held}: support {m.sum()} != v3 {want}")
            out[(space, held)] = m
    return out


def forced_task(task):
    """Forced-in: v3 supports and Y unchanged, nuisance forced beside surprisal in the Ridge."""
    t0 = time.time()
    Y, OK, bv, sid, T, lags = task["Y"], task["OK"], task["bv"], task["sid"], task["T"], task["lags"]
    sup = task["supports"]
    meta = {k: task[k] for k in ("subject", "channel", "condition", "mode")}
    rows = []
    for space, X in (("sae", task["X_sae"]), ("residual", task["X_resid"]), ("surprisal", None)):
        fr = np.zeros((3, len(lags)))
        for f, (tr, te, held) in enumerate(v3.loso_outer_splits(sid)):
            s = None if X is None else sup[(space, held)]
            for li in range(len(lags)):
                y = Y[:, li].astype(np.float64)
                a, b = _rows(tr, y, OK[:, li], bv), _rows(te, y, OK[:, li], bv)
                if a.size < 20 or b.size < 5 or np.std(y[a]) < 1e-12:
                    continue
                fr[f, li] = _score_forced(X, s, task["surp"], T[:, li, :], y, a, b)
        _emit(rows, meta, space, lags, fr)
    return pd.DataFrame(rows), time.time() - t0, meta


def glove_task(subject, channel, Y, OK, sid, lags, T, G, gvalid):
    """Dense GloVe-300 Ridge (Goldstein analogue): alone, partial, forced."""
    rows = []
    splits = v3.loso_outer_splits(sid)
    out = {m: np.zeros((3, len(lags))) for m in ("raw", "partial", "forced")}
    for f, (tr, te, _h) in enumerate(splits):
        for li in range(len(lags)):
            y = Y[:, li].astype(np.float64)
            a, b = _rows(tr, y, OK[:, li], gvalid), _rows(te, y, OK[:, li], gvalid)
            if a.size < 20 or b.size < 5:
                continue
            Tl = T[:, li, :]
            def ridge(D, yy):
                sc = StandardScaler().fit(D[a])
                m = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit(sc.transform(D[a]), yy[a])
                return m.predict(sc.transform(D[b]))
            out["raw"][f, li] = _fold_r(ridge(G, y), y[b], True)
            yr, _ = _residualize(y, Tl, a, np.r_[a, b])
            out["partial"][f, li] = _fold_r(ridge(G, yr), yr[b], True)
            out["forced"][f, li] = _fold_r(ridge(np.hstack([G, Tl]), y), y[b], True)
    for m, fr in out.items():
        _emit(rows, {"subject": subject, "channel": channel, "condition": "glove300", "mode": m},
              "glove", lags, fr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver

def _append(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def _done(path: Path, cols, n_rows: int = 0):
    """Keys already written. With ``n_rows``, only keys with at least that many rows
    (a preempted append can leave a partial group; it is refit, and readers keep the last copy)."""
    if not path.exists():
        return set()
    d = pd.read_csv(path, usecols=cols).astype(str)
    if n_rows:
        ct = d.groupby(cols).size()
        return set(ct[ct >= n_rows].index)
    return set(map(tuple, d.drop_duplicates().to_numpy()))


def run_forced(el, args):
    from joblib import Parallel, delayed
    if args.max_electrodes:
        el = el.head(args.max_electrodes)
    v3t = pd.read_csv(ROOT.parent / "tables" / "v3_loso_lag_scores.csv")
    v3t = v3t[(v3t.readout == "matched_lasso") & (v3t.lag_index == 0) & v3t.feature_space.isin(["sae", "residual"])]
    v3t = v3t.drop_duplicates(["subject", "channel", "feature_space"], keep="last")
    nsup = {}
    for r in v3t.itertuples(index=False):
        for i in range(3):
            nsup[(r.subject, r.channel, r.feature_space, int(getattr(r, f"fold_section{i}")))] = \
                int(getattr(r, f"n_support_fold{i}"))
    X, R, s, bv = feature_condition("v2", None)
    out = TABLES / "tc_forced_lag_scores.csv"
    done = _done(out, ["subject", "channel"])
    T_ext, T_main = load_covariates(EXT_LAGS), load_covariates(MAIN_LAGS)
    tasks = []
    for subj, g in el.groupby("subject"):
        chans, sid = load_main_neural(subj)
        both_ch = list(g[g.both].channel)
        extd = extended_neural(subj, both_ch)[0] if both_ch else {}
        for row in g.itertuples(index=False):
            if (subj, row.channel) in done:
                continue
            Ym, OKm = chans[row.channel]
            if row.both:
                Ye, OKe = extd[row.channel]
                Y, OK, lags, T = np.concatenate([Ye, Ym], 1), np.concatenate([OKe, OKm], 1), EXT_LAGS, T_ext
            else:
                Y, OK, lags, T = Ym, OKm, MAIN_LAGS, T_main
            want = {(sp, h): nsup[(subj, row.channel, sp, h)] for sp in ("sae", "residual") for h in v3.SECTIONS
                    if (subj, row.channel, sp, h) in nsup}
            if len(want) != 6:
                raise RuntimeError(f"{subj} {row.channel}: v3 LOSO support counts missing")
            tasks.append({"subject": subj, "channel": row.channel, "condition": "v2", "mode": "forced",
                          "Y": Y, "OK": OK, "bv": bv, "sid": sid, "T": T, "lags": lags, "surp": s,
                          "X_sae": X, "X_resid": R,
                          "supports": v3_supports(subj, row.channel, X.shape[1], R.shape[1], want)})
    print(f"[tc] forced (v3 supports): {len(tasks)} electrodes ({len(done)} done)", flush=True)
    t0 = time.time()
    gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
        delayed(forced_task)(t) for t in tasks)
    for k, (df, el_s, meta) in enumerate(gen, 1):
        _append(out, df)
        print(f"  forced {k}/{len(tasks)} {meta['subject']} {meta['channel']} {el_s:.0f}s total {time.time() - t0:.0f}s", flush=True)


def run_partial_indices(el, args):
    """Refit v2 partial for is_lang ∪ sig_glove and save the LASSO supports.

    Scores already on disk do not contain feature indices, so those electrodes
    are refit. The support is the one chosen inside ``fit_space`` (partial
    residualization, training-fold peak on the v3 grid, then LASSO). Existing
    score CSVs and the ``sig_glove`` column are not written.
    """
    from joblib import Parallel, delayed
    if int((el.is_lang | el.sig_glove).sum()) != 359 or len(el) != 359:
        raise RuntimeError(f"expected 359 is_lang ∪ sig_glove electrodes, got {len(el)}")
    if args.max_electrodes:
        el = el.head(args.max_electrodes)
    X, R, s, bv = feature_condition("v2", None)
    T_ext, T_main = load_covariates(EXT_LAGS), load_covariates(MAIN_LAGS)
    neural, sid = {}, None
    for subj, g in el.groupby("subject"):
        chans, s_id = load_main_neural(subj)
        sid = s_id if sid is None else sid
        if not np.array_equal(sid, s_id):
            raise RuntimeError("section ids differ")
        both_ch = list(g[g.both].channel)
        extd = extended_neural(subj, both_ch)[0] if both_ch else {}
        for row in g.itertuples(index=False):
            Ym, OKm = chans[row.channel]
            if row.both:
                Ye, OKe = extd[row.channel]
                neural[(subj, row.channel)] = (
                    np.concatenate([Ye, Ym], 1), np.concatenate([OKe, OKm], 1), EXT_LAGS, T_ext)
            else:
                neural[(subj, row.channel)] = (Ym, OKm, MAIN_LAGS, T_main)
    tasks, n_skip = [], 0
    for (subj, ch), (Y, OK, lags, T) in neural.items():
        part = _index_part_path(subj, ch, "v2", "partial")
        if part.exists() and part.stat().st_size > 0:
            n_skip += 1
            continue
        tasks.append({"subject": subj, "channel": ch, "condition": "v2", "mode": "partial",
                      "Y": Y, "OK": OK, "bv": bv, "sid": sid, "T": T, "lags": lags, "surp": s,
                      "X_sae": X, "X_resid": R, "timing_only": False})
    print(f"[tc] partial indices: {len(tasks)} refits ({n_skip} already saved) -> {INDEX_PARTS}", flush=True)
    if tasks:
        t0 = time.time()
        gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
            delayed(fit_task)(t) for t in tasks)
        for k, (df, el_s, meta, recs) in enumerate(gen, 1):
            del df
            _write_index_part(meta, recs)
            print(f"  indices {k}/{len(tasks)} {meta['subject']} {meta['channel']} "
                  f"n_idx={len(recs)} {el_s:.0f}s total {time.time() - t0:.0f}s", flush=True)
    _concat_index_parts()
    print(f"[tc] wrote {INDEX_COMBINED}", flush=True)


PROTECTED_TABLES = {
    "tc_both_lag_scores.csv",
    "tc_wide_lag_scores.csv",
    "tc_forced_lag_scores.csv",
    "tc_glove_both_lag_scores.csv",
    "sig_glove_timing.csv",
    "tc_selected_indices_partial.csv",
}


def _stage_jobs(stage: str, spec: str):
    """Default job list, or an explicit ``condition:mode`` list.

    An empty spec keeps the historical defaults. A non-empty spec is that list
    in order, so ``raw`` can be requested without changing the defaults.
    """
    default = [("v2", "partial"), ("v2", "forced")]
    if stage == "both":
        default += [("v2_shuffle", "partial"), ("v2_shuffle", "forced"), ("v2_prevproj", "partial")]
    if not spec:
        return default
    allowed_c = {"v2", "v2_shuffle", "v2_prevproj"}
    allowed_m = {"partial", "forced", "raw"}
    jobs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cond, mode = item.split(":")
        if cond not in allowed_c or mode not in allowed_m:
            raise ValueError(f"unknown job {item}")
        jobs.append((cond, mode))
    return jobs


def main():
    from joblib import Parallel, delayed
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["covariates", "both", "glove", "wide", "forced", "indices", "sig_glove"], required=True)
    p.add_argument("--n_jobs", type=int, default=20)
    p.add_argument("--max_electrodes", type=int, default=0)
    p.add_argument("--jobs", default="")  # comma list condition:mode filter
    p.add_argument("--cache-suffix", default=None,
                   help="Single v3 cache suffix, e.g. _trfresid_m200. Default keeps '' and _sig_glove.")
    p.add_argument("--out", default="",
                   help="CSV file name under tables/. Required with --cache-suffix. Refuses the finished tables.")
    args = p.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    if args.cache_suffix and not args.out:
        raise RuntimeError("--out is required with --cache-suffix so the finished tables are not appended")
    if args.stage == "sig_glove" and not args.out:
        raise RuntimeError("--out is required for stage sig_glove")
    if args.out and Path(args.out).name in PROTECTED_TABLES:
        raise RuntimeError(f"refusing to write {Path(args.out).name}")

    if args.stage == "covariates":
        T = load_covariates(EXT_LAGS)
        old = np.load(ROOT.parent / "diagnostics" / "preonset" / "timing_covariates_full_grid.npy")
        Tm = load_covariates(MAIN_LAGS)
        for k in range(6):
            print(f"  {NUIS_NAMES[k]:15s} corr with preonset grid: "
                  f"{np.corrcoef(Tm[:, :, k].ravel(), old[:, :, k].ravel())[0, 1]:.4f}")
        print("nuisance", T.shape, "finite", np.isfinite(T).all())
        return

    el = electrode_table()
    if args.stage == "indices":
        return run_partial_indices(el, args)
    if args.stage == "forced":
        return run_forced(el, args)
    if args.stage in ("both", "glove"):
        el = el[el.both]
    elif args.stage == "sig_glove":
        el = el[el.sig_glove]
    else:
        el = el[~el.both]
    if args.max_electrodes:
        el = el.head(args.max_electrodes)
    # A non-default cache is the n51 grid only. Do not prepend raw-EEG lags.
    ext = args.stage in ("both", "glove") and args.cache_suffix is None
    lags = EXT_LAGS if ext else MAIN_LAGS
    T = load_covariates(lags)

    neural, sid = {}, None
    skipped = []
    for subj, g in el.groupby("subject"):
        chans, s = load_main_neural(subj, args.cache_suffix)
        sid = s if sid is None else sid
        if not np.array_equal(sid, s):
            raise RuntimeError("section ids differ")
        present = [ch for ch in g.channel if ch in chans]
        if args.cache_suffix is None and len(present) != len(g):
            missing = [ch for ch in g.channel if ch not in chans]
            raise KeyError(f"{subj} missing from v3 cache: {missing[:5]}")
        skipped.extend((subj, ch) for ch in g.channel if ch not in chans)
        if ext:
            extd, ext_lags = extended_neural(subj, list(present))
        for ch in present:
            Ym, OKm = chans[ch]
            if ext:
                Ye, OKe = extd[ch]
                neural[(subj, ch)] = (np.concatenate([Ye, Ym], 1), np.concatenate([OKe, OKm], 1))
            else:
                neural[(subj, ch)] = (Ym, OKm)
    if skipped and args.stage == "sig_glove":
        raise RuntimeError(f"sig_glove cache is missing {len(skipped)} electrodes, e.g. {skipped[:3]}")
    if skipped:
        print(f"[tc] skipped {len(skipped)} electrodes absent from cache suffix {args.cache_suffix}", flush=True)
    print(f"[tc] stage={args.stage} electrodes={len(neural)} lags={len(lags)} ({lags[0]}..{lags[-1]})", flush=True)

    if args.stage == "glove":
        G, gv = [], []
        for s_ in v3.SECTIONS:
            G.append(np.load(ap.FEATURES_DIR / f"section_{s_:03d}" / "X_word_glove.npy"))
        G = np.vstack(G).astype(np.float64)
        _X, _R, _s, bv = v3._load_features(TAG)
        gvalid = (np.abs(G).sum(1) > 0) & bv
        out = TABLES / "tc_glove_both_lag_scores.csv"
        if out.exists():
            out.unlink()
        gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(glove_task)(s_, c, Y, OK, sid, lags, T, G, gvalid) for (s_, c), (Y, OK) in neural.items())
        for k, df in enumerate(gen, 1):
            _append(out, df); print(f"  glove {k}/{len(neural)}", flush=True)
        return

    jobs = _stage_jobs(args.stage, args.jobs)
    if args.out:
        out = TABLES / Path(args.out).name
    else:
        out = TABLES / ("tc_both_lag_scores.csv" if args.stage == "both" else "tc_wide_lag_scores.csv")
    done = _done(out, ["subject", "channel", "condition", "mode"], n_rows=3 * len(lags))
    feats = {c: feature_condition(c, sid) for c in sorted({j[0] for j in jobs})}
    tasks = []
    for cond, mode in jobs:
        X, R, s, bv = feats[cond]
        for (subj, ch), (Y, OK) in neural.items():
            if (subj, ch, cond, mode) in done:
                continue
            tasks.append({"subject": subj, "channel": ch, "condition": cond, "mode": mode,
                          "Y": Y, "OK": OK, "bv": bv, "sid": sid, "T": T, "lags": lags,
                          "surp": s, "X_sae": X, "X_resid": R,
                          "timing_only": cond == "v2" and mode == "partial"})
    print(f"[tc] {len(tasks)} fits ({len(done)} done) -> {out.name}", flush=True)
    t0 = time.time()
    gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", max_nbytes="1M")(
        delayed(fit_task)(t) for t in tasks)
    for k, (df, el_s, meta, recs) in enumerate(gen, 1):
        _append(out, df)
        if meta["mode"] == "partial" and not args.out:
            _write_index_part(meta, recs)
        print(f"  {k}/{len(tasks)} {meta['condition']}:{meta['mode']} {meta['subject']} {meta['channel']} "
              f"{el_s:.0f}s total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
