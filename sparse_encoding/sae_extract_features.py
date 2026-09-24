#!/usr/bin/env python3
"""
sae_extract_features.py
=======================
Extract **interpretable sparse features** (SAE latents) and **word surprisal**
from an LM (default: gemma-2-2b + Matryoshka SAE, matching Lepori et al.),
aligned to the word onsets in
``extracted_linguistic_features/section_00X/word_timing.csv``.

This implements the feature side of "Augmented Sparse Encoding Models"
    (Lepori, Kay & Tuckute; https://github.com/mlepori1/Interpretable_Encoding_Models)
adapted to word-locked iEEG on running Chinese speech.

Token → word alignment (repaired)
---------------------------------
Default (``unique_final_word``): each LM token has **exactly one owner**, the
final overlapping segmented word (the word containing the token's last
character). Tokens are never shared across adjacent word rows. Words with no
owned tokens are marked invalid and excluded downstream (no neighbor
duplication, no surprisal imputation).

``--ownership shared_char_weighted`` (v2): a token that overlaps several words
contributes to each of them, weighted by the fraction of the token's characters
inside that word. The word vector is the weight-normalized mean of those SAE
activations (still SAE-then-mean). Outputs use a ``_v2`` feature tag so v1
files are left in place.

Aggregation order matches Lepori: encode each token residual with the SAE
first, then average SAE activations over tokens owned by a word. Surprisal is
the mean negative log-probability across owned tokens with finite NLL.

Default backbone (paper-matched)
-------------------------------
- LM: ``google/gemma-2-2b`` for residual stream + word surprisal.
- SAE: sae_lens release ``gemma-2-2b-res-matryoshka-dc``
  (``chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40``).
- Layer: ``blocks.12.hook_resid_post`` / hook ``model.layers.12`` (paper default).
- Feature tag: ``sae_gemma2_2b_mat_l12``.

Chinese-friendly override (Qwen3)::

    python sae_extract_features.py --preset qwen3_8b_l18

Hierarchical Qwen SAE (Chanin Matryoshka, prefixes 2k / 16k / 65k)::

    python sae_extract_features.py --preset qwen35_4b_mat_l15 --device cuda

Requires a transformers build that knows ``qwen3_5`` (5.x). The encoding
env (4.57) cannot load Qwen3.5; use the extract venv documented in
``run_qwen35_matryoshka_chain.sh``.

Outputs per section (tagged so models do not overwrite each other)
-----------------------------------------------------------------
- ``X_word_<tag>.npz``                 sparse latents (n_words × d_sae)
- ``X_word_<tag>_feature_names.txt``
- ``X_word_<tag>_surprisal.npy``        (n_words × 1)
- ``X_word_<tag>_surprisal_feature_names.txt``
- ``X_word_<tag>_valid.npy``            bool mask (n_words,)
- ``X_word_<tag>_resid.npy``            dense residual (n_words × d_model),
                                       same owned-token mean + valid mask
- ``X_word_<tag>_meta.json``            coverage / L0 / imputation provenance

``--resid_only`` re-runs the LM forward pass, writes ``_resid.npy``, and leaves
existing SAE latents / surprisal / valid masks untouched.

A CPU-only alignment check (no model weights) is available:
    python sae_extract_features.py --dry_run_alignment
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import core.analysis_paths as ap

# ======================================================================
# CONFIGURATION  (edit here or override on the CLI)
# ======================================================================

MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-2-2b-res-matryoshka-dc"
SAE_ID = "blocks.12.hook_resid_post"
SAE_LAYER = 12
FEATURE_TAG = "sae_gemma2_2b_mat_l12"

# Named backbones. ``--preset`` fills model / SAE / layer / tag.
# Qwen3.5-4B-Base L15 is the hosted hierarchical Qwen SAE (not Qwen3-8B).
PRESETS: Dict[str, Dict[str, Any]] = {
    "gemma2_2b_mat_l12": dict(
        model_name="google/gemma-2-2b",
        sae_release="gemma-2-2b-res-matryoshka-dc",
        sae_id="blocks.12.hook_resid_post",
        sae_layer=12,
        feature_tag="sae_gemma2_2b_mat_l12",
        trust_remote_code=False,
        matryoshka_widths=(128, 512, 2048, 8192, 32768),
    ),
    "qwen3_8b_l18": dict(
        model_name="Qwen/Qwen3-8B",
        sae_release="qwen-scope-3-8b-base-w64k-l100",
        sae_id="layer18",
        sae_layer=18,
        feature_tag="sae_qwen3_8b_l18",
        trust_remote_code=False,
        matryoshka_widths=None,
    ),
    "qwen35_4b_mat_l15": dict(
        model_name="Qwen/Qwen3.5-4B-Base",
        sae_release="decoderesearch/qwen-3.5-saes",
        sae_id="qwen-3.5-4b-base/btk-mat-layer-15-k-100",
        sae_layer=15,
        feature_tag="sae_qwen35_4b_mat_l15",
        trust_remote_code=True,
        matryoshka_widths=(2048, 16384, 65536),
    ),
}

# Word-level reduction after SAE encoding of owned tokens.
AGGREGATION = "mean"  # Lepori default: mean over owned-token SAE activations

CONTEXT_LEN = 512
STRIDE = 384
SCALE_BY_DECODER = False

SECTIONS = (1, 2, 3)
DEVICE = "cuda"
DTYPE = "bfloat16"
WORD_COL = "word"

# Default keeps the repaired unique-owner rule. v2 shares character mass.
OWNERSHIP_UNIQUE = "unique_final_word"
OWNERSHIP_SHARED = "shared_char_weighted"
OWNERSHIP_CHOICES = (OWNERSHIP_UNIQUE, OWNERSHIP_SHARED)
OWNERSHIP_META = {
    OWNERSHIP_UNIQUE: "unique_final_overlapping_word",
    OWNERSHIP_SHARED: "shared_char_weighted",
}


def feature_tag_for_ownership(tag: str, ownership: str) -> str:
    """v2 tags end in ``_v2`` so shared-ownership extracts cannot replace v1."""
    tag = str(tag)
    if ownership == OWNERSHIP_SHARED and not tag.endswith("_v2"):
        return f"{tag}_v2"
    return tag


# ======================================================================
# Word / token alignment  (CPU-only, no model needed)
# ======================================================================

@dataclass
class SectionText:
    text: str
    word_spans: List[Tuple[int, int]]
    n_words: int


def build_section_text(word_timing, word_col: str = WORD_COL) -> SectionText:
    """Concatenate words into one string, recording per-word char spans."""
    spans: List[Tuple[int, int]] = []
    parts: List[str] = []
    cursor = 0
    for w in word_timing[word_col].astype(str).tolist():
        if w == "nan":
            w = ""
        start = cursor
        parts.append(w)
        cursor += len(w)
        spans.append((start, cursor))
    return SectionText(text="".join(parts), word_spans=spans, n_words=len(spans))


def assign_tokens_to_words(
    offsets: List[Tuple[int, int]],
    word_spans: List[Tuple[int, int]],
) -> Tuple[List[List[int]], List[int]]:
    """Map each LM token to exactly one segmented word (unique ownership).

    Owner rule: the word containing the token's last character
    (``char_end - 1``). If that fails, fall back to the last overlapping word.
    Tokens are **never** shared across adjacent word rows — only the owner
    receives the token for residual / SAE / surprisal aggregation.

    Returns
    -------
    tokens_per_word : list[list[int]]
        Owned token indices per word (may be empty for some words).
    token_owner : list[int]
        For each token, the unique owning word index, or -1 for empty/special.
    """
    n_words = len(word_spans)
    n_tokens = len(offsets)
    tokens_per_word: List[List[int]] = [[] for _ in range(n_words)]
    token_owner: List[int] = [-1] * n_tokens

    wi = 0
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:  # special / empty token
            continue
        while wi < n_words and word_spans[wi][1] <= ts:
            wi += 1
        wj = wi
        owner = -1
        last_overlap = -1
        while wj < n_words and word_spans[wj][0] < te:
            last_overlap = wj
            if word_spans[wj][0] <= te - 1 < word_spans[wj][1]:
                owner = wj
            wj += 1
        if owner == -1:
            owner = last_overlap
        if owner < 0:
            continue
        token_owner[ti] = owner
        tokens_per_word[owner].append(ti)
    return tokens_per_word, token_owner


@dataclass
class SharedTokenAssignment:
    """Character-weighted token contributions (a token may hit several words)."""

    tokens_per_word: List[List[int]]
    weights_per_word: List[List[float]]
    shared_token: List[bool]
    n_shared_tokens: int


def assign_tokens_shared_char_weighted(
    offsets: List[Tuple[int, int]],
    word_spans: List[Tuple[int, int]],
) -> SharedTokenAssignment:
    """Give every overlapped word a share of the token.

    Weight on word *w* is the number of token characters that fall inside *w*,
    divided by the token's character length. A word's ``shared_token`` flag is
    true when any contributing token also overlaps a different word.
    """
    n_words = len(word_spans)
    tokens_per_word: List[List[int]] = [[] for _ in range(n_words)]
    weights_per_word: List[List[float]] = [[] for _ in range(n_words)]
    shared_token = [False] * n_words
    n_shared_tokens = 0

    wi = 0
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        span = float(te - ts)
        while wi < n_words and word_spans[wi][1] <= ts:
            wi += 1
        overlaps: List[Tuple[int, float]] = []
        wj = wi
        while wj < n_words and word_spans[wj][0] < te:
            ws, we = word_spans[wj]
            ov = min(te, we) - max(ts, ws)
            if ov > 0:
                overlaps.append((wj, float(ov) / span))
            wj += 1
        is_shared = len(overlaps) > 1
        if is_shared:
            n_shared_tokens += 1
        for word_i, weight in overlaps:
            tokens_per_word[word_i].append(ti)
            weights_per_word[word_i].append(weight)
            if is_shared:
                shared_token[word_i] = True
    return SharedTokenAssignment(
        tokens_per_word=tokens_per_word,
        weights_per_word=weights_per_word,
        shared_token=shared_token,
        n_shared_tokens=n_shared_tokens,
    )


def _shared_ownership_mismatches(
    tokens_per_word: List[List[int]],
    weights_per_word: List[List[float]],
) -> int:
    """Words whose token/weight lists are inconsistent, or whose token mass ≠ 1."""
    n = len(tokens_per_word)
    if len(weights_per_word) != n:
        return n
    from collections import defaultdict

    mass: Dict[int, float] = defaultdict(float)
    bad_words = set()
    for wi, (toks, wts) in enumerate(zip(tokens_per_word, weights_per_word)):
        if len(toks) != len(wts) or len(toks) != len(set(toks)):
            bad_words.add(wi)
            continue
        for tok, weight in zip(toks, wts):
            if not np.isfinite(weight) or weight <= 0:
                bad_words.add(wi)
                break
            mass[int(tok)] += float(weight)
    bad_tokens = {tok for tok, total in mass.items() if abs(total - 1.0) > 1e-5}
    if bad_tokens:
        for wi, toks in enumerate(tokens_per_word):
            if any(int(tok) in bad_tokens for tok in toks):
                bad_words.add(wi)
    return int(len(bad_words))


def alignment_report(
    tokens_per_word: List[List[int]],
    token_owner: Optional[List[int]] = None,
    weights_per_word: Optional[List[List[float]]] = None,
) -> Dict[str, float]:
    counts = np.array([len(t) for t in tokens_per_word], dtype=int)
    n = len(counts)
    owned_cov = float(100.0 * np.mean(counts > 0)) if n else 0.0
    shared = 0
    if weights_per_word is not None:
        shared = _shared_ownership_mismatches(tokens_per_word, weights_per_word)
    elif token_owner is not None:
        # Each positive owner index should appear exactly once per owned token;
        # rebuild multiset vs tokens_per_word to detect accidental sharing.
        rebuilt = [[] for _ in range(n)]
        for ti, o in enumerate(token_owner):
            if o >= 0:
                rebuilt[o].append(ti)
        shared = int(sum(
            1 for wi in range(n)
            if sorted(rebuilt[wi]) != sorted(tokens_per_word[wi])
        ))
    return {
        "n_words": int(n),
        "words_with_0_tokens": int(np.sum(counts == 0)),
        "coverage_pct": owned_cov,
        "owned_token_coverage_pct": owned_cov,
        "mean_tokens_per_word": float(np.mean(counts)) if n else 0.0,
        "max_tokens_per_word": int(np.max(counts)) if n else 0,
        "excluded_word_count": int(np.sum(counts == 0)),
        "ownership_mismatch_words": int(shared),
    }


def adjacent_duplicate_row_rate(latents) -> float:
    """Fraction of adjacent word pairs with identical SAE rows (dense compare)."""
    from scipy import sparse as sp

    if latents.shape[0] < 2:
        return 0.0
    if sp.issparse(latents):
        # Compare CSR rows via equality of data/indices slices is awkward;
        # densify only the difference check via row hashes.
        indptr = latents.indptr
        data = latents.data
        indices = latents.indices
        n = latents.shape[0]
        same = 0
        for i in range(n - 1):
            a0, a1 = indptr[i], indptr[i + 1]
            b0, b1 = indptr[i + 1], indptr[i + 2]
            if (a1 - a0) != (b1 - b0):
                continue
            if np.array_equal(indices[a0:a1], indices[b0:b1]) and np.array_equal(
                data[a0:a1], data[b0:b1]
            ):
                same += 1
        return float(same / (n - 1))
    rows = np.asarray(latents)
    same = int(np.sum(np.all(rows[:-1] == rows[1:], axis=1)))
    return float(same / (rows.shape[0] - 1))


# ======================================================================
# Model forward pass  (needs torch + transformers)
# ======================================================================

def apply_preset(args, argv: Optional[Sequence[str]] = None):
    """Fill backbone fields from ``--preset`` unless the user overrode them."""
    if not getattr(args, "preset", None):
        return args
    cfg = PRESETS[args.preset]
    specified = set()
    for raw in (argv if argv is not None else sys.argv[1:]):
        if raw.startswith("--"):
            specified.add(raw.split("=", 1)[0].lstrip("-").replace("-", "_"))
    for key, value in cfg.items():
        if key == "matryoshka_widths":
            if "matryoshka_widths" not in specified:
                args.matryoshka_widths = value
            continue
        if key not in specified:
            setattr(args, key, value)
    return args


def _load_model_and_tokenizer(
    model_name: str, device: str, dtype: str, trust_remote_code: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
    load_kw = dict(
        torch_dtype=torch_dtype,
        output_hidden_states=False,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    model.to(device).eval()
    return model, tok


def _load_sae(release: str, sae_id: str, device: str):
    """Load a pretrained SAE across sae_lens API variants."""
    from sae_lens import SAE

    try:
        out = SAE.from_pretrained(release, sae_id, device=device)
    except TypeError:
        out = SAE.from_pretrained(release, subfolder=sae_id, device=device)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _model_hidden_size(model) -> int:
    cfg = model.config
    h = getattr(cfg, "hidden_size", None)
    if h:
        return int(h)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return int(text_cfg.hidden_size)
    raise AttributeError("Could not resolve hidden_size from model.config")


def _expected_n_layers(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    n = getattr(cfg, "num_hidden_layers", None)
    if n:
        return int(n)
    text_cfg = getattr(cfg, "text_config", None)
    n = getattr(text_cfg, "num_hidden_layers", None) if text_cfg is not None else None
    return int(n) if n else None


def _resolve_decoder_layers(model):
    """Return the ``nn.ModuleList`` of decoder blocks across common architectures."""
    import torch.nn as nn

    n_expected = _expected_n_layers(model)
    found: List[Tuple[str, Any, int]] = []

    def walk(mod, path: str, depth: int = 0):
        if depth > 6:
            return
        for name, child in mod.named_children():
            p = f"{path}.{name}" if path else name
            if isinstance(child, nn.ModuleList) and len(child) >= 4:
                found.append((p, child, len(child)))
            walk(child, p, depth + 1)

    walk(model, "")
    if n_expected:
        match = [item for item in found if item[2] == n_expected]
        if match:
            # Prefer language_model / model.layers over vision towers.
            match.sort(key=lambda t: (
                0 if "language" in t[0] or t[0].endswith("layers") else 1,
                len(t[0]),
            ))
            return match[0][1]
    if found:
        found.sort(key=lambda t: (0 if "language" in t[0] else 1, -t[2], len(t[0])))
        return found[0][1]
    raise AttributeError("Could not locate decoder layers on this model.")


def forward_capture(
    model,
    tok,
    section: SectionText,
    sae_layer: int,
    context_len: int,
    stride: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Run the LM in sliding windows; return token residuals + surprisal."""
    import torch

    enc = tok(
        section.text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    if "offset_mapping" not in enc:
        raise RuntimeError(
            f"Tokenizer {type(tok).__name__} did not return offset_mapping. "
            "Need a fast tokenizer for unique token→word ownership."
        )
    input_ids = enc["input_ids"]
    n_tokens = len(input_ids)
    d_model = _model_hidden_size(model)

    resid = np.full((n_tokens, d_model), np.nan, dtype=np.float32)
    surprisal_tok = np.full((n_tokens,), np.nan, dtype=np.float32)
    filled = np.zeros(n_tokens, dtype=bool)

    layers = _resolve_decoder_layers(model)
    target_layer = layers[sae_layer]
    captured = {}

    def hook(_module, _inp, out):
        captured["resid"] = out[0] if isinstance(out, tuple) else out

    handle = target_layer.register_forward_hook(hook)
    overlap = max(0, context_len - stride)
    try:
        with torch.no_grad():
            start = 0
            while start < n_tokens:
                end = min(start + context_len, n_tokens)
                window = input_ids[start:end]
                ids = torch.tensor([window], device=device)
                out = model(ids)
                logits = out.logits[0].float()
                win_resid = captured["resid"][0].float()
                keep_lo = 0 if start == 0 else overlap
                logprobs = torch.log_softmax(logits, dim=-1)
                for p in range(keep_lo, end - start):
                    gpos = start + p
                    resid[gpos] = win_resid[p].cpu().numpy()
                    filled[gpos] = True
                    if p >= 1:
                        tgt = window[p]
                        surprisal_tok[gpos] = float(-logprobs[p - 1, tgt].cpu())
                if end >= n_tokens:
                    break
                start += stride
    finally:
        handle.remove()

    if not filled.all():
        missing = np.flatnonzero(~filled)
        print(f"  [WARN] {missing.size} tokens uncovered by windows; leaving NaN")

    return resid, surprisal_tok, enc["offset_mapping"]


# ======================================================================
# SAE-then-mean aggregation (Lepori order)
# ======================================================================

def encode_tokens_with_sae(
    resid: np.ndarray,
    sae,
    scale_by_decoder: bool,
    device: str,
    batch_size: int = 4096,
):
    """Encode every token residual through the SAE -> dense float32 latents."""
    import torch

    n = int(resid.shape[0])
    if n == 0:
        d = int(getattr(sae.cfg, "d_sae", 0) or 0)
        return np.zeros((0, d), dtype=np.float32)
    batch_size = max(1, int(batch_size or n))
    sae_dtype = next(sae.parameters()).dtype
    out = None
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = torch.tensor(
                resid[start:start + batch_size], dtype=torch.float32, device=device)
            finite = torch.isfinite(chunk).all(dim=1)
            x_safe = torch.where(torch.isfinite(chunk), chunk, torch.zeros_like(chunk))
            lat = sae.encode(x_safe.to(sae_dtype))
            if scale_by_decoder:
                w_dec = sae.W_dec
                norms = torch.linalg.norm(w_dec, dim=1)
                lat = lat * norms.unsqueeze(0)
            lat = lat.float().cpu().numpy()
            lat[~finite.cpu().numpy()] = np.nan
            lat[np.isfinite(lat) & (np.abs(lat) < 1e-8)] = 0.0
            if out is None:
                out = np.empty((n, lat.shape[1]), dtype=np.float32)
            out[start:start + lat.shape[0]] = lat
            del chunk, x_safe, lat
            if device == "cuda":
                torch.cuda.empty_cache()
    return out


def _reduce_vectors(rows: np.ndarray, weights: Optional[np.ndarray], aggregation: str):
    """Unweighted mean, or weight-normalized mean. Equal weights use ``mean``."""
    if aggregation == "last" or rows.shape[0] == 1:
        return rows[-1] if aggregation == "last" else rows.mean(axis=0)
    if weights is None:
        return rows.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != rows.shape[0]:
        raise ValueError(
            f"weight length {w.shape[0]} != n token rows {rows.shape[0]}")
    if not np.isfinite(w).all() or float(w.sum()) <= 0:
        raise ValueError("token weights must be finite and positive")
    if np.allclose(w, w[0]):
        return rows.mean(axis=0)
    return np.average(rows, axis=0, weights=w).astype(np.float32)


def _reduce_scalar(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if weights is None or values.size == 1 or np.allclose(weights, weights[0]):
        return float(np.mean(values))
    return float(np.average(values, weights=np.asarray(weights, dtype=np.float64)))


def aggregate_owned_token_sae(
    token_latents: np.ndarray,
    surprisal_tok: np.ndarray,
    tokens_per_word: List[List[int]],
    aggregation: str = "mean",
    weights_per_word: Optional[List[List[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Average SAE activations / surprisal over owned tokens.

    ``weights_per_word`` is the per-token character fraction for
    ``shared_char_weighted``. The word vector is the weight-normalized mean
    (equal weights reduce to the v1 unweighted mean). Words with no owned
    tokens, or no finite surprisal, are invalid. No residual imputation and
    no surprisal mean-imputation are performed.
    """
    from scipy import sparse

    n_words = len(tokens_per_word)
    d_sae = token_latents.shape[1]
    word_lat = np.zeros((n_words, d_sae), dtype=np.float32)
    word_surp = np.full((n_words,), np.nan, dtype=np.float32)
    valid = np.zeros(n_words, dtype=bool)
    token_l0s: List[float] = []
    n_imputed = 0  # must remain 0

    for wi, toks in enumerate(tokens_per_word):
        if not toks:
            continue
        tok_idx = np.asarray(toks, dtype=int)
        rows = token_latents[tok_idx]
        good = np.isfinite(rows).all(axis=1)
        rows = rows[good]
        if rows.shape[0] == 0:
            continue
        w = None
        if weights_per_word is not None:
            w = np.asarray(weights_per_word[wi], dtype=np.float64)[good]
            if w.size == 0 or float(np.sum(w)) <= 0:
                continue
        token_l0s.extend(float(np.count_nonzero(r)) for r in rows)
        word_lat[wi] = _reduce_vectors(rows, w, aggregation)
        valid[wi] = True

        s_all = surprisal_tok[tok_idx[good]]
        s_ok = np.isfinite(s_all)
        svals = s_all[s_ok]
        if svals.size:
            w_s = None if w is None else w[s_ok]
            word_surp[wi] = _reduce_scalar(svals, w_s)
        else:
            # Owned tokens exist but no finite NLL (e.g. first token) → invalid.
            valid[wi] = False
            word_lat[wi] = 0.0
            word_surp[wi] = np.nan

    # Zero out invalid rows explicitly (sparse-friendly).
    word_lat[~valid] = 0.0
    latents = sparse.csr_matrix(word_lat)
    latents.eliminate_zeros()

    word_l0 = np.asarray(latents.getnnz(axis=1), dtype=float)
    stats = {
        "surprisal_imputation_count": int(n_imputed),
        "excluded_word_count": int((~valid).sum()),
        "owned_token_coverage_pct": float(100.0 * valid.mean()) if n_words else 0.0,
        "mean_token_L0": float(np.mean(token_l0s)) if token_l0s else 0.0,
        "mean_word_L0": float(word_l0[valid].mean()) if valid.any() else 0.0,
        "adjacent_duplicate_row_rate": adjacent_duplicate_row_rate(latents),
    }
    if stats["surprisal_imputation_count"] != 0:
        raise RuntimeError(
            "Surprisal imputation is forbidden; got "
            f"{stats['surprisal_imputation_count']} imputed words."
        )
    return latents, word_surp, valid, stats


def aggregate_owned_token_dense(
    token_mat: np.ndarray,
    tokens_per_word: List[List[int]],
    aggregation: str = "mean",
    weights_per_word: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """Mean (or last) dense vector over owned tokens. Invalid rows stay 0."""
    n_words = len(tokens_per_word)
    d = int(token_mat.shape[1])
    out = np.zeros((n_words, d), dtype=np.float32)
    for wi, toks in enumerate(tokens_per_word):
        if not toks:
            continue
        rows = token_mat[np.asarray(toks)]
        good = np.isfinite(rows).all(axis=1)
        rows = rows[good]
        if rows.shape[0] == 0:
            continue
        w = None
        if weights_per_word is not None:
            w = np.asarray(weights_per_word[wi], dtype=np.float64)[good]
            if w.size == 0 or float(np.sum(w)) <= 0:
                continue
        out[wi] = _reduce_vectors(rows, w, aggregation)
    return out


# ======================================================================
# Per-section driver
# ======================================================================

def process_section(
    section_id: int,
    model,
    tok,
    sae,
    args,
) -> Optional[Dict[str, float]]:
    import pandas as pd

    sec_dir = ap.FEATURES_DIR / f"section_{section_id:03d}"
    wt_path = sec_dir / "word_timing.csv"
    if not wt_path.exists():
        print(f"[skip] {wt_path} not found")
        return None

    word_timing = pd.read_csv(wt_path)
    sect = build_section_text(word_timing, args.word_col)

    enc = tok(sect.text, return_offsets_mapping=True, add_special_tokens=False)
    if "offset_mapping" not in enc:
        raise RuntimeError(
            f"Tokenizer for {args.model_name} did not return offset_mapping "
            "(need a fast tokenizer)."
        )
    ownership = getattr(args, "ownership", OWNERSHIP_UNIQUE)
    weights_per_word = None
    shared_flags = None
    n_shared_tokens = 0
    if ownership == OWNERSHIP_SHARED:
        shared = assign_tokens_shared_char_weighted(
            enc["offset_mapping"], sect.word_spans)
        tokens_per_word = shared.tokens_per_word
        weights_per_word = shared.weights_per_word
        shared_flags = np.asarray(shared.shared_token, dtype=bool)
        n_shared_tokens = int(shared.n_shared_tokens)
        rep = alignment_report(tokens_per_word, weights_per_word=weights_per_word)
        if rep["ownership_mismatch_words"] != 0:
            raise RuntimeError(
                f"shared-token assignment mismatches: {rep['ownership_mismatch_words']}")
    else:
        tokens_per_word, token_owner = assign_tokens_to_words(
            enc["offset_mapping"], sect.word_spans)
        rep = alignment_report(tokens_per_word, token_owner)
        # Ownership uniqueness invariant
        owned_counts = np.zeros(sect.n_words, dtype=int)
        for o in token_owner:
            if o >= 0:
                owned_counts[o] += 1
        assert all(
            len(tokens_per_word[i]) == owned_counts[i] for i in range(sect.n_words)
        )
    rep["shared_token_count"] = int(n_shared_tokens)
    rep["shared_token_word_count"] = (
        int(shared_flags.sum()) if shared_flags is not None else 0)
    print(f"[section {section_id}] {rep}")

    if args.dry_run_alignment:
        return rep

    resid, surprisal_tok, _ = forward_capture(
        model, tok, sect,
        sae_layer=args.sae_layer,
        context_len=args.context_len,
        stride=args.stride,
        device=args.device,
    )
    tag = args.feature_tag
    word_resid = aggregate_owned_token_dense(
        resid, tokens_per_word, aggregation=args.aggregation,
        weights_per_word=weights_per_word)

    if getattr(args, "resid_only", False):
        valid_path = sec_dir / f"X_word_{tag}_valid.npy"
        if not valid_path.exists():
            raise FileNotFoundError(
                f"{valid_path} missing; run a full SAE extract before --resid_only")
        valid = np.load(valid_path).astype(bool)
        if valid.shape[0] != word_resid.shape[0]:
            raise ValueError(
                f"[{tag} section {section_id}] resid n_words={word_resid.shape[0]} "
                f"!= valid n_words={valid.shape[0]}")
        word_resid[~valid] = 0.0
        resid_path = sec_dir / f"X_word_{tag}_resid.npy"
        np.save(resid_path, word_resid.astype(np.float32))
        print(
            f"  saved residual {tag}: {word_resid.shape}, "
            f"valid={int(valid.sum())}/{sect.n_words} -> {resid_path.name}"
        )
        return {**rep, "d_model": int(word_resid.shape[1]),
                "resid_only": True}

    token_lat = encode_tokens_with_sae(
        resid, sae, args.scale_by_decoder, args.device,
        batch_size=int(getattr(args, "sae_encode_batch", 4096) or 4096),
    )
    latents, word_surp, valid, stats = aggregate_owned_token_sae(
        token_lat, surprisal_tok, tokens_per_word, aggregation=args.aggregation,
        weights_per_word=weights_per_word,
    )
    word_resid[~valid] = 0.0

    from scipy import sparse

    if ownership == OWNERSHIP_SHARED and not str(tag).endswith("_v2"):
        raise RuntimeError(
            f"shared_char_weighted refused to write non-v2 tag {tag!r}")
    sparse.save_npz(sec_dir / f"X_word_{tag}.npz", latents)
    d_sae = latents.shape[1]
    with open(sec_dir / f"X_word_{tag}_feature_names.txt", "w") as f:
        f.write("\n".join(f"{tag}_{i}" for i in range(d_sae)) + "\n")

    # Tagged surprisal + validity (do not overwrite other models).
    np.save(sec_dir / f"X_word_{tag}_surprisal.npy",
            word_surp.reshape(-1, 1).astype(np.float32))
    with open(sec_dir / f"X_word_{tag}_surprisal_feature_names.txt", "w") as f:
        f.write(f"surprisal_{tag}\n")
    np.save(sec_dir / f"X_word_{tag}_valid.npy", valid.astype(np.bool_))
    np.save(sec_dir / f"X_word_{tag}_resid.npy", word_resid.astype(np.float32))
    if shared_flags is not None:
        np.save(sec_dir / f"X_word_{tag}_shared_token.npy", shared_flags)

    meta = {
        "model_name": args.model_name,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "sae_layer": args.sae_layer,
        "d_sae": int(d_sae),
        "d_model": int(word_resid.shape[1]),
        "aggregation": args.aggregation,
        "aggregation_order": "sae_then_mean",
        "token_ownership": OWNERSHIP_META.get(ownership, ownership),
        "coverage_pct": rep.get("coverage_pct"),
        "shared_token_count": int(n_shared_tokens),
        "shared_token_word_count": rep["shared_token_word_count"],
        "context_len": args.context_len,
        "stride": args.stride,
        "scale_by_decoder": bool(args.scale_by_decoder),
        "matryoshka_widths": list(args.matryoshka_widths)
        if getattr(args, "matryoshka_widths", None) else None,
        "n_words": int(sect.n_words),
        "nnz": int(latents.nnz),
        "mean_L0": float(latents.nnz / max(1, int(valid.sum()))),
        "alignment": rep,
        **stats,
    }
    with open(sec_dir / f"X_word_{tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(
        f"  saved {tag}: {latents.shape}, valid={int(valid.sum())}/{sect.n_words}, "
        f"mean word L0={stats['mean_word_L0']:.1f}, "
        f"adj_dup_rate={stats['adjacent_duplicate_row_rate']:.4f}, "
        f"surprisal finite on valid="
        f"{int(np.isfinite(word_surp[valid]).sum())}/{int(valid.sum())}"
    )
    return {**rep, **stats}


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset", choices=sorted(PRESETS), default=None,
        help="Fill model / SAE / layer / feature_tag from a named backbone.",
    )
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--sae_release", default=SAE_RELEASE)
    p.add_argument("--sae_id", default=SAE_ID)
    p.add_argument("--sae_layer", type=int, default=SAE_LAYER)
    p.add_argument("--feature_tag", default=FEATURE_TAG)
    p.add_argument("--aggregation", choices=["last", "mean"], default=AGGREGATION)
    p.add_argument(
        "--ownership", choices=list(OWNERSHIP_CHOICES), default=OWNERSHIP_UNIQUE,
        help="unique_final_word (default, v1) or shared_char_weighted (v2). "
             "shared_char_weighted appends _v2 to --feature_tag and does not "
             "overwrite v1 npz/meta.",
    )
    p.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--scale_by_decoder", action="store_true", default=SCALE_BY_DECODER)
    p.add_argument("--sections", type=int, nargs="+", default=list(SECTIONS))
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dtype", default=DTYPE)
    p.add_argument("--word_col", default=WORD_COL)
    p.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code to tokenizer/model from_pretrained.",
    )
    p.add_argument(
        "--sae_encode_batch", type=int, default=4096,
        help="Token-residual batch size for SAE.encode (65k-d SAEs).",
    )
    p.add_argument("--dry_run_alignment", action="store_true",
                   help="Only tokenize + align to words; no model/SAE needed.")
    p.add_argument(
        "--resid_only", action="store_true",
        help="Write X_word_<tag>_resid.npy from a fresh LM forward pass. "
             "Does not overwrite SAE latents, surprisal, or valid masks. "
             "Requires an existing X_word_<tag>_valid.npy.",
    )
    args = p.parse_args(argv)
    args.matryoshka_widths = None
    apply_preset(args, argv if argv is not None else sys.argv[1:])
    args.feature_tag = feature_tag_for_ownership(args.feature_tag, args.ownership)
    return args


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    trust = bool(getattr(args, "trust_remote_code", False))

    model = tok = sae = None
    if args.dry_run_alignment:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=trust)
    elif args.resid_only:
        model, tok = _load_model_and_tokenizer(
            args.model_name, args.device, args.dtype, trust_remote_code=trust)
        hidden = _model_hidden_size(model)
        print(f"Loaded {args.model_name} for residual dump "
              f"(hidden={hidden}, layer={args.sae_layer}, tag={args.feature_tag})")
    else:
        model, tok = _load_model_and_tokenizer(
            args.model_name, args.device, args.dtype, trust_remote_code=trust)
        sae = _load_sae(args.sae_release, args.sae_id, args.device)
        sae.eval()
        d_in = int(getattr(sae.cfg, "d_in", 0) or 0)
        hidden = _model_hidden_size(model)
        if d_in and d_in != hidden:
            raise ValueError(
                f"SAE d_in={d_in} != model hidden_size={hidden}. "
                f"Check that --model_name matches the SAE backbone "
                f"({args.sae_release}/{args.sae_id}).")
        print(
            f"Loaded {args.model_name} + SAE {args.sae_release}/{args.sae_id} "
            f"(hidden={hidden}, d_sae={getattr(sae.cfg, 'd_sae', '?')}, "
            f"layer={args.sae_layer}, tag={args.feature_tag}, "
            f"matryoshka_widths={getattr(args, 'matryoshka_widths', None)})"
        )

    for sid in args.sections:
        process_section(sid, model, tok, sae, args)


if __name__ == "__main__":
    main()
