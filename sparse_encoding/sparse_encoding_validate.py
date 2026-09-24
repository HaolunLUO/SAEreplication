#!/usr/bin/env python3
"""
sparse_encoding_validate.py
===========================
Staged validation / cohort runner for the sparse-encoding pipeline.

**Temporal precision is the primary analysis** (lag-resolved screening +
continuous-time deconvolution kernels). Fixed single-lag regression is kept
as an optional baseline (``single_lag_baseline``).

Steps
-----
1. Alignment-only checks (unique ownership) for configured models.
2. Optional feature extraction (requires model weights; use --device cpu|cuda).
3. Primary temporal suite: lag-resolved + deconvolution (``temporal_full``).
4. Optional single-lag baseline / dense baselines / nulls / qualitative.
5. Subject-level left-MFG and temporal contrast report.

Examples
--------
    # Alignment + temporal primary path (default)
    python -m sparse_encoding.sparse_encoding_validate \\
        --stage align,extract,temporal_full,report \\
        --models qwen35 --all_subjects --device cuda

    # Small smoke (subset subjects / channels)
    python -m sparse_encoding.sparse_encoding_validate \\
        --stage align,smoke --subjects Subject04 --max_channels 4

    # Legacy single-lag baseline only
    python -m sparse_encoding.sparse_encoding_validate \\
        --stage single_lag_baseline,report --models qwen35 --all_subjects
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import core.analysis_paths as ap
from sparse_encoding.sae_extract_features import (
    OWNERSHIP_SHARED, OWNERSHIP_UNIQUE,
    alignment_report, assign_tokens_shared_char_weighted,
    assign_tokens_to_words, build_section_text, feature_tag_for_ownership,
)
from sparse_encoding.sparse_encoding_summary import (
    LEFT_MFG_REGION, bootstrap_subject_means, language_mask, merge_taxonomy,
)

PY = sys.executable
ROOT = Path(__file__).resolve().parent

GEMMA = dict(
    feature_tag="sae_gemma2_2b_mat_l12",
    model_name="google/gemma-2-2b",
    sae_release="gemma-2-2b-res-matryoshka-dc",
    sae_id="blocks.12.hook_resid_post",
    sae_layer=12,
)
# Chinese companion (non-nested SAE).
QWEN = dict(
    feature_tag="sae_qwen3_8b_l18",
    model_name="Qwen/Qwen3-8B",
    sae_release="",
    sae_id="",
    sae_layer=18,
)
# Main hierarchical SAE for Study 3 analogue.
QWEN35 = dict(
    feature_tag="sae_qwen35_4b_mat_l15",
    model_name="Qwen/Qwen3.5-4B-Base",
    sae_release="decoderesearch/qwen-3.5-saes",
    sae_id="qwen-3.5-4b-base/btk-mat-layer-15-k-100",
    sae_layer=15,
    trust_remote_code=True,
    preset="qwen35_4b_mat_l15",
)

# Stage aliases: new names preferred; old names still accepted.
STAGE_ALIASES = {
    "regress": "single_lag_baseline",
    "lag": "lag_resolved",
    "deconv": "temporal_kernels",
}


def canonicalize_stages(raw: list[str]) -> list[str]:
    out = []
    for s in raw:
        out.append(STAGE_ALIASES.get(s, s))
    return out


def run(cmd: list[str], check: bool = True) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def stage_align(
    models: list[dict],
    sections: list[int],
    ownership: str = OWNERSHIP_UNIQUE,
) -> Path:
    """Tokenizer-only ownership report. Shared mode uses the v2 rule and root."""
    from transformers import AutoTokenizer

    ap.ensure_pipeline_dirs()
    shared_mode = ownership == OWNERSHIP_SHARED
    rows = []
    for cfg in models:
        tag = feature_tag_for_ownership(cfg["feature_tag"], ownership)
        print(f"\n[align] {tag} / {cfg['model_name']} ownership={ownership}")
        tok = AutoTokenizer.from_pretrained(
            cfg["model_name"],
            trust_remote_code=bool(cfg.get("trust_remote_code")),
        )
        for sid in sections:
            wt = pd.read_csv(ap.FEATURES_DIR / f"section_{sid:03d}" / "word_timing.csv")
            sect = build_section_text(wt)
            enc = tok(sect.text, return_offsets_mapping=True, add_special_tokens=False)
            if shared_mode:
                assigned = assign_tokens_shared_char_weighted(
                    enc["offset_mapping"], sect.word_spans)
                rep = alignment_report(
                    assigned.tokens_per_word,
                    weights_per_word=assigned.weights_per_word,
                )
                rep["shared_token_count"] = int(assigned.n_shared_tokens)
                rep["shared_token_word_count"] = int(sum(assigned.shared_token))
                if rep["ownership_mismatch_words"] != 0:
                    raise AssertionError(
                        f"shared ownership mismatches: {rep['ownership_mismatch_words']}"
                    )
            else:
                tpw, owner = assign_tokens_to_words(
                    enc["offset_mapping"], sect.word_spans)
                rep = alignment_report(tpw, owner)
                seen = []
                for toks in tpw:
                    seen.extend(toks)
                assert len(seen) == len(set(seen)), "shared tokens detected"
            rows.append({
                "feature_tag": tag,
                "section": sid,
                "token_ownership": ownership,
                **rep,
            })
            print(f"  section {sid}: {rep}")
    if shared_mode:
        # Never write this QC into the v1 sparse_encoding reports directory.
        root = ap.RESULTS_ROOT / "sparse_encoding_v2"
        reports = root / "reports"
        tables = root / "tables"
        reports.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        out = reports / "sparse_encoding_alignment_shared_v2.txt"
        csv_path = tables / "sparse_encoding_alignment_shared_v2.csv"
        title = "SHARED CHAR-WEIGHTED TOKEN OWNERSHIP — ALIGNMENT CHECK"
    else:
        out = ap.SAE_REPORTS / "sparse_encoding_alignment_repair.txt"
        csv_path = ap.SAE_TABLES / "sparse_encoding_alignment_repair.csv"
        title = "UNIQUE TOKEN OWNERSHIP — ALIGNMENT CHECK"
    df = pd.DataFrame(rows)
    lines = [title, "=" * 60, df.to_string(index=False)]
    out.write_text("\n".join(lines))
    df.to_csv(csv_path, index=False)
    print(f"Wrote {out}")
    return out


def stage_extract(
    models: list[dict],
    sections: list[int],
    device: str,
    dtype: str,
    ownership: str = OWNERSHIP_UNIQUE,
):
    for cfg in models:
        if cfg.get("preset"):
            cmd = [
                PY, "-m", "sparse_encoding.sae_extract_features",
                "--preset", cfg["preset"],
                "--ownership", ownership,
                "--device", device,
                "--dtype", dtype if device == "cuda" else "float32",
                "--sections", *[str(s) for s in sections],
            ]
        else:
            cmd = [
                PY, "-m", "sparse_encoding.sae_extract_features",
                "--model_name", cfg["model_name"],
                "--sae_release", cfg["sae_release"],
                "--sae_id", cfg["sae_id"],
                "--sae_layer", str(cfg["sae_layer"]),
                "--feature_tag", feature_tag_for_ownership(cfg["feature_tag"], ownership),
                "--ownership", ownership,
                "--aggregation", "mean",
                "--device", device,
                "--dtype", dtype if device == "cuda" else "float32",
                "--sections", *[str(s) for s in sections],
            ]
            if cfg.get("trust_remote_code"):
                cmd.append("--trust_remote_code")
        run(cmd)


def stage_smoke(models: list[dict], subjects: list[str], max_channels: int, sections: list[int]):
    for cfg in models:
        tag = cfg["feature_tag"]
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_lag_screen",
            "--feature_tag", tag,
            "--subjects", *subjects,
            "--max_channels", str(max_channels),
            "--sections", *[str(s) for s in sections],
            "--out_suffix", f"_{tag}_smoke",
        ])
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_deconvolution",
            "--feature_tag", tag,
            "--subjects", *subjects,
            "--max_channels", str(max_channels),
            "--sections", *[str(s) for s in sections],
            "--out_suffix", f"_{tag}_smoke",
        ])
        # Optional single-lag baseline for comparison.
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_regression",
            "--feature_tag", tag,
            "--subjects", *subjects,
            "--max_channels", str(max_channels),
            "--sections", *[str(s) for s in sections],
            "--out", str(ap.SAE_TABLES / f"sparse_encoding_results_{tag}_smoke.csv"),
        ])


def stage_single_lag_baseline(models: list[dict], subjects: list[str] | None, sections: list[int]):
    """Deprecated primary path — kept for baselines / Matryoshka bin refits."""
    print(
        "[note] single_lag_baseline is a simplified fixed-lag (default 300 ms) "
        "comparison. Prefer lag_resolved + temporal_kernels for primary claims.",
        flush=True,
    )
    for cfg in models:
        tag = cfg["feature_tag"]
        cmd = [
            PY, "-m", "sparse_encoding.sparse_encoding_regression",
            "--feature_tag", tag,
            "--sections", *[str(s) for s in sections],
            "--out", str(ap.SAE_TABLES / f"sparse_encoding_results_{tag}.csv"),
        ]
        if subjects:
            cmd += ["--subjects", *subjects]
        run(cmd)
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_summary",
            "--results_csv", str(ap.SAE_TABLES / f"sparse_encoding_results_{tag}.csv"),
        ])


def stage_lag_resolved(models: list[dict], subjects: list[str] | None,
                       max_channels: int, lang_only: bool = True):
    for cfg in models:
        tag = cfg["feature_tag"]
        cmd = [
            PY, "-m", "sparse_encoding.sparse_encoding_lag_screen",
            "--feature_tag", tag,
            "--out_suffix", f"_{tag}_lang" if lang_only else f"_{tag}",
        ]
        if lang_only:
            cmd.append("--lang_only")
        if subjects:
            cmd += ["--subjects", *subjects]
        if max_channels:
            cmd += ["--max_channels", str(max_channels)]
        run(cmd)


def stage_temporal_kernels(models: list[dict], subjects: list[str] | None,
                           sections: list[int], max_channels: int,
                           lang_only: bool = True):
    for cfg in models:
        tag = cfg["feature_tag"]
        cmd = [
            PY, "-m", "sparse_encoding.sparse_encoding_deconvolution",
            "--feature_tag", tag,
            "--sections", *[str(s) for s in sections],
            "--out_suffix", f"_{tag}_lang" if lang_only else f"_{tag}",
        ]
        if lang_only:
            cmd += ["--roi_filter", "lang"]
        if subjects:
            cmd += ["--subjects", *subjects]
        if max_channels:
            cmd += ["--max_channels", str(max_channels)]
        run(cmd)
        # Explicit plot pass (also invoked inside deconv; safe to re-run).
        k_path = ap.SAE_TABLES / (
            f"sparse_encoding_kernels_{tag}_lang.csv" if lang_only
            else f"sparse_encoding_kernels_{tag}.csv"
        )
        if k_path.exists():
            run([
                PY, "-m", "sparse_encoding.sparse_encoding_plot_kernels",
                "--kernels_csv", str(k_path),
                "--out_suffix", f"_{tag}_lang" if lang_only else f"_{tag}",
            ], check=False)


def stage_temporal_full(models: list[dict], subjects: list[str] | None,
                        sections: list[int], max_channels: int,
                        lang_only: bool = True):
    """Primary combined temporal suite: lag-resolved + continuous kernels."""
    stage_lag_resolved(models, subjects, max_channels, lang_only=lang_only)
    stage_temporal_kernels(
        models, subjects, sections, max_channels, lang_only=lang_only)


def stage_null(models: list[dict], subjects: list[str] | None, sections: list[int],
               max_channels: int, n_perm: int,
               roi_filter: str = "mfg_temporal"):
    """Fast frozen-support circular + block nulls (default 100 perms on ROI)."""
    for cfg in models:
        tag = cfg["feature_tag"]
        for null in ("circular", "block"):
            cmd = [
                PY, "-m", "sparse_encoding.sparse_encoding_deconvolution",
                "--feature_tag", tag,
                "--sections", *[str(s) for s in sections],
                "--roi_filter", roi_filter,
                "--null", null,
                "--null_perms", str(n_perm),
                "--null_seed", "19",
                "--out_suffix", f"_{tag}_roi_{null}{n_perm}",
            ]
            if subjects:
                cmd += ["--subjects", *subjects]
            if max_channels:
                cmd += ["--max_channels", str(max_channels)]
            run(cmd)


def stage_temporal_report(models: list[dict]):
    """Append temporal-precision summary from lag peaks + kernel shapes."""
    lines = [
        "=" * 70,
        "TEMPORAL PRECISION SUMMARY",
        "=" * 70,
        "",
        "Primary analyses: lag_resolved (0–800 ms) + temporal_kernels (FIR).",
        "Single-lag 300 ms regression is a baseline only.",
        "",
    ]
    for cfg in models:
        tag = cfg["feature_tag"]
        lines.append("-" * 70)
        lines.append(f"MODEL {tag}")
        lines.append("-" * 70)
        peaks = ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks_{tag}_lang.csv"
        if not peaks.exists():
            peaks = ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks_{tag}.csv"
        if peaks.exists():
            pdf = pd.read_csv(peaks)
            lines.append(f"[lag peaks] {peaks.name}  nE={len(pdf)}")
            for col in ("full_peak_lag_ms", "surprisal_peak_lag_ms",
                        "full_fwhm_ms", "lag_diff_full_vs_surprisal_ms"):
                if col not in pdf.columns:
                    continue
                subj = pdf.groupby("subject")[col].mean()
                lines.append(
                    f"  {col}: subject-mean={subj.mean():+.1f}  "
                    f"median={pdf[col].median():.1f}  nS={subj.notna().sum()}"
                )
        else:
            lines.append(f"[lag peaks] missing ({peaks.name})")
        shapes = ap.SAE_TABLES / f"sparse_encoding_kernel_shapes_{tag}_lang.csv"
        if not shapes.exists():
            shapes = ap.SAE_TABLES / f"sparse_encoding_kernel_shapes_{tag}.csv"
        if shapes.exists():
            sdf = pd.read_csv(shapes)
            lines.append(f"[kernel shapes] {shapes.name}  nE={len(sdf)}")
            for col in ("surprisal_peak_lag_ms", "sae_l2_peak_lag_ms"):
                if col in sdf.columns:
                    subj = sdf.groupby("subject")[col].mean()
                    lines.append(
                        f"  {col}: subject-mean={subj.mean():+.1f}  "
                        f"nS={subj.notna().sum()}"
                    )
        else:
            lines.append(f"[kernel shapes] missing ({shapes.name})")
        lines.append("")
    out = ap.SAE_REPORTS / "sparse_encoding_temporal_summary.txt"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out


def stage_report(models: list[dict]):
    """Aggregate subject-level left-MFG and temporal contrasts across models."""
    lines = [
        "=" * 70,
        "REPAIRED SPARSE ENCODING — SUBJECT-LEVEL CONTRASTS",
        "=" * 70,
        "",
        f"Left MFG region label: {LEFT_MFG_REGION}",
        "Paper-comparable content gain = full − surprisal (not full − content).",
        "MFG n_subjects is typically < 8 → underpowered extension, not a "
        "direct 8-participant replication.",
        "PRIMARY temporal outputs: lag_screen_* and deconv/kernels_* tables.",
        "",
    ]
    for cfg in models:
        tag = cfg["feature_tag"]
        lines.append("-" * 70)
        lines.append(f"MODEL {tag}")
        lines.append("-" * 70)
        candidate_tables = [
            ("lag-peaks-lang", ap.SAE_TABLES / f"sparse_encoding_lag_screen_peaks_{tag}_lang.csv"),
            ("deconv-lang", ap.SAE_TABLES / f"sparse_encoding_deconv_{tag}_lang.csv"),
            ("deconv", ap.SAE_TABLES / f"sparse_encoding_deconv_{tag}.csv"),
            ("word-locked-cohort", ap.SAE_TABLES / f"sparse_encoding_results_{tag}_cohort.csv"),
            ("word-locked", ap.SAE_TABLES / f"sparse_encoding_results_{tag}.csv"),
            ("word-locked-smoke", ap.SAE_TABLES / f"sparse_encoding_results_{tag}_smoke.csv"),
            ("deconv-mfg", ap.SAE_TABLES / f"sparse_encoding_deconv_{tag}_mfg.csv"),
            ("deconv-smoke", ap.SAE_TABLES / f"sparse_encoding_deconv_{tag}_smoke.csv"),
            ("dense-baselines", ap.SAE_TABLES / "sparse_encoding_dense_baselines_cohort.csv"),
        ]
        for kind, path in candidate_tables:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if kind.startswith("lag-peaks"):
                lines.append(f"[{kind}] {path.name}")
                for col in ("full_peak_lag_ms", "lag_diff_full_vs_surprisal_ms"):
                    if col in df.columns and "subject" in df.columns:
                        subj = df.groupby("subject")[col].mean()
                        lines.append(
                            f"  {col}: mean={subj.mean():+.1f}  nS={len(subj)}"
                        )
                lines.append("")
                continue

            df = merge_taxonomy(df)
            lang = language_mask(df)
            mfg = lang & df["region"].astype(str).eq(LEFT_MFG_REGION)
            temp = lang & df["region"].astype(str).str.contains(
                "Temp|STG|MTG|ITG|STS|temporal", case=False, na=False)
            lines.append(f"[{kind}] {path.name}")

            if kind == "dense-baselines" or "dense__R_fisher" in df.columns:
                score = ("dense__R_fisher_paired"
                         if "dense__R_fisher_paired" in df.columns
                         else "dense__R_fisher")
                for feat, g in df.groupby("feature_name"):
                    sub = g.loc[language_mask(g)] if "is_lang" in g.columns else g
                    if sub.empty:
                        continue
                    v = sub.groupby("subject")[score].mean()
                    lines.append(
                        f"  {feat} lang subject-mean: "
                        f"{v.mean():+.4f} (n_subj={len(v)}, n_elec={len(sub)})"
                    )
                lines.append("")
                continue

            df = df.copy()
            if "full__R_fisher" in df.columns or "full__R_fisher_paired" in df.columns:
                full_c = ("full__R_fisher_paired"
                          if "full__R_fisher_paired" in df.columns
                          else "full__R_fisher")
                surp_c = ("surprisal_only__R_fisher_paired"
                          if "surprisal_only__R_fisher_paired" in df.columns
                          else "surprisal_only__R_fisher")
                content_c = ("content__R_fisher_paired"
                             if "content__R_fisher_paired" in df.columns
                             else "content__R_fisher")
                df["sae_gain"] = df[full_c] - df[surp_c]
                df["surprisal_gain"] = df[full_c] - df[content_c]
                df["full_R"] = df[full_c]
            else:
                df["full_R"] = df.get("full__R", np.nan)
                if "sae_gain" not in df.columns:
                    df["sae_gain"] = np.nan
                if "surprisal_gain" not in df.columns:
                    df["surprisal_gain"] = np.nan

            for label, mask in (
                ("localizer-positive", lang),
                ("left-MFG", mfg),
                ("temporal-ish", temp),
            ):
                sub = df.loc[mask]
                if sub.empty:
                    lines.append(f"  {label}: n=0")
                    continue
                g = sub.groupby("subject")[["full_R", "sae_gain", "surprisal_gain"]].mean()
                lines.append(
                    f"  {label}: n_elec={len(sub)} n_subj={g.shape[0]}")
                for col in ("full_R", "sae_gain", "surprisal_gain"):
                    mu, lo, hi, n = bootstrap_subject_means(g.reset_index(), col)
                    lines.append(
                        f"    {col}: mean={mu:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  n={n}"
                    )
            lines.append("")

        for null_glob, label in (
            (f"sparse_encoding_deconv_nulls_{tag}_roi_circular*.csv", "ROI circular"),
            (f"sparse_encoding_deconv_nulls_{tag}_roi_block*.csv", "ROI block"),
            (f"sparse_encoding_deconv_nulls_{tag}*.csv", "deconv nulls"),
        ):
            nulls = sorted(ap.SAE_TABLES.glob(null_glob))
            if not nulls:
                continue
            nd = pd.concat([pd.read_csv(p) for p in nulls], ignore_index=True)
            if "sae_gain" not in nd.columns or "obs_sae_gain" not in nd.columns:
                continue
            obs = (nd.drop_duplicates(["subject", "channel"])
                     .groupby("subject")["obs_sae_gain"].mean().mean())
            null_means = [
                g.groupby("subject")["sae_gain"].mean().mean()
                for _, g in nd.groupby("perm")
            ]
            null_means = np.asarray(null_means, dtype=float)
            p = float(np.mean(null_means >= obs)) if len(null_means) else np.nan
            lines.append(
                f"  {label} SAE gain (subject-mean): obs={obs:+.4f}  "
                f"null_mean={np.nanmean(null_means):+.4f}  "
                f"p_ge_obs={p:.3f}  n_perm={len(null_means)}  "
                f"files={len(nulls)}"
            )
            lines.append("")

    out = ap.SAE_REPORTS / "sparse_encoding_repair_validation.txt"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    stage_temporal_report(models)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        default="align,extract,temporal_full,report",
        help="Comma-separated stages. Primary default is temporal_full. "
             "Names: align,extract,smoke,temporal_full,lag_resolved,"
             "temporal_kernels,single_lag_baseline,baselines,null,"
             "qualitative,task_type,paper,dominance,report. "
             "Aliases: lag→lag_resolved, deconv→temporal_kernels, "
             "regress→single_lag_baseline.",
    )
    p.add_argument("--models", nargs="+", default=["qwen35"],
                   choices=["gemma", "qwen", "qwen35"])
    p.add_argument("--subjects", nargs="+", default=["Subject04"])
    p.add_argument("--sections", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max_channels", type=int, default=0,
                   help="0 = all channels (production). Smoke uses a small n.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--n_perm", type=int, default=100,
                   help="Fast frozen-support null permutations for --stage null")
    p.add_argument("--all_subjects", action="store_true",
                   help="Use full SUBJECTS map (ignores --subjects).")
    p.add_argument("--lang_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Lang-only electrodes for temporal stages (default on).")
    p.add_argument(
        "--ownership",
        choices=[OWNERSHIP_UNIQUE, OWNERSHIP_SHARED],
        default=OWNERSHIP_UNIQUE,
        help="Token→word rule for align/extract. shared_char_weighted is v2.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ap.ensure_pipeline_dirs()
    # ``qwen`` keeps the 8B companion; ``qwen35`` is the hierarchical main model.
    model_map = {"gemma": GEMMA, "qwen": QWEN, "qwen35": QWEN35}
    models = [model_map[m] for m in args.models]
    stages = canonicalize_stages(
        [s.strip() for s in args.stage.split(",") if s.strip()])
    subjects = None if args.all_subjects else args.subjects

    t0 = time.time()
    if args.ownership == OWNERSHIP_SHARED:
        rewritten = []
        for cfg in models:
            cfg = dict(cfg)
            cfg["feature_tag"] = feature_tag_for_ownership(
                cfg["feature_tag"], args.ownership)
            rewritten.append(cfg)
        models = rewritten
    if "align" in stages:
        stage_align(models, args.sections, ownership=args.ownership)
    if "extract" in stages:
        stage_extract(
            models, args.sections, args.device, args.dtype,
            ownership=args.ownership)
    if "smoke" in stages:
        stage_smoke(models, args.subjects, args.max_channels or 4, args.sections)
    if "temporal_full" in stages:
        stage_temporal_full(
            models, subjects, args.sections, args.max_channels,
            lang_only=args.lang_only)
    else:
        if "lag_resolved" in stages:
            stage_lag_resolved(
                models, subjects, args.max_channels, lang_only=args.lang_only)
        if "temporal_kernels" in stages:
            stage_temporal_kernels(
                models, subjects, args.sections, args.max_channels,
                lang_only=args.lang_only)
    if "single_lag_baseline" in stages:
        stage_single_lag_baseline(models, subjects, args.sections)
    if "baselines" in stages:
        cmd = [
            PY, "-m", "sparse_encoding.sparse_encoding_dense_baseline",
            "--features", "glove", "gpt2cn_l24",
            "--surprisal_tag", models[0]["feature_tag"],
            "--with_surprisal", "--lang_only",
            "--out_suffix", "_cohort",
        ]
        if subjects:
            cmd += ["--subjects", *subjects]
        run(cmd)
    if "null" in stages:
        stage_null(models, subjects, args.sections,
                   args.max_channels, args.n_perm)
    primary_tag = models[0]["feature_tag"]
    if "qualitative" in stages:
        qwen_cfg = next(
            (c for c in models if "qwen" in c["feature_tag"]), models[0])
        results = ap.SAE_TABLES / f"sparse_encoding_results_{qwen_cfg['feature_tag']}_cohort.csv"
        if not results.exists():
            results = ap.SAE_TABLES / f"sparse_encoding_results_{qwen_cfg['feature_tag']}.csv"
        cmd = [
            PY, "-m", "sparse_encoding.sparse_encoding_qualitative",
            "--feature_tag", qwen_cfg["feature_tag"],
            "--results_csv", str(results),
            "--all_channels", "--top_n", "0",
            "--suffix", f"_{qwen_cfg['feature_tag']}_all_channels",
        ]
        if subjects:
            cmd += ["--subjects", *subjects]
        run(cmd)
    if "task_type" in stages:
        feat = ap.SAE_TABLES / f"sparse_encoding_features_{primary_tag}_all_channels.csv"
        if not feat.exists():
            feat = ap.SAE_FEATURES_QWEN_ALL
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_task_type_analysis",
            "--feature_tag", primary_tag if "qwen" in primary_tag else QWEN["feature_tag"],
            "--features_csv", str(feat),
            "--n_perm", str(min(args.n_perm, 500)),
        ])
    if "paper" in stages:
        results = ap.SAE_TABLES / f"sparse_encoding_results_{primary_tag}_cohort.csv"
        if not results.exists():
            results = ap.SAE_TABLES / f"sparse_encoding_results_{primary_tag}.csv"
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_paper_aligned",
            "--results_csv", str(results),
            "--n_perm", str(args.n_perm),
        ])
    if "dominance" in stages:
        results = ap.SAE_TABLES / f"sparse_encoding_results_{primary_tag}_cohort.csv"
        if not results.exists():
            results = ap.SAE_TABLES / f"sparse_encoding_results_{primary_tag}.csv"
        run([
            PY, "-m", "sparse_encoding.sparse_encoding_surprisal_dominance",
            "--results_csv", str(results),
            "--n_perm", str(args.n_perm),
        ])
    if "report" in stages:
        stage_report(models)
    print(f"\nValidation finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
