"""
Canonical paths for the SAE sparse-encoding pipelines.

This module lives in the **sae_sparse_encoding** scripts repo. Data and outputs
stay in the parent **analysisEV** project directory (one level up by default
when this repo sits under analysisEV/, or set via env).

Override the data root with the environment variable ANALYSISEV_DATA_ROOT if
your data live elsewhere:

    export ANALYSISEV_DATA_ROOT=/path/to/analysisEV
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Roots ────────────────────────────────────────────────────────────────────
# This module lives in ev_analysis/core/, so the repo root is one level up.
SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
DATA_ROOT = Path(
    os.environ.get("ANALYSISEV_DATA_ROOT", REPO_ROOT.parent)
).resolve()
RESULTS_ROOT = DATA_ROOT / "group_encoding_results"
GROUP_ENCODING_RESULTS = RESULTS_ROOT

# ── Atlas volumes (fetched by download_atlases.sh; NIfTI blobs gitignored) ───
ATLAS_DIR = REPO_ROOT / "atlases"
AAL3_NIFTI = ATLAS_DIR / "aal3" / "AAL3v1_1mm.nii.gz"
AAL3_LUT = ATLAS_DIR / "aal3" / "AAL3v1_1mm.nii.txt"
AAL3_ALIAS_CSV = ATLAS_DIR / "aal3_alias.csv"
SCHAEFER200_17_NIFTI = (
    ATLAS_DIR
    / "schaefer_200_17"
    / "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.nii.gz"
)
SCHAEFER200_17_LUT = (
    ATLAS_DIR
    / "schaefer_200_17"
    / "Schaefer2018_200Parcels_17Networks_order.lut"
)
ANATOMY_QC_DIR = GROUP_ENCODING_RESULTS / "anatomy_qc"

# Pre-extracted linguistic features (GloVe, GPT2-CN, Qwen, SAE latents, …)
FEATURES_DIR = DATA_ROOT / "extracted_linguistic_features"
FEATURES_DIR = FEATURES_DIR

# ── Shared caches (encoding_group writes, taxonomy reads) ────────────────────
CACHE_DIR = RESULTS_ROOT / "cache"
PER_SUBJECT_DIR = CACHE_DIR / "per_subject"
PER_SUBJECT_VP_DIR = CACHE_DIR / "per_subject_vp"
APERIC_CACHE = CACHE_DIR / "aperiodic_cache.pkl"

# ── encoding_group pipeline ──────────────────────────────────────────────────
ENC_ROOT = RESULTS_ROOT / "encoding_group"
ENC_TABLES = ENC_ROOT / "tables"
ENC_FIGURES = ENC_ROOT / "figures"
ENC_SUMMARY = ENC_ROOT / "group_summary.pkl"

GROUP_ELECTRODE_TABLE = ENC_TABLES / "group_electrode_table.csv"
GROUP_ROI_STATS = ENC_TABLES / "group_roi_stats.csv"
LANG_VS_NONLANG_STATS = ENC_TABLES / "lang_vs_nonlang_stats.csv"
ENCODING_VS_LOCALIZER_OVERLAP = ENC_TABLES / "encoding_vs_localizer_overlap.csv"
WM_HARD_EASY_BY_CATEGORY = ENC_TABLES / "wm_hard_easy_by_category.csv"
APERIC_ENCODING_CORRELATION = ENC_TABLES / "aperiodic_encoding_correlation.csv"
APERIC_BY_CHANNEL_CATEGORY = ENC_TABLES / "aperiodic_by_channel_category.csv"
CHANNEL_TYPE_ANATOMY = ENC_TABLES / "channel_type_anatomy.csv"
ROI_CONTINGENCY_TABLE = ENC_TABLES / "roi_contingency_table.csv"
TEMPORAL_HIERARCHY_STATS = ENC_TABLES / "temporal_hierarchy_stats.csv"

# ── functional_taxonomy pipeline ─────────────────────────────────────────────
TAX_ROOT = RESULTS_ROOT / "functional_taxonomy"
TAX_TABLES = TAX_ROOT / "tables"
TAX_REPORTS = TAX_ROOT / "reports"
TAX_FIGURES = TAX_ROOT / "figures"
TAX_SUMMARY = TAX_ROOT / "functional_taxonomy_summary.pkl"

TAX_ELECTRODE_TABLE = TAX_TABLES / "functional_taxonomy_electrode_table.csv"
BUILDUP_LINKAGE = TAX_TABLES / "buildup_linkage.csv"
PER_SUBJECT_BUILDUP = TAX_TABLES / "per_subject_buildup.csv"
ROI_FUNCTIONAL_PROFILES = TAX_TABLES / "roi_functional_profiles.csv"
ECS_ALIGNMENT_REPORT = TAX_TABLES / "ecs_alignment_report.csv"
ECS_CURRENT_CORRELATIONS = TAX_TABLES / "ecs_current_correlations.csv"
ECS_DISTANCE_PREDICTIONS = TAX_TABLES / "ecs_distance_predictions.csv"
BUILDUP_REGRESSION = TAX_REPORTS / "buildup_regression.txt"
ECS_LOGISTIC_SUMMARY = TAX_REPORTS / "ecs_logistic_summary.txt"
ECS_DISTANCE_REGRESSION = TAX_REPORTS / "ecs_distance_regression.txt"

# ── Augmented Sparse Encoding (Lepori et al.) ────────────────────────────────
# SAE-feature + surprisal interpretable encoding models.
# SAE_RESULTS_DIRNAME=sparse_encoding_v2 sends regression tables/reports to the
# v2 root. The default stays the v1 directory.
SAE_ROOT = RESULTS_ROOT / os.environ.get("SAE_RESULTS_DIRNAME", "sparse_encoding")
SAE_TABLES = SAE_ROOT / "tables"
SAE_REPORTS = SAE_ROOT / "reports"
SAE_FIGURES = SAE_ROOT / "figures"
SAE_REGRESSION_TABLE = SAE_TABLES / "sparse_encoding_results.csv"
SAE_QUALITATIVE_TABLE = SAE_TABLES / "sparse_encoding_features.csv"
SAE_KERNEL_TABLE = SAE_TABLES / "sparse_encoding_kernels.csv"
SAE_NULL_TABLE = SAE_TABLES / "sparse_encoding_deconv_nulls.csv"
SAE_DENSE_TABLE = SAE_TABLES / "sparse_encoding_dense_baselines.csv"
SAE_DOMINANCE_TABLE = SAE_TABLES / "sparse_encoding_surprisal_dominance_all_channels.csv"
SAE_DOMINANCE_BY_LOCALIZER = SAE_TABLES / "sparse_encoding_dominance_by_localizer.csv"
SAE_DOMINANCE_LOCALIZER_STATS = SAE_TABLES / "sparse_encoding_dominance_localizer_stats.csv"
SAE_DOMINANCE_TASK_TYPE_STATS = SAE_TABLES / "sparse_encoding_dominance_task_type_pairwise.csv"
# Paper-aligned (Lepori et al.): full vs surprisal-only is primary.
SAE_PAPER_CHANNEL_TABLE = SAE_TABLES / "sparse_encoding_paper_aligned_channels.csv"
SAE_PAPER_SUMMARY_TABLE = SAE_TABLES / "sparse_encoding_paper_aligned_summary.csv"
SAE_PAPER_STATS_TABLE = SAE_TABLES / "sparse_encoding_paper_aligned_stats.csv"
SAE_PAPER_REPORT = SAE_REPORTS / "sparse_encoding_paper_aligned.txt"
SAE_FEATURES_QWEN_ALL = SAE_TABLES / "sparse_encoding_features_sae_qwen3_8b_l18_all_channels.csv"
SAE_FEATURES_QWEN35_ALL = SAE_TABLES / "sparse_encoding_features_sae_qwen35_4b_mat_l15_all_channels.csv"
SAE_STUDY3_ELECTRODE_TABLE = SAE_TABLES / "lepori_study3_electrodes.csv"
SAE_STUDY3_SUMMARY_TABLE = SAE_TABLES / "lepori_study3_froi_summary.csv"
SAE_STUDY3_STATS_TABLE = SAE_TABLES / "lepori_study3_froi_stats.csv"
SAE_STUDY3_DENSE_TABLE = SAE_TABLES / "lepori_study3_dense_vs_sae.csv"
SAE_STUDY3_FEATURE_TABLE = SAE_TABLES / "lepori_study3_feature_sharing.csv"
SAE_STUDY3_NAMED_TABLE = SAE_TABLES / "lepori_study3_named_latents.csv"
SAE_STUDY3_BIN_TABLE = SAE_TABLES / "lepori_study3_matryoshka_bins.csv"
SAE_STUDY3_REPORT = SAE_REPORTS / "lepori_study3_ieeg.txt"
SAE_STUDY3_QWEN_RESID = SAE_TABLES / "sparse_encoding_dense_baselines_qwen_resid_lang.csv"
SAE_STUDY3_GEMMA_RESID = SAE_TABLES / "sparse_encoding_dense_baselines_gemma_resid_lang.csv"
SAE_STUDY3_QWEN35_RESID = SAE_TABLES / "sparse_encoding_dense_baselines_qwen35_resid_lang.csv"
LANA_NII = DATA_ROOT / "SPM" / "LanA_n806.nii"
SAE_STUDY3_LANA_ELECTRODES = SAE_TABLES / "lepori_study3_lana_electrodes.csv"
SAE_STUDY3_LANA_SUMMARY = SAE_TABLES / "lepori_study3_lana_summary.csv"
SAE_STUDY3_LANA_STATS = SAE_TABLES / "lepori_study3_lana_stats.csv"
SAE_STUDY3_LANA_SWEEP = SAE_TABLES / "lepori_study3_lana_threshold_sweep.csv"
SAE_STUDY3_LANA_REPORT = SAE_REPORTS / "lepori_study3_lana.txt"
SAE_TASK_TYPE_JACCARD_UNION = SAE_TABLES / "sparse_encoding_task_type_jaccard_union.csv"
SAE_TASK_TYPE_JACCARD_PAIRWISE = SAE_TABLES / "sparse_encoding_task_type_jaccard_pairwise.csv"
SAE_TASK_TYPE_JACCARD_SUBJECT = SAE_TABLES / "sparse_encoding_task_type_jaccard_subject.csv"
SAE_TASK_TYPE_JACCARD_STATS = SAE_TABLES / "sparse_encoding_task_type_jaccard_stats.csv"
SAE_TASK_TYPE_FEATURE_FREQ = SAE_TABLES / "sparse_encoding_task_type_feature_freq.csv"
SAE_TASK_TYPE_REPORT = SAE_REPORTS / "sparse_encoding_task_type_analysis.txt"
# Temporal-precision primary outputs (lag-resolved + kernels).
SAE_LAG_SCREEN_TABLE = SAE_TABLES / "sparse_encoding_lag_screen.csv"
SAE_LAG_SCREEN_LONG = SAE_TABLES / "sparse_encoding_lag_screen_long.csv"
SAE_LAG_SCREEN_PEAKS = SAE_TABLES / "sparse_encoding_lag_screen_peaks.csv"
SAE_LAG_BY_REGION = SAE_TABLES / "sparse_encoding_lag_by_region.csv"
SAE_LAG_REGION_PAIRWISE = SAE_TABLES / "sparse_encoding_lag_region_pairwise.csv"
SAE_KERNEL_SHAPES = SAE_TABLES / "sparse_encoding_kernel_shapes.csv"
SAE_TEMPORAL_REPORT = SAE_REPORTS / "sparse_encoding_temporal_summary.txt"
SAE_LAG_SCREEN_REPORT = SAE_REPORTS / "sparse_encoding_lag_screen.txt"
CROSS_TASK_RESPONSES = ENC_TABLES / "cross_task_electrode_responses.csv"

# ── Boundary-gated contextual integration ────────────────────────────────────
BOUNDARY_ROOT = RESULTS_ROOT / "boundary_gating"
BOUNDARY_TABLES = BOUNDARY_ROOT / "tables"
BOUNDARY_FIGURES = BOUNDARY_ROOT / "figures"
BOUNDARY_REPORTS = BOUNDARY_ROOT / "reports"
BOUNDARY_CACHE = BOUNDARY_ROOT / "cache"

BOUNDARY_SYNTAX_CSV = BOUNDARY_TABLES / "word_syntax_features.csv"
BOUNDARY_SYNTAX_QC = BOUNDARY_REPORTS / "syntax_alignment_qc.txt"
BOUNDARY_LM_META = BOUNDARY_TABLES / "boundary_lm_extraction_meta.json"
BOUNDARY_ELECTRODE_TABLE = BOUNDARY_TABLES / "boundary_gate_electrode_table.csv"
BOUNDARY_LAG_TABLE = BOUNDARY_TABLES / "boundary_gate_lag_curves.csv"
BOUNDARY_LINKAGE_TABLE = BOUNDARY_TABLES / "boundary_gate_linkage.csv"
BOUNDARY_LINKAGE_STATS = BOUNDARY_TABLES / "boundary_gate_linkage_stats.csv"
BOUNDARY_CLOSING_TABLE = BOUNDARY_TABLES / "constituent_closing_responses.csv"
BOUNDARY_CLOSING_STATS = BOUNDARY_TABLES / "constituent_closing_stats.csv"
BOUNDARY_SUMMARY = BOUNDARY_REPORTS / "boundary_gating_summary.txt"

# ── Confirmatory boundary-gating (revised estimands / controls) ──────────────
CONF_ROOT = RESULTS_ROOT / "boundary_gating_confirmatory"
CONF_TABLES = CONF_ROOT / "tables"
CONF_FIGURES = CONF_ROOT / "figures"
CONF_REPORTS = CONF_ROOT / "reports"
CONF_CACHE = CONF_ROOT / "cache"

CONF_EDGES_CSV = CONF_TABLES / "boundary_edges.csv"
CONF_WORD_CSV = CONF_TABLES / "word_syntax_features_confirmatory.csv"
CONF_PSEUDO_CSV = CONF_TABLES / "pseudo_boundary_edges.csv"
CONF_EDGE_QC = CONF_REPORTS / "edge_boundary_qc.txt"
CONF_LM_META = CONF_TABLES / "confirmatory_lm_extraction_meta.json"
CONF_CROSSFIT_DIR = CONF_TABLES / "localizer_crossfit"
CONF_CROSSFIT_CSV = CONF_TABLES / "localizer_crossfit_electrode_table.csv"
CONF_ELECTRODE_TABLE = CONF_TABLES / "boundary_gate_electrode_table.csv"
CONF_SUBJECT_TABLE = CONF_TABLES / "boundary_gate_subject_table.csv"
CONF_LAG_TABLE = CONF_TABLES / "boundary_gate_lag_curves.csv"
CONF_MATCH_QC = CONF_TABLES / "within_post_match_diagnostics.csv"
CONF_DECONV_TABLE = CONF_TABLES / "joint_deconv_electrode_table.csv"
CONF_LINKAGE_STATS = CONF_TABLES / "boundary_gate_linkage_stats.csv"
CONF_SUMMARY = CONF_REPORTS / "boundary_gating_confirmatory_summary.txt"

# ── Confirmatory boundary-gating v2 (robustness audit; preserves v1) ─────────
CONF_V2_ROOT = RESULTS_ROOT / "boundary_gating_confirmatory_v2"
CONF_V2_TABLES = CONF_V2_ROOT / "tables"
CONF_V2_FIGURES = CONF_V2_ROOT / "figures"
CONF_V2_REPORTS = CONF_V2_ROOT / "reports"
CONF_V2_CACHE = CONF_V2_ROOT / "cache"
CONF_V2_AUDIT = CONF_V2_ROOT / "audit"

CONF_V2_EDGES_CSV = CONF_V2_TABLES / "boundary_edges.csv"
CONF_V2_WORD_CSV = CONF_V2_TABLES / "word_syntax_features_confirmatory.csv"
CONF_V2_PSEUDO_CSV = CONF_V2_TABLES / "pseudo_boundary_edges.csv"
CONF_V2_EVENTS_CSV = CONF_V2_TABLES / "boundary_events.csv"
CONF_V2_SHAM_EVENTS_CSV = CONF_V2_TABLES / "sham_boundary_events.csv"
CONF_V2_EVENT_QC = CONF_V2_REPORTS / "boundary_event_qc.txt"
CONF_V2_CONTEXT_DOSE = CONF_V2_TABLES / "context_dose_diagnostics.csv"
CONF_V2_LM_META = CONF_V2_TABLES / "confirmatory_v2_lm_extraction_meta.json"
CONF_V2_ELECTRODE_TABLE = CONF_V2_TABLES / "boundary_gate_electrode_table.csv"
CONF_V2_SUBJECT_TABLE = CONF_V2_TABLES / "boundary_gate_subject_table.csv"
CONF_V2_EVENT_LOSSES = CONF_V2_TABLES / "boundary_gate_event_losses.csv"
CONF_V2_LAG_TABLE = CONF_V2_TABLES / "boundary_gate_lag_curves.csv"
CONF_V2_MATCH_QC = CONF_V2_TABLES / "within_post_match_diagnostics.csv"
CONF_V2_DECONV_TABLE = CONF_V2_TABLES / "joint_deconv_electrode_table.csv"
CONF_V2_DECONV_LAGS = CONF_V2_TABLES / "joint_deconv_lag_interaction.csv"
CONF_V2_LINKAGE_STATS = CONF_V2_TABLES / "boundary_gate_linkage_stats.csv"
CONF_V2_SUMMARY = CONF_V2_REPORTS / "boundary_gating_confirmatory_v2_summary.txt"

# ── Nature-aligned surprisal follow-up (exploratory; preserves v1/v2) ────────
NATURE_ROOT = RESULTS_ROOT / "boundary_gating_nature_followup"
NATURE_TABLES = NATURE_ROOT / "tables"
NATURE_FIGURES = NATURE_ROOT / "figures"
NATURE_REPORTS = NATURE_ROOT / "reports"
NATURE_CACHE = NATURE_ROOT / "cache"
NATURE_AUDIT = NATURE_ROOT / "audit"

NATURE_WORD_CSV = NATURE_TABLES / "word_syntax_features_nature.csv"
NATURE_EDGES_CSV = NATURE_TABLES / "boundary_edges_nature.csv"
NATURE_EVENTS_CSV = NATURE_TABLES / "major_boundary_events.csv"
NATURE_SHAM_EVENTS_CSV = NATURE_TABLES / "sham_major_boundary_events.csv"
NATURE_SENT_BENCH_CSV = NATURE_TABLES / "sentence_boundary_benchmark_pairs.csv"
NATURE_EVENT_QC = NATURE_REPORTS / "nature_event_qc.txt"
NATURE_SURP_META = NATURE_TABLES / "nature_surprisal_extraction_meta.json"
NATURE_SURP_ELECTRODE = NATURE_TABLES / "surprisal_gate_electrode_table.csv"
NATURE_SURP_SUBJECT = NATURE_TABLES / "surprisal_gate_subject_table.csv"
NATURE_SURP_EVENT_LOSSES = NATURE_TABLES / "surprisal_gate_event_losses.csv"
NATURE_SURP_LAG = NATURE_TABLES / "surprisal_gate_lag_curves.csv"
NATURE_BENCH_ELECTRODE = NATURE_TABLES / "sentence_benchmark_electrode_table.csv"
NATURE_BENCH_SUBJECT = NATURE_TABLES / "sentence_benchmark_subject_table.csv"
NATURE_HISTORY_ELECTRODE = NATURE_TABLES / "history_gate_electrode_table.csv"
NATURE_HISTORY_SUBJECT = NATURE_TABLES / "history_gate_subject_table.csv"
NATURE_DECONV_TABLE = NATURE_TABLES / "surprisal_deconv_electrode_table.csv"
NATURE_DECONV_LAGS = NATURE_TABLES / "surprisal_deconv_lag_curves.csv"
NATURE_INFERENCE_JSON = NATURE_TABLES / "nature_followup_inference_tiers.json"
NATURE_SUMMARY = NATURE_REPORTS / "boundary_gating_nature_followup_summary.txt"

# ── Nature-style β/Spearman replication attempt (preserves nature_followup) ──
NATURE_REP_ROOT = RESULTS_ROOT / "boundary_gating_nature_replication"
NATURE_REP_TABLES = NATURE_REP_ROOT / "tables"
NATURE_REP_FIGURES = NATURE_REP_ROOT / "figures"
NATURE_REP_REPORTS = NATURE_REP_ROOT / "reports"
NATURE_REP_CACHE = NATURE_REP_ROOT / "cache"
NATURE_REP_AUDIT = NATURE_REP_ROOT / "audit"

NATURE_REP_METHODS = NATURE_REP_REPORTS / "zenodo_method_match.md"
NATURE_REP_ELECTRODE = NATURE_REP_TABLES / "coupling_electrode_table.csv"
NATURE_REP_SUBJECT = NATURE_REP_TABLES / "coupling_subject_table.csv"
NATURE_REP_SENT_ELECTRODE = NATURE_REP_TABLES / "sentence_coupling_electrode_table.csv"
NATURE_REP_SENT_SUBJECT = NATURE_REP_TABLES / "sentence_coupling_subject_table.csv"
NATURE_REP_LAG = NATURE_REP_TABLES / "coupling_lag_curves.csv"
NATURE_REP_INFERENCE = NATURE_REP_TABLES / "nature_replication_inference.json"
NATURE_REP_SUMMARY = NATURE_REP_REPORTS / "boundary_gating_nature_replication_summary.txt"

# ── Nature-style TRF de-enveloping + full lag surface (preserves word-level rep) ──
NATURE_REP_TRF_ROOT = RESULTS_ROOT / "boundary_gating_nature_replication_trf"
NATURE_REP_TRF_TABLES = NATURE_REP_TRF_ROOT / "tables"
NATURE_REP_TRF_FIGURES = NATURE_REP_TRF_ROOT / "figures"
NATURE_REP_TRF_REPORTS = NATURE_REP_TRF_ROOT / "reports"
NATURE_REP_TRF_CACHE = NATURE_REP_TRF_ROOT / "cache"
NATURE_REP_TRF_AUDIT = NATURE_REP_TRF_ROOT / "audit"

NATURE_REP_TRF_METHODS = NATURE_REP_TRF_REPORTS / "zenodo_mne_trf_method_match.md"
NATURE_REP_TRF_ELECTRODE = NATURE_REP_TRF_TABLES / "coupling_electrode_table.csv"
NATURE_REP_TRF_SUBJECT = NATURE_REP_TRF_TABLES / "coupling_subject_table.csv"
NATURE_REP_TRF_SENT_ELECTRODE = NATURE_REP_TRF_TABLES / "sentence_coupling_electrode_table.csv"
NATURE_REP_TRF_SENT_SUBJECT = NATURE_REP_TRF_TABLES / "sentence_coupling_subject_table.csv"
NATURE_REP_TRF_LAG = NATURE_REP_TRF_TABLES / "coupling_lag_curves.csv"
NATURE_REP_TRF_SENT_LAG = NATURE_REP_TRF_TABLES / "sentence_coupling_lag_curves.csv"
NATURE_REP_TRF_INFERENCE = NATURE_REP_TRF_TABLES / "nature_replication_trf_inference.json"
NATURE_REP_TRF_SUMMARY = NATURE_REP_TRF_REPORTS / "boundary_gating_nature_replication_trf_summary.txt"
NATURE_REP_TRF_RESID_DIR = NATURE_REP_TRF_CACHE / "trf_resid"
NATURE_REP_TRF_SENT_NULL_DIR = NATURE_REP_TRF_CACHE / "sent_circshift"
NATURE_REP_TRF_BOUNDARY_VAL = NATURE_REP_TRF_REPORTS / "boundary_restore_validation.md"
NATURE_REP_TRF_SENT_BENCH_RESTORE = (
    NATURE_REP_TRF_TABLES / "sentence_boundary_benchmark_pairs_restore.csv"
)
NATURE_REP_TRF_SENT_BENCH_CTB = (
    NATURE_REP_TRF_TABLES / "sentence_boundary_benchmark_pairs_ctb.csv"
)
NATURE_REP_TRF_PERCLASS = NATURE_REP_TRF_TABLES / "sentence_coupling_perclass_restore.csv"
# Archived CTB-prior tables (side-by-side with restore)
NATURE_REP_TRF_INFERENCE_CTB_PRIOR = (
    NATURE_REP_TRF_TABLES / "nature_replication_trf_inference_ctb_prior.json"
)

# ── M400c electrode-gate repair (post-hoc; does not overwrite legacy tables) ──
NATURE_REP_TRF_M400C_CACHE = NATURE_REP_TRF_CACHE / "m400c_repair"
NATURE_REP_TRF_M400C_QC_JSON = NATURE_REP_TRF_REPORTS / "m400c_surprisal_qc.json"
NATURE_REP_TRF_M400C_QC_TABLE = NATURE_REP_TRF_TABLES / "m400c_surprisal_qc.csv"
NATURE_REP_TRF_M400C_LAG = NATURE_REP_TRF_TABLES / "m400c_corrected_lag_curves.csv"
NATURE_REP_TRF_M400C_NULL = NATURE_REP_TRF_TABLES / "m400c_corrected_null_table.csv"
NATURE_REP_TRF_M400C_CHANNELS = NATURE_REP_TRF_TABLES / "m400c_corrected_channel_masks.csv"
NATURE_REP_TRF_M400C_CTB = NATURE_REP_TRF_TABLES / "m400c_corrected_ctb_heldout.csv"
NATURE_REP_TRF_M400C_FIR_KERNELS = NATURE_REP_TRF_TABLES / "m400c_fir_kernels.csv"
NATURE_REP_TRF_M400C_FIR_JSON = NATURE_REP_TRF_TABLES / "m400c_fir_confirmation.json"
NATURE_REP_TRF_REPAIR_INFERENCE = (
    NATURE_REP_TRF_TABLES / "nature_replication_trf_m400c_repair_inference.json"
)
NATURE_REP_TRF_REPAIR_REPORT = NATURE_REP_TRF_REPORTS / "m400c_gate_repair_summary.txt"

# ── Zou et al. literal notebook port (NeuralSurChunk_deenv_{Cal,Vis}.ipynb) ───
# Minimal, faithful reproduction of the two Zenodo notebooks end-to-end:
# in-sample TRF residual, pooled ±2000ms lag grid electrode selection,
# 101-draw + surprisal-matched balancing, literal LME_Test, Fig 7A/7B.
ZOU_LITERAL_ROOT = RESULTS_ROOT / "zou_literal_port"
ZOU_LITERAL_TABLES = ZOU_LITERAL_ROOT / "tables"
ZOU_LITERAL_FIGURES = ZOU_LITERAL_ROOT / "figures"
ZOU_LITERAL_REPORTS = ZOU_LITERAL_ROOT / "reports"
ZOU_LITERAL_CACHE = ZOU_LITERAL_ROOT / "cache"
ZOU_LITERAL_RESID_DIR = ZOU_LITERAL_CACHE / "trf_resid_insample"
# Cal notebook intermediate_plot/ analogue (ERP before/pred/resid + TRF kernels)
ZOU_LITERAL_INTERMEDIATE = ZOU_LITERAL_FIGURES / "intermediate_plot"

ZOU_LITERAL_PICKLE_DIR = ZOU_LITERAL_CACHE / "neural_surp_pickles"
ZOU_LITERAL_ELECTRODE = ZOU_LITERAL_TABLES / "zou_literal_electrode_table.csv"
ZOU_LITERAL_SUBJECT = ZOU_LITERAL_TABLES / "zou_literal_subject_table.csv"
ZOU_LITERAL_LME = ZOU_LITERAL_TABLES / "zou_literal_lme.json"
ZOU_LITERAL_SUMMARY = ZOU_LITERAL_REPORTS / "zou_literal_port_summary.txt"

# Zou Cal/Vis recipe with confirmatory sentence-pair roles (not Zou Boundaries).
# Shares ZOU_LITERAL_RESID_DIR; writes its own tables/figures/reports.
ZOU_ALIGNED_OUR_ROLES_ROOT = RESULTS_ROOT / "zou_aligned_our_roles"
ZOU_ALIGNED_OUR_ROLES_TABLES = ZOU_ALIGNED_OUR_ROLES_ROOT / "tables"
ZOU_ALIGNED_OUR_ROLES_FIGURES = ZOU_ALIGNED_OUR_ROLES_ROOT / "figures"
ZOU_ALIGNED_OUR_ROLES_REPORTS = ZOU_ALIGNED_OUR_ROLES_ROOT / "reports"
ZOU_ALIGNED_OUR_ROLES_ELECTRODE = (
    ZOU_ALIGNED_OUR_ROLES_TABLES / "zou_literal_electrode_table.csv"
)
ZOU_ALIGNED_OUR_ROLES_LME = ZOU_ALIGNED_OUR_ROLES_TABLES / "zou_literal_lme.json"
ZOU_ALIGNED_OUR_ROLES_SUMMARY = (
    ZOU_ALIGNED_OUR_ROLES_REPORTS / "zou_aligned_our_roles_summary.txt"
)

# Zou recipe with CTB is_sent_start roles (not punctuation-restore Boundaries).
# Shares ZOU_LITERAL_RESID_DIR; writes its own tables/figures/reports.
ZOU_LITERAL_CTB_ROLES_ROOT = RESULTS_ROOT / "zou_literal_ctb_roles"
ZOU_LITERAL_CTB_ROLES_TABLES = ZOU_LITERAL_CTB_ROLES_ROOT / "tables"
ZOU_LITERAL_CTB_ROLES_FIGURES = ZOU_LITERAL_CTB_ROLES_ROOT / "figures"
ZOU_LITERAL_CTB_ROLES_REPORTS = ZOU_LITERAL_CTB_ROLES_ROOT / "reports"
ZOU_LITERAL_CTB_ROLES_SUMMARY = (
    ZOU_LITERAL_CTB_ROLES_REPORTS / "zou_literal_ctb_roles_summary.txt"
)

# Zou recipe with CTB phrase-initial roles (NP/VP/PP/LCP spans, not S).
ZOU_LITERAL_CTB_PHRASE_ROOT = RESULTS_ROOT / "zou_literal_ctb_phrase_roles"
ZOU_LITERAL_CTB_PHRASE_TABLES = ZOU_LITERAL_CTB_PHRASE_ROOT / "tables"
ZOU_LITERAL_CTB_PHRASE_FIGURES = ZOU_LITERAL_CTB_PHRASE_ROOT / "figures"
ZOU_LITERAL_CTB_PHRASE_REPORTS = ZOU_LITERAL_CTB_PHRASE_ROOT / "reports"
ZOU_LITERAL_CTB_PHRASE_SUMMARY = (
    ZOU_LITERAL_CTB_PHRASE_REPORTS / "zou_literal_ctb_phrase_roles_summary.txt"
)


def zou_port_layout(role_source: str = "zou_boundaries"):
    """Output layout for the Zou notebook port.

    ``zou_boundaries`` (default) is the literal Cal Boundaries role set.
    ``sentence_pairs`` keeps Zou's M400c / TRF / Spearman / LME recipe but uses
    the confirmatory restore sentence-benchmark pairs as initial/non-initial.
    ``ctb`` uses all CTB ``is_sent_start`` words vs the rest; writes under
    ``zou_literal_ctb_roles/`` so the punctuation-restore run is preserved.
    ``ctb_phrase`` uses first words of CTB NP/VP/PP/LCP spans (length ≥ 2)
    vs the rest; writes under ``zou_literal_ctb_phrase_roles/``.
    """
    from types import SimpleNamespace

    src = str(role_source).strip().lower()
    if src in ("sentence_pairs", "ours", "restore_pairs"):
        root = ZOU_ALIGNED_OUR_ROLES_ROOT
        summary = ZOU_ALIGNED_OUR_ROLES_SUMMARY
        role = "sentence_pairs" if src != "ours" else src
    elif src == "ctb":
        root = ZOU_LITERAL_CTB_ROLES_ROOT
        summary = ZOU_LITERAL_CTB_ROLES_SUMMARY
        role = "ctb"
    elif src in ("ctb_phrase", "phrase"):
        root = ZOU_LITERAL_CTB_PHRASE_ROOT
        summary = ZOU_LITERAL_CTB_PHRASE_SUMMARY
        role = "ctb_phrase"
    else:
        root = ZOU_LITERAL_ROOT
        summary = ZOU_LITERAL_SUMMARY
        role = "zou_boundaries"
    tables = root / "tables"
    figures = root / "figures"
    reports = root / "reports"
    return SimpleNamespace(
        role_source=role,
        root=root,
        tables=tables,
        figures=figures,
        reports=reports,
        electrode=tables / "zou_literal_electrode_table.csv",
        lme=tables / "zou_literal_lme.json",
        summary=summary,
        lag_curves=tables / "zou_literal_lag_curves.csv",
        chunk_effect=tables / "zou_literal_chunk_effect.csv",
        encoding_meta=tables / "zou_literal_encoding_meta.json",
    )

# ── Goldstein et al. (2025) temporal-hierarchy replication ───────────────────
# Layer-wise GPT2-CN encoding ↔ peak-lag hierarchy along the ventral stream.
GOLDSTEIN_ROOT = RESULTS_ROOT / "goldstein"
GOLDSTEIN_TABLES = GOLDSTEIN_ROOT / "tables"
GOLDSTEIN_FIGURES = GOLDSTEIN_ROOT / "figures"
GOLDSTEIN_REPORTS = GOLDSTEIN_ROOT / "reports"
GOLDSTEIN_SUMMARY_CSV = GOLDSTEIN_TABLES / "goldstein_summary.csv"
GOLDSTEIN_ENCODING_MATRICES = GOLDSTEIN_TABLES / "encoding_matrices.npz"
GOLDSTEIN_LMM_REPORT = GOLDSTEIN_REPORTS / "goldstein_lmm.txt"
GOLDSTEIN_PEAK_LAGS_CSV = GOLDSTEIN_TABLES / "goldstein_electrode_peak_lags.csv"

# ── Tuckute et al. (bioRxiv 2025.05.21.655330) sentence-PC adaptation ────────
# Method-matched 2D sentence decomposition on Mandarin naturalistic iEEG
# (not a stimulus-matched 7T English sentence-fMRI replication).
TUCKUTE_ROOT = RESULTS_ROOT / "tuckute_2d"
TUCKUTE_TABLES = TUCKUTE_ROOT / "tables"
TUCKUTE_FIGURES = TUCKUTE_ROOT / "figures"
TUCKUTE_REPORTS = TUCKUTE_ROOT / "reports"
TUCKUTE_CACHE = TUCKUTE_ROOT / "cache"
TUCKUTE_SUMMARY_JSON = TUCKUTE_TABLES / "tuckute_2d_summary.json"
TUCKUTE_SUMMARY_TXT = TUCKUTE_REPORTS / "tuckute_2d_summary.txt"
TUCKUTE_SENTENCE_TABLE = TUCKUTE_TABLES / "sentence_properties.csv"
TUCKUTE_PC_SCORES = TUCKUTE_TABLES / "sentence_pc_scores.csv"
TUCKUTE_PC_PROPERTY = TUCKUTE_TABLES / "pc_property_correlations.csv"
TUCKUTE_CV_TABLE = TUCKUTE_TABLES / "loso_nested_cv.csv"
TUCKUTE_ELECTRODE_WEIGHTS = TUCKUTE_TABLES / "electrode_pc_weights.csv"
TUCKUTE_SPATIAL_TABLE = TUCKUTE_TABLES / "spatial_pc_weights.csv"
TUCKUTE_FIELDTRIP_LABELS = TUCKUTE_TABLES / "fieldtrip_anatomy_labels.csv"
TUCKUTE_SPATIAL_FIELDTRIP = TUCKUTE_TABLES / "spatial_pc_weights_fieldtrip.csv"

# Word-level PC adaptation (iEEG temporal resolution).
TUCKUTE_WORD_PROPERTY_TABLE = TUCKUTE_TABLES / "word_properties.csv"
TUCKUTE_WORD_PC_SCORES = TUCKUTE_TABLES / "word_pc_scores.csv"
TUCKUTE_WORD_PC_PROPERTY = TUCKUTE_TABLES / "word_pc_property_correlations.csv"
TUCKUTE_WORD_CV_TABLE = TUCKUTE_TABLES / "word_loso_nested_cv.csv"
TUCKUTE_WORD_ELECTRODE_WEIGHTS = TUCKUTE_TABLES / "word_electrode_pc_weights.csv"
TUCKUTE_WORD_SPATIAL_TABLE = TUCKUTE_TABLES / "word_spatial_pc_weights.csv"
TUCKUTE_WORD_SUMMARY_JSON = TUCKUTE_TABLES / "word_pcs_summary.json"
TUCKUTE_WORD_SUMMARY_TXT = TUCKUTE_REPORTS / "word_pcs_summary.txt"

# ── SRM-Encoding (Bhattacharjee/Zada Nat Comput Sci port) ────────────────────
# Shared Response Model encoding on sig_glove / is_lang electrodes.
SRM_ENCODING_ROOT = RESULTS_ROOT / "srm_encoding"
SRM_ENCODING_CACHE = SRM_ENCODING_ROOT / "cache"
SRM_ENCODING_TABLES = SRM_ENCODING_ROOT / "tables"
SRM_ENCODING_FIGURES = SRM_ENCODING_ROOT / "figures"
SRM_ENCODING_REPORTS = SRM_ENCODING_ROOT / "reports"

# SRM-aligned word-level Tuckute 2D replication.
TUCKUTE_SRM_WORD_ROOT = TUCKUTE_ROOT / "srm_word_2d"
TUCKUTE_SRM_WORD_TABLES = TUCKUTE_SRM_WORD_ROOT / "tables"
TUCKUTE_SRM_WORD_FIGURES = TUCKUTE_SRM_WORD_ROOT / "figures"
TUCKUTE_SRM_WORD_REPORTS = TUCKUTE_SRM_WORD_ROOT / "reports"
TUCKUTE_SRM_WORD_SUMMARY_JSON = TUCKUTE_SRM_WORD_TABLES / "srm_word_2d_summary.json"
TUCKUTE_SRM_WORD_SUMMARY_TXT = TUCKUTE_SRM_WORD_REPORTS / "srm_word_2d_summary.txt"

# Within-subject Word-PCs (no cross-subject alignment).
TUCKUTE_WITHIN_SUBJECT_ROOT = TUCKUTE_ROOT / "within_subject_pcs"
TUCKUTE_WITHIN_SUBJECT_TABLES = TUCKUTE_WITHIN_SUBJECT_ROOT / "tables"
TUCKUTE_WITHIN_SUBJECT_FIGURES = TUCKUTE_WITHIN_SUBJECT_ROOT / "figures"
TUCKUTE_WITHIN_SUBJECT_REPORTS = TUCKUTE_WITHIN_SUBJECT_ROOT / "reports"

# Robustness checks for the Tuckute 2D adaptation.
TUCKUTE_ROBUSTNESS_ROOT = TUCKUTE_ROOT / "robustness"
TUCKUTE_ROBUSTNESS_TABLES = TUCKUTE_ROBUSTNESS_ROOT / "tables"
TUCKUTE_ROBUSTNESS_REPORTS = TUCKUTE_ROBUSTNESS_ROOT / "reports"
TUCKUTE_ROBUSTNESS_FIGURES = TUCKUTE_ROBUSTNESS_ROOT / "figures"
TUCKUTE_ROBUSTNESS_SUMMARY_JSON = (
    TUCKUTE_ROBUSTNESS_TABLES / "robustness_summary.json"
)
TUCKUTE_ROBUSTNESS_TIMING_REAL_TABLE = (
    TUCKUTE_ROBUSTNESS_TABLES / "timing_grid_real_k2.csv"
)
TUCKUTE_ROBUSTNESS_TIMING_RANDOMFOLDS_TABLE = (
    TUCKUTE_ROBUSTNESS_TABLES / "timing_grid_randomfolds_real_k2.csv"
)
TUCKUTE_ROBUSTNESS_NULL_MAXSTAT_TABLE = (
    TUCKUTE_ROBUSTNESS_TABLES / "narrative_shift_null_maxstat_k2.csv"
)
TUCKUTE_ROBUSTNESS_WITHIN_SUBJECT_TABLE = (
    TUCKUTE_ROBUSTNESS_TABLES / "within_subject_positive_control.csv"
)

# ── Fresh Tuckute sentence-PC replication (naturalistic HG) ─────────────────
TUCKUTE_SENTENCE_ROOT = RESULTS_ROOT / "tuckute_sentence_naturalistic"
TUCKUTE_SENTENCE_TABLES = TUCKUTE_SENTENCE_ROOT / "tables"
TUCKUTE_SENTENCE_FIGURES = TUCKUTE_SENTENCE_ROOT / "figures"
TUCKUTE_SENTENCE_REPORTS = TUCKUTE_SENTENCE_ROOT / "reports"
TUCKUTE_SENTENCE_CACHE = TUCKUTE_SENTENCE_ROOT / "cache"
TUCKUTE_SENTENCE_CONTROLS = TUCKUTE_SENTENCE_ROOT / "controls"
TUCKUTE_SENTENCE_MANIFEST = TUCKUTE_SENTENCE_ROOT / "manifest.json"
TUCKUTE_SENTENCE_QC = TUCKUTE_SENTENCE_TABLES / "qc_by_subject.csv"
TUCKUTE_SENTENCE_PROPERTIES = TUCKUTE_SENTENCE_TABLES / "sentence_properties.csv"
TUCKUTE_SENTENCE_CV = TUCKUTE_SENTENCE_TABLES / "loso_nested_cv.csv"
TUCKUTE_SENTENCE_PC_SCORES = TUCKUTE_SENTENCE_TABLES / "sentence_pc_scores_exploratory.csv"
TUCKUTE_SENTENCE_PC_PROPERTY = TUCKUTE_SENTENCE_TABLES / "pc_property_correlations_exploratory.csv"
TUCKUTE_SENTENCE_VARIANCE = TUCKUTE_SENTENCE_TABLES / "variance_exploratory.csv"
TUCKUTE_SENTENCE_WEIGHTS = TUCKUTE_SENTENCE_TABLES / "electrode_pc_weights_loso.csv"
TUCKUTE_SENTENCE_NULL_PC = TUCKUTE_SENTENCE_CONTROLS / "null_pc_permutation.csv"
TUCKUTE_SENTENCE_NULL_SHIFT = TUCKUTE_SENTENCE_CONTROLS / "null_narrative_shift.csv"
TUCKUTE_SENTENCE_SUMMARY_JSON = TUCKUTE_SENTENCE_REPORTS / "summary.json"
TUCKUTE_SENTENCE_SUMMARY_TXT = TUCKUTE_SENTENCE_REPORTS / "summary.txt"

# External MITSWJN trial-level crunched files (cross-fitting)
MITSWJN_CRUNCHED_ROOT = Path(
    os.environ.get(
        "MITSWJN_CRUNCHED_ROOT",
        "/mnt/f/seeg/luohong/analysisEV",
    )
)
MATLAB_EXE = Path(os.environ.get("MATLAB_EXE", "/mnt/d/2024b/bin/matlab.exe"))

# Parse resources (story-level)
LPP_CN_TREE = DATA_ROOT / "lppCN_tree.txt"
LPP_CN_DEPENDENCY = DATA_ROOT / "lppCN_dependency.csv"

# Convenience aliases used by older code
GROUP_DIR = RESULTS_ROOT
GROUP_OUTDIR = ENC_ROOT
ELECTRODE_TABLE_PATH = GROUP_ELECTRODE_TABLE
GROUP_SUMMARY_PATH = ENC_SUMMARY
FIG_DIR = TAX_FIGURES
VP_PER_SUBJECT_DIR = PER_SUBJECT_VP_DIR


def ensure_pipeline_dirs() -> None:
    """Create all standard output directories under DATA_ROOT."""
    for d in (
        CACHE_DIR,
        PER_SUBJECT_DIR,
        PER_SUBJECT_VP_DIR,
        ENC_TABLES,
        ENC_FIGURES,
        TAX_TABLES,
        TAX_REPORTS,
        TAX_FIGURES,
        SAE_TABLES,
        SAE_REPORTS,
        SAE_FIGURES,
        BOUNDARY_TABLES,
        BOUNDARY_FIGURES,
        BOUNDARY_REPORTS,
        BOUNDARY_CACHE,
        CONF_TABLES,
        CONF_FIGURES,
        CONF_REPORTS,
        CONF_CACHE,
        CONF_CROSSFIT_DIR,
        CONF_V2_TABLES,
        CONF_V2_FIGURES,
        CONF_V2_REPORTS,
        CONF_V2_CACHE,
        CONF_V2_AUDIT,
        NATURE_TABLES,
        NATURE_FIGURES,
        NATURE_REPORTS,
        NATURE_CACHE,
        NATURE_AUDIT,
        NATURE_REP_TABLES,
        NATURE_REP_FIGURES,
        NATURE_REP_REPORTS,
        NATURE_REP_CACHE,
        NATURE_REP_AUDIT,
        NATURE_REP_TRF_TABLES,
        NATURE_REP_TRF_FIGURES,
        NATURE_REP_TRF_REPORTS,
        NATURE_REP_TRF_CACHE,
        NATURE_REP_TRF_AUDIT,
        NATURE_REP_TRF_RESID_DIR,
        NATURE_REP_TRF_SENT_NULL_DIR,
        NATURE_REP_TRF_M400C_CACHE,
        ZOU_LITERAL_TABLES,
        ZOU_LITERAL_FIGURES,
        ZOU_LITERAL_REPORTS,
        ZOU_LITERAL_CACHE,
        ZOU_LITERAL_RESID_DIR,
        ZOU_LITERAL_INTERMEDIATE,
        ZOU_LITERAL_PICKLE_DIR,
        ZOU_ALIGNED_OUR_ROLES_TABLES,
        ZOU_ALIGNED_OUR_ROLES_FIGURES,
        ZOU_ALIGNED_OUR_ROLES_REPORTS,
        ZOU_LITERAL_CTB_ROLES_TABLES,
        ZOU_LITERAL_CTB_ROLES_FIGURES,
        ZOU_LITERAL_CTB_ROLES_REPORTS,
        ZOU_LITERAL_CTB_PHRASE_TABLES,
        ZOU_LITERAL_CTB_PHRASE_FIGURES,
        ZOU_LITERAL_CTB_PHRASE_REPORTS,
        ANATOMY_QC_DIR,
        GOLDSTEIN_TABLES,
        GOLDSTEIN_FIGURES,
        GOLDSTEIN_REPORTS,
        TUCKUTE_TABLES,
        TUCKUTE_FIGURES,
        TUCKUTE_REPORTS,
        TUCKUTE_CACHE,
        TUCKUTE_ROBUSTNESS_TABLES,
        TUCKUTE_ROBUSTNESS_FIGURES,
        TUCKUTE_ROBUSTNESS_REPORTS,
        SRM_ENCODING_CACHE,
        SRM_ENCODING_TABLES,
        SRM_ENCODING_FIGURES,
        SRM_ENCODING_REPORTS,
        TUCKUTE_SRM_WORD_TABLES,
        TUCKUTE_SRM_WORD_FIGURES,
        TUCKUTE_SRM_WORD_REPORTS,
        TUCKUTE_SENTENCE_TABLES,
        TUCKUTE_SENTENCE_FIGURES,
        TUCKUTE_SENTENCE_REPORTS,
        TUCKUTE_SENTENCE_CACHE,
        TUCKUTE_SENTENCE_CONTROLS,
    ):
        d.mkdir(parents=True, exist_ok=True)


# Legacy flat filenames → new locations (for migrate_results_layout.py)
_LEGACY_FILE_MAP: dict[Path, Path] = {
    RESULTS_ROOT / "group_electrode_table.csv": GROUP_ELECTRODE_TABLE,
    RESULTS_ROOT / "group_roi_stats.csv": GROUP_ROI_STATS,
    RESULTS_ROOT / "lang_vs_nonlang_stats.csv": LANG_VS_NONLANG_STATS,
    RESULTS_ROOT / "encoding_vs_localizer_overlap.csv": ENCODING_VS_LOCALIZER_OVERLAP,
    RESULTS_ROOT / "wm_hard_easy_by_category.csv": WM_HARD_EASY_BY_CATEGORY,
    RESULTS_ROOT / "aperiodic_encoding_correlation.csv": APERIC_ENCODING_CORRELATION,
    RESULTS_ROOT / "aperiodic_by_channel_category.csv": APERIC_BY_CHANNEL_CATEGORY,
    RESULTS_ROOT / "channel_type_anatomy.csv": CHANNEL_TYPE_ANATOMY,
    RESULTS_ROOT / "roi_contingency_table.csv": ROI_CONTINGENCY_TABLE,
    RESULTS_ROOT / "temporal_hierarchy_stats.csv": TEMPORAL_HIERARCHY_STATS,
    RESULTS_ROOT / "group_summary.pkl": ENC_SUMMARY,
    RESULTS_ROOT / "aperiodic_cache.pkl": APERIC_CACHE,
    RESULTS_ROOT / "functional_taxonomy_electrode_table.csv": TAX_ELECTRODE_TABLE,
    RESULTS_ROOT / "buildup_linkage.csv": BUILDUP_LINKAGE,
    RESULTS_ROOT / "per_subject_buildup.csv": PER_SUBJECT_BUILDUP,
    RESULTS_ROOT / "roi_functional_profiles.csv": ROI_FUNCTIONAL_PROFILES,
    RESULTS_ROOT / "ecs_alignment_report.csv": ECS_ALIGNMENT_REPORT,
    RESULTS_ROOT / "ecs_current_correlations.csv": ECS_CURRENT_CORRELATIONS,
    RESULTS_ROOT / "ecs_distance_predictions.csv": ECS_DISTANCE_PREDICTIONS,
    RESULTS_ROOT / "buildup_regression.txt": BUILDUP_REGRESSION,
    RESULTS_ROOT / "ecs_logistic_summary.txt": ECS_LOGISTIC_SUMMARY,
    RESULTS_ROOT / "ecs_distance_regression.txt": ECS_DISTANCE_REGRESSION,
    RESULTS_ROOT / "functional_taxonomy_summary.pkl": TAX_SUMMARY,
}

_LEGACY_DIR_MAP: dict[Path, Path] = {
    RESULTS_ROOT / "per_subject": PER_SUBJECT_DIR,
    RESULTS_ROOT / "per_subject_vp": PER_SUBJECT_VP_DIR,
    RESULTS_ROOT / "figures": ENC_FIGURES,
    RESULTS_ROOT / "figures" / "functional_taxonomy": TAX_FIGURES,
}
