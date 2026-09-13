#!/usr/bin/env bash
# Staged sparse-encoding cohort run with **temporal precision primary**.
#
# Order:
#   1) lag-resolved screening (peaks / FWHM / region contrasts)
#   2) continuous-time deconvolution + kernels + figures
#   3) optional single-lag baseline (300 ms) for Matryoshka / paper tables
#   4) dense baselines, nulls, qualitative, paper-aligned, report
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/home/ryan/miniforge3/envs/ieeg-encoding/bin/python}"
LOG_DIR="/home/ryan/analysisEV/group_encoding_results/sparse_encoding/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cohort_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "==== COHORT START $(date -Is) ===="
echo "LOG=$LOG"
echo "PRIMARY: lag_resolved + temporal_kernels (single-lag is baseline only)"

MODELS=(sae_gemma2_2b_mat_l12 sae_qwen3_8b_l18)
# Prefer Qwen3.5 Matryoshka when features exist.
if [[ -f "/home/ryan/analysisEV/extracted_linguistic_features/section_001/X_word_sae_qwen35_4b_mat_l15.npz" ]]; then
  MODELS+=(sae_qwen35_4b_mat_l15)
fi
N_PERM="${N_PERM:-100}"
RUN_SINGLE_LAG_BASELINE="${RUN_SINGLE_LAG_BASELINE:-1}"

echo "==== 1) LAG-RESOLVED (PRIMARY), lang-only ===="
for tag in "${MODELS[@]}"; do
  $PY -u -m sparse_encoding.sparse_encoding_lag_screen --feature_tag "$tag" --lang_only \
    --out_suffix "_${tag}_lang"
done

echo "==== 2) TEMPORAL KERNELS / DECONV (PRIMARY), lang-only ===="
for tag in "${MODELS[@]}"; do
  $PY -u -m sparse_encoding.sparse_encoding_deconvolution --feature_tag "$tag" \
    --roi_filter lang --out_suffix "_${tag}_lang"
  $PY -u -m sparse_encoding.sparse_encoding_plot_kernels \
    --kernels_csv "/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_kernels_${tag}_lang.csv" \
    --out_suffix "_${tag}_lang" || true
done

if [[ "$RUN_SINGLE_LAG_BASELINE" == "1" ]]; then
  echo "==== 3) SINGLE-LAG BASELINE (optional; default 300 ms) ===="
  for tag in "${MODELS[@]}"; do
    $PY -u -m sparse_encoding.sparse_encoding_regression --feature_tag "$tag" --n_jobs -1 \
      --out "/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_results_${tag}_cohort.csv"
    $PY -u -m sparse_encoding.sparse_encoding_summary \
      --results_csv "/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_results_${tag}_cohort.csv"
  done
else
  echo "==== 3) SINGLE-LAG BASELINE skipped (RUN_SINGLE_LAG_BASELINE=0) ===="
fi

echo "==== 4) DENSE BASELINES (glove + gpt2cn_l24), lang-only ===="
$PY -u -m sparse_encoding.sparse_encoding_dense_baseline \
  --features glove gpt2cn_l24 \
  --surprisal_tag sae_gemma2_2b_mat_l12 \
  --with_surprisal --lang_only \
  --out_suffix "_cohort"

echo "==== 5) ROI NULL CALIBRATION (circular ${N_PERM} + block ${N_PERM_BLOCK:-20}, mfg+temporal) ===="
N_PERM_BLOCK="${N_PERM_BLOCK:-20}"
for tag in "${MODELS[@]}"; do
  $PY -u -m sparse_encoding.sparse_encoding_deconvolution --feature_tag "$tag" \
    --roi_filter mfg_temporal \
    --null circular --null_perms "$N_PERM" --null_seed 19 \
    --out_suffix "_${tag}_roi_circular${N_PERM}"
  $PY -u -m sparse_encoding.sparse_encoding_deconvolution --feature_tag "$tag" \
    --roi_filter mfg_temporal \
    --null block --null_perms "$N_PERM_BLOCK" --null_seed 19 \
    --out_suffix "_${tag}_roi_block${N_PERM_BLOCK}"
done

echo "==== 6) QUALITATIVE + TASK-TYPE JACCARD ===="
# Prefer Qwen3.5 all-channel features when present; else Qwen3-8B.
QWEN_TAG=sae_qwen3_8b_l18
if [[ -f "/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_results_sae_qwen35_4b_mat_l15_cohort.csv" ]]; then
  QWEN_TAG=sae_qwen35_4b_mat_l15
fi
QWEN_RESULTS="/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_results_${QWEN_TAG}_cohort.csv"
if [[ -f "$QWEN_RESULTS" ]]; then
  $PY -u -m sparse_encoding.sparse_encoding_qualitative \
    --feature_tag "$QWEN_TAG" \
    --results_csv "$QWEN_RESULTS" \
    --all_channels --top_n 0 \
    --suffix "_${QWEN_TAG}_all_channels"
  $PY -u -m sparse_encoding.sparse_encoding_task_type_analysis \
    --feature_tag "$QWEN_TAG" \
    --features_csv "/home/ryan/analysisEV/group_encoding_results/sparse_encoding/tables/sparse_encoding_features_${QWEN_TAG}_all_channels.csv"

  echo "==== 7) PAPER-ALIGNED FULL vs SURPRISAL-ONLY ===="
  $PY -u -m sparse_encoding.sparse_encoding_paper_aligned \
    --results_csv "$QWEN_RESULTS"

  echo "==== 8) EXPLORATORY DOMINANCE ===="
  $PY -u -m sparse_encoding.sparse_encoding_surprisal_dominance \
    --results_csv "$QWEN_RESULTS"
else
  echo "[skip] qualitative/paper/dominance — missing $QWEN_RESULTS"
fi

echo "==== 9) COMBINED + TEMPORAL REPORT ===="
$PY -u -m sparse_encoding.sparse_encoding_validate --stage report --models gemma qwen qwen35 --all_subjects

echo "==== COHORT DONE $(date -Is) ===="
echo "LOG=$LOG"
