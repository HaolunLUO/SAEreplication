#!/usr/bin/env bash
# Extract Qwen3.5-4B-Base Matryoshka L15 SAE features, then lang-only encoding
# + nested-prefix refits (0-2048 vs 2048-end).
#
# NOTE: This chain uses single-lag regression for Matryoshka *bin* refits
# (column slices). For the primary temporal-precision story (lag curves +
# kernels), run:
#   python -m sparse_encoding.sparse_encoding_validate \
#     --stage temporal_full,report --models qwen35 --all_subjects
# or ./run_sparse_encoding_cohort.sh
#
# Extract is skipped when the three npz files already exist (OVERWRITE=1
# forces a re-extract). Regression always starts: incomplete CSVs resume
# per subject, completed CSVs skip those people without reloading EEG.
# Default N_JOBS=1 because the 63k-column 2048-end slice OOMs under
# joblib n_jobs=4 in this WSL setup.
#
# Transformers 4.57 (ieeg-encoding) cannot load qwen3_5. Extract uses a
# system-site-packages venv with transformers 5.x; regression stays on
# ieeg-encoding.
set -eu
set -o pipefail

ROOT="${ANALYSISEV_DATA_ROOT:-/home/ryan/analysisEV}"
SCRIPTS="${ROOT}/ev_analysis"
LOGDIR="${ROOT}/group_encoding_results/sparse_encoding/logs"
TAG="sae_qwen35_4b_mat_l15"
EXTRACT_PY="${EXTRACT_PY:-${HOME}/.venvs/qwen35-extract/bin/python}"
REGRESS_PY="${REGRESS_PY:-${HOME}/miniforge3/envs/ieeg-encoding/bin/python}"
N_JOBS="${N_JOBS:-1}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p "${LOGDIR}"
export ANALYSISEV_DATA_ROOT="${ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPTS}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SCRIPTS}"

need() {
  local p="$1"
  if [[ ! -x "${p}" ]]; then
    echo "Missing executable: ${p}" >&2
    exit 1
  fi
}
need "${EXTRACT_PY}"
need "${REGRESS_PY}"

echo "=== $(date -Iseconds) Qwen3.5 Matryoshka extract ==="
missing_npz=0
for sid in 1 2 3; do
  npz="${ROOT}/extracted_linguistic_features/section_$(printf '%03d' "${sid}")/X_word_${TAG}.npz"
  if [[ ! -f "${npz}" ]]; then
    missing_npz=1
  fi
done
if [[ "${missing_npz}" -eq 0 && "${OVERWRITE}" != "1" ]]; then
  echo "skip extract (npz already present)"
else
  "${EXTRACT_PY}" -m sparse_encoding.sae_extract_features \
    --preset qwen35_4b_mat_l15 \
    --device cuda --dtype bfloat16 \
    --sections 1 2 3
fi

for sid in 1 2 3; do
  npz="${ROOT}/extracted_linguistic_features/section_$(printf '%03d' "${sid}")/X_word_${TAG}.npz"
  [[ -f "${npz}" ]] || { echo "extract missing ${npz}" >&2; exit 1; }
done

echo "=== $(date -Iseconds) lang_only full SAE ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
  --feature_tag "${TAG}" --lang_only --n_jobs "${N_JOBS}"

echo "=== $(date -Iseconds) nested prefix 0-2048 ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
  --feature_tag "${TAG}" --lang_only \
  --sae_col_start 0 --sae_col_end 2048 --n_jobs "${N_JOBS}"

echo "=== $(date -Iseconds) nested prefix 2048-end ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
  --feature_tag "${TAG}" --lang_only \
  --sae_col_start 2048 --n_jobs "${N_JOBS}"

echo "=== $(date -Iseconds) nested prefix 2048-16384 ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
  --feature_tag "${TAG}" --lang_only \
  --sae_col_start 2048 --sae_col_end 16384 --n_jobs "${N_JOBS}"

echo "=== $(date -Iseconds) nested prefix 16384-end ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
  --feature_tag "${TAG}" --lang_only \
  --sae_col_start 16384 --n_jobs "${N_JOBS}"

echo "=== $(date -Iseconds) same-model residual Ridge ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_dense_baseline \
  --features "${TAG}_resid" --surprisal_tag "${TAG}" \
  --with_surprisal --lang_only --out_suffix _qwen35_resid \
  --n_jobs 1 --resume

echo "=== $(date -Iseconds) Study 3 report ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_lepori_study3

echo "=== ALL DONE $(date -Iseconds) ==="
