#!/usr/bin/env bash
# Shared-token ownership v2: extract + the v1 regression chain.
# Writes X_word_<tag>_v2.* next to v1 features and tables under
# group_encoding_results/sparse_encoding_v2/. Does not overwrite v1 npz/meta
# or v1 sparse_encoding tables.
#
# Mirrors run_qwen35_matryoshka_chain.sh (F-test k=8000, LassoCV, Ridge,
# 5 contiguous within-section folds, random_state=19, surprisal force-included).
set -eu
set -o pipefail

ROOT="${ANALYSISEV_DATA_ROOT:-/home/ryan/analysisEV}"
EV="${ROOT}/ev_analysis"
SAE_SRC="${ROOT}/sae_sparse_encoding"
V2="${ROOT}/group_encoding_results/sparse_encoding_v2"
LOGDIR="${V2}/logs"
TABLES="${V2}/tables"
EXTRACT_PY="${EXTRACT_PY:-${HOME}/.venvs/qwen35-extract/bin/python}"
REGRESS_PY="${REGRESS_PY:-${HOME}/miniforge3/envs/ieeg-encoding/bin/python}"
N_JOBS="${N_JOBS:-4}"
N_JOBS_WIDE="${N_JOBS_WIDE:-1}"
mkdir -p "${LOGDIR}" "${TABLES}" "${V2}/reports"
# Keep the runtime tree on the sae_sparse_encoding sources.
cp -a "${SAE_SRC}/sparse_encoding/sae_extract_features.py" "${EV}/sparse_encoding/sae_extract_features.py"
cp -a "${SAE_SRC}/sparse_encoding/sparse_encoding_validate.py" "${EV}/sparse_encoding/sparse_encoding_validate.py"
cp -a "${SAE_SRC}/sparse_encoding/sparse_encoding_lepori_study3.py" "${EV}/sparse_encoding/sparse_encoding_lepori_study3.py"
export ANALYSISEV_DATA_ROOT="${ROOT}"
export SAE_RESULTS_DIRNAME="sparse_encoding_v2"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${EV}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${EV}"

need() {
  local p="$1"
  if [[ ! -x "${p}" ]]; then
    echo "Missing executable: ${p}" >&2
    exit 1
  fi
}
need "${EXTRACT_PY}"
need "${REGRESS_PY}"
"${REGRESS_PY}" -c "import core.analysis_paths as ap; assert ap.SAE_ROOT.name == 'sparse_encoding_v2', ap.SAE_ROOT"

extract_preset() {
  local preset="$1"
  shift
  local tag="sae_${preset}_v2"
  local todo=()
  local sid
  for sid in "$@"; do
    local npz="${ROOT}/extracted_linguistic_features/section_$(printf '%03d' "${sid}")/X_word_${tag}.npz"
    if [[ ! -f "${npz}" ]]; then
      todo+=("${sid}")
    fi
  done
  if [[ "${#todo[@]}" -eq 0 ]]; then
    echo "skip extract ${tag} (npz already present)"
    return 0
  fi
  echo "=== $(date -Iseconds) extract ${preset} sections ${todo[*]} ==="
  "${EXTRACT_PY}" -m sparse_encoding.sae_extract_features \
    --preset "${preset}" \
    --ownership shared_char_weighted \
    --device cuda --dtype bfloat16 \
    --sections "${todo[@]}"
}

echo "=== $(date -Iseconds) shared-ownership alignment QC ==="
"${EXTRACT_PY}" -m sparse_encoding.sparse_encoding_validate \
  --stage align --models gemma qwen qwen35 \
  --ownership shared_char_weighted

extract_preset qwen35_4b_mat_l15 1 2 3
extract_preset gemma2_2b_mat_l12 1 2 3
extract_preset qwen3_8b_l18 1 2 3

regress_cohort_lang() {
  local tag="$1"
  echo "=== $(date -Iseconds) cohort ${tag} ==="
  "${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
    --feature_tag "${tag}" --n_jobs "${N_JOBS}" \
    --out "${TABLES}/sparse_encoding_results_${tag}_cohort.csv"
  echo "=== $(date -Iseconds) is_lang ${tag} ==="
  "${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression \
    --feature_tag "${tag}" --lang_only --n_jobs "${N_JOBS}" \
    --out "${TABLES}/sparse_encoding_results_${tag}_lang.csv"
}

regress_bin() {
  local tag="$1" start="$2" end="${3:-}" jobs="$4"
  local end_s="end"
  local cmd=(
    "${REGRESS_PY}" -m sparse_encoding.sparse_encoding_regression
    --feature_tag "${tag}" --lang_only
    --sae_col_start "${start}" --n_jobs "${jobs}"
  )
  if [[ -n "${end}" ]]; then
    end_s="${end}"
    cmd+=(--sae_col_end "${end}")
  fi
  echo "=== $(date -Iseconds) bin ${tag} ${start}-${end_s} ==="
  cmd+=(--out "${TABLES}/sparse_encoding_results_${tag}_lang_bin${start}-${end_s}.csv")
  "${cmd[@]}"
}

QWEN35="sae_qwen35_4b_mat_l15_v2"
GEMMA="sae_gemma2_2b_mat_l12_v2"
QWEN="sae_qwen3_8b_l18_v2"

regress_cohort_lang "${QWEN35}"
regress_cohort_lang "${GEMMA}"
regress_cohort_lang "${QWEN}"

regress_bin "${QWEN35}" 0 2048 "${N_JOBS}"
regress_bin "${QWEN35}" 2048 "" "${N_JOBS_WIDE}"
regress_bin "${QWEN35}" 2048 16384 "${N_JOBS_WIDE}"
regress_bin "${QWEN35}" 16384 "" "${N_JOBS_WIDE}"
regress_bin "${GEMMA}" 0 128 "${N_JOBS}"
regress_bin "${GEMMA}" 128 "" "${N_JOBS}"

echo "=== $(date -Iseconds) residual baselines ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_dense_baseline \
  --features "${QWEN35}_resid" --surprisal_tag "${QWEN35}" \
  --with_surprisal --lang_only --out_suffix _qwen35_v2_resid \
  --n_jobs 1 --resume
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_dense_baseline \
  --features "${GEMMA}_resid" --surprisal_tag "${GEMMA}" \
  --with_surprisal --lang_only --out_suffix _gemma_v2_resid \
  --n_jobs 1 --resume
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_dense_baseline \
  --features "${QWEN}_resid" --surprisal_tag "${QWEN}" \
  --with_surprisal --lang_only --out_suffix _qwen_v2_resid \
  --n_jobs 1 --resume

echo "=== $(date -Iseconds) Qwen3.5 qualitative (bin occupancy) ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_qualitative \
  --feature_tag "${QWEN35}" \
  --results_csv "${TABLES}/sparse_encoding_results_${QWEN35}_cohort.csv" \
  --all_channels --top_n 0 \
  --suffix "_${QWEN35}_all_channels"

echo "=== $(date -Iseconds) Study 3 analogue ==="
"${REGRESS_PY}" -m sparse_encoding.sparse_encoding_lepori_study3 --tag_suffix _v2

echo "=== ALL DONE $(date -Iseconds) ==="
