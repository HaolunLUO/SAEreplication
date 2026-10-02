#!/usr/bin/env bash
# SAE replication v3. Logs under group_encoding_results/sparse_encoding_v3/logs/.
set -euo pipefail
cd /home/ryan/analysisEV/sae_sparse_encoding
export ANALYSISEV_DATA_ROOT=/home/ryan/analysisEV
export SAE_RESULTS_DIRNAME=sparse_encoding_v3
export PYTHONPATH=/home/ryan/analysisEV/sae_sparse_encoding:/home/ryan/analysisEV/ev_analysis
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
LOG=/home/ryan/analysisEV/group_encoding_results/sparse_encoding_v3/logs/v3_all.log
mkdir -p "$(dirname "$LOG")"
exec /home/ryan/miniforge3/envs/ieeg-encoding/bin/python -u -m sparse_encoding.sparse_encoding_v3 \
  --stage all --n-jobs 16 --n-null 20 >> "$LOG" 2>&1
