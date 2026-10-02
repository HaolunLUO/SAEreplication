#!/bin/bash
#SBATCH --job-name=fc_gen
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_high
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency/logs/slurm-gen-%A_%a.out

# Sign-constrained generalization, one source electrode at a time.
# ou_bcs_high allows 4 submitted jobs per user and is the partition that
# starts now. A 151-task array is rejected there. This array has 3 shards
# (the 300 ms LASSO job is the fourth). Shard k runs sources k, k+3, ...
# Writes only under feature_consistency/generalization/.
set -euo pipefail
export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency:$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
cd /orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency
echo "[$(date '+%F %T')] shard ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} on $(hostname)"
pids=()
for src in $(seq "${SLURM_ARRAY_TASK_ID}" 3 150); do
  "$PY" fc_b_source.py --source "$src" &
  pids+=("$!")
  if [ "${#pids[@]}" -ge 4 ]; then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[$(date '+%F %T')] shard ${SLURM_ARRAY_TASK_ID} done"
