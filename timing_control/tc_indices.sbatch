#!/bin/bash
#SBATCH --job-name=v3_tc_partial_indices
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-indices-%j.out

# Resumable: each electrode's selected-index file is replaced only when that
# electrode finishes, so a preemption requeue skips electrodes already saved.
# Does not rewrite tc_*_lag_scores.csv and does not touch sig_glove.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
LOG=$TC/logs
# At least 4 GB per worker: 256G / 4G = 64 workers, never more than the CPUs allocated.
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-262144} / 1024 ))
NJ=$(( MEM_GB / 4 ))
[ "$NJ" -gt 64 ] && NJ=64
[ "$NJ" -gt "${SLURM_CPUS_PER_TASK:-128}" ] && NJ=${SLURM_CPUS_PER_TASK:-128}
cd "$TC" || exit 1
mkdir -p "$LOG"
stamp() { date '+%F %T'; }

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/indices_progress.log"
echo "[$(stamp)] stage indices: v2 partial selected SAE indices (359 is_lang ∪ sig_glove)" >> "$LOG/indices_progress.log"
if ! "$PY" timing_control.py --stage indices --n_jobs "$NJ" >> "$LOG/indices_orcd.log" 2>&1; then
    echo "[$(stamp)] stage indices crashed" >> "$LOG/indices_progress.log"
    exit 1
fi
echo "[$(stamp)] indices stage done" >> "$LOG/indices_progress.log"
