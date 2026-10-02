#!/bin/bash
#SBATCH --job-name=sig_glove_timing
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-sig-glove-timing-%j.out

# Resumable: each subject is appended to tables/sig_glove_timing.csv only after
# all of its lags finish, so a preemption requeue skips completed subjects.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
LOG=$TC/logs
# Each lag worker peaks near 2 GB. Leave headroom: 256G / 3G, and never more than the CPUs.
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-262144} / 1024 ))
NJ=$(( MEM_GB / 3 ))
[ "$NJ" -lt 1 ] && NJ=1
[ "$NJ" -gt "${SLURM_CPUS_PER_TASK:-128}" ] && NJ=${SLURM_CPUS_PER_TASK:-128}
cd "$TC" || exit 1
mkdir -p "$LOG" "$TC/tables" "$TC/cache"
stamp() { date '+%F %T'; }

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/sig_glove_timing_progress.log"

if ! "$PY" sig_glove_timing.py --n-jobs "$NJ" >> "$LOG/sig_glove_timing.log" 2>&1; then
    echo "[$(stamp)] sig_glove_timing crashed" >> "$LOG/sig_glove_timing_progress.log"
    exit 1
fi
echo "[$(stamp)] sig_glove_timing done" >> "$LOG/sig_glove_timing_progress.log"
