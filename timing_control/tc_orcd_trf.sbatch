#!/bin/bash
#SBATCH --job-name=v3_tc_trf
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-trf-%j.out

# TRF-residual lag curves. Writes only the four new CSVs. Does not touch
# tc_both_lag_scores.csv, tc_wide_lag_scores.csv, or sig_glove_timing.csv.
# Resumable: each stage skips electrodes already complete in its own CSV.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
LOG=$TC/logs
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-262144} / 1024 ))
NJ=$(( MEM_GB / 4 ))
[ "$NJ" -gt 64 ] && NJ=64
[ "$NJ" -gt "${SLURM_CPUS_PER_TASK:-128}" ] && NJ=${SLURM_CPUS_PER_TASK:-128}
cd "$TC" || exit 1
mkdir -p "$LOG"
stamp() { date '+%F %T'; }
SUF=_trfresid_m200

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/trf_progress.log"

run() {
    local stage=$1 jobs=$2 out=$3
    echo "[$(stamp)] stage $stage jobs $jobs -> $out" >> "$LOG/trf_progress.log"
    if ! "$PY" timing_control.py --stage "$stage" --jobs "$jobs" \
        --cache-suffix "$SUF" --out "$out" --n_jobs "$NJ" >> "$LOG/${out%.csv}.log" 2>&1; then
        echo "[$(stamp)] stage $stage crashed ($out)" >> "$LOG/trf_progress.log"
        exit 1
    fi
}

run both v2_shuffle:raw,v2:raw,v2_prevproj:raw tc_both_lag_scores_trf.csv
run both v2_shuffle:partial,v2:partial,v2_prevproj:partial tc_both_lag_scores_trf_partial.csv
run wide v2:raw tc_wide_lag_scores_trf.csv
run wide v2:partial tc_wide_lag_scores_trf_partial.csv
echo "[$(stamp)] trf stages done" >> "$LOG/trf_progress.log"
