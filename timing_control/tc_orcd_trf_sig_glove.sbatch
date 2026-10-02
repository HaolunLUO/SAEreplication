#!/bin/bash
#SBATCH --job-name=v3_tc_trf_glove
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-trf-glove-%j.out

# v2 SAE on the 217 sig_glove channels, TRF residual cache only.
# Does not rewrite the 45-both or 187-language tables, or sig_glove_timing.csv.
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
SUF=_sig_glove_trfresid_m200

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/trf_glove_progress.log"

run() {
    local jobs=$1 out=$2
    echo "[$(stamp)] stage sig_glove jobs $jobs -> $out" >> "$LOG/trf_glove_progress.log"
    if ! "$PY" timing_control.py --stage sig_glove --jobs "$jobs" \
        --cache-suffix "$SUF" --out "$out" --n_jobs "$NJ" >> "$LOG/${out%.csv}.log" 2>&1; then
        echo "[$(stamp)] crashed ($out)" >> "$LOG/trf_glove_progress.log"
        exit 1
    fi
}

run v2:raw tc_sig_glove_lag_scores_trf.csv
run v2:partial tc_sig_glove_lag_scores_trf_partial.csv
echo "[$(stamp)] sig_glove trf stages done" >> "$LOG/trf_glove_progress.log"
