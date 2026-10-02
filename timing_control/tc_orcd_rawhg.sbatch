#!/bin/bash
#SBATCH --job-name=v3_tc_rawhg
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-rawhg-%j.out

# v2 SAE on the original high-gamma caches (no --cache-suffix).
# Omitting the suffix keeps the raw pair "" and _sig_glove, and the both
# stage prepends lags down to -2000 ms. wide and sig_glove stay on -500..2000.
# New tables only. Does not touch the finished partial, TRF, or sig_glove_timing tables.
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

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/rawhg_progress.log"

run() {
    local stage=$1 out=$2
    echo "[$(stamp)] stage $stage jobs v2:raw -> $out" >> "$LOG/rawhg_progress.log"
    if ! "$PY" timing_control.py --stage "$stage" --jobs v2:raw \
        --out "$out" --n_jobs "$NJ" >> "$LOG/${out%.csv}.log" 2>&1; then
        echo "[$(stamp)] stage $stage crashed ($out)" >> "$LOG/rawhg_progress.log"
        exit 1
    fi
}

run both tc_both_lag_scores_rawhg.csv
run wide tc_wide_lag_scores_rawhg.csv
run sig_glove tc_sig_glove_lag_scores_rawhg.csv
echo "[$(stamp)] raw HG stages done" >> "$LOG/rawhg_progress.log"
