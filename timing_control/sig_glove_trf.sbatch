#!/bin/bash
#SBATCH --job-name=v3_sig_glove_trf
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-sig-glove-trf-%j.out

# Phase-permutation GloVe test on n161 TRF residuals.
# Writes tables/sig_glove_trf.csv only. Does not touch sig_glove_timing.csv.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
LOG=$TC/logs
cd "$TC" || exit 1
mkdir -p "$LOG"
stamp() { date '+%F %T'; }
echo "[$(stamp)] job ${SLURM_JOB_ID:-?} on $(hostname)" >> "$LOG/sig_glove_trf_progress.log"
n=$(ls "$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/cache"/Subject*_n161_a-2000.0_b2000.0_sh0_m0_trfresid_m200.npz | wc -l)
if [ "$n" -ne 10 ]; then
    echo "[$(stamp)] expected 10 n161 caches, found $n" >> "$LOG/sig_glove_trf_progress.log"
    exit 1
fi
"$PY" sig_glove_timing.py --trf-resid --n-jobs 16 >> "$LOG/sig_glove_trf.log" 2>&1
rc=$?
echo "[$(stamp)] exit $rc" >> "$LOG/sig_glove_trf_progress.log"
exit $rc
