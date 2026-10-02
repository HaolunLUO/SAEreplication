#!/bin/bash
#SBATCH --job-name=v3_tc_bin_refit
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-binrefit-%j.out

# Fig 5C only. Writes tables/tc_bin_refit_partial.csv and nothing else.
# Resumable: electrodes that already have all three bins are skipped.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
cd "$TC" || exit 1
mkdir -p "$TC/logs"
echo "[$(date '+%F %T')] bin refit job ${SLURM_JOB_ID:-?} on $(hostname)" >> "$TC/logs/binrefit_progress.log"
"$PY" paper_match_partial.py --mode refit --n-jobs "${SLURM_CPUS_PER_TASK:-16}" >> "$TC/logs/binrefit_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] bin refit exit $rc" >> "$TC/logs/binrefit_progress.log"
exit "$rc"
