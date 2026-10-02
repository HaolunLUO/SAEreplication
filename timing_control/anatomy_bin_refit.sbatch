#!/bin/bash
#SBATCH --job-name=v3_tc_bin_anat
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
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-bin-anat-%j.out

# Fig 5C for the AAL anatomy panels. New CSVs only:
#   tables/tc_bin_refit_anatomy_partial.csv
#   tables/tc_bin_refit_anatomy_rawhg.csv
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
cd "$TC" || exit 1
mkdir -p "$TC/logs"
echo "[$(date '+%F %T')] anatomy bin refit job ${SLURM_JOB_ID:-?} on $(hostname)" >> "$TC/logs/bin_anat_progress.log"
"$PY" anatomy_bin_refit.py --mode both --n-jobs "${SLURM_CPUS_PER_TASK:-16}" >> "$TC/logs/bin_anat_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] anatomy bin refit exit $rc" >> "$TC/logs/bin_anat_progress.log"
exit "$rc"
