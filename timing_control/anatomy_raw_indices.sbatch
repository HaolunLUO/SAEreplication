#!/bin/bash
#SBATCH --job-name=v3_tc_raw_idx_anat
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-raw-idx-%j.out

# Raw-HG LASSO supports for the 127 Fig 5 anatomy electrodes.
# Writes tables/tc_selected_indices_rawhg_anatomy.csv and the matching parts.
# Does not touch tc_selected_indices_partial.csv or the lag-score tables.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
cd "$TC" || exit 1
mkdir -p "$TC/logs"
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-262144} / 1024 ))
NJ=$(( MEM_GB / 4 ))
[ "$NJ" -gt 64 ] && NJ=64
[ "$NJ" -gt "${SLURM_CPUS_PER_TASK:-128}" ] && NJ=${SLURM_CPUS_PER_TASK:-128}
echo "[$(date '+%F %T')] raw anatomy indices job ${SLURM_JOB_ID:-?} on $(hostname) n_jobs=$NJ" >> "$TC/logs/raw_anat_indices_progress.log"
"$PY" anatomy_raw_indices.py --n-jobs "$NJ" >> "$TC/logs/raw_anat_indices_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] raw anatomy indices exit $rc" >> "$TC/logs/raw_anat_indices_progress.log"
exit "$rc"
