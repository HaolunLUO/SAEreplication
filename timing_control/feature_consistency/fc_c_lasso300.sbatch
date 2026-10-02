#!/bin/bash
#SBATCH --job-name=fc_lasso300
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency/logs/slurm-lasso300-%j.out

# LASSO at 300 ms only, for the 151 left language electrodes.
# Does not read or write the stored peak-lag masks.
set -euo pipefail
export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency:$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
cd /orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency
echo "[$(date '+%F %T')] lasso300 job ${SLURM_JOB_ID} on $(hostname)"
"$PY" fc_c_lasso300.py --n-jobs "${SLURM_CPUS_PER_TASK}"
