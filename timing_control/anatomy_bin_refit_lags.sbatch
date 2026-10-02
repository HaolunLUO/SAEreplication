#!/bin/bash
#SBATCH --job-name=v3_tc_bin_lags
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-bin-lags-%j.out

# Full-lag bin refit for the 127 anatomy electrodes, partial and raw.
# New CSVs only, then the two lag-trajectory figures.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
cd "$TC" || exit 1
mkdir -p "$TC/logs"
echo "[$(date '+%F %T')] bin lag refit job ${SLURM_JOB_ID:-?} on $(hostname)" >> "$TC/logs/bin_lags_progress.log"
"$PY" anatomy_bin_refit_lags.py --mode both --n-jobs "${SLURM_CPUS_PER_TASK:-32}" >> "$TC/logs/bin_lags_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] bin lag refit exit $rc" >> "$TC/logs/bin_lags_progress.log"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
python3 draw_fig5_bin_lags.py >> "$TC/logs/bin_lags_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] bin lag figures exit $rc" >> "$TC/logs/bin_lags_progress.log"
exit "$rc"
