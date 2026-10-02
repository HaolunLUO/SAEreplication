#!/bin/bash
#SBATCH --job-name=v3_tc_count_sets
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-count-sets-%j.out

# Per-lag joint LASSO counts: glove partial, language raw, glove raw.
# Writes the three count CSVs, then the pool pngs. Does not touch the session store.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control:$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
cd "$TC" || exit 1
mkdir -p "$TC/logs"
echo "[$(date '+%F %T')] count sets job ${SLURM_JOB_ID:-?} on $(hostname)" >> "$TC/logs/count_sets_progress.log"
"$PY" anatomy_count_sets.py --n-jobs "${SLURM_CPUS_PER_TASK:-32}" >> "$TC/logs/count_sets_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] count sets fit exit $rc" >> "$TC/logs/count_sets_progress.log"
if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi
python3 draw_fig5b_count_sets.py >> "$TC/logs/count_sets_progress.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] count sets figure exit $rc" >> "$TC/logs/count_sets_progress.log"
exit "$rc"
