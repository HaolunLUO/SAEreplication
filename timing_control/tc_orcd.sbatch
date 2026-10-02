#!/bin/bash
#SBATCH --job-name=v3_timing_control
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/orcd/pool/005/haolun52/analysisEV/group_encoding_results/sparse_encoding_v3/timing_control/logs/slurm-%j.out

# Resumable: every stage skips electrodes already complete in its output CSV,
# so a preemption requeue continues from the last written electrode.
set -uo pipefail

export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
export PYTHONPATH=$ANALYSISEV_DATA_ROOT/sae_sparse_encoding:$ANALYSISEV_DATA_ROOT/ev_analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/haolun52/.conda/envs/sparse_encoding/bin/python
TC=$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding_v3/timing_control
LOG=$TC/logs
# At least 4 GB per worker: 256G / 4G = 64 workers, never more than the CPUs allocated.
MEM_GB=$(( ${SLURM_MEM_PER_NODE:-262144} / 1024 ))
NJ=$(( MEM_GB / 4 ))
[ "$NJ" -gt 64 ] && NJ=64
[ "$NJ" -gt "${SLURM_CPUS_PER_TASK:-128}" ] && NJ=${SLURM_CPUS_PER_TASK:-128}
cd "$TC" || exit 1
mkdir -p "$LOG"
stamp() { date '+%F %T'; }

echo "[$(stamp)] job ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) n_jobs=$NJ" >> "$LOG/orcd_progress.log"

if [ -f "$LOG/sanity_FAILED" ]; then
    status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$LOG/sanity.json" 2>/dev/null || true)
    if [ "$status" != "pass" ]; then
        echo "[$(stamp)] sanity failed earlier, not continuing" >> "$LOG/orcd_progress.log"
        exit 0
    fi
    echo "[$(stamp)] sanity.json passes; continuing past stale sanity_FAILED" >> "$LOG/orcd_progress.log"
fi

echo "[$(stamp)] stage both: v2_shuffle, v2, v2_prevproj partial (45 electrodes)" >> "$LOG/orcd_progress.log"
"$PY" timing_control.py --stage both --jobs v2_shuffle:partial,v2:partial,v2_prevproj:partial \
    --n_jobs "$NJ" >> "$LOG/both_orcd.log" 2>&1 || { echo "[$(stamp)] stage both crashed" >> "$LOG/orcd_progress.log"; exit 1; }

"$PY" tc_sanity.py > "$LOG/sanity.json" 2>> "$LOG/sanity.err"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[$(stamp)] SANITY FAILED (rc=$rc, see sanity.json); stopping before the 217 + 187 run" >> "$LOG/orcd_progress.log"
    touch "$LOG/sanity_FAILED"
    exit 0
fi
echo "[$(stamp)] sanity passed; stage wide: v2 partial (remaining sig_glove and is_lang electrodes)" >> "$LOG/orcd_progress.log"

if ! "$PY" timing_control.py --stage wide --jobs v2:partial --n_jobs "$NJ" >> "$LOG/wide_orcd.log" 2>&1; then
    echo "[$(stamp)] stage wide crashed" >> "$LOG/orcd_progress.log"
    exit 1
fi
echo "[$(stamp)] all stages done" >> "$LOG/orcd_progress.log"
