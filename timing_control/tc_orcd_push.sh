#!/bin/bash
# Run from the workstation in an interactive terminal (ORCD asks for password + Duo once).
# Copies only what the pool lacks for the timing-control fit, then submits tc_orcd.sbatch.
set -euo pipefail
REMOTE=haolun52@orcd-login.mit.edu
POOL=/orcd/pool/005/haolun52/analysisEV
SOCK=/tmp/orcd-cm-%r@%h:%p
SSH="ssh -o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=15m"
cd /home/ryan/analysisEV

$SSH "$REMOTE" "mkdir -p $POOL/group_encoding_results/sparse_encoding_v3/timing_control/logs"

# Scripts, caches (extended-lag neural + nuisance grid) and the tables that hold the 21 finished electrodes.
rsync -av -e "$SSH" --relative --exclude='__pycache__' \
    ./group_encoding_results/sparse_encoding_v3/timing_control/ \
    ./group_encoding_results/sparse_encoding_v3/diagnostics/preonset/preonset_refit.py \
    ./group_encoding_results/sparse_encoding_v3/diagnostics/preonset/timing_covariates_full_grid.npy \
    "$REMOTE:$POOL/"

# Inputs the pool may not have; never overwrite what is already there.
rsync -av -e "$SSH" --relative --ignore-existing \
    ./group_encoding_results/sparse_encoding_v3/cache/*_n51_a-500.0_b2000.0_sh0_m0.npz \
    ./group_encoding_results/sparse_encoding_v3/cache/*_n51_a-500.0_b2000.0_sh0_m0_sig_glove.npz \
    ./group_encoding_results/sparse_encoding_v3/tables/v3_loso_lag_scores.csv \
    ./group_encoding_results/functional_taxonomy/tables/functional_taxonomy_electrode_table.csv \
    "$REMOTE:$POOL/"

# Report (do not copy) if the pool's v3 module differs from the workstation copy.
rsync -avnc -e "$SSH" --relative ./sae_sparse_encoding/sparse_encoding/sparse_encoding_v3.py \
    ./sae_sparse_encoding/sparse_encoding/sparse_encoding_regression.py \
    ./sae_sparse_encoding/core/analysis_paths.py "$REMOTE:$POOL/" | sed 's/^/[dry-run differs] /'

$SSH "$REMOTE" "cd $POOL/group_encoding_results/sparse_encoding_v3/timing_control && sbatch tc_orcd.sbatch"
sleep 5
$SSH "$REMOTE" "squeue -u haolun52 -n v3_timing_control -o '%.12i %.12P %.20j %.8T %.20S %.10M %R'; squeue -u haolun52 -n v3_timing_control --start"
