# Temporal precision in SAE sparse encoding

sEEG gives millisecond-scale timing that a single 300 ms lag cannot use.
This package treats **lag-resolved screening** and **continuous-time
deconvolution kernels** as the primary analysis. Fixed single-lag regression
remains available as a simplified baseline (Matryoshka bin refits, paper
tables that still quote one lag).

## Why it matters

| Approach | Timepoints | What you learn |
|----------|------------|----------------|
| Single-lag regression (baseline) | 1 (default 300 ms) | Encoding strength at one window |
| Lag screen (primary) | 17 (0–800 ms, 50 ms) | Per-electrode peak lag, FWHM, SAE vs surprisal timing |
| Deconvolution (primary) | FIR kernels 0–800 ms | Overlap-corrected impulse responses |

## Recommended order

```bash
export ANALYSISEV_DATA_ROOT=/home/ryan/analysisEV

# Primary validation path (default stages)
python -m sparse_encoding.sparse_encoding_validate \
  --stage align,extract,temporal_full,report \
  --models qwen35 --all_subjects --device cuda

# Or the cohort shell (lag → kernels → optional single-lag)
./run_sparse_encoding_cohort.sh
```

Stage aliases (old → new):

- `lag` → `lag_resolved`
- `deconv` → `temporal_kernels`
- `regress` → `single_lag_baseline`
- `temporal_full` = lag_resolved + temporal_kernels

## Interpreting results

### Lag screen (`sparse_encoding_lag_screen.py`)

- **Peak lag**: lag of maximum held-out Fisher-z *R* after training-only
  feature selection.
- **FWHM**: temporal selectivity of that lag curve (narrower = sharper).
- **`lag_diff_full_vs_surprisal_ms`**: full-model peak lag minus surprisal-only.
  Positive → SAE/full peaks later than surprisal.

Primary tables:

- `tables/sparse_encoding_lag_screen_{tag}_lang.csv` (wide)
- `tables/sparse_encoding_lag_screen_peaks_{tag}_lang.csv`
- `tables/sparse_encoding_lag_by_region_{tag}_lang.csv`
- `figures/lag_timecourse_lang_{tag}_lang.png`
- `figures/lag_by_region_{tag}_lang.png`

### Deconvolution (`sparse_encoding_deconvolution.py`)

- Fold-averaged FIR kernels for surprisal, SAE L2, nuisance L2.
- Shape metrics: peak latency, FWHM, onset/offset, polarity, multi-peak count.
- Figures via `sparse_encoding_plot_kernels.py`.

Primary tables / figures:

- `tables/sparse_encoding_deconv_{tag}_lang.csv`
- `tables/sparse_encoding_kernels_{tag}_lang.csv`
- `tables/sparse_encoding_kernel_shapes_{tag}_lang.csv`
- `figures/kernel_grand_average_{tag}_lang.png`
- `figures/kernel_heatmap_{tag}_lang.png`
- `figures/kernel_surprisal_vs_sae_{tag}_lang.png`

### Single-lag baseline (`sparse_encoding_regression.py`)

Use for:

- Quick encoding *R* at one lag
- Matryoshka nested-bin refits (`--sae_col_start` / `--sae_col_end`)
- Paper-aligned SAE-gain tables that historically used 300 ms

Do **not** treat it as the temporal story.

## Defaults (open to change)

| Setting | Default | Notes |
|---------|---------|-------|
| Lag range | 0–800 ms | No anticipatory (negative) lags yet |
| Lag step | 50 ms | Match 50 ms response bins / FIR step |
| Region buckets | temporal / left_MFG / frontal / deep / other | From AAL3 labels |
| Peak-lag inference | subject-mean sign-flip | On electrode-averaged peaks |

## QC checks

`temporal_qc_report` (embedded in lag-screen reports) flags:

1. Clear peaks: fraction with FWHM &lt; 400 ms (target ≥ 80%)
2. Temporal diversity: range of peak lags &gt; 300 ms
3. Fold stability: when fold-std is available, std &lt; 100 ms

## File map

| File | Role |
|------|------|
| `sparse_encoding_lag_screen.py` | Primary discrete lag analysis |
| `sparse_encoding_lag_by_region.py` | Region peak-lag contrasts |
| `sparse_encoding_deconvolution.py` | Primary continuous kernels |
| `sparse_encoding_plot_kernels.py` | Kernel figure suite |
| `temporal_metrics.py` | Peak / FWHM / QC helpers |
| `sparse_encoding_regression.py` | Single-lag baseline (deprecated as primary) |
| `sparse_encoding_validate.py` | Orchestrator (`temporal_full` default) |
