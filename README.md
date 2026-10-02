# sae_sparse_encoding

Word-locked iEEG **Augmented Sparse Encoding Models** (Lepori, Kay & Tuckute
analogue): SAE latents + word surprisal → Lasso → Ridge, with lag-resolved and
continuous-time deconvolution as the primary temporal analyses.

Extracted from the larger `ev_analysis` scripts repo so the SAE pipeline can
live and version on its own.

## Layout

```
core/                 # shared paths + word-locked EEG loader
sparse_encoding/      # SAE extract, regression, temporal, Study 3 analogue
timing_control/       # speech-timing residual, lag curves, feature consistency
tests/
run_sparse_encoding_cohort.sh
run_qwen35_matryoshka_chain.sh
```

## Where the data are

Recordings, feature matrices, and fit tables are not in this git repo. Paths below are relative to `ANALYSISEV_DATA_ROOT`. If that variable is unset, `core/analysis_paths.py` uses the parent directory of this repo. The timing-control and feature-consistency fits were run with:

```bash
export ANALYSISEV_DATA_ROOT=/orcd/pool/005/haolun52/analysisEV
```

Story sections are 1–3. The feature tag for those fits is `sae_qwen35_4b_mat_l15_v2`.

### Inputs

| What | Path |
| --- | --- |
| High-gamma, sections 1–3 | `output/Naturalistic/segments/<Subject>_NA*_HG.mat` |
| Electrodes (`is_lang`, `sig_glove`, `region`, `hemisphere`) | `group_encoding_results/functional_taxonomy/tables/functional_taxonomy_electrode_table.csv` |
| Word onsets and offsets | `extracted_linguistic_features/section_001/word_timing.csv` (and `section_002`, `section_003`) |
| Matryoshka SAE, per section | `extracted_linguistic_features/section_00N/X_word_sae_qwen35_4b_mat_l15_v2.npz` |
| Surprisal and row mask | `X_word_sae_qwen35_4b_mat_l15_v2_surprisal.npy`, `X_word_sae_qwen35_4b_mat_l15_v2_valid.npy` in the same section folder |
| Dense residual | `X_word_sae_qwen35_4b_mat_l15_v2_resid.npy` in the same section folder |
| Story audio (envelope covariates) | `syntax_tree_probe/data/audio/task-lppCN_section_1.wav` (sections 2 and 3 the same way) |
| Story text and dependencies | `lppCN_tree.txt`, `lppCN_dependency.csv` |

`timing_control/feature_consistency/fc_c_words.py` currently points at a second copy of the story text, `/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/lppCN_tree.txt`, and at `extracted_linguistic_features` under the ORCD root above.

### Saved outputs the later scripts read

| What | Path |
| --- | --- |
| Word-locked high-gamma caches | `group_encoding_results/sparse_encoding_v3/cache/<Subject>_n51_a-500.0_b2000.0_sh0_m0.npz` and the `_sig_glove` pair |
| Timing-covariate cache | `group_encoding_results/sparse_encoding_v3/timing_control/cache/nuisance_ext_grid.npy` |
| Lag scores | `group_encoding_results/sparse_encoding_v3/timing_control/tables/tc_both_lag_scores.csv`, `tc_wide_lag_scores.csv` |
| Selected SAE indices | `group_encoding_results/sparse_encoding_v3/timing_control/tables/tc_selected_indices_partial.csv` and `tc_selected_indices_partial_parts/` |
| Feature-consistency tables | `group_encoding_results/sparse_encoding_v3/timing_control/feature_consistency/` |

Earlier SAE cohort reports stay under `group_encoding_results/sparse_encoding/` and `group_encoding_results/sparse_encoding_v2/`.

## Quick start

```bash
cd sae_sparse_encoding
pip install -r requirements.txt
# GPU extraction also needs: torch transformers sae_lens

# alignment sanity check (CPU)
python -m sparse_encoding.sae_extract_features --dry_run_alignment

# extract Qwen3.5 Matryoshka SAE (GPU)
python -m sparse_encoding.sae_extract_features --preset qwen35_4b_mat_l15 --device cuda

# temporal-primary cohort (lag screen → deconv → optional single-lag)
./run_sparse_encoding_cohort.sh
```

See `sparse_encoding/README_TEMPORAL.md` for lag / kernel interpretation.

## Main model

| Tag | Backbone | Notes |
|-----|----------|--------|
| `sae_qwen35_4b_mat_l15` | Qwen3.5-4B-Base Chanin Matryoshka L15 | Primary hierarchical SAE |
| `sae_qwen3_8b_l18` | Qwen3-8B layer 18 | Chinese companion |
| `sae_gemma2_2b_mat_l12` | Gemma-2-2B Matryoshka L12 | Paper backbone (method check) |

## Tests

```bash
pytest tests/ -q
```
