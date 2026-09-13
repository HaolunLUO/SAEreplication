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
tests/
run_sparse_encoding_cohort.sh
run_qwen35_matryoshka_chain.sh
```

Data and results stay under the parent **analysisEV** tree (not in this repo):

- Features: `$ANALYSISEV_DATA_ROOT/extracted_linguistic_features/`
- Outputs: `$ANALYSISEV_DATA_ROOT/group_encoding_results/sparse_encoding/`

```bash
export ANALYSISEV_DATA_ROOT=/home/ryan/analysisEV   # default if this repo is under analysisEV/
```

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
