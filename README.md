# InferenceX PCA Demo

This is a Streamlit analysis app built on the local InferenceX benchmark dump to understand which inference configuration features explain structural benchmark variance and which features predict selected performance targets.

## What This Project Does

- Loads local InferenceX benchmark/config data.
- Joins `benchmark_results.config_id = configs.id`.
- Builds a data dictionary and data-understanding layer.
- Explains feature families, missingness, cardinality, distributions, and data quality checks.
- Runs PCA on setup/configuration features only.
- Maps PC1, PC2, and other synthetic PCA axes back to original features through loadings and contribution summaries.
- Uses outcome metrics as overlays or targets, not PCA inputs.
- Runs target-aware `RandomForestRegressor` models with permutation importance.
- Uses grouped train/test splitting by `config_id` when available.
- Produces executive Findings and downloadable summaries.

## Why This Matters

Inference performance is not only a model/hardware story. Serving architecture, parallelism, disaggregation, prefill/decode split, concurrency, framework, precision, and speculative method are key parts of inference infrastructure value. This demo supports datacenter and inference infrastructure analysis by tying benchmark configuration choices to performance outcomes.

## Data Source

Expected local dump folder:

```text
inferencex-dump-2026-06-29/
```

Required files:

- `benchmark_results.json`
- `configs.json`

Optional/supported later:

- `availability.json`
- `eval_results.json`
- `run_stats.json`
- `changelog_entries.json`

Intentionally skipped:

- `server_logs.json`
- `eval_samples.json`

Those skipped files can be extremely large and are unnecessary for this PCA workflow.

## Internal Data Setup

### A. Internal Approved Storage Path

Download the approved InferenceX dump from:

```text
<INTERNAL_STORAGE_LINK_OR_PATH>
```

Ask the project owner if you do not have access. Place the unpacked dump folder at the repo root:

```text
InferenceX-PCA-demo/
  inferencex-dump-2026-06-29/
    benchmark_results.json
    configs.json
```

### B. Public GitHub Release Fallback, If Allowed

If your team allows downloading from the public GitHub release, use:

```bash
gh release download 'db-dump/2026-06-29' \
  --repo SemiAnalysisAI/InferenceX-app \
  -p 'inferencex-dump-*.tar.xz.part*'

cat inferencex-dump-2026-06-29.tar.xz.part* | xz -d -T0 | tar -x
```

`gh` may require:

```bash
brew install gh
gh auth login
```

Do not commit the dump. Do not upload the dump to GitHub. Do not commit `server_logs.json` or `eval_samples.json`.

## Local Setup

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python3 -m pip install -r requirements-streamlit.txt
python3 -m streamlit run apps/inferencex_pca_demo.py
```

Open the Streamlit localhost URL shown in the terminal. If the dump is in a different location, update the sidebar dump directory in the app.

## Repo Layout

```text
apps/inferencex_pca_demo.py
requirements-streamlit.txt
README.md
.gitignore
```

`inferencex-dump-*` and `exports/` are intentionally excluded from git.

## Data Model

- `benchmark_results` contains measured benchmark rows, workload shape, outcome metrics, and provenance ids.
- `configs` contains model, hardware, framework, precision, and parallelism setup.
- The core join is `benchmark_results.config_id = configs.id`.
- One joined row means one benchmark workload/config observation.

## Feature Families

### Workload Shape

- `isl`: input sequence length.
- `osl`: output sequence length.
- `conc`: concurrency / concurrent requests.
- `benchmark_type`: test or workload type.

### Model / Hardware / Software

- `config_model`: model key.
- `config_hardware`: GPU/system target.
- `config_framework`: inference serving framework.
- `config_precision`: numerical precision.
- `config_spec_method`: speculative decoding / speculative method.

### Serving Topology / Parallelism

- `config_disagg`: whether prefill and decode are separated.
- `config_is_multinode`: whether the run spans multiple nodes.
- `config_prefill_tp`: prefill tensor parallelism.
- `config_prefill_ep`: prefill expert parallelism.
- `config_prefill_dp_attention`: prefill DP-attention flag.
- `config_prefill_num_workers`: prefill worker count.
- `config_decode_tp`: decode tensor parallelism.
- `config_decode_ep`: decode expert parallelism.
- `config_decode_dp_attention`: decode DP-attention flag.
- `config_decode_num_workers`: decode worker count.
- `config_num_prefill_gpu`: GPUs allocated to prefill.
- `config_num_decode_gpu`: GPUs allocated to decode.

### Outcome Metrics

- `metrics_*_ttft`: time to first token.
- `metrics_*_tpot`: time per output token.
- `metrics_*_itl`: inter-token latency.
- `metrics_*_e2el`: end-to-end latency.
- Throughput metrics: higher is generally better.
- Latency and energy metrics: lower is generally better.

## Analysis Units

The app supports four analysis units:

| Analysis unit | Verified row count |
| --- | ---: |
| Raw benchmark rows | 79,830 |
| Latest row per config/workload/concurrency | 7,462 |
| Median aggregate per config/workload/concurrency | 7,462 |
| One row per config | 1,197 |

For config/workload/concurrency analysis, grouping keys are:

- `config_id`
- `benchmark_type`
- `isl`
- `osl`
- `conc`

The default is **Median aggregate per config/workload/concurrency** because it reduces repeat-run bias, preserves workload shape, and is more defensible for team findings.

## PCA Methodology

PCA is unsupervised. It finds structural variance, not value.

In this app:

- PCA inputs should be setup/configuration features only.
- Outcome metrics should not be PCA inputs.
- Metrics can be used to color the PCA scatter or as target variables.
- Loadings and feature contributions translate synthetic PC axes back to real features.

## Current PCA Findings

Observed with the current default analysis setup:

- First five PCs explain about 70.8% of configuration variance.
- PC1 explains about 28.8%.
- PC2 explains about 16.1%.
- PC3 explains about 11.2%.
- PC4 explains about 8.3%.

Top structural variance drivers are mostly serving/parallelism fields:

- `config_prefill_tp`
- `config_decode_tp`
- `config_prefill_ep`
- `config_is_multinode`
- `config_num_prefill_gpu`
- `config_disagg`
- `config_decode_dp_attention`
- `config_prefill_dp_attention`
- `config_decode_num_workers`
- `config_decode_ep`

Interpretation: the benchmark configuration space is shaped more by serving architecture and parallelism strategy than by raw model or hardware labels alone.

## Component Interpretation

- **PC1:** disaggregated/multinode serving axis.
- **PC2:** tensor-parallel/GPU-scaling axis.
- **PC3:** expert-parallel/prefill-vs-disagg split.
- **PC4:** workload-shape axis driven by ISL/OSL/concurrency.

## Target-Aware Modeling

The target-aware layer uses `RandomForestRegressor` as a baseline supervised model and computes permutation importance on the test split.

Defaults:

- Predictors are config/setup fields.
- Targets are selected outcome metric columns.
- Split mode defaults to grouped split by `config_id` when available.

Random splits can overstate performance when repeated configurations appear in both train and test. Grouped split by `config_id` is safer for this dataset.

## Current Target-Aware Finding

Example observed target:

- Target: `metrics_p99_itl`
- Test R2: about 0.783
- Test MAE: about 0.151

Top predictors:

- `conc`
- `config_spec_method`
- `config_framework`
- `isl`
- `config_precision`
- `config_hardware`
- `config_model`

Interpretation: for p99 inter-token latency, concurrency is the dominant predictor, followed by speculative method and serving framework. Hardware and model matter, but were less important than runtime/workload choices in this selected target model.

## Findings Summary

PCA tells us what structures the benchmark space. Target-aware modeling tells us what predicts selected outcomes. Together, they show that inference infrastructure value depends on workload shape, concurrency, serving framework, speculative method, and parallelism strategy, not just chip or model choice.

## Limitations

- Descriptive, not causal.
- PCA is sensitive to feature encoding and selected analysis unit.
- RandomForest feature importance is target-specific.
- Dataset coverage is uneven.
- Repeated runs can overweight frequently tested configs unless aggregated.
- Grouped splits are better than random splits for this dataset, but still not full causal validation.
- Cloud deployment requires a safe data access plan.

## What Not To Commit

Do not commit:

- `inferencex-dump-*/`
- `exports/`
- `.venv*/`
- `__pycache__/`
- `.pytest_cache/`
- `.streamlit/secrets.toml`
- `*.pyc`
- `.DS_Store`
- `server_logs.json`
- `eval_samples.json`

## Team Workflow

1. Clone the repo.
2. Download/place the approved dump locally.
3. Create a Python virtual environment.
4. Install `requirements-streamlit.txt`.
5. Run Streamlit.
6. Use the default median aggregate analysis unit.
7. Export findings from the app.
8. Share `findings_summary.md` or screenshots with the team.

## Future Improvements

- Add cloud data mounting.
- Add cost/token valuation.
- Add energy metrics if available.
- Add TabFM/TabPFN/TabICL comparison for predicting missing benchmark outcomes.
- Add benchmark coverage gap analysis.
- Add quality/performance tradeoff analysis using `eval_results.json`.
