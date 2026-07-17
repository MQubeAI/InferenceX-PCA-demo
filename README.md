# InferenceX PCA Demo

This is a Streamlit analysis app built on local InferenceX benchmark exports to understand which inference configuration features explain structural benchmark variance and which features predict selected performance targets.

## What This Project Does

- Loads local InferenceX benchmark/config CSV data, with JSON dump fallback for local development.
- Joins `benchmark_results.config_id = configs.id`.
- Builds a data dictionary and data-understanding layer.
- Explains feature families, missingness, cardinality, distributions, and data quality checks.
- Runs PCA on setup/configuration features only.
- Maps PC1, PC2, and other synthetic PCA axes back to original features through loadings and contribution summaries.
- Uses outcome metrics as overlays or targets, not PCA inputs.
- Runs target-aware `RandomForestRegressor` models with permutation importance.
- Uses deterministic grouped cross-validation by `config_id` when available.
- Produces executive Findings and downloadable summaries.

## Why This Matters

Inference performance is not only a model/hardware story. Serving architecture, parallelism, disaggregation, prefill/decode split, concurrency, framework, precision, and speculative method are key parts of inference infrastructure value. This demo supports datacenter and inference infrastructure analysis by tying benchmark configuration choices to performance outcomes.

## Data Source

Official data folder: [Google Drive](https://drive.google.com/drive/u/2/folders/1RQYKaliWtJym1kbGH9A4SwfzqSh1RBvR)

Required files:

- `benchmark_results.csv`
- `configs.csv`

Optional/supported later:

- `availability.csv`
- `eval_results.csv`
- `run_stats.csv`
- `workflow_runs.csv`
- `changelog_entries.csv`

Intentionally skipped:

- `server_logs.json`
- `eval_samples.json`

Those skipped files can be extremely large and are unnecessary for this PCA workflow. The app can still fall back to JSON dump mode for local development if `inferencex-pca-data/benchmark_results.csv` and `inferencex-pca-data/configs.csv` are not present.

## Internal Data Setup

Download the approved InferenceX PCA CSV export folder from: [Google Drive](https://drive.google.com/drive/u/2/folders/1RQYKaliWtJym1kbGH9A4SwfzqSh1RBvR).

Place the downloaded folder at the repo root and rename it to:

```text
inferencex-pca-data/
```

## Local Setup

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python3 -m pip install -r requirements-streamlit.txt
python3 -m streamlit run apps/inferencex_pca_demo.py
```

Open the Streamlit localhost URL shown in the terminal. If the CSV folder is in a different location, update the sidebar data directory in the app.

## Reproducible Analysis Run

The default result snapshot is generated outside Streamlit using the same production analysis functions:

```bash
python3 scripts/reproducible_analysis.py \
  --data-dir inferencex-pca-data \
  --output artifacts/reproducible-results.json \
  --max-rows 20000 --seed 42 --stability-runs 5
```

The committed `artifacts/reproducible-results.json` contains aggregate metadata only: source file names/sizes/mtimes and bounded file fingerprints, aggregate PCA/RF results, package versions, warnings, and signatures. It contains no benchmark rows. Re-run this command after changing code or data before updating the findings below.

## Repo Layout

```text
apps/inferencex_pca_demo.py
scripts/reproducible_analysis.py
tests/test_analysis_workflow.py
artifacts/reproducible-results.json
requirements-streamlit.txt
README.md
.gitignore
```

`inferencex-pca-data/`, `inferencex-dump-*`, and `exports/` are intentionally excluded from git.

## Data Model

- `benchmark_results.csv` contains measured benchmark rows, workload shape, outcome metrics, and provenance ids.
- `configs.csv` contains model, hardware, framework, precision, and parallelism setup.
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

| Analysis unit                                    | Verified row count |
| ------------------------------------------------ | -----------------: |
| Raw benchmark rows                               |             79,830 |
| Latest row per config/workload/concurrency       |              7,462 |
| Median aggregate per config/workload/concurrency |              7,462 |
| One row per config                               |              1,197 |

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

## Reproducible PCA Findings

Snapshot date: **2026-07-17**. Source commit: `d2ec92073fc87e4c46777a02b9dfe6417e34d0c2`. The full dataset manifest is recorded in `artifacts/reproducible-results.json`.

- Dataset fingerprint: `1a19986135342bc31961497d2fbc6d423408217c7afb9e66e1d2ee85a02c8a09`.
- Default analysis unit: **Median aggregate per config/workload/concurrency** (7,462 rows from 79,830 joined benchmark rows).
- PCA feature set: `isl`, `osl`, `conc`, 12 setup numeric fields through `config_num_prefill_gpu`, plus hardware, framework, model, precision, speculative method, disaggregation, and multinode fields (19 source features total).
- Sample limit: 20,000; all 7,462 default analysis rows were used. Seed: 42.
- PC1–PC5 explain **28.25%, 13.51%, 8.44%, 7.72%, and 6.87%** respectively (**64.78% cumulative**).
- Top five retained-PC contribution groups are `config_disagg`, `config_is_multinode`, `config_decode_tp`, `config_prefill_ep`, and `config_prefill_tp`.

PCA stability uses five deterministic 80% samples (5,970 rows each), with component-loading cosine similarity after sign alignment and top-10 driver frequency. PC1–PC5 minimum loading similarities were 0.9999, 0.9995, 0.9929, 0.9900, and 0.9912. Ten of eleven observed top drivers appeared in at least four of five runs; `config_prefill_dp_attention` appeared once and is flagged as unstable. Treat axis labels and individual signed sides as descriptive rather than fixed findings.

## Target-Aware Modeling

The target-aware layer uses `RandomForestRegressor` as a baseline supervised model. Its primary report uses deterministic grouped cross-validation and computes permutation importance on each validation fold only.

Defaults:

- Predictors are config/setup fields.
- Targets are selected outcome metric columns.
- Split mode defaults to five-fold grouped cross-validation by `config_id` when available (with safe fold reduction if fewer groups exist).

Random K-fold is explicitly labelled as a fallback because repeated configurations can appear across its validation folds.

## Reproducible Target-Aware Finding

For target `metrics_p99_itl`, the same 19 default features, seed 42, and 150-tree forest were evaluated with five grouped `config_id` folds. Every fold had zero train/validation config overlap.

- Aggregate R2: **0.466 ± 0.196**, range **0.093 to 0.634**.
- Aggregate MAE: **0.568 ± 0.122**, range **0.355 to 0.707**.
- Fold validation sizes were 1,492–1,493 rows and 239–240 config groups.
- Top aggregate held-out permutation predictors were `conc`, `config_framework`, `config_spec_method`, `config_hardware`, `config_precision`, and `isl`.

These cross-validated values replace the old one-holdout R2/MAE claim. The variability across held-out config groups is material and should remain visible in conclusions.

## Findings Summary

PCA tells us what structures the benchmark space. Target-aware modeling tells us what predicts selected outcomes. Together, they show that inference infrastructure value depends on workload shape, concurrency, serving framework, speculative method, and parallelism strategy, not just chip or model choice.

## Limitations

- Descriptive, not causal.
- PCA is sensitive to feature encoding and selected analysis unit.
- RandomForest feature importance is target-specific.
- Dataset coverage is uneven.
- Repeated runs can overweight frequently tested configs unless aggregated.
- Grouped cross-validation is safer than random splits, but still not full causal validation.
- PCA stability results are sample-sensitive diagnostics, not proof that an interpretation will generalize to a different dataset snapshot.
- Cloud deployment requires a safe data access plan.

## What Not To Commit

Do not commit:

- `inferencex-dump-*/`
- `inferencex-pca-data/`
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
2. Download/place the approved CSV data folder locally.
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
