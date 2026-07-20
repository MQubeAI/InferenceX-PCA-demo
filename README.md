# InferenceX PCA Demo

This repository contains a Streamlit research dashboard and reproducible modeling pipeline for understanding which inference configuration features structure benchmark performance and which features predict selected outcomes.

## What This Project Does

- Loads local InferenceX benchmark and configuration CSV data, with JSON fallback for local development.
- Joins `benchmark_results.config_id = configs.id`.
- Builds a data-understanding layer for missingness, cardinality, distributions, workload coverage, and data quality.
- Runs PCA on configuration and workload features only.
- Uses outcome metrics as overlays or supervised targets, not PCA inputs.
- Evaluates Random Forest, CatBoost, and TabFM with deterministic grouped validation by `config_id`.
- Diagnoses residual behavior and compares conformal uncertainty methods.
- Reads aggregate research artifacts in Streamlit without fitting TabFM during normal startup.

## Why This Matters

Inference performance depends on more than model and hardware. Serving framework, precision, concurrency, workload length, parallelism, disaggregation, speculative decoding, and worker allocation can all affect throughput and latency. Benchmarking every possible combination is expensive, so this project tests whether performance can be predicted for configurations that were not present in training.

## Data Source

Official data folder: [Google Drive](https://drive.google.com/drive/u/2/folders/1RQYKaliWtJym1kbGH9A4SwfzqSh1RBvR)

Required files:

- `benchmark_results.csv`
- `configs.csv`

Optional or supported later:

- `availability.csv`
- `eval_results.csv`
- `run_stats.csv`
- `workflow_runs.csv`
- `changelog_entries.csv`

Intentionally skipped:

- `server_logs.json`
- `eval_samples.json`

The skipped files can be extremely large and are unnecessary for this workflow.

## Local Data Setup

Place the approved CSV export at the repository root and name the folder:

```text
inferencex-pca-data/
```

The application prefers CSV. It can fall back to compatible JSON or JSONL dumps for local development.

## Local App Setup

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python3 -m pip install -r requirements-streamlit.txt
PYTHONPATH=. streamlit run apps/inferencex_pca_demo.py
```

## Reproducible Analysis Run

```bash
python3 scripts/reproducible_analysis.py \
  --data-dir inferencex-pca-data \
  --output artifacts/reproducible-results.json \
  --max-rows 20000 --seed 42 --stability-runs 5
```

## Completed Model Research

The selected research point model is full fold-context TabFM for `metrics_tput_per_gpu`.

| Result | Value |
|---|---:|
| Rows in experiment | 4,096 |
| Validation | 3 grouped folds by `config_id` |
| R2 | **0.961979 +/- 0.008605** |
| MAE | **338.540384 tokens/s/GPU** |

Throughput residuals were strongly heteroskedastic and non-Gaussian. Conditional-scale split conformal was selected as the single uncertainty research method. At 95% nominal coverage it achieved 95.34% empirical coverage, 2,485.442 average interval width, and a 4,739.169 interval score. Its intervals were 34.62% narrower than global conformal.

The uncertainty experiment used only about half of each outer-training fold as TabFM context because separate rows were required for uncertainty training and calibration. Its internal point-model R2 was 0.913897. That result must not replace the 0.961979 full-context result. The intervals remain research-only and are not calibrated around the selected full-context point model.

Median-TPOT two-stage tail modeling did not consistently improve the weaker global latency baseline. Latency segmentation, residual modeling, VAE, and CRVAE were not pursued.

See `docs/model-research-conclusion.md` for the final decision record.

# Full Model Comparison Ledger

## How to Read These Results

Not every row below is directly comparable.

- A direct comparison requires the same target, sampled rows, feature set, and fold assignment.
- A dash means that model was not run or the metric was not preserved for that exact setup.
- Negative R2 means the model performed worse than predicting the validation-set mean.
- All primary evaluations hold out complete `config_id` groups unless explicitly labeled as a smoke test or interpolation diagnostic.
- The final throughput result used 4,096 sampled rows, not all 7,462 aggregate rows.
- "Full context" means all fold-training rows available inside that experiment were used as TabFM context. It does not mean every source column or every row in the complete dataset was used.

## 1. Direct Head-to-Head Experiments

These are the cleanest Random Forest, CatBoost, and TabFM comparisons.

| Target and experiment | Rows | Validation | Random Forest | CatBoost | TabFM | Winner |
|---|---:|---|---|---|---|---|
| p99 ITL smoke test | 96 | One grouped split | R2 `-5.674`, MAE `1.551` | R2 `-1.290`, MAE `0.973` | R2 `-1.323`, MAE `1.175` | CatBoost, but all unusable |
| p99 ITL fair bounded run | 1,024 | One grouped fold | R2 `0.270` | R2 `0.120` | R2 `0.635` | TabFM |
| p99 ITL main bounded comparison | 1,024 | Three grouped folds | R2 `-0.044 +/- 0.403` | R2 `0.145 +/- 0.036` | R2 `0.294 +/- 0.134`, MAE `0.642 +/- 0.147` | TabFM |
| p99 ITL with `log1p` target | 1,024 | Three grouped folds | R2 `0.328` | R2 `0.229` | Not run | Random Forest |
| p99 ITL known-configuration diagnostic | 1,024 | Interpolation only | R2 `-0.060`, MAE `0.564` | R2 `-0.334`, MAE `0.717` | Not run | Random Forest, but both weak |
| p99 ITL full aggregate table | 7,462 | Five grouped folds | R2 `0.465 +/- 0.195`, MAE `0.568 +/- 0.122` | R2 `0.337 +/- 0.136`, MAE `0.811 +/- 0.140` | Not run | Random Forest |

### 96-Row Smoke-Test Details

The smoke test verified that all three model adapters worked locally. It was not intended as a quality conclusion.

| Model | Train rows | Validation rows | R2 | MAE | Runtime |
|---|---:|---:|---:|---:|---:|
| Random Forest | 48 | 48 | `-5.674155` | `1.551187` | `0.565 s` |
| CatBoost | 48 | 48 | `-1.289875` | `0.972537` | `0.119 s` |
| TabFM | 48 available, 32 used as context | 48 | `-1.323274` | `1.175118` | `20.400 s` |

### Full 7,462-Row p99 ITL Fold Results

| Fold | Random Forest R2 | Random Forest MAE | CatBoost R2 | CatBoost MAE |
|---:|---:|---:|---:|---:|
| 1 | `0.095898` | `0.704520` | `0.170178` | `0.875647` |
| 2 | `0.605779` | `0.563509` | `0.385961` | `0.847494` |
| 3 | `0.538133` | `0.662711` | `0.253291` | `0.973158` |
| 4 | `0.633757` | `0.353146` | `0.570187` | `0.554708` |
| 5 | `0.451159` | `0.555376` | `0.305019` | `0.802908` |
| **Mean** | **`0.464945`** | **`0.567852`** | **`0.336927`** | **`0.810783`** |
| **Std. dev.** | **`0.194974`** | **`0.121614`** | **`0.136112`** | **`0.139683`** |

Runtime:

| Model | Total runtime |
|---|---:|
| Random Forest | `20.10 s` |
| CatBoost | `0.95 s` |

For full-data p99 ITL, Random Forest was more accurate while CatBoost was much faster.

## 2. Target-Screening Comparison

This 1,024-row CatBoost diagnostic identified which targets were worth additional TabFM compute.

| Target | CatBoost raw R2 | CatBoost `log1p` R2 | Random Forest result | TabFM result |
|---|---:|---:|---|---|
| Throughput per GPU | **`0.643`** | `0.634` | No preserved comparable run | Later reached `0.962` |
| Median TPOT | `0.588` | `0.589` | No preserved comparable run | Later ranged from `0.48` to `0.79` |
| Mean E2EL | `0.487` | `0.436` | Not run | Not run |
| Median ITL | `0.462` | `0.443` | Not run | Not run |
| Mean ITL | `0.367` | `0.379` | Not run | Not run |
| p99 ITL | `0.145` | `0.229` | Raw `-0.044`, log `0.328` | Raw `0.294 +/- 0.134` |
| Median TTFT | `0.107` | `0.069` | Not run | Not run |

Throughput was the strongest screened target. Median TPOT was the strongest latency candidate. p99 ITL remained tail-heavy and unstable.

## 3. Throughput Comparison and TabFM Context Scaling

After throughput emerged as the strongest target, most later experiments focused on how TabFM changed with more context.

### 1,024-Row Context Experiments

| Model | Context strategy | Context rows | R2 | MAE |
|---|---|---:|---:|---:|
| CatBoost | Standard supervised baseline | Full fold training data | `0.643` | Not preserved in the screening summary |
| TabFM | Coverage | 128 | `0.66586` | Not preserved |
| TabFM | Random | 128 | `0.67695` | Not preserved |
| TabFM | Coverage | 256 | `0.79541` | Not preserved |
| TabFM | Random | 256 | `0.83714` | Not preserved |
| TabFM | Coverage | 512 | `0.90325` | Not preserved |
| TabFM | Random | 512 | `0.90231` | Not preserved |
| TabFM | Nearest | 512 | `0.77774` | Not preserved |
| TabFM | Stratified | 512 | `0.88043` | Not preserved |
| TabFM | Random | 640 | `0.91543` | Not preserved |
| TabFM | Full fold context | About 682 | **`0.924474 +/- 0.009809`** | `461.217` |
| Random Forest | Same final setup | Not run | Not run | Not run |

### Larger Throughput Experiments

| Model | Total sampled rows | Context rows per fold | R2 | MAE |
|---|---:|---:|---:|---:|
| CatBoost screening baseline | 1,024 | Standard fold training | `0.643` | Not preserved in the screening summary |
| TabFM | 1,024 | About 682 | `0.924474 +/- 0.009809` | `461.217` |
| TabFM | 2,048 | About 1,024 | `0.934478 +/- 0.022792` | `408.911` |
| TabFM | 4,096 | About 2,730 | **`0.961979 +/- 0.008605`** | **`338.540384`** |
| TabFM uncertainty experiment | 4,096 | About 1,350 to 1,400 | `0.913897` | `449.803343` |
| Random Forest | These exact setups | Not run | Not run | Not run |

### Final 4,096-Row Throughput Folds

| Fold | TabFM R2 | TabFM MAE |
|---:|---:|---:|
| 1 | `0.972651` | `301.225` |
| 2 | `0.951579` | `373.555` |
| 3 | `0.961707` | `340.840` |
| **Mean** | **`0.961979`** | **`338.540`** |
| **R2 std. dev.** | **`0.008605`** | - |

The supported conclusion is that TabFM strongly outperformed the preserved CatBoost throughput screening baseline. A final Random Forest throughput comparison under the exact 4,096-row folds was not completed, so the repository does not claim that TabFM beat every possible Random Forest implementation under an identical final setup.

## 4. Median TPOT Comparison

Median TPOT initially looked promising, but larger TabFM experiments became much less stable.

| Model | Rows | Context or setup | R2 | MAE |
|---|---:|---|---:|---:|
| CatBoost | 1,024 | Three grouped folds | `0.588` | Not preserved in the screening summary |
| TabFM | 1,024 | Coverage, 512 context | `0.726524 +/- 0.051359` | `0.006412` |
| TabFM | 1,024 | Random, 512 context | `0.723389 +/- 0.024264` | `0.006210` |
| TabFM | 1,024 | Full context, about 682 | **`0.789370 +/- 0.046334`** | **`0.005644`** |
| TabFM | 2,048 | 1,024 context | `0.609838 +/- 0.188622` | `0.006582` |
| TabFM | 2,048 | Full context, about 1,365 | `0.605943 +/- 0.182013` | `0.006122` |
| TabFM, seed 123 | 2,048 | Full context, about 1,365 | `0.481594 +/- 0.187365` | `0.007142` |
| TabFM | 4,096 | Full fold context | `0.604782 +/- 0.198737` | `0.005893` |
| TabFM tail-diagnostic rerun | 4,096 | Full fold context | `0.602290 +/- 0.198851` | `0.005913` |
| Random Forest | Median TPOT | No preserved grouped run | Not run | Not run |

### Final 4,096-Row Median TPOT Folds

| Fold | TabFM R2 | TabFM MAE |
|---:|---:|---:|
| 1 | `0.718840` | `0.005953` |
| 2 | `0.765655` | `0.005269` |
| 3 | `0.322375` | `0.006518` |
| **Mean** | **`0.602290`** | **`0.005913`** |

The large Fold 3 drop was the main reason median TPOT was treated as unstable.

## 5. CatBoost and TabFM Latency Hybrids

These were two-stage systems rather than standalone CatBoost models. A classifier attempted to identify difficult tail cases and route them to a specialized model.

| Architecture | Mean R2 | Overall MAE | Ordinary-case MAE | True-tail MAE | Worst-decile MAE |
|---|---:|---:|---:|---:|---:|
| Global TabFM | **`0.602290`** | **`0.005913`** | `0.003642` | **`0.049681`** | **`0.039893`** |
| CatBoost classifier -> CatBoost tail model | `0.506481` | `0.006432` | `0.003473` | `0.063141` | `0.045965` |
| CatBoost classifier -> fallback tail model | `0.223170` | `0.007823` | **`0.003136`** | `0.097284` | `0.059624` |
| CatBoost classifier -> TabFM tail model | `0.554491` | `0.006101` | `0.003336` | `0.059114` | `0.042747` |
| Logistic classifier -> CatBoost tail model | `-0.770218` | `0.020185` | `0.018074` | `0.060690` | `0.122285` |
| Weighted CatBoost tail model | `0.535279` | `0.006981` | `0.004396` | `0.056611` | `0.044819` |

The global TabFM baseline had the best overall R2, overall MAE, true-tail MAE, and worst-decile MAE. Some hybrids marginally improved ordinary-case MAE, but they harmed the difficult cases they were designed to fix.

### Tail Classifier Behavior

| Classifier | Fold 1 precision / recall | Fold 2 precision / recall | Fold 3 precision / recall |
|---|---:|---:|---:|
| CatBoost | `0.688 / 0.349` | `0.705 / 0.388` | `0.938 / 0.469` |
| Logistic regression | `0.188 / 0.762` | `0.230 / 0.813` | `0.205 / 0.844` |

CatBoost was precise but missed most real tail cases. Logistic regression found more tail cases but incorrectly routed too many ordinary rows.

## 6. Model-by-Model Record

### Random Forest

| Experiment | Result |
|---|---|
| 96-row p99 smoke | R2 `-5.674`, MAE `1.551` |
| 1,024-row p99 one-fold | R2 `0.270` |
| 1,024-row p99 three-fold raw | R2 `-0.044 +/- 0.403` |
| 1,024-row p99 `log1p` | R2 `0.328` |
| Known-config p99 | R2 `-0.060`, MAE `0.564` |
| Full 7,462-row p99 | R2 `0.465 +/- 0.195`, MAE `0.568 +/- 0.122` |
| Throughput | No preserved comparable final run |
| Median TPOT | No preserved comparable grouped run |

### CatBoost

| Experiment | Result |
|---|---|
| 96-row p99 smoke | R2 `-1.290`, MAE `0.973` |
| 1,024-row p99 one-fold | R2 `0.120` |
| 1,024-row p99 three-fold raw | R2 `0.145 +/- 0.036` |
| 1,024-row p99 `log1p` | R2 `0.229` |
| Known-config p99 | R2 `-0.334`, MAE `0.717` |
| Full 7,462-row p99 | R2 `0.337 +/- 0.136`, MAE `0.811 +/- 0.140` |
| 1,024-row throughput screening | R2 `0.643` |
| 1,024-row median TPOT screening | R2 `0.588` |
| Mean E2EL | R2 `0.487` |
| Median ITL | R2 `0.462` |
| Mean ITL | R2 `0.367` |
| Median TTFT | R2 `0.107` |
| Best latency hybrid | CatBoost classifier -> TabFM tail, R2 `0.554`, MAE `0.006101` |

### TabFM

| Experiment | Result |
|---|---|
| 96-row p99 smoke | R2 `-1.323`, MAE `1.175` |
| 1,024-row p99 one-fold | R2 `0.635` |
| 1,024-row p99 three-fold | R2 `0.294 +/- 0.134`, MAE `0.642 +/- 0.147` |
| 1,024-row throughput full context | R2 `0.924 +/- 0.010`, MAE `461.217` |
| 2,048-row throughput | R2 `0.934 +/- 0.023`, MAE `408.911` |
| 4,096-row throughput | **R2 `0.962 +/- 0.009`, MAE `338.540`** |
| Split-context throughput | R2 `0.914`, MAE `449.803` |
| Best 1,024-row median TPOT | R2 `0.789 +/- 0.046`, MAE `0.005644` |
| 2,048-row median TPOT | About R2 `0.606`, MAE `0.006122` |
| 4,096-row median TPOT | R2 `0.602 +/- 0.199`, MAE `0.005913` |

## 7. Final Comparison Summary

| Target | Random Forest | CatBoost | TabFM | Final interpretation |
|---|---|---|---|---|
| p99 ITL | Sometimes beat CatBoost, but unstable | Weak but comparatively stable | Best in the bounded three-model run, but still weak | Do not use as the central target |
| Throughput | No final identical-fold comparison | Strongest screened tree result at `0.643` | Reached `0.962` | TabFM selected |
| Median TPOT | No preserved grouped comparison | Screening R2 `0.588` | Reached `0.789` early, then fell to about `0.602` at scale | Keep as a weaker research baseline |
| Tail-specialized latency | Not used | Routing and tail components worsened results | Global TabFM remained best | Reject the two-stage approach |

The central conclusion is:

> TabFM produced the strongest throughput result. Random Forest was the strongest conventional model for full-data p99 ITL, while CatBoost was substantially faster. TabFM also beat both conventional models in the bounded p99 comparison, but p99 remained too unstable. Median TPOT showed early promise, but larger TabFM runs were inconsistent, and CatBoost-based tail specialization failed to improve it.

## Reproducing the Conventional p99 Comparison

```bash
.venv-streamlit/bin/python scripts/model_comparison.py \
  --data-dir inferencex-pca-data \
  --target metrics_p99_itl \
  --models random_forest,catboost \
  --folds 5 --seed 42 --max-rows 20000 \
  --output artifacts/model-comparison.json
```

## Running the TabFM Smoke Test

TabFM is research-only, lazy-loaded, and run through the separate `.venv-tabfm` environment. Normal Streamlit startup does not import TabFM or load its checkpoint.

```bash
.venv-tabfm/bin/python scripts/model_comparison.py \
  --data-dir inferencex-pca-data \
  --target metrics_p99_itl \
  --models tabfm \
  --folds 1 --seed 42 --max-rows 96 \
  --tabfm-max-context 32 \
  --output artifacts/tabfm-smoke-results.json
```

## Missingness and Preprocessing Policy

- Rows with a missing target are excluded for that target. Target labels are never imputed.
- Within each training fold, numerical predictors receive the training-fold median plus a missingness indicator.
- Categorical predictors receive the explicit `__MISSING__` category.
- All fold preprocessing is fitted on training rows only.
- Model artifacts contain aggregate metadata and metrics, not source rows or row-level predictions.

## Repository Layout

```text
apps/inferencex_pca_demo.py
modeling/
scripts/
tests/
artifacts/
docs/
requirements-streamlit.txt
README.md
```

`inferencex-pca-data/`, `inferencex-dump-*`, and `exports/` are intentionally excluded from Git.

## Data Model

- `benchmark_results.csv` contains measured benchmark rows, workload shape, outcome metrics, and provenance IDs.
- `configs.csv` contains model, hardware, framework, precision, and parallelism setup.
- The core join is `benchmark_results.config_id = configs.id`.
- One joined row represents one benchmark workload and configuration observation.

## Feature Families

### Workload Shape

- `isl`: input sequence length.
- `osl`: output sequence length.
- `conc`: concurrency or concurrent requests.
- `benchmark_type`: benchmark or workload type.

### Model, Hardware, and Software

- `config_model`
- `config_hardware`
- `config_framework`
- `config_precision`
- `config_spec_method`

### Serving Topology and Parallelism

- `config_disagg`
- `config_is_multinode`
- `config_prefill_tp`
- `config_prefill_ep`
- `config_prefill_dp_attention`
- `config_prefill_num_workers`
- `config_decode_tp`
- `config_decode_ep`
- `config_decode_dp_attention`
- `config_decode_num_workers`
- `config_num_prefill_gpu`
- `config_num_decode_gpu`

### Outcome Metrics

- `metrics_*_ttft`: time to first token.
- `metrics_*_tpot`: time per output token.
- `metrics_*_itl`: inter-token latency.
- `metrics_*_e2el`: end-to-end latency.
- Throughput metrics: higher is generally better.
- Latency and energy metrics: lower is generally better.

## Analysis Units

| Analysis unit | Verified row count |
|---|---:|
| Raw benchmark rows | 79,830 |
| Latest row per config/workload/concurrency | 7,462 |
| Median aggregate per config/workload/concurrency | 7,462 |
| One row per config | 1,197 |

The default analysis unit is **Median aggregate per config/workload/concurrency**.

Grouping keys:

- `config_id`
- `benchmark_type`
- `isl`
- `osl`
- `conc`

## PCA Methodology

PCA is unsupervised. It identifies structural variance, not causal value.

- PCA inputs are configuration and workload features only.
- Outcome metrics are not PCA inputs.
- Metrics may be used as color overlays or supervised targets.
- Encoded loading contributions are regrouped to their original source features.

## Reproducible PCA Findings

Snapshot date: **2026-07-17**.

- Dataset fingerprint: `1a19986135342bc31961497d2fbc6d423408217c7afb9e66e1d2ee85a02c8a09`.
- Default analysis unit: 7,462 median aggregate rows from 79,830 joined benchmark rows.
- PCA feature set: 19 configuration and workload source features.
- PC1 through PC5 explain 28.25%, 13.51%, 8.44%, 7.72%, and 6.87% of variance.
- Cumulative explained variance for PC1 through PC5: **64.78%**.
- Top retained-component contribution groups: `config_disagg`, `config_is_multinode`, `config_decode_tp`, `config_prefill_ep`, and `config_prefill_tp`.

PCA stability used five deterministic 80% samples. Minimum sign-aligned loading similarities were:

| Component | Minimum similarity |
|---|---:|
| PC1 | `0.9999` |
| PC2 | `0.9995` |
| PC3 | `0.9929` |
| PC4 | `0.9900` |
| PC5 | `0.9912` |

Ten of eleven observed top drivers appeared in at least four of five runs. `config_prefill_dp_attention` appeared once and was flagged as unstable.

## Limitations

- The analysis is descriptive, not causal.
- Several experiment rows are historical bounded runs with different sample sizes or context limits.
- Not every target received a complete Random Forest, CatBoost, and TabFM comparison under identical folds.
- The final throughput experiment used 4,096 sampled aggregate rows rather than all 7,462 rows.
- TabFM was CPU-intensive on the available hardware.
- Dataset coverage is uneven across configurations and workloads.
- Grouped validation reduces leakage but does not eliminate every source of dataset shift.
- Conditional uncertainty achieved strong overall coverage but undercovered the hardest high-throughput subgroup.
- The uncertainty model is research-only and is not calibrated around the selected full-context point model.
- PCA is sensitive to feature encoding and the selected analysis unit.
- Repeated runs can overweight frequently tested configurations unless aggregated.

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

1. Clone the repository.
2. Download and place the approved CSV data folder locally.
3. Create the appropriate Python virtual environment.
4. Install `requirements-streamlit.txt`.
5. Run Streamlit with `PYTHONPATH=.`.
6. Use the median aggregate analysis unit.
7. Read completed aggregate research artifacts in the dashboard.
8. Re-run experiments only when the data, task, or evaluation contract changes.

## Future Improvements

- Audit whether the data supports configuration ranking and regret evaluation.
- Build an observed-candidate recommendation layer if workloads contain enough comparable configurations.
- Add cost-per-token valuation.
- Add energy metrics when coverage becomes sufficient.
- Add benchmark coverage-gap analysis.
- Add quality and performance tradeoff analysis using `eval_results.csv`.
