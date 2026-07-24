# InferenceX Representation Analysis Demo

This repository contains a Streamlit research dashboard and reproducible modeling pipeline for understanding which inference configuration features structure benchmark performance and which features predict selected outcomes.

## What This Project Does

- Loads local InferenceX benchmark and configuration CSV data, with JSON fallback for local development.
- Joins `benchmark_results.config_id = configs.id`.
- Builds a data-understanding layer for missingness, cardinality, distributions, workload coverage, and data quality.
- Compares PCA, a deterministic autoencoder, and a variational autoencoder on the same
  configuration/workload representation cohort.
- Uses outcome metrics as post-fit overlays or supervised evaluation targets, never as
  representation-training inputs.
- Evaluates Random Forest, CatBoost, and TabFM with deterministic grouped validation by `config_id`.
- Diagnoses residual behavior and compares conformal uncertainty methods.
- Reads completed representation and model artifacts in Streamlit without importing PyTorch or
  fitting any model during normal startup.

## Why This Matters

Inference performance depends on more than model and hardware. Serving framework, precision, concurrency, workload length, parallelism, disaggregation, speculative decoding, and worker allocation can all affect throughput and latency. Benchmarking every possible combination is expensive, so this project tests whether performance can be predicted for configurations that were not present in training.

## Data Source

Active snapshot: **`db-dump/2026-07-20`**, published in the official
[SemiAnalysisAI/InferenceX-app release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20).
The official InferenceX About page identifies these weekly GitHub releases as the full-database
snapshot source. The older `inferencex-pca-data/` June export remains a rollback source.

Required files (official raw export):

- `benchmark_results_raw.csv`
- `configs.csv`

The legacy flattened export uses `benchmark_results.csv` with the same `configs.csv`.

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

Download and verify the official July 20 dump outside the repository, restore only the required
tables, and provide `benchmark_results_raw.csv` plus `configs.csv`. Activate it with:

```bash
export INFERENCEX_DATA_DIR=/absolute/path/to/db-dump-2026-07-20
```

For the verified local audit, the default is
`/tmp/inferencex-dump-comparison/db-dump-2026-07-20/`. The loader expands the official
`metrics` JSON column deterministically. It also supports the older flattened
`benchmark_results.csv` format and compatible JSON/JSONL dumps. No dump is committed.

Rollback to the preserved June export with:

```bash
export INFERENCEX_DATA_DIR=inferencex-pca-data
```

## Local App Setup

```bash
python3 -m venv .venv-streamlit
source .venv-streamlit/bin/activate
python3 -m pip install -r requirements-streamlit.txt
PYTHONPATH=. streamlit run apps/inferencex_pca_demo.py
```

The four top-level pages are **Overview**, **Data Understanding**, **Representation Analysis**, and
**Model Results**. Representation Analysis contains **Principal Component Analysis**,
**Autoencoder**, **Variational Autoencoder**, **Results and Comparison**, and
**Research Validation** subpages.

Neural training is never performed in Streamlit. Install the separate research dependencies into
a Python 3.11 environment before running the scripts:

```bash
python3.11 -m venv .venv-representation
source .venv-representation/bin/activate
python -m pip install -r requirements-representation.txt
```

## Observed Energy Measurements

The **Model Results** tab includes an observed-only Energy Measurements explorer. It performs a
complete configuration-and-workload lookup against rows with a real
`metrics_joules_per_output_token` value. Exact matches show the observed median, range, count,
dates, throughput, average power, and mathematical energy/cost conversions. When no exact row
exists, the dashboard shows nearby measured configurations as comparisons—not predictions.

July measured support remains narrow:

- 5,175 usable raw rows, aggregating to 2,766 config/workload/concurrency groups.
- 305 of 1,368 configurations have measured output-token energy (22.30%).
- Only `single_turn`, OSL 1024, and ISL 1024 or 8192 are represented.
- Measurements run from May 27 through July 18, 2026.
- The official formula is average per-GPU power × GPU count × benchmark duration ÷ actual output
  tokens. Individual dump rows do not pin the metric-code version used at collection time.

Energy prediction remains blocked by narrow workload/category/time coverage and the lack of a
row-level metric-code version. The dashboard does not train or load an energy model, impute labels,
or generate modeled or extrapolated energy values.

## Reproducible Analysis Runs

The legacy June command below also fits the historical Random Forest baseline; it is retained for
provenance and is not part of the July refresh:

```bash
python3 scripts/reproducible_analysis.py \
  --data-dir inferencex-pca-data \
  --output artifacts/reproducible-results.json \
  --max-rows 20000 --seed 42 --stability-runs 5
```

The cumulative July 20 snapshot target-overlay PCA artifact is regenerated independently and does not train any
supervised model:

```bash
PYTHONPATH=. .venv-streamlit/bin/python scripts/build_july_pca_artifact.py \
  --july-data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --june-data-dir inferencex-pca-data \
  --output artifacts/pca-db-dump-2026-07-20.json
```

The preserved Stage 2 bounded neural screens are generated separately:

```bash
PYTHONPATH=. .venv-representation/bin/python scripts/train_autoencoder_representation.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --output artifacts/representation-ae-db-dump-2026-07-20.json \
  --weights artifacts/representation-ae-db-dump-2026-07-20.pt

PYTHONPATH=. .venv-representation/bin/python scripts/train_vae_representation.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --output artifacts/representation-vae-db-dump-2026-07-20.json \
  --weights artifacts/representation-vae-db-dump-2026-07-20.pt

PYTHONPATH=. .venv-streamlit/bin/python scripts/build_representation_comparison_artifact.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --output artifacts/representation-comparison-db-dump-2026-07-20.json
```

The bounded Stage 3 final experiment uses the fixed 15-dimensional architecture and seeds 42,
123, and 2026:

```bash
PYTHONPATH=. .venv-representation/bin/python scripts/train_autoencoder_representation_final.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --maximum-epochs 250

PYTHONPATH=. .venv-representation/bin/python scripts/diagnose_vae_representation_beta.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20

PYTHONPATH=. .venv-representation/bin/python scripts/train_vae_representation_final.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20

PYTHONPATH=. .venv-streamlit/bin/python scripts/build_representation_comparison_final.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20
```

Stage 4 is a fixed methodological validation, not another model search. It fits every imputer,
scaler, categorical mapping, and representation on grouped-fold training rows only; repeats the
evaluation across three independent grouped partition assignments; and adds source-balanced
reconstruction, bounded robustness interventions, cross-method cluster agreement, and two
feature-family ablations:

```bash
PYTHONPATH=. .venv-representation/bin/python \
  scripts/run_representation_validation_stage4.py \
  --data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20
```

The command writes
`artifacts/representation-validation-stage4-db-dump-2026-07-20.json`. It does not replace any
Stage 3 artifact. The optional `--augment-warmup-diagnostics` mode recreates only the three fixed
VAE warm-up folds when per-dimension diagnostic arrays need to be refreshed; it is not a search.

The preserved PCA artifact is `artifacts/pca-db-dump-2026-07-20.json`. Neural JSON artifacts store
cohort and row-key hashes, exact feature order, preprocessing and split metadata, training
histories, validation/evaluation results, and software versions. Final v2 artifacts move row-level
embeddings and row keys to Zstandard-compressed Parquet companions and use versioned `.pt` bundles
for all seed/fold weights. The dashboard rejects artifacts with incompatible schemas, snapshot
versions, cohorts, feature order, companion checksum, row order, or seeds. See
`reports/representation_analysis_protocol.md` for the fixed pre-interpretation protocol.
The completed one-seed screening results and the explicit stop decision before Stage 3 are in
`reports/representation_analysis_stage2.md`. Final multi-seed evidence is in
`reports/representation_analysis_stage3.md`. Leakage, partition, source-feature, interpretability,
cluster-agreement, and ablation evidence is in `reports/representation_analysis_stage4.md`.

For reproducibility, use the active July dump, fixed seeds 42/123/2026, the recorded three
`config_id` folds, and the commands above. Neural training is never performed in Streamlit. Do not
add database dumps to Git.

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
| Raw benchmark rows | 81,851 |
| Median aggregate per config/workload/concurrency | 8,239 |
| One row per benchmarked config | 1,368 |

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
- The July 20 dump is cumulative. The updated full-dataset basis uses all 8,063 eligible
  `single_turn` aggregate groups in that snapshot. All 7,462 groups from the prior snapshot are
  retained, alongside 601 new eligible groups. The contributing source observations span
  2025-09-29 through 2026-07-18 and include 79,975 pre-July rows plus 1,660 July rows. It is not fit
  only on July-dated observations.
- The 176 `agentic_traces` aggregates are excluded because their null ISL/OSL fields do not share
  the same workload semantics as the three target overlays.
- A single shared basis supports direct comparison between the three overlays.
- **Median TPOT** is the primary latency-focused PCA overlay. It uses raw
  `metrics_median_tpot` in seconds/output token (identity transform, lower is better). An optional
  `log1p` color scale is display-only and never changes the stored raw target.
- **Throughput per GPU** separately overlays raw `metrics_tput_per_gpu`
  (tokens/second/GPU, identity transform, higher is better) because it was the final selected
  supervised target.
- **Joules per output token** overlays observed `metrics_joules_per_output_token` only (lower is
  better).
- None of the three targets, nor any other latency, throughput, power, or energy metric, enters PCA
  preprocessing.

## July 20 PCA Findings

Snapshot release: **`db-dump/2026-07-20`**.

- Default analysis unit: 8,239 median aggregate rows from 81,851 joined benchmark rows.
- Shared PCA basis: all 8,063 eligible aggregate groups in the cumulative snapshot. Source
  observations span 2025-09-29 through 2026-07-18; aggregate representative dates span through
  2026-07-17. No July-only date filter is applied.
- PCA feature set: 19 configuration and workload source features.
- PC1 through PC5 explain 28.11%, 13.47%, 8.14%, 7.71%, and 6.93% of variance.
- Cumulative explained variance for PC1 through PC5: **64.37%**, versus 64.78% for the June basis.
- The first-five loading cosine similarities versus June range from 0.987 to 0.999; five-dimensional
  principal angles range from 0.81° to 4.23°.
- Raw median TPOT is most strongly rank-aligned with PC5 among the first five (Pearson −0.120;
  Spearman −0.210). Throughput is also most strongly aligned with PC5 (Pearson −0.345; Spearman
  −0.408). Observed energy is most strongly rank-aligned with PC3 (Pearson 0.288; Spearman 0.600).
  These are descriptive associations, not predictions or causal effects.

The versioned artifact stores preprocessing state, encoded and source loadings, explained variance,
component-bin summaries for all three overlays, target associations, full-snapshot scope metadata,
cohort filters, and dump provenance. It does not
contain supervised predictions.

## Limitations

- The analysis is descriptive, not causal.
- Several experiment rows are historical bounded runs with different sample sizes or context limits.
- Not every target received a complete Random Forest, CatBoost, and TabFM comparison under identical folds.
- The historical final throughput experiment used 4,096 sampled June aggregate rows rather than
  all 7,462 June rows; it was not retrained on July data.
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
- Revisit energy modeling only after workload, category, and temporal support broadens enough for
  grouped and temporal validation.
- Add benchmark coverage-gap analysis.
- Add quality and performance tradeoff analysis using `eval_results.csv`.
