# Multi-source source inventory

## InferenceX

The active research source is the frozen July checkpoint at
`.data/inferencex-db-dump-2026-07-20/`, corresponding to SemiAnalysis
`db-dump/2026-07-20`. Its two source tables are
`benchmark_results_raw.csv` and `configs.csv`. The application flattens the
raw `metrics` JSON, prefixes configuration fields, and performs the validated
many-to-one join `benchmark_results.config_id = configs.id`.

`config_id` is the InferenceX configuration identity: a benchmark row can
repeat a configuration under multiple workloads/runs, but `configs.id` is
unique. The current structural and throughput cohort is built by the existing
`apps.inferencex_pca_demo.build_analysis_frame` as the **Median aggregate per
config/workload/concurrency**, grouped by `config_id`, `benchmark_type`, `isl`,
`osl`, and `conc`. Metrics are medians in each group. This integration reuses
that builder through `scripts.build_july_pca_artifact.load_aggregate`; it does
not recreate or replace the cohort.

| Item | Finding |
|---|---:|
| Raw joined benchmark rows | 81,851 |
| Aggregate cohort rows | 8,239 |
| Configurations in joined data | 1,368 |
| `metrics_tput_per_gpu` non-null aggregate rows | 8,239 |
| Target unit | tokens/s/GPU |
| Observed target range | 0.1706–63,535.8684 tokens/s/GPU |

The current throughput feature convention is 11 numerical workload/topology
fields (`isl`, `osl`, `conc`, prefill/decode TP/EP/worker counts, and prefill /
decode GPU counts) and eight categorical fields: benchmark type, hardware,
framework, model, precision, speculative method, disaggregation, and
multinode state. Fold preprocessing in `modeling.comparison.FoldPreprocessor`
fits numerical medians plus missingness indicators and categorical
`__MISSING__` values on training rows only. Grouped validation is by
`config_id`.

Verified InferenceX hardware vocabulary: `b200`, `b300`, `gb200`, `gb300`,
`h100`, `h200`, `mi300x`, `mi325x`, `mi355x`. Verified model vocabulary:
`dsr1`, `dsv4`, `glm5`, `glm5.1`, `glm5.2`, `gptoss120b`, `kimik2.5`,
`llama70b`, `minimaxm2.5`, `minimaxm3`, `qwen3.5`.

The historical TabFM work is implemented through `scripts/model_comparison.py`,
`scripts/model_diagnostics.py`, and the throughput diagnostics scripts. The
dashboard is `apps/inferencex_pca_demo.py`; it has four top-level tabs and
reads completed research artifacts. The reported R²≈0.962 TabFM result is an
aggregate experiment result, not a reusable deployed model artifact: no
serialized TabFM point predictor is present or loaded by the application.

## Epoch raw snapshots

The immutable files are under `data/external/epoch/raw/`. The local
`SHA256SUMS.txt` manifest records a 2026-08-20 source-snapshot capture. The
integration builder checks all five raw SHA-256 identities before and after it
runs.

| Source | Rows | Columns | Identity | Role |
|---|---:|---:|---|---|
| `ml_hardware.csv` | 176 | 39 | `Hardware name` (176 unique; no duplicates) | CanonicalHardware descriptors |
| `all_ai_models.csv` | 3,592 | 57 | `Model` (3,592 unique; no duplicates) | CanonicalModel metadata |
| extracted benchmark data | 77 files | heterogeneous | source-specific model/version identifiers | BenchmarkObservation layer |
| `gpu_clusters.csv` | 482 | 57 | `Name` | GPUCluster infrastructure layer |
| `data_centers.csv` | 83 | 16 | `Name` | DataCenter infrastructure layer |

The complete per-column dtype, missingness, cardinality, exact extracted-file
schema, extracted-file SHA-256, date/model identifier columns, and duplicate
identity checks are machine-readable in
`data/derived/integration/reports/epoch_source_audit.json`. It is the
authoritative complete column inventory, rather than a lossy hand-maintained
wide table.

`ml_hardware.csv` contains direct hardware descriptors including `Memory
(bytes)`, `Memory bandwidth (byte/s)`, `Tensor-FP16/BF16 performance
(FLOP/s)`, FP8/FP4 performance, INT8/INT4 performance, `TDP (W)`, intranode
bandwidth, internode bandwidth, and release date. It also contains composite or
derived columns—`Max performance`, `Energy efficiency`, `Total processing
performance (bit-OP/s)`, `Price-performance`, and `ML OP/s`—which are audited
but excluded from the initial predictive feature set. Missing fields remain
missing: they are never converted to zero. The direct fields are sparse in the
source; in particular, the initial nine accepted hardware records have no
supported internode-bandwidth value.

`all_ai_models.csv` includes model, organization, publication date, parameters,
base model, Hugging Face developer ID, training compute/data/time/hardware,
accessibility, and source/provenance fields. Training variables are not added
to the throughput predictor in this milestone. The source contains model
versions rather than an unambiguous mapping from InferenceX shorthand.

Epoch benchmark files have 77 distinct source tables and many schemas. Common
fields often include a model version and release date, but benchmark-specific
score, harness, prompting, provider, source, and evaluation-date fields vary.
`epoch_capabilities_index.csv` is retained as a benchmark observation schema;
it is not flattened with unrelated score tables.

The 482 GPU-cluster observations describe clusters, owners, chip quantities,
power/cost capacity, dates, locations, and sources. The 83 datacenter rows
describe site-level H100-equivalent capacity, power, capital cost, owners,
chip types, country, and address. Neither unit has a valid entity relationship
to an individual InferenceX deployment operating point. They remain separate
downstream infrastructure/capacity inputs and are deliberately absent from the
throughput view.

## Artificial Analysis boundary

No Artificial Analysis raw export was present. Its intended raw location is
`data/external/artificial_analysis/raw/`; absence is non-fatal. See
[artificial-analysis-integration.md](artificial-analysis-integration.md) for
the strict future contract and metric-semantics boundary.
