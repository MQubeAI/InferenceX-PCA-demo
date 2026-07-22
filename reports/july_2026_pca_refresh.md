# July 2026 InferenceX PCA refresh

Refresh date: 2026-07-22  
Active release: `db-dump/2026-07-20`  
Official source: [SemiAnalysisAI/InferenceX-app release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20)  
Local verified source: `/tmp/inferencex-dump-comparison/db-dump-2026-07-20/`

## Decision summary

The July snapshot reproduces all six required count gates under the existing join and median
aggregation policy. The main dashboard now uses one shared July `single_turn` PCA basis for direct
comparison between the output-performance and observed-energy overlays. The 176 new
`agentic_traces` aggregate rows are excluded from the shared basis because they have no ISL/OSL and
therefore do not share the workload semantics used by either overlay.

No supervised model was trained. Historical TabFM and uncertainty artifacts remain unchanged and
are presented only as June target-selection context, never applied to July rows.

## Dataset refresh

| Measure | June 29 | July 20 | Change |
|---|---:|---:|---:|
| Raw benchmark rows | 79,830 | 81,851 | +2,021 |
| Median config/workload/concurrency rows | 7,462 | 8,239 | +777 |
| Benchmarked configurations | 1,197 | 1,368 | +171 |
| Usable output-energy rows | 3,794 | 5,175 | +1,381 |
| Output-energy aggregate groups | 2,135 | 2,766 | +631 |
| Energy-measured configurations | 251 | 305 | +54 |

The July canonical loader expands the official `metrics` JSON column to the existing
`metrics_*` shape before the unchanged many-to-one configuration join. Aggregation remains median
by `config_id`, `benchmark_type`, `isl`, `osl`, and `conc`. The configuration schema is unchanged;
new raw benchmark fields do not alter grouping semantics.

## Verified output target

PCA Analysis A uses the final selected target established by the existing experiment evidence:

- raw target: `metrics_tput_per_gpu`
- transformation: identity
- inverse transformation: identity
- units: tokens/second/GPU
- direction: higher is better
- final selection context: 4,096 rows; three grouped `config_id` folds; TabFM R²
  `0.961979 +/- 0.008605`; MAE `338.540384` tokens/s/GPU

`log1p(metrics_median_tpot)` is historical diagnostic context only. It is not a primary PCA mode.

## Shared PCA feature matrix

The 19 source features, in their frozen order, are:

1. `isl`
2. `osl`
3. `conc`
4. `config_prefill_tp`
5. `config_prefill_ep`
6. `config_prefill_dp_attention`
7. `config_prefill_num_workers`
8. `config_decode_tp`
9. `config_decode_ep`
10. `config_decode_dp_attention`
11. `config_decode_num_workers`
12. `config_num_prefill_gpu`
13. `config_hardware`
14. `config_framework`
15. `config_model`
16. `config_precision`
17. `config_spec_method`
18. `config_disagg`
19. `config_is_multinode`

Numeric fields use median imputation followed by standard scaling. The topology booleans preserve
the June numeric encoding after deterministic `t`/`f` normalization. Categorical fields use
most-frequent imputation and one-hot encoding with the existing 30-category cap. Outcome metrics,
including throughput, latency, power, and energy, are rejected from the PCA feature schema.

The shared fit contains 8,063 `single_turn` aggregate rows. Both target cohorts are projected into
this identical feature order and basis.

## Explained variance and basis stability

| Component | June variance | July variance | Loading cosine | Sign-aligned correlation |
|---|---:|---:|---:|---:|
| PC1 | 28.25% | 28.11% | 0.9995 | 0.9993 |
| PC2 | 13.51% | 13.47% | 0.9991 | 0.9991 |
| PC3 | 8.44% | 8.14% | 0.9916 | 0.9916 |
| PC4 | 7.72% | 7.71% | 0.9867 | 0.9867 |
| PC5 | 6.87% | 6.93% | 0.9898 | 0.9899 |

PC1–PC5 cumulative variance declines modestly from 64.78% to 64.37%. The five-dimensional
principal angles are 4.23°, 2.53°, 1.95°, 1.16°, and 0.81°. This is strong evidence that the first
five structural directions remain stable enough for a shared-basis refresh.

July needs 4, 7, 10, 15, and 21 components to reach 50%, 70%, 80%, 90%, and 95% cumulative
variance, respectively.

The largest July PC1 loading magnitudes are disaggregation, multinode topology, prefill DP
attention, framework, decode EP, and prefill worker count. PC4 is primarily an opposing ISL/OSL
workload axis. PC5 combines decode workers and prefill EP against concurrency, ISL, and prefill
workers. Component signs are arbitrary; interpretation uses the stored sign alignment.

## Analysis A: output performance

- usable aggregate rows: 8,063
- unique configurations: 1,354
- workload support: `single_turn`; ISL 1024 or 8192; OSL 1024 or 8192
- raw median: 1,049.81 tokens/second/GPU
- interquartile range: 328.92–2,916.50
- 95th percentile: 10,305.65
- maximum: 63,535.87

Among PC1–PC5, PC5 has the largest absolute associations with raw throughput (Pearson −0.345;
Spearman −0.408). Median throughput falls from 2,623.18 tokens/s/GPU in the lowest PC5 quintile to
515.40 in the highest quintile. This is an alignment within the observed July benchmark mix, not a
causal effect and not a new predictive validation result.

## Analysis B: observed energy

- raw measured rows: 5,175
- aggregate measured groups: 2,766
- measured configurations: 305 of 1,368 (22.30%)
- dates: 2026-05-27 through 2026-07-18
- workload support: `single_turn`; ISL 1024 or 8192; OSL 1024; 15 observed concurrency values
- hardware values: 8
- framework values: 7
- model values: 9
- precision values: BF16, FP4, FP8, and INT4

Observed joules per output token have median 2.9723, interquartile range 1.2898–6.6298, 90th
percentile 13.9985, 95th percentile 21.4255, 99th percentile 43.4152, minimum 0.0785, and maximum
86.9951. `log1p` is available only as a distribution visualization; all displayed values and
associations use the observed raw target.

PC3 has the strongest rank association with observed energy (Pearson 0.288; Spearman 0.600).
Median energy rises from 0.9418 joules/output token in the lowest PC3 quintile to 7.1672 in the
highest quintile. PC3 is most aligned with prefill TP, decode TP, decode DP attention, framework,
disaggregation, and model mix. This describes the measured subset only and does not imply that a
component predicts or causes energy use.

Monthly observed medians are 2.1816 in May, 3.3194 in June, and 3.0446 in July. These shifts justify
the date-colored view and continued instrumentation caution. Sparse-category warnings remain for
one hardware category, two framework categories, one model category, and one precision category
with fewer than 30 measured aggregate rows.

The official target formula is average per-GPU power × GPU count × benchmark duration ÷ actual
output tokens. General energy prediction remains blocked: workload support is still confined to
one benchmark type, one OSL, two ISLs, a short collection period, and concentrated category support;
the dump also does not pin the metric-code version to each row.

## Dashboard and artifact changes

The four top-level tabs remain Overview, Data Understanding, PCA, and Model Results. The PCA tab now
contains:

- shared-basis variance and June/July stability diagnostics;
- Output Performance PCA with raw throughput definition, direction, distribution, associations,
  component bins, loadings, and interactive PC1/PC2 and PC1/PC3 projections;
- Energy PCA with measured-only support, distribution, associations, component bins, sparse-group
  warnings, date-colored projection, and observed subgroup summaries.

The observed Energy Measurements explorer continues to return exact measurements or comparison
rows only and now derives all July support counts dynamically. No energy estimator is loaded.

The versioned artifact is `artifacts/pca-db-dump-2026-07-20.json`. It includes dump provenance,
row counts, cohort filters, exact feature order, preprocessing state, PCA state, explained variance,
loadings, basis alignment, target distribution summaries, association tables, and component-bin
summaries. The prior `artifacts/reproducible-results.json` remains unchanged as the June rollback
artifact.

## Commands

```bash
# Regenerate July PCA only (no supervised training)
PYTHONPATH=. .venv-streamlit/bin/python scripts/build_july_pca_artifact.py \
  --july-data-dir /tmp/inferencex-dump-comparison/db-dump-2026-07-20 \
  --june-data-dir inferencex-pca-data \
  --output artifacts/pca-db-dump-2026-07-20.json

# Focused tests
.venv-streamlit/bin/python -m unittest \
  tests.test_pca_target_analysis tests.test_dashboard_ui tests.test_energy_measurements

# Full tests and syntax validation
.venv-streamlit/bin/python -m unittest discover -s tests
.venv-streamlit/bin/python -m compileall apps modeling scripts tests

# Dashboard validation
PYTHONPATH=. .venv-streamlit/bin/streamlit run apps/inferencex_pca_demo.py \
  --server.headless true --server.port 8765
```

## Limitations and remaining decisions

- PCA is descriptive, encoding-sensitive, and not causal.
- The common basis deliberately excludes agentic traces; adding agentic analysis would require a
  workload representation that does not fabricate ISL/OSL.
- Output throughput is strongly skewed; the requested primary overlay remains raw/identity.
- Energy subgroup summaries are suppressed from broad interpretation when support is sparse.
- The historical supervised artifacts are not July models and are never silently applied to July
  observations.
- Energy model training remains out of scope until support broadens and a separate grouped plus
  temporal validation experiment is approved.
