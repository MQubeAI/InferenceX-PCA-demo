# July 2026 InferenceX PCA refresh

Refresh date: 2026-07-22  
Active release: `db-dump/2026-07-20`  
Official source: [SemiAnalysisAI/InferenceX-app release](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20)  
Local verified source: `/tmp/inferencex-dump-comparison/db-dump-2026-07-20/`

## Decision summary

The July 20 snapshot is cumulative and reproduces all six required count gates under the existing
join and median aggregation policy. The main dashboard uses one shared `single_turn` PCA basis fit
on all 8,063 eligible aggregate groups in that snapshot. All 7,462 prior-snapshot groups are
retained alongside 601 new groups. The 81,635 contributing eligible source rows span 2025-09-29
through 2026-07-18 and include 79,975 pre-July plus 1,660 July observations. It is not a July-only
PCA. Median TPOT, throughput, and observed energy are three separate
overlays on the same scores. The 176
`agentic_traces` aggregate rows are excluded from the shared basis because they have no ISL/OSL and
therefore do not share the workload semantics used by the overlays.

No supervised model was trained. Historical TabFM and uncertainty artifacts remain unchanged and
are presented only as June target-selection context, never applied to cumulative-snapshot rows.

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

## Verified outcome overlays

The primary latency-focused PCA overlay is:

- raw target: `metrics_median_tpot`
- definition: median time per output token
- transformation: identity for all primary analysis
- units: seconds/output token
- direction: lower is better
- missing values: excluded, never imputed
- valid values: 8,063 aggregate rows across 1,354 configurations; no zero or negative values

An optional `log1p` color scale is available because the distribution is strongly right-skewed.
It is display-only; the stored target, associations, component bins, and user-facing statistics
remain in raw seconds/output token. Median TPOT is the main latency outcome studied in PCA, not the
final selected supervised target.

Throughput remains a separate overlay because it was the final selected supervised target:

- raw target: `metrics_tput_per_gpu`
- transformation: identity
- inverse transformation: identity
- units: tokens/second/GPU
- direction: higher is better
- final selection context: 4,096 rows; three grouped `config_id` folds; TabFM R²
  `0.961979 +/- 0.008605`; MAE `338.540384` tokens/s/GPU

The supervised metrics above come from the historical grouped TabFM experiment, not from PCA.

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

The shared fit contains all 8,063 eligible `single_turn` aggregate rows in the cumulative snapshot.
Contributing source observations span 2025-09-29 through 2026-07-18; the aggregate representative
date field spans through 2026-07-17 because non-metric metadata uses the existing first-non-null
aggregation rule. All three target overlays use this identical feature order, preprocessing state,
basis, and score matrix.

## Explained variance and basis stability

| Component | June variance | Updated snapshot variance | Loading cosine | Sign-aligned correlation |
|---|---:|---:|---:|---:|
| PC1 | 28.25% | 28.11% | 0.9995 | 0.9993 |
| PC2 | 13.51% | 13.47% | 0.9991 | 0.9991 |
| PC3 | 8.44% | 8.14% | 0.9916 | 0.9916 |
| PC4 | 7.72% | 7.71% | 0.9867 | 0.9867 |
| PC5 | 6.87% | 6.93% | 0.9898 | 0.9899 |

PC1–PC5 cumulative variance declines modestly from 64.78% to 64.37%. The five-dimensional
principal angles are 4.23°, 2.53°, 1.95°, 1.16°, and 0.81°. This is strong evidence that the first
five structural directions remain stable enough for a shared-basis refresh.

The updated full-dataset basis needs 4, 7, 10, 15, and 21 components to reach 50%, 70%, 80%, 90%, and 95% cumulative
variance, respectively.

The largest updated-basis PC1 loading magnitudes are disaggregation, multinode topology, prefill DP
attention, framework, decode EP, and prefill worker count. PC4 is primarily an opposing ISL/OSL
workload axis. PC5 combines decode workers and prefill EP against concurrency, ISL, and prefill
workers. Component signs are arbitrary; interpretation uses the stored sign alignment.

## Analysis A: median TPOT latency

- usable aggregate rows: 8,063
- unique configurations: 1,354
- dates: 2025-09-29 through 2026-07-17
- workload support: `single_turn`; ISL 1024 or 8192; OSL 1024 or 8192
- raw median: 0.015831 seconds/output token
- interquartile range: 0.009240–0.028479
- 90th / 95th / 99th percentiles: 0.050433 / 0.074486 / 0.176121
- minimum / maximum: 0.001868 / 1.459967

Among PC1–PC5, PC5 has the largest absolute rank association with raw median TPOT (Pearson −0.120;
Spearman −0.210), closely followed by PC3 on rank correlation (−0.203). These associations are
modest. PC5 combines positive decode-worker, prefill-EP, OSL, and prefill-GPU loadings with negative
concurrency, ISL, prefill-worker, and decode-TP loadings. Median TPOT varies from 0.022884 seconds
per output token in the lowest PC5 quintile to 0.013784 in the highest quintile. This is descriptive
alignment within the cumulative snapshot, not a predictive or causal result.

## Analysis B: throughput per GPU

- usable aggregate rows: 8,063
- unique configurations: 1,354
- workload support: `single_turn`; ISL 1024 or 8192; OSL 1024 or 8192
- raw median: 1,049.81 tokens/second/GPU
- interquartile range: 328.92–2,916.50
- 95th percentile: 10,305.65
- maximum: 63,535.87

Among PC1–PC5, PC5 has the largest absolute associations with raw throughput (Pearson −0.345;
Spearman −0.408). Median throughput falls from 2,623.18 tokens/s/GPU in the lowest PC5 quintile to
515.40 in the highest quintile. This is an alignment within the observed cumulative-snapshot benchmark mix, not a
causal effect and not a new predictive validation result.

## Analysis C: observed energy

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

- shared-basis variance and June/updated-snapshot stability diagnostics;
- Median TPOT with raw latency definition, optional display-only log color, direction,
  distribution, associations, component bins, positive/negative loadings, and interactive views;
- Throughput per GPU with raw throughput definition, supervised-target context, direction, associations,
  component bins, loadings, and interactive PC1/PC2 and PC1/PC3 projections;
- Joules per output token with measured-only support, distribution, associations, component bins, sparse-group
  warnings, date-colored projection, and observed subgroup summaries.

The observed Energy Measurements explorer continues to return exact measurements or comparison
rows only and now derives all July support counts dynamically. No energy estimator is loaded.

The versioned artifact is `artifacts/pca-db-dump-2026-07-20.json`. It includes dump provenance,
row counts, cumulative-snapshot scope and date counts, cohort filters, exact feature order,
preprocessing state, PCA state, explained variance, loadings, basis alignment, all three target
distribution summaries, association tables, and component-bin
summaries. The prior `artifacts/reproducible-results.json` remains unchanged as the June rollback
artifact.

The shared preprocessing and PCA state is byte-structurally identical to the prior artifact:
`basis_sha256 = 03ac94b1463c26567d0bfa55448b610faaa24735469a06eeac5cbb50ee943898`.
Only cumulative-scope metadata and the median-TPOT overlay were added; the basis, explained
variance, and loading tables were not changed.

## Commands

```bash
# Regenerate the cumulative July 20 snapshot PCA only (no supervised training)
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
- The historical supervised artifacts were not retrained on the cumulative July 20 snapshot and
  are never silently applied to its observations.
- Energy model training remains out of scope until support broadens and a separate grouped plus
  temporal validation experiment is approved.
