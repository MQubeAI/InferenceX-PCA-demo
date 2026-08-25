# Throughput cohort reconciliation

## Finding

The historical selected TabFM result and the first Epoch Random Forest result
were not comparable. The 4,096 rows were a deterministic computational sample
from a legacy aggregate, not an analytical filter. The Epoch study used a newer
July 20 snapshot and then applied a separate accepted-hardware-mapping filter.

| Stage | Historical TabFM | New integration |
|---|---:|---:|
| Raw joined benchmark rows | 79,830 / 1,197 configs | 81,851 / 1,368 configs |
| Median aggregate (`config_id`, benchmark type, ISL, OSL, concurrency) | 7,462 / 1,197 | 8,239 / 1,368 |
| Non-null `metrics_tput_per_gpu` | 7,462 / 1,197 | 8,239 / 1,368 |
| Workload filter | none | none |
| Deterministic `max_rows` sample | 4,096 / 993 | none |
| Accepted Epoch mapping | not applied | 7,168 / 1,243 |
| Final initial experiment rows | 4,096 / 993 | 7,168 / 1,243 |

The old source is the legacy `inferencex-pca-data/benchmark_results.csv` plus
`configs.csv` pair (recorded historical fingerprint
`1a199861…` under the then-current manifest convention). The newer source is
the frozen raw `db-dump/2026-07-20` release, whose metrics JSON is flattened
before the identical many-to-one config join and median aggregation.

## Throughput target distribution

`metrics_tput_per_gpu` is the raw median-aggregate per-GPU throughput target;
it is not log transformed for the recorded TabFM or controlled benchmark. The
mapping restriction changes the target mix, which is a further reason the
historical and Epoch-resolved scores are not directly comparable.

| Frame | Rows | Min | Median | Mean | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy aggregate | 7,462 | 0.17 | 1,005.54 | 2,518.10 | 10,326.57 | 18,946.38 | 63,535.87 |
| Historical sampled input | 4,096 | 2.66 | 1,047.91 | 2,611.91 | 10,644.71 | 19,487.59 | 53,262.60 |
| July 20 aggregate | 8,239 | 0.17 | 1,081.96 | 2,833.79 | 11,560.62 | 22,748.47 | 63,535.87 |
| July 20 Epoch-resolved primary cohort | 7,168 | 0.17 | 1,220.47 | 3,092.99 | 12,321.61 | 24,965.83 | 63,535.87 |

## Exact causes

1. The legacy aggregate has 7,462 rows; the July 20 aggregate has 8,239. The
   777-row difference is source-snapshot growth before any Epoch enrichment.
2. Historical TabFM called `prepare_model_frame(..., max_rows=4096, seed=42)`.
   It samples first, then excludes missing targets; all 4,096 sampled targets
   were present. It did not filter to `single_turn`, hardware, model, precision,
   or a configuration subset.
3. The new 7,168-row number is the 8,239-row July aggregate after restricting
   to accepted B200/B300/GB200/GB300/MI300X/MI325X/MI355X mappings. The 1,071
   H100/H200 rows remain in the integration view but are excluded from
   capability experiments because variant resolution is unresolved.
4. The initial integration RF feature contract also drifted from the historical
   TabFM contract. Historical TabFM used the two DP-attention fields and did
   not use `config_num_decode_gpu`; the first RF integration did the reverse.

## Controlled replacement protocol

The controlled Epoch-resolved study fixes the July 20 accepted-map cohort at
7,168 rows / 1,243 configs, the raw target, the artifact-recorded 19
non-hardware/structural fields, three deterministic grouped `config_id` folds,
seed 42, and all conventional-model hyperparameters. A/B/C/D differ only in
the allowed hardware representation. Fold preprocessing is training-only:
numeric training median plus missingness indicator, categorical `__MISSING__`.

The full machine-readable audit is
`cohort_reconciliation.json`; the controlled benchmark is in
`../validation/controlled_hardware_validation.json`.
