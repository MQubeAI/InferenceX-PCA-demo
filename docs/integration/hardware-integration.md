# Epoch hardware integration

## Derived view and validation

`scripts/build_hardware_integration.py` builds the task-specific view
`data/derived/integration/views/inferencex_throughput_epoch_hardware_v1.csv`.
It reuses the existing July aggregate cohort and left-joins one accepted
canonical Epoch hardware record per InferenceX operating point. It checks Epoch
raw SHA-256 identities before and after the pipeline.

| Validation gate | Result |
|---|---:|
| Rows before / after enrichment | 8,239 / 8,239 PASS |
| Unique `config_id` before / after | 1,368 / 1,368 PASS |
| Ordered throughput target values unchanged | PASS |
| Duplicate aggregate row identities | 0 PASS |
| GPU-cluster/datacenter fields in throughput view | 0 PASS |
| Accepted hardware mapping row coverage | 7,168 / 8,239 (87.00%) |

The unresolved H100/H200 rows (1,071) remain in the view with a mapping status
and missing Epoch capability columns. Missing hardware fields are preserved as
missing; no capability is encoded as zero. The detailed PASS/FAIL report,
mapping coverage, target distribution before/after, field missingness, and
temporal status are in
`data/derived/integration/reports/inferencex_throughput_enrichment_validation.json`.

The 10 direct descriptor columns are available in the view. Internode bandwidth
is missing for every accepted mapped record in this snapshot, so it is retained
for provenance/display but excluded from the experiment feature vector rather
than forcing an invalid all-missing imputation. The other partially missing
physical fields use fold-local median plus missingness indicator preprocessing.

## Hardware-representation experiment

This first study intentionally uses the repository’s conventional Random Forest
protocol rather than retraining TabFM: 150 trees, `min_samples_leaf=2`,
fold-local preprocessing, and deterministic three-fold `GroupKFold` by
`config_id`. The common capability-covered, release-date-eligible cohort has
7,168 rows, 1,243 configs, and seven accepted hardware labels. It contains no
target/latency/energy/benchmark-outcome features, no Epoch composite score, and
no UMAP/t-SNE coordinates.

| Representation | Mean R² | Mean MAE | Mean median AE | Median APE |
|---|---:|---:|---:|---:|
| A — existing hardware identity | 0.8376 | 774.01 | 221.86 | 20.26% |
| B — identity + Epoch capabilities | 0.8394 | 760.60 | 213.18 | 19.89% |
| C — Epoch capabilities only | 0.8406 | 759.05 | 211.85 | 19.93% |

Fold-level results and hardware subgroup metrics are retained in
`data/derived/integration/reports/hardware_representation_experiment.json`.
The small mean differences do not establish superiority. They do show that, in
this limited post-hoc cohort and protocol, representation C remains usable when
the accelerator name is removed. The result is not directly comparable to the
historical 4,096-row full-context TabFM result, which used a different snapshot,
sample, point model, and feature/model protocol. No old conformal interval is
attached to any of these RF predictions.

## Controlled reconciliation and validation

The initial table above is preserved as the first integration run, but it is
not used for the scientific conclusion. The controlled study reconciles the
legacy 4,096-row TabFM cohort with the July 20 Epoch study in
`data/derived/integration/reports/cohort_reconciliation/`.

The historical 4,096 was a deterministic `max_rows=4096`, seed-42 sample from
a 7,462-row legacy aggregate. It was not a hardware, model, precision,
single-turn, or target-availability restriction. The current July 20 aggregate
has 8,239 rows; accepted Epoch hardware mappings retain 7,168 rows / 1,243
`config_id`s and explicitly exclude only the 1,071 unresolved H100/H200 rows.

The controlled ordinary-prediction cohort uses the artifact-recorded historical
19-column structural contract, the same 7,168 rows, raw
`metrics_tput_per_gpu`, fixed three-fold grouped `config_id` splits, seed 42,
and fold-local preprocessing for every representation. The only intentional
change is hardware representation:

| Random Forest representation | Pooled R² | Pooled MAE | Median AE | Median APE | p90 APE | Within 20% |
|---|---:|---:|---:|---:|---:|---:|
| A — identity only | 0.8377 | 784.23 | 227.08 | 20.21% | 73.83% | 49.62% |
| B — identity + capabilities | 0.8382 | 767.36 | 216.37 | 19.64% | 70.82% | 50.56% |
| C — capabilities only | 0.8375 | 768.20 | 217.12 | 19.67% | 70.42% | 50.67% |
| D — C + matched precision peak | 0.8349 | 780.86 | 218.81 | 19.89% | 71.90% | 50.18% |

B beats A by MAE in all three folds (−21.70, −24.16, and −4.74
tokens/s/GPU); C also beats A in all three (−23.99, −21.27, and −2.82).
However, paired configuration-level absolute-error wins are approximately
50/50. The ordinary-prediction evidence is therefore a small, consistent
improvement in fold MAE—not material superiority or a significance claim.

The conventional baseline ladder on exactly the same cohort/folds places RF
first (R² 0.8377, MAE 784.23 for A); CatBoost reaches R² 0.7755 / MAE 1,084.17,
nearest non-hardware configuration analogue R² 0.5422 / MAE 1,236.69,
workload-conditioned training median R² 0.2742 / MAE 1,861.57, ridge R²
0.5023 / MAE 2,107.23, and global training median R² −0.1308 / MAE 2,632.98.

Historical full-context TabFM was rerun using its exact helper, legacy CSV
source, 4,096-row seed-42 sample, raw target, and grouped folds. It reproduced
exactly: R² 0.9619788 ± 0.0086047 and MAE 338.5404. This verifies the old
research result but does not make it comparable to the Epoch-resolved RF
cohort.

The exact historical TabFM reproduction is available, but the primary-cohort
A/B/C/D companion did not complete in the available execution budget (it ran
for more than three CPU-saturated hours without a materialized checkpoint).
It is therefore not included in the controlled model ladder. It must be run as
a separately checkpointed job before any TabFM capability-ablation claim is
made; no substitute score is reported.

## Controlled held-out SKU transfer

The transfer model is representation C, trained after removing every row of one
accepted SKU. All preprocessing, workload medians, hardware-analogue choice,
and model fits use only the remaining SKUs. Capability RF beats the global,
workload-conditioned, nearest-hardware-capability analogue, and ridge baselines
in MAE for every resolved SKU. This supports useful transfer information, but
relative errors vary substantially.

| Held-out SKU | Test rows | MAE | MdAPE | p90 APE | Within 20% | R² | Nearest hardware / distance | Outside dimensions |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| B200 | 1,994 | 1,016.31 | 23.21% | 51.55% | 42.08% | 0.759 | GB200 / 0.413 | 3 |
| B300 | 1,237 | 1,386.78 | 26.43% | 49.87% | 37.19% | 0.748 | GB300 / 0.416 | 1 |
| GB200 | 681 | 1,514.69 | 23.60% | 169.48% | 44.93% | 0.750 | B200 / 0.413 | 0 |
| GB300 | 531 | 1,211.18 | 25.03% | 165.65% | 44.63% | 0.818 | B300 / 0.425 | 2 |
| MI300X | 279 | 560.99 | 38.24% | 74.03% | 26.88% | 0.211 | MI325X / 0.416 | 2 |
| MI325X | 386 | 355.00 | 23.02% | 113.86% | 44.30% | 0.715 | MI300X / 0.337 | 0 |
| MI355X | 2,060 | 850.69 | 39.22% | 141.01% | 26.60% | 0.673 | B200 / 0.438 | 3 |

GB200 and MI325X are interpolation-like by the available descriptor ranges;
the other held-out profiles have one or more fields outside the training range.
The seven-SKU Spearman association between capability-space distance and MdAPE
is 0.82, but this is descriptive only and does not define a support threshold.
MI300X and MI355X are the clearest weak-transfer cases by relative error.

Held-out permutation importance for the best conventional capability tree
(RF B) ranks FP8 peak compute highest. INT8 compute, BF16/FP16 peak, memory
bandwidth, FP4 peak, TDP, and memory capacity are smaller predictive signals.
The INT4 missingness indicator has nonzero importance; four distinct
missingness patterns across seven SKUs demonstrate that missingness can be an
identity proxy. These are predictive associations, not causal effects.

The controlled evidence is **Outcome B**: capability enrichment does not
materially improve ordinary interpolation, but capability-only RF preserves
nearly all identity-only accuracy and provides useful, uneven held-out-SKU
transfer relative to simple training-only baselines. It does not support
unreleased-hardware, as-of historical, or calibrated arbitrary what-if
prediction claims.

## Reproduction

```bash
.venv-streamlit/bin/python scripts/build_hardware_integration.py \
  --data-dir .data/inferencex-db-dump-2026-07-20
```

The command writes only under `data/derived/integration/`, apart from normal
Python bytecode cache behavior. It does not modify Epoch raw files or existing
research artifacts.
