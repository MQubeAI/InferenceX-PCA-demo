# Epoch × InferenceX hardware validation summary

## Answer in one sentence

The evidence supports **Outcome B**: Epoch's physical capability descriptors
do not materially improve ordinary, in-distribution throughput prediction over
hardware identity, but a capability-only Random Forest preserves essentially
the same accuracy and beats training-only transfer baselines for every tested
held-out SKU; the transfer quality is uneven and is weakest for the less
supported AMD profiles.

## 1. Why 4,096 differs from 8,239 / 7,168

These are not competing definitions of one cohort.

| Stage | Historical selected TabFM | July 20 integration |
|---|---:|---:|
| Joined benchmark rows | 79,830 / 1,197 configs | 81,851 / 1,368 configs |
| Median aggregate | 7,462 / 1,197 configs | 8,239 / 1,368 configs |
| Non-null throughput target | 7,462 / 1,197 configs | 8,239 / 1,368 configs |
| Single-turn/workload/model/precision filter | none | none |
| Deterministic `max_rows=4096`, seed 42 sample | 4,096 / 993 configs | not applied |
| Accepted Epoch mapping | not applied | 7,168 / 1,243 configs |

The first 777-row difference is snapshot growth: legacy
`inferencex-pca-data/` versus the July 20 frozen dump. The historical 4,096
was a computational sample from the legacy aggregate, **not** a scientific
eligibility filter. The remaining 1,071 July rows are H100/H200 and remain in
the derived view but are excluded only from capability experiments because the
available sources do not prove their exact variants. The initial integration
RF also used a slightly different structural feature contract; the controlled
benchmark restores the 19 columns recorded with the historical TabFM artifact.

See [cohort reconciliation](cohort_reconciliation/reconciliation.md) and its
machine-readable companion `cohort_reconciliation/cohort_reconciliation.json`.

## 2. Historical result reproduction

The exact full-context historical procedure was rerun in the dedicated TabFM
environment: legacy source, 4,096-row seed-42 sample, raw
`metrics_tput_per_gpu`, recorded 19 features, and three grouped `config_id`
folds. It reproduces the stored result exactly:

| Rows / configs | Mean R² (fold SD) | Mean MAE |
|---|---:|---:|
| 4,096 / 993 | 0.961979 (0.008605) | 338.540 tokens/s/GPU |

Fold R² values are 0.972651, 0.951579, and 0.961707; fold MAEs are 301.225,
373.555, and 340.840. The artifact records TabFM 1.0.1, the command contract,
software versions, and source provenance in
`validation/historical_tabfm_reproduction.json`. This validates the historical
result; it does not make it comparable to the newer, mapping-restricted cohort.

## 3. Primary apples-to-apples ordinary-prediction cohort

All primary comparisons use the same 7,168 July 20 rows / 1,243 configs, raw
target, historical 19-column structural contract, deterministic three-fold
grouped `config_id` splits, seed 42, and fold-local preprocessing. Numeric
imputation and missingness indicators are fit on each training fold only.
`epoch_internode_bandwidth_bytes_per_second` is all missing and is not modeled;
no missing physical measurement is converted to zero. A/B/C/D are the only
intentional feature difference.

| RF representation | Pooled R² | MAE | Median AE | MdAPE | p90 APE | Within ±20% |
|---|---:|---:|---:|---:|---:|---:|
| A — hardware identity | 0.8377 | 784.23 | 227.08 | 20.21% | 73.83% | 49.62% |
| B — identity + capabilities | 0.8382 | 767.36 | 216.37 | 19.64% | 70.82% | 50.56% |
| C — capabilities only | 0.8375 | 768.20 | 217.12 | 19.67% | 70.42% | 50.67% |
| D — C + matched-precision peak | 0.8349 | 780.86 | 218.81 | 19.89% | 71.90% | 50.18% |

The matched feature is added alongside, not in place of, raw capabilities. It
uses BF16/FP16, FP8, FP4, INT8, or INT4 peak only when the InferenceX precision
label maps semantically; unrecognized labels stay missing.

## 4. Fixed-cohort baseline ladder

The A baseline ladder below uses the exact same rows and folds. Non-hardware
baselines are invariant across A/B/C/D. CatBoost was available. TabFM exactly
reproduced the historical experiment, but its 7,168-row A/B/C/D companion was
stopped after more than three CPU-saturated hours without a checkpointed
artifact; it is reported as not completed, not replaced by another score.

| Model | Pooled R² | MAE | MdAPE | p90 APE | Within ±20% |
|---|---:|---:|---:|---:|---:|
| Global training median | -0.1308 | 2,632.98 | 81.28% | 781.94% | 9.92% |
| Workload-conditioned training median | 0.2742 | 1,861.57 | 52.71% | 209.04% | 20.38% |
| Nearest measured configuration analogue | 0.5422 | 1,236.69 | 30.68% | 98.43% | 36.66% |
| Ridge | 0.5023 | 2,107.23 | 100.22% | 968.92% | 14.86% |
| CatBoost (A) | 0.7755 | 1,084.17 | 37.89% | 192.26% | 28.21% |
| Random Forest (A) | 0.8377 | 784.23 | 20.21% | 73.83% | 49.62% |

Raw percentage errors have a low-throughput tail; the detailed artifact also
reports median absolute log ratio as a stable scale-relative companion.

## 5. Is the small RF difference real?

Against A, B improves fold MAE by 21.70, 24.16, and 4.74 tokens/s/GPU, and C
by 23.99, 21.27, and 2.82, respectively: each wins all three fold aggregates.
At the more granular grouped-config level, however, B wins only 49.96% of
configs and C 49.72%; median error deltas are slightly adverse. This is a
small, consistent fold-MAE improvement, not a material difference and not a
statistical-significance claim. D wins only two folds and is worse overall.

## 6. Held-out hardware SKU transfer

For each row below, every observation and config for the named SKU was removed
before all preprocessing, baseline construction, capability-analogue selection,
and fitting. The capability RF (C) beats global, workload median, nearest
hardware-capability analogue, and ridge MAE for each SKU. R² is included only
as a descriptive within-SKU metric.

| Held-out SKU | Train / test rows | Test configs | RF MAE | MdAPE | p90 APE | Within ±20% | R² | Nearest physical SKU / distance | Outside dims |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| B200 | 5,174 / 1,994 | 284 | 1,016.31 | 23.21% | 51.55% | 42.08% | 0.759 | GB200 / 0.413 | 3 |
| B300 | 5,931 / 1,237 | 211 | 1,386.78 | 26.43% | 49.87% | 37.19% | 0.748 | GB300 / 0.416 | 1 |
| GB200 | 6,487 / 681 | 249 | 1,514.69 | 23.60% | 169.48% | 44.93% | 0.750 | B200 / 0.413 | 0 |
| GB300 | 6,637 / 531 | 254 | 1,211.18 | 25.03% | 165.65% | 44.63% | 0.818 | B300 / 0.425 | 2 |
| MI300X | 6,889 / 279 | 20 | 560.99 | 38.24% | 74.03% | 26.88% | 0.211 | MI325X / 0.416 | 2 |
| MI325X | 6,782 / 386 | 34 | 355.00 | 23.02% | 113.86% | 44.30% | 0.715 | MI300X / 0.337 | 0 |
| MI355X | 5,108 / 2,060 | 191 | 850.69 | 39.22% | 141.01% | 26.60% | 0.673 | B200 / 0.438 | 3 |

GB200 and MI325X are interpolation-like by descriptor range. B300 has one
out-of-range descriptor; B200, GB300, MI300X, and MI355X have two or three.
Across only seven SKUs, capability distance has a descriptive Spearman 0.82
association with MdAPE (not a threshold or inferential test). MI300X and
MI355X are the clearest weak-transfer cases by relative accuracy; MI300X also
has only 20 test configs. H100/H200 are not transfer-tested because their
variant mappings remain unresolved.

## 7. What descriptors appear useful?

Held-out permutation importance for RF B ranks FP8 peak compute highest.
INT8 compute, BF16/FP16 peak, memory bandwidth, FP4 peak, TDP, and memory
capacity carry smaller positive predictive signal. The INT4-missingness
indicator is also important: four missingness patterns across seven SKU
profiles show that missingness can partially encode vendor/generation identity.
These are predictive associations measured on held-out folds, not causal
claims about hardware performance.

## 8. Scope, product boundary, and next step

The existing prototype is intentionally unchanged. It continues to make exact
InferenceX matches **Observed** and all non-exact what-ifs **Unsupported**;
no aggregate TabFM score or uneven transfer study is treated as an arbitrary
prediction service. The architecture schematic in
`docs/integration/prototype-plan.md` shows Epoch as the implemented hardware
entity path and Artificial Analysis only as the next, dashed model-side source.

This study does **not** support claims of unreleased-hardware prediction,
as-of historical prediction from the newer Epoch snapshot, calibrated what-if
intervals, causal importance, or transfer to the unresolved H100/H200 variants.
The recommended next step is to preserve this controlled hardware benchmark,
resolve H100/H200 only with source evidence, and then add a separate,
source-specific Artificial Analysis ingestion and model/provider observation
layer—without folding its speed or TTFT metrics into the InferenceX target.

## Evidence artifacts

- `validation/controlled_hardware_validation.json` — fixed-fold model ladder,
  A/B/C/D metrics, paired errors, cohort metadata, and protocol.
- `validation/held_out_hardware_transfer.json` — per-SKU models, training-only
  baselines, target distributions, and support diagnostics.
- `validation/feature_importance.json` — held-out permutation importance.
- `validation/historical_tabfm_reproduction.json` — exact legacy reproduction.
- `cohort_reconciliation/` — row-flow report and machine-readable audit.
