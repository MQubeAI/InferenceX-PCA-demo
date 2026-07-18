# Median TPOT stability diagnostic

## Scope and safeguards

This is the raw `metrics_median_tpot` TabFM experiment on 2,048 sampled rows,
three grouped `config_id` folds, deterministic random context selection, and the
complete fold-local training context (1,365/1,366 rows). The seed-42 result is
in `artifacts/model-diagnostics-2048.json`; the independent seed-123 repeat is
in `artifacts/model-diagnostics-2048-seed-123.json`. Both artifacts contain only
aggregate fold distributions and category summaries. Out-of-fold predictions
were held in memory only to form the error aggregates; neither predictions nor
row-level residuals are written.

For every fold, the artifacts report the full training and validation target
distributions (count, mean, median, population standard deviation, minimum,
maximum, p1/p5/p25/p75/p95/p99, skewness, and 1.5-IQR outlier count); complete
value-count distributions for hardware, model, framework, precision,
speculative method, disaggregation, multinode, input length, output length, and
concurrency; validation-only and rare validation categories; and row-weighted
validation coverage in training. Error tables include every category rather
than only a top-N list.

## Seed-42 Fold 1 diagnosis

Fold 1 is weak (R2 **0.377794**, MAE **0.007666**) because its validation
target is a much more extreme workload/target regime than either other
validation fold, not because a configuration category is missing from training.

| Target summary | Train (1,365 rows) | Validation (683 rows) |
|---|---:|---:|
| mean / median / std | 0.024639 / 0.015660 / 0.029756 | 0.028790 / 0.017069 / 0.056359 |
| min / max | 0.002006 / 0.285085 | 0.002666 / 1.074429 |
| p1 / p5 / p25 | 0.003395 / 0.004865 / 0.009114 | 0.003482 / 0.005013 / 0.009859 |
| p75 / p95 / p99 | 0.028194 / 0.069652 / 0.184360 | 0.029676 / 0.078291 / 0.210844 |
| skewness / IQR outliers | 4.183 / 101 | 11.422 / 51 |

The validation maximum is 3.77 times the training maximum, and its standard
deviation is 1.89 times the training standard deviation. Folds 2 and 3 validate
only up to 0.285085 and 0.284877, respectively. Thus a very small number of
tail cases has a disproportionate effect on Fold 1 R2.

All validation hardware, model, framework, precision, speculative-method,
disaggregation, multinode, input-length, and output-length values occur in its
training split (100% row-weighted coverage). Their validation distributions are
recorded in the artifact; notable validation counts are `mi355x=159`,
`dsv4=67`, `sglang=117`, `fp8=334`, `none=540`, `False` disaggregated=472,
`False` multinode=483, input 8,192=286, and output 1,024=604. The only rare
framework is `atom-disagg` (two training and two validation rows). Concurrency
coverage is 95.75%: 24 validation-only values account for 29 rows, 0.150 of
5.236 total absolute error, and zero large residuals. Missing/rare coverage is
therefore not the cause of the weak fold.

## Aggregate error attribution

Large residuals mean `abs(actual - prediction) > 1.5 * validation-target IQR`;
positive residuals are underpredictions. Fold 1 has 5.236 total absolute error,
23 large residuals, 3.731 underprediction error, and 1.505 overprediction
error. Its leading marginal subgroups are:

| Dimension | Largest total error | Largest mean error / large-residual signal |
|---|---|---|
| hardware | `mi355x`: 2.463 | `mi355x`: 0.01549 MAE, 11 large |
| model | `dsv4`: 2.250 | `dsv4`: 0.03357 MAE, 9 large |
| framework | `sglang`: 1.928 | `sglang`: 0.01648 MAE; `vllm`: 12 large |
| precision | `fp8`: 3.516 | `int4`: 0.01923 MAE; `fp8`: 11 large |
| concurrency | 64: 1.372 | 4,096: 0.02320 MAE; 256: 9 large |
| input length | 8,192: 3.541 | 8,192: 0.01238 MAE, 15 large |
| output length | 1,024: 4.966 | 1,024: 21 large |

`dsv4` is only 67 validation rows (9.8%) but contributes 43.0% of Fold 1
absolute error, principally through underprediction (1.933 versus 0.317
overprediction error). `mi355x`, 8,192 input, and output 1,024 are broader
overlapping regimes and also carry the largest total error. This is therefore a
tail-heavy, concentrated workload regime—not a blanket failure across every
category, and not an unseen-category problem. The same categories recur in the
other seed, so the conclusion is not an isolated label-coverage artifact.

## Seed comparison

| Seed | R2 mean ± std (range) | MAE mean ± std (range) | Fold R2 |
|---|---:|---:|---:|
| 42 | 0.605943 ± 0.182013 (0.377794–0.823236) | 0.006122 ± 0.001340 (0.004398–0.007666) | 0.377794, 0.616800, 0.823236 |
| 123 | 0.481594 ± 0.187365 (0.330644–0.745667) | 0.007142 ± 0.000221 (0.006834–0.007340) | 0.368472, 0.330644, 0.745667 |

Seed 123 also has one distinctly weak fold, but it is Fold 2 rather than Fold
1. Its validation target has max 1.074429, std 0.060189, skewness 12.229, and
52 IQR outliers; the other seed-123 validations have maxima 0.227252 and
0.425570. Its largest error groups are again `dsv4` (3.179), `mi355x` (2.767),
FP8 (3.462), input 8,192 (3.589), output 1,024 (4.857), and concurrency 64
(1.348). The recurring difficult regime is consequently real, while the exact
fold that receives its tail cases is sample/split dependent.

The earlier 1,024-row, seed-42, full-context result (R2 0.789370 ± 0.046334)
was very likely an easier sample: its validation maxima were only 0.285085,
0.420677, and 0.207412, compared with the 1.074429 tail now present in the
2,048-row experiments. It should not be treated as the more representative
central-latency estimate.

## Recommendation

Raw median TPOT is **not reliable enough yet as a latency central model**. Keep
the 2,048-row, full-fold-context setup as the minimum reproducible diagnostic,
but do not select a context cap by score-chasing: use all available fold-local
training rows. Before any central-model decision, run at least one more
deterministic seed and increase the sample to at least 4,096 rows (or the full
available data if feasible), preserving grouped `config_id` evaluation and
reporting the same tail/regime aggregates.

Another seed is necessary because both completed seeds contain a low-R2 fold
and their aggregate R2 differs by 0.124. Do not begin CRVAE/residual work for
either target yet: latency fails the stability gate, and throughput has not yet
been repeated under this same multi-seed protocol. The appropriate decision is
**neither**, not throughput-only or both, until that validation is complete.
