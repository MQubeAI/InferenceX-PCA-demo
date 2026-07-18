# Central-model selection diagnostic

## Scope

This is a grouped-configuration diagnostic, not a VAE evaluation. All primary
scores hold out complete `config_id` groups. Diagnostic artifacts contain only
aggregate distributions, category summaries, and fold metrics.

## Why the p99 ITL folds differ

At 1,024 sampled rows and three grouped folds, the validation target distribution
is materially different by fold. Fold 2 has mean p99 ITL 1.433, standard
deviation 4.605, and 44 IQR-rule outliers, versus fold 1 mean 0.793, standard
deviation 1.579, and 38 outliers. Its median is only 0.292, so the high MAE is
primarily a long-tail target-scale issue rather than a uniform shift.

Configuration category coverage is otherwise high for hardware, framework,
model, precision, disaggregation, and multinode state. Concurrency is the
exception: validation coverage in training is 95.6% in fold 2 and 94.7% in fold
3, with 12 and 18 unseen concurrency values respectively. This is a plausible
additional source of uneven generalization. The aggregate artifact records the
largest absolute-error category contributions per fold without retaining rows.

## Target and transformation comparison

The table reports three-fold unseen-configuration CatBoost baselines on the same
1,024-row cap. MAE is in each target's native unit.

| Target | raw R2 | log1p R2 | Selection note |
|---|---:|---:|---|
| `metrics_tput_per_gpu` | 0.643 | 0.634 | strongest measured target; raw is preferred |
| `metrics_median_tpot` | 0.588 | 0.589 | stable latency alternative |
| `metrics_mean_e2el` | 0.487 | 0.436 | raw is preferred |
| `metrics_median_itl` | 0.462 | 0.443 | raw is preferred |
| `metrics_mean_itl` | 0.367 | 0.379 | log1p marginally helps R2 |
| `metrics_p99_itl` | 0.145 | 0.229 | tail-heavy and unstable |
| `metrics_median_ttft` | 0.107 | 0.069 | weak |

For p99 ITL, the RF baseline improves from R2 -0.044 raw to 0.328 with log1p;
CatBoost improves from 0.145 to 0.229. Primary MAE/R2 are always calculated after
inverse transformation. No outliers were filtered or winsorized.

## TabFM and context selection

The prior bounded p99 ITL experiment (512 context rows) remains the only completed
three-fold TabFM score: R2 0.294 +/- 0.134 and MAE 0.642 +/- 0.147. It exceeds the
raw p99 CatBoost/RF baselines but is not a sufficient basis for choosing a central
model, because p99 is not the strongest measured target and variation remains
material.

Four deterministic, training-only strategies are implemented: `random`,
`stratified`, `coverage`, and `nearest`. They record context count, groups,
categorical/workload coverage, signature, and runtime. `nearest` uses fold-local
median/scaled numerical distances plus categorical mismatch distance, averaged
over a bounded validation panel; no validation targets are used. The 128/256/512
three-fold rerun was stopped after CPU runtime exceeded the bounded diagnostic
window, so there is **no measured strategy or context-size winner yet**. Do not
infer one from context coverage alone.

## Unseen versus known configurations

The known-configuration diagnostic is explicitly interpolation-only: validation
rows are disjoint but each validation `config_id` remains in training. For raw
p99 ITL, it produced RF R2 -0.060 / MAE 0.564 and CatBoost R2 -0.334 / MAE 0.717.
It does not show an interpolation advantage in this sampled diagnostic; target
tail behavior and workload coverage remain important.

## Recommendation

This historical diagnostic is superseded for final throughput selection by the
completed 4,096-row full-context TabFM experiment: grouped R2 **0.961979 +/-
0.008605** and MAE **338.540384** for raw `metrics_tput_per_gpu`. Select that
throughput point model for research reporting.

The latency-focused median-TPOT two-stage experiment failed to improve the
weaker global latency baseline consistently. Do not pursue latency segmentation
or residual modeling. Do not implement CRVAE/VAE, and do not schedule another
expensive context or latency run at this time.
