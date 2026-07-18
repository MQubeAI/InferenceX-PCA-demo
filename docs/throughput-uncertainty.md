# Throughput uncertainty diagnostics

This experiment evaluates prediction intervals for `metrics_tput_per_gpu` on
unseen `config_id` groups. It is deliberately separate from the median-TPOT
work; it does not fit a TPOT tail model, VAE, or CRVAE.

Run it on a Mac with the cached TabFM environment:

```bash
cd /Users/vivaanbhargava/Documents/InferenceX-PCA-demo-clean
scripts/run_tabfm_mac.sh --threads 8 --script throughput_uncertainty_diagnostics.py -- \
  --data-dir inferencex-pca-data --max-rows 4096 --seed 42 --folds 3 \
  --output artifacts/throughput-uncertainty-4096-seed-42.json
```

The artifact contains aggregate metrics only. It never writes source rows,
point predictions, residuals, or interval endpoints.

## Split and leakage policy

Each deterministic outer `GroupKFold` validation fold holds out complete
`config_id` groups. Its remaining configurations are deterministically divided
into three mutually exclusive roles (approximately 50% / 25% / 25% by group):

- TabFM context: the only rows whose targets are supplied to TabFM.
- uncertainty-model training: residuals train CatBoost scale and quantile
  models.
- conformal calibration: residuals calibrate all interval methods.

The outer-validation target is never used when fitting a model or choosing a
conformal quantile. Throughput-bin edges used for reporting are derived from
the outer-training target distribution, not validation targets.

## One TabFM call per fold

The installed TabFM implementation makes this safe and inexpensive. Its
`fit()` fixes feature transforms and target scaling from the context rows;
`predict()` transforms query features and appends them after the context.
The PyTorch predictor's context-attention mask exposes only the labeled context
rows to the induced context representation. Query rows are unlabeled and do
not become context. Therefore the script predicts the concatenated
uncertainty-training, calibration, and validation feature rows once, then
slices the resulting predictions by partition.

With the default three outer folds, the expected TabFM invocation count is
exactly **3**.

## Interval methods and reports

At nominal 50%, 80%, and 95%, the artifact evaluates:

1. Global split-conformal intervals using calibration absolute residuals.
2. CatBoost conditional scale intervals: CatBoost predicts
   `log1p(abs(residual))` from configuration/workload predictors; normalized
   residuals are conformally calibrated on the separate calibration split.
3. CatBoost lower/upper residual-quantile models with quantile loss, followed
   by conformalized quantile regression calibration.

For every method and nominal level, it reports empirical coverage, mean and
median width, interval score, coverage and width by outer-training throughput
quintile, coverage by hardware/model/framework/precision/concurrency/input
length/output length, and the subgroup with the largest undercoverage. It also
reports point-model R2/MAE, fold runtimes, partition counts, and TabFM call
counts.

All subgroup rows remain in the artifact. The reporting-only
`--subgroup-minimum-support` argument defaults to **20**; groups below that
support are excluded only from the headline worst-undercoverage ranking. The
artifact records the applied support and the number of smaller groups excluded,
so one-row concurrency categories cannot become the headline failure.

## Final interpretation

Conditional-scale split conformal is the selected **research** uncertainty
method. At 95%, it achieved 95.34% coverage, 2485.442 average width, and a
4739.169 interval score; global conformal scored 8235.473 and was 34.62% wider.
Conditional quantiles narrowly won at 50%, but conditional scale is preferred as
one method across coverage levels.

This is not production calibration. The uncertainty evaluation's point model had
R2 0.913897 because approximately half of each outer-training fold was available
as TabFM context. The selected full-context point-model result is R2 0.961979 +/-
0.008605 (MAE 338.540384); do not merge or compare the two as equivalent runs.
