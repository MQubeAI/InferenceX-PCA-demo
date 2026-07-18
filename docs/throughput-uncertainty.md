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
