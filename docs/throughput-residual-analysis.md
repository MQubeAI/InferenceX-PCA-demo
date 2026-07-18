# Throughput residual analysis

## Protocol

`scripts/tail_model_diagnostics.py --kind throughput` performs the required
three-fold grouped out-of-fold TabFM evaluation for raw
`metrics_tput_per_gpu`: a deterministic 4,096-row seed-42 sample, grouped by
`config_id`, and full fold-local training context. Predictions and residuals
remain in memory; its output is aggregate-only.

The generated artifact reports fold R2/MAE, residual location/scale/skewness
and percentiles, under/over-prediction rates, absolute-error percentiles,
target-bin and categorical/workload residual scales, Levene target-bin test,
Spearman magnitude correlations, D'Agostino normality diagnostics, histogram
mode proxy, and descriptive Gaussian/empirical uncertainty coverage.

## Established model evidence

The selected full-context throughput result (`artifacts/model-diagnostics-4096.json`)
is R2 **0.961979 +/- 0.008605** and MAE **338.540384** on 4,096 rows with grouped
`config_id` folds. The residual diagnostic confirms strong conditional
heteroskedasticity (Levene p=9.30e-212) and non-Gaussian residuals (D'Agostino
p=0). This supports conditional-scale conformal research, rather than a global
Gaussian residual model.

## Decision gate

The artifact is complete. Its interval evidence remains experimental and not
deployment-calibrated around the final full-context point model. Do not start a
new throughput residual-model run: the research decision is conditional-scale
split conformal only, pending future production calibration work.

```bash
scripts/run_tabfm_mac.sh --threads 4 --log logs/throughput-residual.log -- \
  --max-rows 4096 --folds 3 --seed 42 --targets metrics_tput_per_gpu \
  --models tabfm --tabfm-context-sizes 2730 --resume \
  --output artifacts/throughput-tabfm-checkpoints.json
.venv-tabfm/bin/python scripts/tail_model_diagnostics.py --kind throughput \
  --max-rows 4096 --folds 3 --seed 42 \
  --output artifacts/throughput-residual-diagnostics.json
```

The first command is the resumable, per-fold TabFM run; the second creates the
aggregate residual diagnostic artifact. Do not commit logs or row-level data.
