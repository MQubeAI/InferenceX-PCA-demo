# Median-TPOT leakage-safe tail model

## Definition and split safety

For every grouped held-out fold, candidate tail thresholds are derived only
from that fold's training `metrics_median_tpot` values: p95, p97.5, p99, and
`Q3 + 1.5 * IQR`. Validation labels are then created by applying that fixed
training threshold. The implementation never calculates a validation-derived
threshold and retains all tail rows in primary raw-scale evaluation.

The pre-registered selection rule is to prefer the most prevalent stable
threshold that supplies at least 64 training tail rows in every fold, before
looking at final regressor metrics. This normally selects **p95**; the artifact
records actual per-fold prevalence, threshold standard deviation, and support
before model comparison.

## Models

The experiment compares a global full-context TabFM baseline with leakage-safe
two-stage candidates. Stage 1 trains CatBoost and class-balanced logistic
classifiers from predictor fields only. It reports tail precision, recall,
PR-AUC, and confusion counts. Stage 2 trains an ordinary TabFM model and
compares TabFM, CatBoost, and a tail-training-median fallback for the tail
component. Hard gating and probability-weighted predictions are evaluated on
the original raw target scale. A deterministic median fallback is used when a
fold has fewer than 64 training tail rows.

## Existing baseline and final decision

At 4,096 rows, seed 42, the global TabFM baseline has R2
**0.604782 +/- 0.198737** (folds 0.718840, 0.770212, 0.325294) and MAE
**0.005893**. Earlier seed-123 evidence at 2,048 rows was materially weaker
(R2 0.481594 +/- 0.187365), so a single favorable split must not choose a
segmented model.

The completed seed-42 two-stage experiment did not consistently improve the
global TabFM baseline, including tail-focused error. The median TPOT global model
therefore remains a weaker research baseline only. Do not continue latency
segmentation, a latency residual model, or additional expensive latency runs.
Do not implement CRVAE or VAE.

The prior repeat command is retained only as historical reproducibility context;
it is not a currently requested experiment:

```bash
.venv-tabfm/bin/python scripts/tail_model_diagnostics.py --kind median-tpot \
  --max-rows 4096 --folds 3 --seed 42 \
  --output artifacts/median-tpot-tail-model-4096-seed-42.json
.venv-tabfm/bin/python scripts/tail_model_diagnostics.py --kind median-tpot \
  --max-rows 4096 --folds 3 --seed 123 \
  --output artifacts/median-tpot-tail-model-4096-seed-123.json
```

No segmented latency model is selected. The two-stage result is rejected because
it did not beat the global baseline consistently on aggregate and tail metrics.

## Long-run Mac guidance

Plug in the laptop, disable Low Power Mode, close CPU-heavy applications, keep
adequate free memory, and do not start duplicate TabFM jobs. Use the launcher:

```bash
scripts/run_tabfm_mac.sh --threads 4 --log logs/tabfm-4096-seed42.log -- \
  --max-rows 4096 --folds 3 --seed 42 --targets metrics_median_tpot \
  --models tabfm --tabfm-context-sizes 2730 --resume \
  --output artifacts/tabfm-4096-seed-42-checkpoints.json
```

It prevents sleep with `caffeinate`, uses unbuffered output, runs sequentially,
and forwards all diagnostic arguments. It detects four performance cores on
this host; thread counts are explicit only after a representative one-fold
benchmark proves they improve runtime without materially changing predictions.

TabFM's installed loader accepts `device=`, and PyTorch reports MPS available,
but no bounded end-to-end MPS run has shown that every operation remains native
or faster with reasonable predictions. CPU is therefore the supported path;
the launcher never silently enables or falls back from MPS.
