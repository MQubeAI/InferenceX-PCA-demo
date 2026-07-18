"""Leakage-safe aggregate diagnostics for residuals and latency tail models.

All row-level targets, predictions, residuals, and tail assignments stay local to
the running process.  Returned values are grouped summaries only.
"""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling.comparison import FoldPreprocessor, _catboost_fit_predict, _tabfm_fit_predict, deterministic_grouped_folds, prepare_model_frame
from modeling.diagnostics import select_context

REGIME_COLUMNS = ("config_hardware", "config_model", "config_framework", "config_precision", "conc", "isl", "osl")


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def _percentiles(values: np.ndarray, points: tuple[float, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[str, float]:
    values = _finite(values)
    return {f"p{point:g}": float(np.percentile(values, point)) for point in points} if len(values) else {}


def tail_threshold(y_train: pd.Series, method: str) -> float:
    """A deterministic threshold based exclusively on a fold's training labels."""
    values = pd.to_numeric(y_train, errors="coerce").dropna()
    if method.startswith("p"):
        return float(values.quantile(float(method[1:]) / 100))
    if method == "iqr":
        q1, q3 = values.quantile([.25, .75])
        return float(q3 + 1.5 * (q3 - q1))
    raise ValueError(f"Unknown tail threshold method: {method}")


def tail_labels(y: pd.Series, threshold: float) -> np.ndarray:
    return (pd.to_numeric(y, errors="coerce").to_numpy(dtype=float) > threshold).astype(np.int8)


def combine_predictions(ordinary: np.ndarray, tail: np.ndarray, tail_probability: np.ndarray, hard: bool) -> np.ndarray:
    """Deterministic two-stage gate; probabilities come from training-only fit."""
    ordinary, tail, probability = map(lambda x: np.asarray(x, dtype=float), (ordinary, tail, tail_probability))
    return np.where(probability >= .5, tail, ordinary) if hard else probability * tail + (1 - probability) * ordinary


def conservative_tail_prediction(y_train: pd.Series, tail_indices: np.ndarray, rows: int, minimum_support: int = 64) -> tuple[np.ndarray, str]:
    """Predict a training-only median whenever a tail component is unsupported."""
    values = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=float)
    if len(tail_indices) < minimum_support:
        return np.full(rows, float(np.median(values))), "train_median_fallback"
    return np.full(rows, float(np.median(values[tail_indices]))), "tail_training_median"


def _group_scale(frame: pd.DataFrame, residual: np.ndarray, columns: tuple[str, ...] = REGIME_COLUMNS) -> dict[str, list[dict[str, Any]]]:
    absolute = np.abs(residual)
    out: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].astype("string").fillna("__MISSING__").astype(str)
        grouped = pd.DataFrame({"value": values, "absolute_error": absolute, "residual": residual}).groupby("value", sort=True)
        records = grouped.agg(rows=("absolute_error", "size"), mae=("absolute_error", "mean"), residual_std=("residual", lambda x: float(np.std(x, ddof=0)))).reset_index()
        out[column] = [{"value": str(row.value), "rows": int(row.rows), "mae": float(row.mae), "residual_std": float(row.residual_std)} for row in records.sort_values(["mae", "value"], ascending=[False, True], kind="stable").itertuples(index=False)]
    return out


def _residual_summary(frame: pd.DataFrame, observed: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = observed - prediction
    absolute = np.abs(residual)
    bins = pd.qcut(pd.Series(observed), q=min(5, len(np.unique(observed))), duplicates="drop")
    scale_by_target = pd.DataFrame({"bin": bins.astype(str), "absolute_error": absolute, "residual": residual}).groupby("bin", sort=True).agg(rows=("absolute_error", "size"), mae=("absolute_error", "mean"), residual_std=("residual", lambda x: float(np.std(x, ddof=0)))).reset_index()
    # These tests describe the OOF residual shape; they are not a claim that an
    # independent conformal interval has been calibrated.
    normality = stats.normaltest(residual) if len(residual) >= 8 else (np.nan, np.nan)
    histogram, _ = np.histogram(residual, bins=min(20, max(5, int(np.sqrt(len(residual))))))
    peaks = int(sum(histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1] for i in range(1, len(histogram) - 1)))
    groups = [group["residual"].to_numpy() for _, group in pd.DataFrame({"bin": bins.astype(str), "residual": residual}).groupby("bin")]
    levene_p = float(stats.levene(*groups).pvalue) if len(groups) >= 2 and all(len(x) > 1 for x in groups) else np.nan
    gaussian = {str(level): float(np.mean(np.abs(residual) <= stats.norm.ppf((1 + level) / 2) * np.std(residual, ddof=0))) for level in (.5, .8, .95)}
    empirical = {str(level): float(np.mean(np.abs(residual) <= np.quantile(np.abs(residual), level))) for level in (.5, .8, .95)}
    return {
        "residual_definition": "observed_minus_predicted",
        "rows": int(len(residual)), "mean": float(np.mean(residual)), "median": float(np.median(residual)), "std": float(np.std(residual, ddof=0)), "skewness": float(stats.skew(residual, bias=False)),
        "percentiles": _percentiles(residual), "absolute_error_percentiles": _percentiles(absolute),
        "underprediction_rate": float(np.mean(residual > 0)), "overprediction_rate": float(np.mean(residual < 0)),
        "correlation_absolute_residual_predicted": float(stats.spearmanr(absolute, prediction).statistic),
        "correlation_absolute_residual_target": float(stats.spearmanr(absolute, observed).statistic),
        "heteroskedasticity": {"target_bin_levene_pvalue": levene_p, "conditional_on_target": bool(np.isfinite(levene_p) and levene_p < .05)},
        "normality": {"dagostino_k2": float(normality.statistic), "pvalue": float(normality.pvalue), "histogram_local_peak_count": peaks},
        "target_value_bins": [{"bin": str(row.bin), "rows": int(row.rows), "mae": float(row.mae), "residual_std": float(row.residual_std)} for row in scale_by_target.itertuples(index=False)],
        "residual_scale_by_feature": _group_scale(frame, residual),
        "uncertainty_baselines": {"note": "In-sample OOF descriptive coverage; calibrate on a separate split before deployment.", "gaussian_nominal_to_coverage": gaussian, "empirical_absolute_residual_nominal_to_coverage": empirical},
    }


def tabfm_oof_diagnostics(frame: pd.DataFrame, features: list[str], target: str, max_rows: int, seed: int, n_splits: int = 3, context_cap: int | None = None, context_strategy: str = "random") -> dict[str, Any]:
    work, preparation = prepare_model_frame(frame, features, target, max_rows, seed)
    folds = deterministic_grouped_folds(work, n_splits)
    observed = np.full(len(work), np.nan); predicted = np.full(len(work), np.nan)
    fold_rows = []
    for number, (train_i, valid_i) in enumerate(folds, 1):
        train_raw, valid_raw = work.iloc[train_i], work.iloc[valid_i]
        processor = FoldPreprocessor.fit(train_raw, features)
        context_raw, context_meta = select_context(train_raw, valid_raw, features, context_cap or len(train_raw), context_strategy, seed + number)
        prediction, _, extra = _tabfm_fit_predict(processor.transform(context_raw), processor.transform(valid_raw), context_raw[target], seed + number, context_cap, context_meta)
        truth = valid_raw[target].to_numpy(dtype=float)
        observed[valid_i], predicted[valid_i] = truth, prediction
        fold_rows.append({"fold": number, "train_rows": int(len(train_raw)), "validation_rows": int(len(valid_raw)), "group_overlap": int(bool(set(train_raw.config_id) & set(valid_raw.config_id))), "r2": float(r2_score(truth, prediction)), "mae": float(mean_absolute_error(truth, prediction)), **extra})
        del processor, context_raw, train_raw, valid_raw, prediction
        gc.collect()
    summary = _residual_summary(work, observed, predicted)
    return {"aggregate_only": True, "target": target, "seed": seed, "preparation": preparation, "folds": fold_rows, "metrics": {"r2_mean": float(np.mean([x["r2"] for x in fold_rows])), "r2_std": float(np.std([x["r2"] for x in fold_rows])), "mae_mean": float(np.mean([x["mae"] for x in fold_rows]))}, "residual_diagnostics": summary}


def threshold_comparison(frame: pd.DataFrame, features: list[str], target: str, max_rows: int, seed: int, global_diagnostics: dict[str, Any], n_splits: int = 3) -> dict[str, Any]:
    """Compare definitions using train-only labels; residual capture is OOF-only."""
    work, _ = prepare_model_frame(frame, features, target, max_rows, seed)
    folds = deterministic_grouped_folds(work, n_splits)
    # Reconstruct only aggregate large-error cutoffs from a concurrent OOF run
    # is intentionally avoided: callers pass aggregate diagnostics, and the
    # threshold selection relies on prevalence/stability rather than score.
    methods = ("p95", "p97.5", "p99", "iqr")
    records: dict[str, Any] = {}
    for method in methods:
        per_fold = []
        for number, (train_i, valid_i) in enumerate(folds, 1):
            threshold = tail_threshold(work.iloc[train_i][target], method)
            train_label, valid_label = tail_labels(work.iloc[train_i][target], threshold), tail_labels(work.iloc[valid_i][target], threshold)
            per_fold.append({"fold": number, "threshold": threshold, "train_tail_count": int(train_label.sum()), "train_tail_prevalence": float(train_label.mean()), "validation_tail_count": int(valid_label.sum()), "validation_tail_prevalence": float(valid_label.mean())})
        prevalences = [x["validation_tail_prevalence"] for x in per_fold]
        counts = [x["train_tail_count"] for x in per_fold]
        records[method] = {"folds": per_fold, "threshold_std": float(np.std([x["threshold"] for x in per_fold])), "validation_prevalence_std": float(np.std(prevalences)), "minimum_training_tail_count": int(min(counts)), "large_enough_to_model": bool(min(counts) >= 64)}
    return {"aggregate_only": True, "selection_rule": "Prefer the most prevalent stable definition with >=64 tail training rows; do not select on validation R2.", "candidate_thresholds": records, "global_residual_reference": {"absolute_error_percentiles": global_diagnostics["residual_diagnostics"]["absolute_error_percentiles"]}}


def _classifier(train: pd.DataFrame, valid: pd.DataFrame, y: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if len(np.unique(y)) < 2: return np.full(len(valid), float(y[0]) if len(y) else 0.0)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        cats = [c for c in train if not pd.api.types.is_numeric_dtype(train[c])]
        model = CatBoostClassifier(iterations=150, depth=5, learning_rate=.05, random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
        model.fit(train, y, cat_features=cats)
        return model.predict_proba(valid)[:, 1]
    cats = [c for c in train if not pd.api.types.is_numeric_dtype(train[c])]; nums = [c for c in train if c not in cats]
    model = Pipeline([("prep", ColumnTransformer([("num", "passthrough", nums), ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed))])
    return model.fit(train, y).predict_proba(valid)[:, 1]


def two_stage_latency(frame: pd.DataFrame, features: list[str], target: str, max_rows: int, seed: int, threshold_method: str = "p95", n_splits: int = 3, context_cap: int | None = None) -> dict[str, Any]:
    """Classifier + ordinary/tail regressors, evaluated on the untouched raw target."""
    work, preparation = prepare_model_frame(frame, features, target, max_rows, seed); folds = deterministic_grouped_folds(work, n_splits)
    totals: dict[str, list[float]] = {"global_tabfm": [], "hard_catboost_tail_tabfm": [], "hard_catboost_tail_catboost": [], "hard_catboost_tail_fallback": [], "weighted_catboost_tail_catboost": [], "hard_logistic_tail_catboost": []}
    fold_rows = []; classifier_rows = []
    for number, (train_i, valid_i) in enumerate(folds, 1):
        train_raw, valid_raw = work.iloc[train_i], work.iloc[valid_i]; processor = FoldPreprocessor.fit(train_raw, features)
        train, valid = processor.transform(train_raw), processor.transform(valid_raw); threshold = tail_threshold(train_raw[target], threshold_method); y_tail = tail_labels(train_raw[target], threshold); true_tail = tail_labels(valid_raw[target], threshold)
        scores = {kind: _classifier(train, valid, y_tail, kind, seed + number) for kind in ("catboost", "logistic")}
        classifier_rows.append({"fold": number, "threshold": threshold, "train_tail_count": int(y_tail.sum()), "validation_tail_count": int(true_tail.sum()), **{kind: {"precision": float(((score >= .5) & (true_tail == 1)).sum() / max(1, (score >= .5).sum())), "recall": float(((score >= .5) & (true_tail == 1)).sum() / max(1, true_tail.sum())), "pr_auc": float(average_precision_score(true_tail, score)) if len(np.unique(true_tail)) > 1 else np.nan, "confusion": confusion_matrix(true_tail, score >= .5, labels=[0, 1]).tolist()} for kind, score in scores.items()}})
        # Global and ordinary components use TabFM with fold-local contexts.
        context_raw, meta = select_context(train_raw, valid_raw, features, context_cap or len(train_raw), "random", seed + number)
        global_pred, _, _ = _tabfm_fit_predict(processor.transform(context_raw), valid, context_raw[target], seed + number, context_cap, meta)
        ordinary_idx = np.flatnonzero(y_tail == 0); tail_idx = np.flatnonzero(y_tail == 1)
        ordinary_pred, _, _ = _tabfm_fit_predict(train.iloc[ordinary_idx], valid, train_raw.iloc[ordinary_idx][target], seed + 100 + number, len(ordinary_idx))
        if len(tail_idx) >= 64:
            tail_catboost, _ = _catboost_fit_predict(train.iloc[tail_idx], valid, train_raw.iloc[tail_idx][target], seed + 200 + number)
            tail_tabfm, _, _ = _tabfm_fit_predict(train.iloc[tail_idx], valid, train_raw.iloc[tail_idx][target], seed + 300 + number, len(tail_idx))
            tail_strategy = "catboost"
        else:
            tail_catboost, _ = conservative_tail_prediction(train_raw[target], tail_idx, len(valid))
            tail_tabfm = tail_catboost.copy()
            tail_strategy = "train_median_fallback"
        tail_fallback, _ = conservative_tail_prediction(train_raw[target], tail_idx, len(valid), minimum_support=len(tail_idx) + 1)
        hard_tabfm = combine_predictions(ordinary_pred, tail_tabfm, scores["catboost"], True)
        hard = combine_predictions(ordinary_pred, tail_catboost, scores["catboost"], True)
        hard_fallback = combine_predictions(ordinary_pred, tail_fallback, scores["catboost"], True)
        weighted = combine_predictions(ordinary_pred, tail_catboost, scores["catboost"], False)
        hard_logistic = combine_predictions(ordinary_pred, tail_catboost, scores["logistic"], True)
        truth = valid_raw[target].to_numpy(float)
        candidates = {"global_tabfm": global_pred, "hard_catboost_tail_tabfm": hard_tabfm, "hard_catboost_tail_catboost": hard, "hard_catboost_tail_fallback": hard_fallback, "weighted_catboost_tail_catboost": weighted, "hard_logistic_tail_catboost": hard_logistic}
        row = {"fold": number, "threshold": threshold, "tail_component": tail_strategy, "ordinary_rows": int(len(ordinary_idx)), "tail_rows": int(len(tail_idx)), "metrics": {}}
        for name, pred in candidates.items():
            residual = truth - pred; tail_mask = true_tail.astype(bool); worst = np.abs(residual) >= np.quantile(np.abs(residual), .9)
            row["metrics"][name] = {"r2": float(r2_score(truth, pred)), "mae": float(mean_absolute_error(truth, pred)), "ordinary_mae": float(mean_absolute_error(truth[~tail_mask], pred[~tail_mask])) if (~tail_mask).any() else np.nan, "true_tail_mae": float(mean_absolute_error(truth[tail_mask], pred[tail_mask])) if tail_mask.any() else np.nan, "large_residual_count": int((np.abs(residual) > 1.5 * (np.quantile(truth, .75) - np.quantile(truth, .25))).sum()), "worst_decile_mae": float(np.abs(residual)[worst].mean())}
            totals[name].append(row["metrics"][name]["mae"])
        fold_rows.append(row); del processor, train, valid, global_pred, ordinary_pred, tail_catboost, tail_tabfm; gc.collect()
    return {"aggregate_only": True, "target": target, "seed": seed, "threshold_method": threshold_method, "preparation": preparation, "classifier": classifier_rows, "folds": fold_rows, "summary": {name: {"mae_mean": float(np.mean(v)), "mae_std": float(np.std(v))} for name, v in totals.items()}}
