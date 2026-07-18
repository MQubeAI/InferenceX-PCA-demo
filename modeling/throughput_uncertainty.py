"""Leakage-safe conditional uncertainty evaluation for throughput.

TabFM is fitted only on a fold-local context partition.  Its predictions for
the residual-training, calibration, and outer-validation rows are kept in
memory, as are all labels and residuals.  The returned structure is strictly
aggregate-only.
"""
from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from modeling.comparison import (
    FoldPreprocessor,
    _tabfm_fit_predict,
    deterministic_grouped_folds,
    prepare_model_frame,
)

NOMINAL_LEVELS = (0.50, 0.80, 0.95)
DEFAULT_SUBGROUP_MINIMUM_SUPPORT = 20
WORKLOAD_PREDICTORS = ("benchmark_type", "isl", "osl", "conc")
REPORT_FEATURES = (
    "config_hardware",
    "config_model",
    "config_framework",
    "config_precision",
    "conc",
    "isl",
    "osl",
)
METHODS = ("global_split_conformal", "conditional_scale", "conditional_quantile")


def throughput_predictors(features: list[str]) -> list[str]:
    """Keep only configuration and workload predictors, in supplied order."""
    selected = [
        column
        for column in features
        if column.startswith("config_") or column in WORKLOAD_PREDICTORS
    ]
    if not selected:
        raise ValueError("Throughput uncertainty needs configuration/workload predictors.")
    return list(dict.fromkeys(selected))


def split_outer_training_groups(train: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Deterministically allocate whole configurations to three disjoint roles."""
    if "config_id" not in train:
        raise ValueError("Grouped uncertainty splitting requires config_id.")
    groups = train["config_id"].fillna("__missing_config_id__")
    values = sorted(groups.unique().tolist(), key=str)
    if len(values) < 3:
        raise ValueError("Each outer-training fold needs at least three config_id groups.")
    shuffled = np.random.RandomState(seed).permutation(len(values))
    ordered = [values[position] for position in shuffled]
    context_count = max(1, len(ordered) // 2)
    uncertainty_count = max(1, (len(ordered) - context_count) // 2)
    calibration_count = len(ordered) - context_count - uncertainty_count
    if calibration_count < 1:
        raise ValueError("Each uncertainty role needs at least one config_id group.")
    partitions = {
        "context": set(ordered[:context_count]),
        "uncertainty_train": set(ordered[context_count : context_count + uncertainty_count]),
        "calibration": set(ordered[context_count + uncertainty_count :]),
    }
    return {name: train.loc[groups.isin(values)].copy() for name, values in partitions.items()}


def conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """Finite-sample split-conformal quantile using the conservative order statistic."""
    values = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    if not len(values):
        raise ValueError("Conformal calibration needs at least one finite score.")
    rank = min(len(values), int(np.ceil((len(values) + 1) * float(nominal))))
    return float(values[rank - 1])


def global_split_conformal_intervals(
    prediction: np.ndarray, calibration_residual: np.ndarray, levels: tuple[float, ...] = NOMINAL_LEVELS
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], dict[float, float]]:
    """Symmetric intervals calibrated only on a separate residual split."""
    prediction = np.asarray(prediction, dtype=float)
    absolute = np.abs(np.asarray(calibration_residual, dtype=float))
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    quantiles: dict[float, float] = {}
    for level in levels:
        quantile = conformal_quantile(absolute, level)
        quantiles[level] = quantile
        intervals[level] = (prediction - quantile, prediction + quantile)
    return intervals, quantiles


def conformalized_scale_intervals(
    prediction: np.ndarray,
    calibration_residual: np.ndarray,
    calibration_scale: np.ndarray,
    validation_scale: np.ndarray,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], dict[float, float]]:
    """Symmetric intervals with a separately calibrated conditional scale."""
    prediction = np.asarray(prediction, dtype=float)
    calibration_scale = np.maximum(np.asarray(calibration_scale, dtype=float), 1e-8)
    validation_scale = np.maximum(np.asarray(validation_scale, dtype=float), 1e-8)
    scores = np.abs(np.asarray(calibration_residual, dtype=float)) / calibration_scale
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    quantiles: dict[float, float] = {}
    for level in levels:
        quantile = conformal_quantile(scores, level)
        quantiles[level] = quantile
        radius = quantile * validation_scale
        intervals[level] = (prediction - radius, prediction + radius)
    return intervals, quantiles


def conformalized_quantile_intervals(
    prediction: np.ndarray,
    calibration_residual: np.ndarray,
    calibration_lower: np.ndarray,
    calibration_upper: np.ndarray,
    validation_lower: np.ndarray,
    validation_upper: np.ndarray,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], dict[float, float]]:
    """Conformalized residual quantile intervals (CQR)."""
    prediction = np.asarray(prediction, dtype=float)
    calibration_residual = np.asarray(calibration_residual, dtype=float)
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    quantiles: dict[float, float] = {}
    for level in levels:
        lower_cal_raw = np.asarray(calibration_lower[level], dtype=float)
        upper_cal_raw = np.asarray(calibration_upper[level], dtype=float)
        lower_cal = np.minimum(lower_cal_raw, upper_cal_raw)
        upper_cal = np.maximum(lower_cal_raw, upper_cal_raw)
        score = np.maximum.reduce((lower_cal - calibration_residual, calibration_residual - upper_cal, np.zeros_like(calibration_residual)))
        quantile = conformal_quantile(score, level)
        quantiles[level] = quantile
        lower_valid_raw = np.asarray(validation_lower[level], dtype=float)
        upper_valid_raw = np.asarray(validation_upper[level], dtype=float)
        intervals[level] = (
            prediction + np.minimum(lower_valid_raw, upper_valid_raw) - quantile,
            prediction + np.maximum(lower_valid_raw, upper_valid_raw) + quantile,
        )
    return intervals, quantiles


def interval_score(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> np.ndarray:
    """Proper interval score; lower values are better."""
    observed, lower, upper = (np.asarray(value, dtype=float) for value in (observed, lower, upper))
    alpha = 1.0 - float(nominal)
    return (upper - lower) + (2.0 / alpha) * np.maximum(lower - observed, 0.0) + (2.0 / alpha) * np.maximum(observed - upper, 0.0)


def _catboost_predict(train: pd.DataFrame, query: pd.DataFrame, target: np.ndarray, loss_function: str, seed: int) -> np.ndarray:
    """Small deterministic CPU CatBoost model used only for residual models."""
    from catboost import CatBoostRegressor

    categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    model = CatBoostRegressor(
        iterations=100,
        depth=5,
        learning_rate=0.05,
        loss_function=loss_function,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train, np.asarray(target, dtype=float), cat_features=categorical)
    return np.asarray(model.predict(query), dtype=float)


def _throughput_bin_labels(training_target: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = np.unique(np.quantile(np.asarray(training_target, dtype=float), np.linspace(0, 1, 6)))
    if len(edges) <= 1:
        return np.full(len(observed), "all", dtype=object), [float(edges[0])]
    positions = np.digitize(np.asarray(observed, dtype=float), edges[1:-1], right=True)
    return np.asarray([f"q{position + 1}" for position in positions], dtype=object), [float(edge) for edge in edges]


def _group_rows(frame: pd.DataFrame, covered: np.ndarray, width: np.ndarray, columns: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].astype("string").fillna("__MISSING__").astype(str)
        grouped = pd.DataFrame({"value": values, "covered": covered, "width": width}).groupby("value", sort=True)
        rows = grouped.agg(rows=("covered", "size"), coverage=("covered", "mean"), average_width=("width", "mean")).reset_index()
        result[column] = [
            {"value": str(row.value), "rows": int(row.rows), "coverage": float(row.coverage), "average_width": float(row.average_width)}
            for row in rows.sort_values(["coverage", "value"], kind="stable").itertuples(index=False)
        ]
    return result


def summarize_interval(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
    report_frame: pd.DataFrame,
    throughput_bins: np.ndarray,
    subgroup_minimum_support: int = DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
) -> dict[str, Any]:
    """Compute aggregate-only interval summaries.

    Every subgroup remains in ``coverage_by_feature``.  Only subgroups with at
    least ``subgroup_minimum_support`` rows are eligible for the headline worst
    undercoverage ranking, so a one-row category cannot dominate that report.
    """
    if subgroup_minimum_support < 1:
        raise ValueError("subgroup_minimum_support must be at least 1.")
    observed, lower, upper = (np.asarray(value, dtype=float) for value in (observed, lower, upper))
    covered = (observed >= lower) & (observed <= upper)
    width = upper - lower
    bin_frame = pd.DataFrame({"throughput_bin": throughput_bins})
    by_bin = _group_rows(bin_frame, covered, width, ("throughput_bin",)).get("throughput_bin", [])
    by_feature = _group_rows(report_frame, covered, width, REPORT_FEATURES)
    all_candidates = [
        {"feature": feature, **row, "undercoverage": max(0.0, float(nominal) - row["coverage"])}
        for feature, rows in by_feature.items()
        for row in rows
    ]
    candidates = [row for row in all_candidates if row["rows"] >= subgroup_minimum_support]
    worst = max(candidates, key=lambda row: (row["undercoverage"], -row["rows"], row["feature"], row["value"])) if candidates else None
    return {
        "nominal_coverage": float(nominal),
        "empirical_coverage": float(np.mean(covered)),
        "average_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score(observed, lower, upper, nominal))),
        "coverage_and_width_by_throughput_bin": by_bin,
        "coverage_by_feature": by_feature,
        "subgroup_minimum_support": int(subgroup_minimum_support),
        "subgroup_groups_excluded_from_worst_ranking": int(len(all_candidates) - len(candidates)),
        "worst_subgroup_undercoverage": worst,
    }


def _split_predictions(prediction: np.ndarray, sizes: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, third = sizes
    if len(prediction) != first + second + third:
        raise AssertionError("Combined TabFM prediction length does not match query partitions.")
    return prediction[:first], prediction[first : first + second], prediction[first + second : first + second + third]


def evaluate_throughput_uncertainty(
    frame: pd.DataFrame,
    features: list[str],
    target: str = "metrics_tput_per_gpu",
    max_rows: int = 4096,
    seed: int = 42,
    n_splits: int = 3,
    subgroup_minimum_support: int = DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
) -> dict[str, Any]:
    """Evaluate three leakage-safe uncertainty baselines by outer fold.

    ``subgroup_minimum_support`` affects reporting only; it never affects model
    fitting, calibration, or the complete subgroup tables retained in the
    aggregate-only artifact.
    """
    if subgroup_minimum_support < 1:
        raise ValueError("subgroup_minimum_support must be at least 1.")
    predictors = throughput_predictors(features)
    work, preparation = prepare_model_frame(frame, predictors, target, max_rows, seed)
    folds = deterministic_grouped_folds(work, n_splits)
    all_truth: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    retained: dict[str, dict[float, list[tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]]]] = {
        method: {level: [] for level in NOMINAL_LEVELS} for method in METHODS
    }
    fold_rows: list[dict[str, Any]] = []
    tabfm_invocations = 0

    for fold_number, (outer_train_i, validation_i) in enumerate(folds, 1):
        started = time.perf_counter()
        outer_train = work.iloc[outer_train_i]
        validation = work.iloc[validation_i]
        partitions = split_outer_training_groups(outer_train, seed + fold_number)
        context = partitions["context"]
        uncertainty_train = partitions["uncertainty_train"]
        calibration = partitions["calibration"]
        partition_groups = {
            name: set(partition["config_id"].fillna("__missing_config_id__"))
            for name, partition in partitions.items()
        }
        if any(partition_groups[left] & partition_groups[right] for left, right in (("context", "uncertainty_train"), ("context", "calibration"), ("uncertainty_train", "calibration"))):
            raise AssertionError("Context, residual-training, and calibration configurations must be disjoint.")
        if set(outer_train["config_id"].fillna("__missing_config_id__")) & set(validation["config_id"].fillna("__missing_config_id__")):
            raise AssertionError("Outer grouped fold overlaps on config_id.")

        # The official TabFM predictor appends query rows after context rows. Its
        # fitted feature transforms and attention context are context-only, so one
        # combined unlabeled query is semantically equivalent to three calls.
        tabfm_processor = FoldPreprocessor.fit(context, predictors)
        query = pd.concat([uncertainty_train, calibration, validation], axis=0)
        combined_prediction, _, tabfm_meta = _tabfm_fit_predict(
            tabfm_processor.transform(context),
            tabfm_processor.transform(query),
            context[target],
            seed + fold_number,
            None,
            {"combined_unlabeled_query_rows": int(len(query)), "combined_query_partitions": ["uncertainty_train", "calibration", "validation"]},
        )
        tabfm_invocations += 1
        uncertainty_prediction, calibration_prediction, validation_prediction = _split_predictions(
            np.asarray(combined_prediction, dtype=float), (len(uncertainty_train), len(calibration), len(validation))
        )
        uncertainty_residual = uncertainty_train[target].to_numpy(dtype=float) - uncertainty_prediction
        calibration_residual = calibration[target].to_numpy(dtype=float) - calibration_prediction
        validation_truth = validation[target].to_numpy(dtype=float)
        all_truth.append(validation_truth)
        all_prediction.append(validation_prediction)

        residual_processor = FoldPreprocessor.fit(uncertainty_train, predictors)
        uncertainty_x = residual_processor.transform(uncertainty_train)
        combined_residual_query = residual_processor.transform(pd.concat([calibration, validation], axis=0))
        calibration_x_rows = len(calibration)

        global_intervals, global_quantiles = global_split_conformal_intervals(validation_prediction, calibration_residual)
        scale_prediction = _catboost_predict(
            uncertainty_x,
            combined_residual_query,
            np.log1p(np.abs(uncertainty_residual)),
            "RMSE",
            seed + 10_000 + fold_number,
        )
        calibration_scale, validation_scale = _split_predictions(scale_prediction, (calibration_x_rows, len(validation), 0))[:2]
        scale_intervals, scale_quantiles = conformalized_scale_intervals(
            validation_prediction,
            calibration_residual,
            np.expm1(calibration_scale),
            np.expm1(validation_scale),
        )

        lower_calibration: dict[float, np.ndarray] = {}
        upper_calibration: dict[float, np.ndarray] = {}
        lower_validation: dict[float, np.ndarray] = {}
        upper_validation: dict[float, np.ndarray] = {}
        for level in NOMINAL_LEVELS:
            tail = (1.0 - level) / 2.0
            lower_prediction = _catboost_predict(uncertainty_x, combined_residual_query, uncertainty_residual, f"Quantile:alpha={tail}", seed + fold_number + int(level * 1_000))
            upper_prediction = _catboost_predict(uncertainty_x, combined_residual_query, uncertainty_residual, f"Quantile:alpha={1.0 - tail}", seed + fold_number + 20_000 + int(level * 1_000))
            lower_calibration[level], lower_validation[level] = lower_prediction[:calibration_x_rows], lower_prediction[calibration_x_rows:]
            upper_calibration[level], upper_validation[level] = upper_prediction[:calibration_x_rows], upper_prediction[calibration_x_rows:]
        quantile_intervals, quantile_quantiles = conformalized_quantile_intervals(
            validation_prediction, calibration_residual, lower_calibration, upper_calibration, lower_validation, upper_validation
        )

        bins, edges = _throughput_bin_labels(outer_train[target].to_numpy(dtype=float), validation_truth)
        interval_sets = {
            "global_split_conformal": global_intervals,
            "conditional_scale": scale_intervals,
            "conditional_quantile": quantile_intervals,
        }
        calibration_summary = {
            "global_split_conformal": {str(level): global_quantiles[level] for level in NOMINAL_LEVELS},
            "conditional_scale": {str(level): scale_quantiles[level] for level in NOMINAL_LEVELS},
            "conditional_quantile": {str(level): quantile_quantiles[level] for level in NOMINAL_LEVELS},
        }
        for method, intervals in interval_sets.items():
            for level, (lower, upper) in intervals.items():
                retained[method][level].append((validation_truth, lower, upper, validation.loc[:, [column for column in REPORT_FEATURES if column in validation]].copy(), bins))
        fold_rows.append({
            "fold": fold_number,
            "outer_train_rows": int(len(outer_train)),
            "validation_rows": int(len(validation)),
            "outer_group_overlap": 0,
            "partition_rows": {name: int(len(partition)) for name, partition in partitions.items()},
            "partition_groups": {name: int(len(groups)) for name, groups in partition_groups.items()},
            "tabfm_invocations": 1,
            "tabfm": tabfm_meta,
            "point_model": {"r2": float(r2_score(validation_truth, validation_prediction)), "mae": float(mean_absolute_error(validation_truth, validation_prediction))},
            "calibration_quantiles": calibration_summary,
            "throughput_bin_edges_from_outer_training": edges,
            "runtime_seconds": float(time.perf_counter() - started),
        })
        del context, uncertainty_train, calibration, query, combined_prediction, combined_residual_query
        gc.collect()

    interval_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for method in METHODS:
        interval_summary[method] = {}
        for level in NOMINAL_LEVELS:
            entries = retained[method][level]
            truth = np.concatenate([entry[0] for entry in entries])
            lower = np.concatenate([entry[1] for entry in entries])
            upper = np.concatenate([entry[2] for entry in entries])
            report_frame = pd.concat([entry[3] for entry in entries], ignore_index=True)
            bins = np.concatenate([entry[4] for entry in entries])
            interval_summary[method][str(level)] = summarize_interval(
                truth,
                lower,
                upper,
                level,
                report_frame,
                bins,
                subgroup_minimum_support,
            )

    point_truth = np.concatenate(all_truth)
    point_prediction = np.concatenate(all_prediction)
    return {
        "aggregate_only": True,
        "target": target,
        "seed": seed,
        "preparation": preparation,
        "predictors": predictors,
        "fold_assignment": {"fold_count": len(folds), "groups": int(work["config_id"].nunique(dropna=False)), "outer_split": "deterministic GroupKFold by config_id"},
        "uncertainty_partition": {"roles": ["tabfm_context", "uncertainty_model_training", "conformal_calibration"], "unit": "whole config_id groups", "allocation": "about 50% / 25% / 25% of each outer-training fold"},
        "tabfm": {"invocation_count": tabfm_invocations, "expected_invocations": len(folds), "query_strategy": "one combined unlabeled uncertainty-training + calibration + validation query per outer fold", "combined_query_semantically_safe": True},
        "point_model": {"r2": float(r2_score(point_truth, point_prediction)), "mae": float(mean_absolute_error(point_truth, point_prediction))},
        "folds": fold_rows,
        "intervals": interval_summary,
    }
