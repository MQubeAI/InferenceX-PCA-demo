"""Shared, fold-safe evaluation for the research model-comparison workflow.

This module deliberately has no Streamlit or TabFM import at module import time.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MISSING_CATEGORY = "__MISSING__"
DEFAULT_MODELS = ("random_forest", "catboost")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def split_feature_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def prepare_model_frame(
    frame: pd.DataFrame, feature_columns: list[str], target: str, max_rows: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample once, then exclude only rows without a real training label."""
    columns = list(dict.fromkeys(feature_columns + [target, "config_id"]))
    work = frame.loc[:, [column for column in columns if column in frame.columns]].replace(
        [np.inf, -np.inf], np.nan
    )
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=seed)
    initial_rows = len(work)
    work = work.dropna(subset=[target]).copy()
    if "config_id" not in work:
        raise ValueError("Grouped evaluation requires config_id.")
    if len(work) < 4:
        raise ValueError("Not enough rows with a non-null target value.")
    return work, {
        "sampled_rows_before_target_exclusion": initial_rows,
        "usable_rows": len(work),
        "missing_target_rows_excluded": initial_rows - len(work),
    }


def deterministic_grouped_folds(work: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = work["config_id"].fillna("__missing_config_id__")
    group_count = int(groups.nunique(dropna=False))
    if group_count < 2:
        raise ValueError("Grouped evaluation needs at least two config_id groups.")
    requested = int(n_splits)
    if requested <= 1:
        # A bounded smoke test still holds out complete configurations.
        return list(GroupKFold(n_splits=2).split(work, groups=groups))[:1]
    folds = min(requested, group_count)
    return list(GroupKFold(n_splits=folds).split(work, groups=groups))


@dataclass
class FoldPreprocessor:
    numeric_features: list[str]
    categorical_features: list[str]
    medians: pd.Series

    @classmethod
    def fit(cls, train: pd.DataFrame, feature_columns: list[str]) -> "FoldPreprocessor":
        numeric, categorical = split_feature_types(train, feature_columns)
        medians = train[numeric].median(numeric_only=True) if numeric else pd.Series(dtype=float)
        return cls(numeric, categorical, medians)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.numeric_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            result[column] = values.fillna(self.medians[column])
            # Keep missingness visible to models without leaking fit information.
            result[f"{column}__missing"] = values.isna().astype("int8")
        for column in self.categorical_features:
            result[column] = frame[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        return result


def missingness_report(frame: pd.DataFrame, features: list[str], targets: list[str]) -> dict[str, Any]:
    columns = [column for column in dict.fromkeys(features + targets) if column in frame]
    total = max(len(frame), 1)
    per_column = [
        {
            "column": column,
            "role": "target" if column in targets else "feature",
            "missing_count": int(frame[column].isna().sum()),
            "missing_percentage": float(frame[column].isna().mean() * 100),
            "complete_case_count": int(frame[column].notna().sum()),
            "likely_structural_or_not_applicable": bool(
                frame[column].isna().mean() >= 0.95 or (frame[column].notna().sum() and frame[column].nunique(dropna=True) <= 1)
            ),
        }
        for column in columns
    ]
    usable = {target: int(frame[target].notna().sum()) for target in targets if target in frame}
    dimensions = [
        column for column in ("config_hardware", "config_framework", "config_model", "benchmark_type", "date") if column in frame
    ]
    by_dimension: dict[str, list[dict[str, Any]]] = {}
    for dimension in dimensions:
        grouped = frame.assign(**{dimension: frame[dimension].fillna(MISSING_CATEGORY)}).groupby(dimension, dropna=False)
        rows = []
        for value, subset in grouped:
            rows.append({
                "value": str(value), "rows": int(len(subset)),
                "missing_percent": {column: float(subset[column].isna().mean() * 100) for column in columns},
            })
        by_dimension[dimension] = rows
    failed_column = next((column for column in ("error", "status", "run_status") if column in frame), None)
    failures: dict[str, Any] = {"detectable": False}
    if failed_column:
        failed = frame[failed_column].astype("string").str.contains("fail|error|abort", case=False, na=False)
        failures = {
            "detectable": True, "column": failed_column, "failed_rows": int(failed.sum()),
            "missing_percent_when_failed": {column: float(frame.loc[failed, column].isna().mean() * 100) if failed.any() else 0.0 for column in columns},
            "missing_percent_when_not_failed": {column: float(frame.loc[~failed, column].isna().mean() * 100) if (~failed).any() else 0.0 for column in columns},
        }
    return {"rows": len(frame), "columns": per_column, "usable_rows_per_target": usable, "by_dimension": by_dimension, "failed_run_relationship": failures}


def _metric_summary(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in ("r2", "mae"):
        values = np.asarray([row[metric] for row in folds if row.get(metric) is not None], dtype=float)
        values = values[np.isfinite(values)]
        result[metric] = ({"mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max())} if len(values) else {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
    return result


def _rf_fit_predict(train: pd.DataFrame, valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cat = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    num = [column for column in train if column not in cat]
    model = Pipeline([
        ("preprocess", ColumnTransformer([("numeric", "passthrough", num), ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)], sparse_threshold=0)),
        ("model", RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    ])
    model.fit(train, y_train)
    prediction = model.predict(valid)
    ranked = permutation_importance(model, valid, y_valid, n_repeats=3, random_state=seed, n_jobs=1, scoring="r2")
    return prediction, [{"feature": feature, "importance": float(value)} for feature, value in zip(train.columns, ranked.importances_mean)]


def _catboost_fit_predict(train: pd.DataFrame, valid: pd.DataFrame, y_train: pd.Series, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from catboost import CatBoostRegressor
    categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05, loss_function="RMSE", random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
    model.fit(train, y_train, cat_features=categorical)
    prediction = model.predict(valid)
    return prediction, [{"feature": feature, "importance": float(value)} for feature, value in zip(train.columns, model.get_feature_importance())]


def _tabfm_fit_predict(train: pd.DataFrame, valid: pd.DataFrame, y_train: pd.Series, seed: int, context_cap: int | None, context_metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Lazy TabFM boundary. This function is called only by an explicit tabfm run."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
    available = len(train)
    if context_cap and available > context_cap:
        context = train.sample(n=context_cap, random_state=seed)
        context_y = y_train.loc[context.index]
    else:
        context, context_y = train, y_train
    model = tabfm_v1_0_0_pytorch.load(model_type="regression", device="cpu")
    regressor = TabFMRegressor(model=model, n_estimators=1, random_state=seed, use_amp=False, verbose=False, max_num_rows=context_cap)
    regressor.fit(context, context_y.to_numpy())
    return regressor.predict(valid), [], {"available_context_rows": available, "used_context_rows": len(context), "device": "cpu", "checkpoint": "tabfm_v1_0_0 regression (local Hugging Face cache)", "package_version": importlib.metadata.version("tabfm"), **(context_metadata or {})}


def model_available(model: str) -> tuple[bool, str]:
    if model == "random_forest":
        return True, "scikit-learn"
    if model == "catboost":
        return importlib.util.find_spec("catboost") is not None, "catboost package unavailable" if importlib.util.find_spec("catboost") is None else "catboost"
    if model == "tabfm":
        return importlib.util.find_spec("tabfm") is not None, "TabFM must be run with .venv-tabfm/bin/python" if importlib.util.find_spec("tabfm") is None else "tabfm"
    return False, "unknown model"


def evaluate_models(frame: pd.DataFrame, feature_columns: list[str], target: str, models: list[str], max_rows: int, seed: int, n_splits: int = 5, tabfm_max_context: int | None = None, target_transform: str = "raw", tabfm_context_strategy: str = "random", folds_override: list[tuple[np.ndarray, np.ndarray]] | None = None, evaluation_label: str = "unseen_config") -> dict[str, Any]:
    from modeling.diagnostics import fold_difficulty, inverse_target, select_context, target_transform as transform_target
    work, preparation = prepare_model_frame(frame, feature_columns, target, max_rows, seed)
    folds = folds_override or deterministic_grouped_folds(work, n_splits)
    numeric, categorical = split_feature_types(work, feature_columns)
    result: dict[str, Any] = {"target": target, "feature_columns": feature_columns, "feature_types": {"numeric": numeric, "categorical": categorical}, "preparation": preparation, "fold_assignment": {"fold_count": len(folds), "groups": int(work["config_id"].nunique(dropna=False))}, "models": {}}
    for model_name in models:
        available, availability_reason = model_available(model_name)
        folds_result: list[dict[str, Any]] = []
        importance_rows: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(folds, 1):
            train_raw, valid_raw = work.iloc[train_index], work.iloc[validation_index]
            processor = FoldPreprocessor.fit(train_raw, feature_columns)
            train, valid = processor.transform(train_raw), processor.transform(valid_raw)
            groups_train = set(train_raw["config_id"].fillna("__missing_config_id__"))
            groups_valid = set(valid_raw["config_id"].fillna("__missing_config_id__"))
            started = time.perf_counter()
            row: dict[str, Any] = {"fold": fold_number, "train_rows": len(train), "validation_rows": len(valid), "train_groups": len(groups_train), "validation_groups": len(groups_valid), "group_overlap": len(groups_train & groups_valid), "r2": None, "mae": None, "runtime_seconds": 0.0, "failure": "", "fallback": "", "available_context_rows": None, "used_context_rows": None}
            if evaluation_label == "unseen_config" and row["group_overlap"] != 0:
                raise AssertionError("Grouped folds overlap on config_id.")
            if not available:
                row["failure"] = availability_reason
            else:
                try:
                    y_train = transform_target(train_raw[target], target_transform)
                    if model_name == "random_forest":
                        prediction, importance = _rf_fit_predict(train, valid, y_train, transform_target(valid_raw[target], target_transform), seed + fold_number)
                        extra = {}
                    elif model_name == "catboost":
                        prediction, importance = _catboost_fit_predict(train, valid, y_train, seed + fold_number)
                        extra = {}
                    elif model_name == "tabfm":
                        context_raw, context_meta = select_context(train_raw, valid_raw, feature_columns, tabfm_max_context or len(train_raw), tabfm_context_strategy, seed + fold_number)
                        context = processor.transform(context_raw)
                        prediction, importance, extra = _tabfm_fit_predict(context, valid, y_train.loc[context_raw.index], seed + fold_number, tabfm_max_context, context_meta)
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")
                    raw_prediction = inverse_target(prediction, target_transform)
                    row["r2"] = float(r2_score(valid_raw[target], raw_prediction))
                    row["mae"] = float(mean_absolute_error(valid_raw[target], raw_prediction))
                    row["target_transform"] = target_transform
                    if target_transform != "raw": row["transformed_r2"] = float(r2_score(transform_target(valid_raw[target], target_transform), prediction))
                    row["difficulty"] = fold_difficulty(train_raw, valid_raw, target, raw_prediction)
                    row.update(extra)
                    importance_rows.extend({"fold": fold_number, **item} for item in importance)
                except Exception as exc:
                    row["failure"] = f"{type(exc).__name__}: {exc}"
            row["runtime_seconds"] = float(time.perf_counter() - started)
            folds_result.append(row)
        importance = pd.DataFrame(importance_rows)
        aggregate_importance = ([] if importance.empty else importance.groupby("feature", as_index=False).agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            folds=("fold", "nunique"),
        ).fillna({"importance_std": 0.0}).sort_values("importance_mean", ascending=False).to_dict(orient="records"))
        result["models"][model_name] = {"available": available, "availability": availability_reason, "folds": folds_result, "metrics": _metric_summary(folds_result), "importance": [{key: _json_safe(value) for key, value in item.items()} for item in aggregate_importance], "runtime_seconds": float(sum(row["runtime_seconds"] for row in folds_result))}
    result["evaluation_label"] = evaluation_label
    result["target_transform"] = target_transform
    result["tabfm_context_strategy"] = tabfm_context_strategy
    return result
