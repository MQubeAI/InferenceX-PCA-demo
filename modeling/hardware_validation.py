"""Controlled validation for the Epoch hardware-capability prototype.

This module intentionally evaluates one task only: whether Epoch's *physical
hardware descriptors* help predict InferenceX throughput or transfer to an
observed, withheld hardware SKU.  It does not ingest Epoch model/benchmark,
cluster, datacenter, or Artificial Analysis observations.

All preprocessing is fitted inside the appropriate training partition.  The
module serializes aggregate diagnostics only; it never writes row-level
predictions or residuals.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apps import inferencex_pca_demo as app
from modeling.comparison import (
    FoldPreprocessor,
    _tabfm_fit_predict,
    deterministic_grouped_folds,
    model_available,
    prepare_model_frame,
    split_feature_types,
)
from modeling.integration import (
    CAPABILITY_FEATURE_COLUMNS,
    INTEGRATION_ROOT,
    TARGET,
    load_existing_throughput_cohort,
)


HISTORICAL_SOURCE_DIR = "inferencex-pca-data"
JULY20_SOURCE_DIR = ".data/inferencex-db-dump-2026-07-20"
HISTORICAL_TABFM_ARTIFACT = (
    INTEGRATION_ROOT.parents[2] / "artifacts" / "model-diagnostics-4096.json"
)
HISTORICAL_FULL_CONTEXT_ARTIFACT = (
    INTEGRATION_ROOT.parents[2] / "artifacts" / "throughput-residual-diagnostics.json"
)
PRIMARY_VIEW_PATH = (
    INTEGRATION_ROOT / "views" / "inferencex_throughput_epoch_hardware_v1.csv"
)
VALIDATION_ROOT = INTEGRATION_ROOT / "reports" / "validation"
RECONCILIATION_ROOT = INTEGRATION_ROOT / "reports" / "cohort_reconciliation"

SEED = 42
N_SPLITS = 3
RF_SETTINGS = {"n_estimators": 150, "min_samples_leaf": 2}
RIDGE_ALPHA = 10.0
WORKLOAD_COLUMNS = ("benchmark_type", "isl", "osl", "conc")
MATCHED_PRECISION_FEATURE = "matched_precision_peak_compute"

PRECISION_CAPABILITY_MAP = {
    "bf16": "epoch_tensor_fp16_bf16_peak_flops",
    "fp16": "epoch_tensor_fp16_bf16_peak_flops",
    "float16": "epoch_tensor_fp16_bf16_peak_flops",
    "fp8": "epoch_fp8_peak_flops",
    "fp4": "epoch_fp4_peak_flops",
    "int8": "epoch_int8_peak_ops",
    "int4": "epoch_int4_peak_ops",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [_json_safe(record) for record in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value


def write_validation_json(path: Path, value: Any) -> None:
    """Write a strict, portable JSON artifact under the derived report root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def target_distribution(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "p01": None, "p05": None, "median": None,
                "mean": None, "p95": None, "p99": None, "max": None, "std": None}
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
        "std": float(numeric.std(ddof=0)),
    }


def prediction_metrics(truth: Iterable[float], prediction: Iterable[float]) -> dict[str, float | int | None]:
    """Metrics in the native target scale plus explicit relative-error diagnostics."""

    y = np.asarray(pd.to_numeric(pd.Series(truth), errors="coerce"), dtype=float)
    p = np.asarray(pd.to_numeric(pd.Series(prediction), errors="coerce"), dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {
            "rows": 0, "r2": None, "mae": None, "median_absolute_error": None,
            "median_absolute_percentage_error": None, "p90_absolute_percentage_error": None,
            "within_10_percent": None, "within_20_percent": None, "within_30_percent": None,
            "median_absolute_log_ratio": None,
        }
    absolute_error = np.abs(y - p)
    positive = np.abs(y) > 1e-12
    ape = np.full(len(y), np.nan, dtype=float)
    ape[positive] = absolute_error[positive] / np.abs(y[positive])
    # Throughput is positive, but its lower tail makes raw percentages high variance.
    # The clipped log-ratio is reported as a scale-stable companion, not a replacement.
    log_ratio = np.abs(np.log(np.clip(p[positive], 1e-12, None) / np.clip(y[positive], 1e-12, None)))
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, p)) if len(y) >= 2 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, p)),
        "median_absolute_error": float(np.median(absolute_error)),
        "median_absolute_percentage_error": float(np.nanmedian(ape)) if positive.any() else None,
        "p90_absolute_percentage_error": float(np.nanquantile(ape, 0.90)) if positive.any() else None,
        "within_10_percent": float(np.nanmean(ape <= 0.10)) if positive.any() else None,
        "within_20_percent": float(np.nanmean(ape <= 0.20)) if positive.any() else None,
        "within_30_percent": float(np.nanmean(ape <= 0.30)) if positive.any() else None,
        "median_absolute_log_ratio": float(np.median(log_ratio)) if len(log_ratio) else None,
    }


def _fold_metric_summary(folds: list[dict[str, Any]], pooled: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "r2", "mae", "median_absolute_error", "median_absolute_percentage_error",
        "p90_absolute_percentage_error", "within_10_percent", "within_20_percent",
        "within_30_percent", "median_absolute_log_ratio",
    )
    result: dict[str, Any] = {"pooled_oof": pooled, "fold_mean": {}, "fold_standard_deviation": {}}
    for metric in metric_names:
        values = np.asarray([row["metrics"].get(metric) for row in folds], dtype=float)
        values = values[np.isfinite(values)]
        result["fold_mean"][metric] = float(values.mean()) if len(values) else None
        result["fold_standard_deviation"][metric] = float(values.std(ddof=0)) if len(values) else None
    return result


def _source_frame(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one existing aggregate without changing its construction protocol."""

    source_info = app.data_source_status(data_dir)[1]
    manifest = app.build_dataset_manifest(source_info)
    _benchmarks, _configs, joined, _ = app.load_joined_data(data_dir, manifest["fingerprint"])
    aggregate, analysis = app.build_analysis_frame(
        joined, "Median aggregate per config/workload/concurrency"
    )
    return joined, aggregate, {
        "data_dir": data_dir,
        "source_mode": source_info["active_mode"],
        "source_manifest": manifest,
        "aggregation": analysis,
    }


def _feature_contract(frame: pd.DataFrame) -> list[str]:
    numeric, categorical = app.default_target_features(frame, TARGET)
    return numeric + categorical


def _display_stage(rows: int, configs: int, note: str) -> dict[str, Any]:
    return {"rows": int(rows), "configs": int(configs), "note": note}


def _value_counts(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    counts = frame[column].astype("string").fillna("__MISSING__").value_counts(dropna=False)
    return [{"value": str(value), "rows": int(count)} for value, count in counts.items()]


def _prior_tabfm_contract() -> dict[str, Any]:
    if not HISTORICAL_TABFM_ARTIFACT.exists():
        return {"available": False, "reason": f"Missing artifact: {HISTORICAL_TABFM_ARTIFACT}"}
    artifact = json.loads(HISTORICAL_TABFM_ARTIFACT.read_text(encoding="utf-8"))
    experiment = artifact.get("experiments", {}).get(
        "unseen:metrics_tput_per_gpu:raw:tabfm:random:2730", {}
    )
    full_context: dict[str, Any] = {}
    if HISTORICAL_FULL_CONTEXT_ARTIFACT.exists():
        full_artifact = json.loads(HISTORICAL_FULL_CONTEXT_ARTIFACT.read_text(encoding="utf-8"))
        full_context = {
            "artifact_path": str(HISTORICAL_FULL_CONTEXT_ARTIFACT),
            "controls": full_artifact.get("controls", {}),
            "preparation": full_artifact.get("result", {}).get("preparation", {}),
            "metrics": full_artifact.get("result", {}).get("metrics", {}),
            "folds": full_artifact.get("result", {}).get("folds", []),
        }
    return {
        "available": bool(experiment),
        "dataset": artifact.get("dataset", {}),
        "controls": artifact.get("controls", {}),
        "feature_columns": experiment.get("feature_columns", []),
        "feature_types": experiment.get("feature_types", {}),
        "preparation": experiment.get("preparation", {}),
        "fold_assignment": experiment.get("fold_assignment", {}),
        "recorded_metrics": experiment.get("models", {}).get("tabfm", {}).get("metrics", {}),
        "selected_full_context_result": full_context,
    }


def reconcile_cohorts(
    historical_data_dir: str = HISTORICAL_SOURCE_DIR,
    july20_data_dir: str = JULY20_SOURCE_DIR,
    integrated_view_path: Path = PRIMARY_VIEW_PATH,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Trace the old 4,096-row TabFM and the new 8,239/7,168 row flows."""

    historical_joined, historical, historical_source = _source_frame(historical_data_dir)
    july_joined, july_aggregate, july_source = _source_frame(july20_data_dir)
    view = pd.read_csv(integrated_view_path, low_memory=False)
    historical_features = _feature_contract(historical)
    prior = _prior_tabfm_contract()
    historical_sample, historical_preparation = prepare_model_frame(
        historical, historical_features, TARGET, 4096, SEED
    )
    historical_y = pd.to_numeric(historical[TARGET], errors="coerce")
    july_y = pd.to_numeric(july_aggregate[TARGET], errors="coerce")
    view_y = pd.to_numeric(view[TARGET], errors="coerce")
    resolved = view.loc[
        view["hardware_mapping_status"].eq("accepted") & view_y.notna()
    ].copy()
    h100_h200 = view.loc[
        view["config_hardware"].astype(str).str.lower().isin({"h100", "h200"})
    ].copy()
    current_features = _feature_contract(july_aggregate)
    initial_features = []
    initial_path = INTEGRATION_ROOT / "reports" / "hardware_representation_experiment.json"
    if initial_path.exists():
        initial_features = json.loads(initial_path.read_text(encoding="utf-8")).get(
            "representations", {}
        ).get("A_identity", {}).get("feature_columns", [])
    table = [
        {
            "stage": "Raw joined benchmark rows",
            "historical_tabfm": _display_stage(len(historical_joined), historical_joined["config_id"].nunique(), "No row filter."),
            "new_integration": _display_stage(len(july_joined), july_joined["config_id"].nunique(), "No row filter."),
        },
        {
            "stage": "Median aggregate by config/workload/concurrency",
            "historical_tabfm": _display_stage(len(historical), historical["config_id"].nunique(), "Same aggregation implementation."),
            "new_integration": _display_stage(len(july_aggregate), july_aggregate["config_id"].nunique(), "Same aggregation implementation."),
        },
        {
            "stage": "Target metrics_tput_per_gpu non-null",
            "historical_tabfm": _display_stage(historical_y.notna().sum(), historical.loc[historical_y.notna(), "config_id"].nunique(), "Target filter after deterministic sampling in the historical helper; it removed zero sampled rows."),
            "new_integration": _display_stage(july_y.notna().sum(), july_aggregate.loc[july_y.notna(), "config_id"].nunique(), "Target filter removed no aggregate rows."),
        },
        {
            "stage": "Workload / benchmark-type filter",
            "historical_tabfm": _display_stage(historical_y.notna().sum(), historical.loc[historical_y.notna(), "config_id"].nunique(), "None. No single_turn-only filter."),
            "new_integration": _display_stage(july_y.notna().sum(), july_aggregate.loc[july_y.notna(), "config_id"].nunique(), "None. No single_turn-only filter."),
        },
        {
            "stage": "Deterministic max_rows sample",
            "historical_tabfm": _display_stage(len(historical_sample), historical_sample["config_id"].nunique(), "pandas sample(n=4096, random_state=42), then non-null target filtering."),
            "new_integration": _display_stage(len(july_aggregate), july_aggregate["config_id"].nunique(), "No sample cap in the original RF integration experiment."),
        },
        {
            "stage": "Accepted Epoch hardware mapping",
            "historical_tabfm": _display_stage(len(historical_sample), historical_sample["config_id"].nunique(), "Not applied in historical TabFM."),
            "new_integration": _display_stage(len(resolved), resolved["config_id"].nunique(), "Retains seven accepted hardware labels; H100/H200 remain unresolved."),
        },
        {
            "stage": "Final model rows",
            "historical_tabfm": _display_stage(len(historical_sample), historical_sample["config_id"].nunique(), "Full-context TabFM input."),
            "new_integration": _display_stage(len(resolved), resolved["config_id"].nunique(), "Initial Epoch RF input."),
        },
    ]
    report = {
        "study": "cohort_reconciliation",
        "target": TARGET,
        "historical_protocol": {
            "source": historical_source,
            "source_table": "benchmark_results.csv joined to configs.csv",
            "aggregation_level": "median by config_id, benchmark_type, isl, osl, conc",
            "filters": {
                "target_non_null": "Applied after max_rows sampling; removed 0 of 4,096 sampled rows.",
                "single_turn": "No filter.", "hardware": "No filter.", "model": "No filter.",
                "precision": "No filter.", "configuration": "No inclusion/exclusion filter beyond available feature columns.",
            },
            "target_distribution_aggregate": target_distribution(historical_y),
            "target_distribution_sample": target_distribution(historical_sample[TARGET]),
            "feature_columns_reconstructed": historical_features,
            "feature_columns_recorded": prior.get("feature_columns", []),
            "feature_contract_matches_artifact": historical_features == prior.get("feature_columns", []),
            "feature_types_recorded": prior.get("feature_types", {}),
            "sample_preparation": historical_preparation,
            "observed_values_before_sampling": {
                column: _value_counts(historical, column)
                for column in ("benchmark_type", "config_hardware", "config_model", "config_precision")
            },
            "preprocessing": "FoldPreprocessor fit on each training fold: numeric training median plus missingness indicators; categorical values receive __MISSING__. The TabFM full-context point model then receives every fold-training row as context.",
            "grouped_split": "GroupKFold(n_splits=3) on config_id; group ordering follows the deterministic sampled row order.",
            "recorded_artifact": prior,
        },
        "new_integration_protocol": {
            "source": july_source,
            "source_table": "benchmark_results_raw.csv (metrics JSON flattened) joined to configs.csv",
            "aggregation_level": "median by config_id, benchmark_type, isl, osl, conc",
            "target_distribution_aggregate": target_distribution(july_y),
            "target_distribution_mapped": target_distribution(resolved[TARGET]),
            "accepted_hardware_labels": sorted(resolved["config_hardware"].dropna().astype(str).unique()),
            "unresolved_hardware_labels": sorted(h100_h200["config_hardware"].dropna().astype(str).unique()),
            "unresolved_rows": int(len(h100_h200)),
            "initial_rf_feature_columns": initial_features,
            "historical_default_feature_columns_on_july20": current_features,
            "observed_values_before_epoch_filter": {
                column: _value_counts(july_aggregate, column)
                for column in ("benchmark_type", "config_hardware", "config_model", "config_precision")
            },
        },
        "row_flow": table,
        "exact_causes": [
            "The historical TabFM artifact used the older local CSV source inferencex-pca-data (its recorded manifest fingerprint differs from the July 20 raw dump).",
            "The historical 4,096 is a deterministic max_rows=4096 computational sample, not a workload, hardware, model, precision, or target-availability cohort filter.",
            "The new July 20 aggregate has more rows/configurations before any Epoch operation because it is a different cumulative source snapshot.",
            "The 7,168 mapped rows additionally exclude 1,071 H100/H200 operating points because their exact Epoch variants are unresolved; this restriction did not exist historically.",
            "The initial integration RF used a different structural feature list from the historical TabFM contract. The controlled validation restores the historical default structural feature contract before adding/removing hardware representations.",
        ],
    }
    return report, historical, resolved


def add_matched_precision_peak(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add one explicit, physics-informed feature without replacing raw capabilities."""

    result = frame.copy()
    precision = result["config_precision"].astype("string").str.lower().str.strip()
    matched = pd.Series(np.nan, index=result.index, dtype=float)
    mapping_rows: list[dict[str, Any]] = []
    for label, capability in PRECISION_CAPABILITY_MAP.items():
        mask = precision.eq(label)
        if capability in result:
            matched.loc[mask] = pd.to_numeric(result.loc[mask, capability], errors="coerce")
        mapping_rows.append({
            "config_precision": label,
            "capability_column": capability,
            "rows": int(mask.sum()),
            "missing_matched_peak": int(matched.loc[mask].isna().sum()),
        })
    result[MATCHED_PRECISION_FEATURE] = matched
    unknown = sorted(precision.loc[~precision.isin(PRECISION_CAPABILITY_MAP)].dropna().unique().tolist())
    return result, {
        "feature": MATCHED_PRECISION_FEATURE,
        "mapping": mapping_rows,
        "unmapped_precision_labels": unknown,
        "missing_feature_rows": int(matched.isna().sum()),
        "policy": "Unknown precision terminology remains missing; no peak is guessed or substituted.",
    }


def primary_epoch_resolved_cohort(view_path: Path = PRIMARY_VIEW_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the one primary cohort used by every A/B/C/D comparison."""

    view = pd.read_csv(view_path, low_memory=False)
    # The CSV view is deliberately portable, but CSV inference can change logical
    # config dtypes (notably true/false fields).  Restore structural fields from
    # the existing aggregate and use the feature columns recorded in the selected
    # historical TabFM artifact, not today's dashboard default.
    _joined, july_aggregate, _source = _source_frame(JULY20_SOURCE_DIR)
    recorded_contract = _prior_tabfm_contract().get("feature_columns", [])
    if not recorded_contract:
        raise RuntimeError("The recorded historical TabFM feature contract is unavailable.")
    base_features = list(recorded_contract)
    identity_keys = ["config_id", "benchmark_type", "isl", "osl", "conc"]
    if july_aggregate.duplicated(identity_keys).any() or view.duplicated(identity_keys).any():
        raise AssertionError("Aggregate identity keys must stay unique during validation.")
    structural_columns = list(dict.fromkeys(identity_keys + base_features))
    structural = july_aggregate[structural_columns].copy()
    replaceable_structural = [column for column in base_features if column not in identity_keys]
    view = view.drop(columns=[column for column in replaceable_structural if column in view], errors="ignore").merge(
        structural, on=identity_keys, how="left", validate="one_to_one"
    )
    if len(view) != len(structural):
        raise AssertionError("Restoring aggregate structural semantics changed integration-view row count.")
    target = pd.to_numeric(view[TARGET], errors="coerce")
    cohort = view.loc[view["hardware_mapping_status"].eq("accepted") & target.notna()].copy()
    cohort, precision_metadata = add_matched_precision_peak(cohort)
    # Check the recorded historical app contract against the actual July 20 view
    # rather than maintaining a second shadow schema.
    missing = [column for column in base_features if column not in cohort]
    if missing:
        raise ValueError(f"Primary view lacks historical structural features: {missing}")
    capability_features = [
        column for column in CAPABILITY_FEATURE_COLUMNS
        if column in cohort and cohort[column].notna().any()
    ]
    excluded_all_missing = [
        column for column in CAPABILITY_FEATURE_COLUMNS
        if column in cohort and not cohort[column].notna().any()
    ]
    if "config_hardware" not in base_features:
        raise ValueError("Historical structural contract no longer contains config_hardware.")
    representations = {
        "A_identity_only": base_features,
        "B_identity_plus_capabilities": [*base_features, *capability_features],
        "C_capabilities_only": [
            column for column in base_features if column != "config_hardware"
        ] + capability_features,
        "D_capabilities_plus_matched_precision": [
            column for column in base_features if column != "config_hardware"
        ] + capability_features + [MATCHED_PRECISION_FEATURE],
    }
    for name, features in representations.items():
        if len(features) != len(set(features)):
            raise AssertionError(f"Duplicate features in {name}")
        prohibited = [column for column in features if column.startswith("metrics_")]
        if prohibited:
            raise AssertionError(f"Target/outcome leakage in {name}: {prohibited}")
    metadata = {
        "name": "Epoch-resolved controlled primary cohort",
        "rows": int(len(cohort)),
        "configs": int(cohort["config_id"].nunique(dropna=False)),
        "target": TARGET,
        "target_distribution": target_distribution(cohort[TARGET]),
        "hardware_labels": sorted(cohort["config_hardware"].dropna().astype(str).unique()),
        "unresolved_rows_excluded": int((view["hardware_mapping_status"] != "accepted").sum()),
        "unresolved_hardware_labels": sorted(
            view.loc[view["hardware_mapping_status"].ne("accepted"), "config_hardware"].dropna().astype(str).unique()
        ),
        "base_feature_contract": base_features,
        "capability_feature_columns": capability_features,
        "excluded_all_missing_capability_columns": excluded_all_missing,
        "representations": representations,
        "matched_precision": precision_metadata,
        "split": {
            "method": "GroupKFold(n_splits=3)", "group": "config_id", "random_seed": SEED,
            "note": "GroupKFold itself is deterministic and unshuffled; seed controls models only.",
        },
        "preprocessing": "Within every training fold: numeric training median plus explicit missingness indicator; categorical __MISSING__; no physical missing value is set to zero.",
    }
    return cohort.reset_index(drop=True), metadata


def _workload_key(frame: pd.DataFrame) -> pd.Series:
    columns = [column for column in WORKLOAD_COLUMNS if column in frame]
    if not columns:
        return pd.Series("__all__", index=frame.index)
    normalized = frame.loc[:, columns].copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string").fillna("__MISSING__")
    return normalized.astype(str).agg("|".join, axis=1)


def workload_median_prediction(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    y_train = pd.to_numeric(train[TARGET], errors="coerce")
    global_median = float(y_train.median())
    grouped = pd.DataFrame({"key": _workload_key(train), "target": y_train}).groupby("key")["target"].median()
    return _workload_key(valid).map(grouped).fillna(global_median).to_numpy(dtype=float)


def nearest_configuration_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    structural_features: list[str],
    chunk_size: int = 128,
) -> np.ndarray:
    """One nearest measured training point in original non-hardware descriptors."""

    features = [column for column in structural_features if column != "config_hardware"]
    numeric, categorical = split_feature_types(train, features)
    train_numeric: np.ndarray | None = None
    valid_numeric: np.ndarray | None = None
    if numeric:
        train_numeric_frame = train[numeric].apply(pd.to_numeric, errors="coerce")
        medians = train_numeric_frame.median()
        train_numeric_frame = train_numeric_frame.fillna(medians)
        valid_numeric_frame = valid[numeric].apply(pd.to_numeric, errors="coerce").fillna(medians)
        scales = train_numeric_frame.quantile(0.75) - train_numeric_frame.quantile(0.25)
        fallback = train_numeric_frame.std(ddof=0)
        scales = scales.where(scales.gt(0), fallback).where(lambda series: series.gt(0), 1.0)
        train_numeric = (train_numeric_frame / scales).to_numpy(dtype=float)
        valid_numeric = (valid_numeric_frame / scales).to_numpy(dtype=float)
    train_categorical: np.ndarray | None = None
    valid_categorical: np.ndarray | None = None
    if categorical:
        train_categorical = train[categorical].astype("string").fillna("__MISSING__").astype(str).to_numpy()
        valid_categorical = valid[categorical].astype("string").fillna("__MISSING__").astype(str).to_numpy()
    terms = len(numeric) + len(categorical)
    if not terms:
        return np.repeat(float(pd.to_numeric(train[TARGET], errors="coerce").median()), len(valid))
    train_y = pd.to_numeric(train[TARGET], errors="coerce").to_numpy(dtype=float)
    predictions = np.empty(len(valid), dtype=float)
    for start in range(0, len(valid), chunk_size):
        stop = min(start + chunk_size, len(valid))
        distance = np.zeros((stop - start, len(train)), dtype=float)
        if train_numeric is not None and valid_numeric is not None:
            distance += np.abs(valid_numeric[start:stop, None, :] - train_numeric[None, :, :]).sum(axis=2)
        if train_categorical is not None and valid_categorical is not None:
            distance += (valid_categorical[start:stop, None, :] != train_categorical[None, :, :]).sum(axis=2)
        nearest = distance.argmin(axis=1)
        predictions[start:stop] = train_y[nearest]
    return predictions


def _ridge_pipeline(train: pd.DataFrame) -> Pipeline:
    categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    numeric = [column for column in train if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ], sparse_threshold=0)),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _random_forest_pipeline(train: pd.DataFrame, seed: int) -> Pipeline:
    categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
    numeric = [column for column in train if column not in categorical]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ], sparse_threshold=0)),
        ("model", RandomForestRegressor(
            n_estimators=RF_SETTINGS["n_estimators"],
            min_samples_leaf=RF_SETTINGS["min_samples_leaf"],
            random_state=seed,
            n_jobs=1,
        )),
    ])


def _predict_model(
    model_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    seed: int,
    tabfm_context_cap: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_train = pd.to_numeric(train[TARGET], errors="coerce")
    if model_name == "global_median":
        return np.repeat(float(y_train.median()), len(valid)), {}
    if model_name == "workload_median":
        return workload_median_prediction(train, valid), {}
    if model_name == "nearest_configuration":
        # This baseline remains a fixed deployment/workload analogue: it never
        # gains an advantage from the representation under evaluation.
        non_hardware = [
            column for column in features
            if column not in CAPABILITY_FEATURE_COLUMNS and column != MATCHED_PRECISION_FEATURE
        ]
        return nearest_configuration_prediction(train, valid, non_hardware), {}
    processor = FoldPreprocessor.fit(train, features)
    x_train, x_valid = processor.transform(train), processor.transform(valid)
    if model_name == "ridge":
        model = _ridge_pipeline(x_train)
        model.fit(x_train, y_train)
        return model.predict(x_valid), {"processor": processor}
    if model_name == "random_forest":
        model = _random_forest_pipeline(x_train, seed)
        model.fit(x_train, y_train)
        return model.predict(x_valid), {"processor": processor}
    if model_name == "catboost":
        if importlib.util.find_spec("catboost") is None:
            raise RuntimeError("catboost package unavailable")
        from catboost import CatBoostRegressor

        categorical = [column for column in x_train if not pd.api.types.is_numeric_dtype(x_train[column])]
        model = CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.05, loss_function="RMSE",
            random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False,
        )
        model.fit(x_train, y_train, cat_features=categorical)
        return model.predict(x_valid), {"processor": processor}
    if model_name == "tabfm":
        prediction, _importance, metadata = _tabfm_fit_predict(
            x_train, x_valid, y_train, seed, tabfm_context_cap
        )
        return prediction, {"processor": processor, **metadata}
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_fixed_folds(
    cohort: pd.DataFrame,
    representations: dict[str, list[str]],
    models: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    tabfm_context_cap: int | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    """Evaluate all requested model/representation combinations on precomputed folds."""

    results: dict[str, Any] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for representation, features in representations.items():
        results[representation] = {
            "feature_columns": features,
            "hardware_identity_included": "config_hardware" in features,
            "epoch_capabilities_included": any(column in features for column in CAPABILITY_FEATURE_COLUMNS),
            "models": {},
        }
        for model_name in models:
            available, availability = model_available(model_name) if model_name in {"catboost", "tabfm"} else (True, "available")
            fold_rows: list[dict[str, Any]] = []
            oof = np.full(len(cohort), np.nan, dtype=float)
            for fold_number, (train_index, valid_index) in enumerate(folds, 1):
                train, valid = cohort.iloc[train_index].copy(), cohort.iloc[valid_index].copy()
                overlap = set(train["config_id"].astype(str)) & set(valid["config_id"].astype(str))
                if overlap:
                    raise AssertionError("config_id overlap in fixed grouped folds")
                row: dict[str, Any] = {
                    "fold": fold_number, "train_rows": int(len(train)), "validation_rows": int(len(valid)),
                    "train_configs": int(train["config_id"].nunique(dropna=False)),
                    "validation_configs": int(valid["config_id"].nunique(dropna=False)),
                    "group_overlap": int(len(overlap)), "metrics": None, "failure": "", "runtime_seconds": 0.0,
                }
                if not available:
                    row["failure"] = availability
                    fold_rows.append(row)
                    continue
                started = time.perf_counter()
                try:
                    prediction, extra = _predict_model(
                        model_name, train, valid, features, SEED + fold_number, tabfm_context_cap
                    )
                    oof[valid_index] = prediction
                    row["metrics"] = prediction_metrics(valid[TARGET], prediction)
                    row.update({key: value for key, value in extra.items() if key != "processor"})
                except Exception as exc:  # Preserve a failed optional model as evidence.
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                row["runtime_seconds"] = float(time.perf_counter() - started)
                fold_rows.append(row)
            successful = [row for row in fold_rows if row["metrics"] is not None]
            pooled = prediction_metrics(cohort.loc[np.isfinite(oof), TARGET], oof[np.isfinite(oof)])
            results[representation]["models"][model_name] = {
                "available": bool(available), "availability": availability, "folds": fold_rows,
                "summary": _fold_metric_summary(successful, pooled) if successful else None,
                "runtime_seconds": float(sum(row["runtime_seconds"] for row in fold_rows)),
            }
            predictions[(representation, model_name)] = oof
    return results, predictions


def paired_error_comparison(
    cohort: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    predictions: dict[tuple[str, str], np.ndarray],
    model_name: str = "random_forest",
) -> dict[str, Any]:
    """Describe fixed-fold A/B/C differences without overstating significance."""

    reference = predictions.get(("A_identity_only", model_name))
    if reference is None or not np.isfinite(reference).any():
        return {"available": False, "reason": f"No successful {model_name} identity predictions."}
    truth = pd.to_numeric(cohort[TARGET], errors="coerce").to_numpy(dtype=float)
    comparisons: dict[str, Any] = {}
    for representation in (
        "B_identity_plus_capabilities", "C_capabilities_only", "D_capabilities_plus_matched_precision"
    ):
        candidate = predictions.get((representation, model_name))
        if candidate is None or not np.isfinite(candidate).any():
            comparisons[representation] = {"available": False}
            continue
        fold_rows: list[dict[str, Any]] = []
        for fold_number, (_train_index, valid_index) in enumerate(folds, 1):
            ref_error = np.abs(truth[valid_index] - reference[valid_index])
            candidate_error = np.abs(truth[valid_index] - candidate[valid_index])
            delta = candidate_error - ref_error
            fold_rows.append({
                "fold": fold_number,
                "candidate_minus_identity_mae": float(delta.mean()),
                "candidate_beats_identity_by_mae": bool(delta.mean() < 0),
                "candidate_better_row_fraction": float(np.mean(delta < 0)),
                "candidate_worse_row_fraction": float(np.mean(delta > 0)),
            })
        config_frame = pd.DataFrame({
            "config_id": cohort["config_id"].astype(str),
            "reference_abs_error": np.abs(truth - reference),
            "candidate_abs_error": np.abs(truth - candidate),
        }).groupby("config_id", as_index=False).mean(numeric_only=True)
        config_delta = config_frame["candidate_abs_error"] - config_frame["reference_abs_error"]
        comparisons[representation] = {
            "available": True,
            "folds": fold_rows,
            "folds_beating_identity": int(sum(row["candidate_beats_identity_by_mae"] for row in fold_rows)),
            "fold_count": int(len(fold_rows)),
            "config_level": {
                "configs": int(len(config_frame)),
                "median_candidate_minus_identity_mae": float(config_delta.median()),
                "mean_candidate_minus_identity_mae": float(config_delta.mean()),
                "candidate_better_config_fraction": float((config_delta < 0).mean()),
                "candidate_worse_config_fraction": float((config_delta > 0).mean()),
            },
            "interpretation": "Descriptive paired fold/config error comparison only; config groups are not independent statistical replicates, so no significance claim is made.",
        }
    return {"available": True, "model": model_name, "comparisons": comparisons}


def _fit_tree_for_importance(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Any:
    if model_name == "random_forest":
        model = _random_forest_pipeline(x_train, seed)
        model.fit(x_train, y_train)
        return model
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        categorical = [column for column in x_train if not pd.api.types.is_numeric_dtype(x_train[column])]
        model = CatBoostRegressor(
            iterations=100, depth=6, learning_rate=0.05, loss_function="RMSE",
            random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False,
        )
        model.fit(x_train, y_train, cat_features=categorical)
        return model
    raise ValueError(model_name)


def held_out_permutation_importance(
    cohort: pd.DataFrame,
    features: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> dict[str, Any]:
    """Permutation importance on each held-out fold, never on training rows."""

    if model_name == "catboost" and importlib.util.find_spec("catboost") is None:
        return {"available": False, "reason": "catboost package unavailable"}
    rows: list[dict[str, Any]] = []
    for fold_number, (train_index, valid_index) in enumerate(folds, 1):
        train, valid = cohort.iloc[train_index], cohort.iloc[valid_index]
        processor = FoldPreprocessor.fit(train, features)
        x_train, x_valid = processor.transform(train), processor.transform(valid)
        model = _fit_tree_for_importance(
            model_name, x_train, pd.to_numeric(train[TARGET], errors="coerce"), SEED + fold_number
        )
        scored = permutation_importance(
            model, x_valid, pd.to_numeric(valid[TARGET], errors="coerce"),
            n_repeats=5, random_state=SEED + fold_number, n_jobs=1,
            scoring="neg_mean_absolute_error",
        )
        for feature, mean, std in zip(x_valid.columns, scored.importances_mean, scored.importances_std, strict=True):
            rows.append({
                "fold": fold_number, "feature": feature,
                "importance_mean": float(mean), "importance_std": float(std),
            })
    summary = pd.DataFrame(rows).groupby("feature", as_index=False).agg(
        held_out_mae_degradation_mean=("importance_mean", "mean"),
        held_out_mae_degradation_std=("importance_mean", "std"),
        folds=("fold", "nunique"),
    ).fillna({"held_out_mae_degradation_std": 0.0}).sort_values(
        "held_out_mae_degradation_mean", ascending=False
    )
    capability_rows = summary.loc[
        summary["feature"].str.startswith("epoch_") | summary["feature"].eq(MATCHED_PRECISION_FEATURE)
    ]
    return {
        "available": True,
        "model": model_name,
        "representation_features": features,
        "method": "Permutation importance on each outer validation fold, scoring neg_mean_absolute_error; positive values indicate held-out MAE degradation after permutation.",
        "all_features": summary.to_dict("records"),
        "epoch_features": capability_rows.to_dict("records"),
        "causal_interpretation": "Predictive association only. Correlated physical descriptors and missingness patterns preclude causal attribution.",
    }


def _hardware_profiles(cohort: pd.DataFrame, capability_features: list[str]) -> pd.DataFrame:
    fields = ["config_hardware", "vendor", "epoch_hardware_name", *capability_features]
    present = [field for field in fields if field in cohort]
    profiles = cohort[present].groupby("config_hardware", as_index=False).first()
    if profiles["config_hardware"].duplicated().any():
        raise AssertionError("Hardware profiles must be one row per InferenceX hardware label.")
    return profiles


def _nearest_hardware_profile(
    profiles: pd.DataFrame,
    held_out_sku: str,
    capability_features: list[str],
) -> dict[str, Any]:
    test = profiles.loc[profiles["config_hardware"].astype(str).eq(str(held_out_sku))]
    train = profiles.loc[~profiles["config_hardware"].astype(str).eq(str(held_out_sku))].copy()
    if len(test) != 1:
        raise AssertionError(f"Expected exactly one profile for {held_out_sku}")
    test_row = test.iloc[0]
    distances: list[dict[str, Any]] = []
    outside: list[str] = []
    missing: list[str] = []
    for capability in capability_features:
        value = pd.to_numeric(pd.Series([test_row.get(capability)]), errors="coerce").iloc[0]
        training_values = pd.to_numeric(train[capability], errors="coerce").dropna()
        if pd.isna(value):
            missing.append(capability)
        elif len(training_values) and (value < training_values.min() or value > training_values.max()):
            outside.append(capability)
    for _, candidate in train.iterrows():
        squared: list[float] = []
        shared: list[str] = []
        for capability in capability_features:
            value = pd.to_numeric(pd.Series([test_row.get(capability)]), errors="coerce").iloc[0]
            other = pd.to_numeric(pd.Series([candidate.get(capability)]), errors="coerce").iloc[0]
            training_values = pd.to_numeric(train[capability], errors="coerce").dropna()
            scale = training_values.max() - training_values.min() if len(training_values) else np.nan
            if pd.isna(value) or pd.isna(other) or not np.isfinite(scale) or scale <= 0:
                continue
            squared.append(float(((value - other) / scale) ** 2))
            shared.append(capability)
        distances.append({
            "training_hardware": str(candidate["config_hardware"]),
            "normalized_distance": float(math.sqrt(np.mean(squared))) if squared else None,
            "shared_capability_dimensions": shared,
        })
    valid = [row for row in distances if row["normalized_distance"] is not None]
    nearest = min(valid, key=lambda row: row["normalized_distance"]) if valid else None
    vendor = str(test_row.get("vendor", "")) if pd.notna(test_row.get("vendor")) else None
    train_vendors = set(train.get("vendor", pd.Series(dtype=str)).dropna().astype(str))
    return {
        "nearest": nearest,
        "all_distances": sorted(valid, key=lambda row: row["normalized_distance"]),
        "dimensions_outside_training_range": outside,
        "outside_dimension_count": int(len(outside)),
        "missing_capability_dimensions": missing,
        "missing_dimension_count": int(len(missing)),
        "vendor": vendor,
        "vendor_novelty": bool(vendor and vendor not in train_vendors),
        "generation_novelty": "not assessed: Epoch fields do not provide a validated common generation taxonomy for these mapped SKUs.",
        "distance_definition": "Root mean squared difference after min-max scaling across training hardware profiles, calculated only on shared nonmissing dimensions with nonzero training range.",
    }


def nearest_hardware_workload_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    nearest_sku: str | None,
) -> np.ndarray:
    if not nearest_sku:
        return np.repeat(float(pd.to_numeric(train[TARGET], errors="coerce").median()), len(valid))
    analogue_rows = train.loc[train["config_hardware"].astype(str).eq(nearest_sku)]
    if analogue_rows.empty:
        return np.repeat(float(pd.to_numeric(train[TARGET], errors="coerce").median()), len(valid))
    return workload_median_prediction(analogue_rows, valid)


def held_out_hardware_transfer(
    cohort: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Transfer to each fully withheld accepted SKU with training-only baselines."""

    capability_features = metadata["capability_feature_columns"]
    c_features = metadata["representations"]["C_capabilities_only"]
    profiles = _hardware_profiles(cohort, capability_features)
    records: list[dict[str, Any]] = []
    for sku in sorted(cohort["config_hardware"].dropna().astype(str).unique()):
        test = cohort.loc[cohort["config_hardware"].astype(str).eq(sku)].copy()
        train = cohort.loc[~cohort["config_hardware"].astype(str).eq(sku)].copy()
        overlap = set(test["config_id"].astype(str)) & set(train["config_id"].astype(str))
        if overlap:
            raise AssertionError(f"Config leakage in held-out SKU {sku}")
        support = _nearest_hardware_profile(profiles, sku, capability_features)
        nearest_sku = (support.get("nearest") or {}).get("training_hardware")
        predictions: dict[str, np.ndarray] = {
            "global_training_median": np.repeat(float(pd.to_numeric(train[TARGET], errors="coerce").median()), len(test)),
            "workload_conditioned_training_median": workload_median_prediction(train, test),
            "nearest_hardware_capability_analogue": nearest_hardware_workload_prediction(train, test, nearest_sku),
        }
        failures: dict[str, str] = {}
        for model_name, label in (("ridge", "capability_ridge"), ("random_forest", "capability_random_forest")):
            try:
                prediction, _extra = _predict_model(model_name, train, test, c_features, SEED)
                predictions[label] = prediction
            except Exception as exc:
                failures[label] = f"{type(exc).__name__}: {exc}"
        records.append({
            "held_out_sku": sku,
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_configs": int(train["config_id"].nunique(dropna=False)),
            "test_configs": int(test["config_id"].nunique(dropna=False)),
            "group_overlap": int(len(overlap)),
            "test_target_distribution": target_distribution(test[TARGET]),
            "support": support,
            "metrics": {label: prediction_metrics(test[TARGET], prediction) for label, prediction in predictions.items()},
            "failures": failures,
            "protocol": "All rows of the held-out SKU, including preprocessing fit, are excluded from training. Hardware-analogue selection uses only physical capability descriptors from training hardware profiles; its target estimate uses training rows only.",
        })
    evaluated = [record for record in records if record["metrics"].get("capability_random_forest")]
    support_frame = pd.DataFrame([
        {
            "sku": row["held_out_sku"],
            "distance": (row["support"].get("nearest") or {}).get("normalized_distance"),
            "outside_dimensions": row["support"].get("outside_dimension_count"),
            "mdape": row["metrics"].get("capability_random_forest", {}).get("median_absolute_percentage_error"),
            "mae": row["metrics"].get("capability_random_forest", {}).get("mae"),
        }
        for row in evaluated
    ])
    correlation: dict[str, Any] = {"available": False}
    if len(support_frame) >= 3:
        correlation = {
            "available": True,
            "rows": int(len(support_frame)),
            "spearman_distance_vs_mdape": support_frame["distance"].corr(support_frame["mdape"], method="spearman"),
            "spearman_distance_vs_mae": support_frame["distance"].corr(support_frame["mae"], method="spearman"),
            "spearman_outside_dimensions_vs_mdape": support_frame["outside_dimensions"].corr(support_frame["mdape"], method="spearman"),
            "interpretation": "Seven SKU-level points are descriptive only; no statistical inference or support threshold is claimed.",
        }
    return {
        "study": "held_out_hardware_sku_transfer",
        "target": TARGET,
        "representation": "C_capabilities_only",
        "features": c_features,
        "records": records,
        "support_error_relationship": correlation,
        "claim_boundary": "Observed held-out-SKU transfer, not unreleased-hardware prediction.",
    }


def missingness_proxy_diagnostic(cohort: pd.DataFrame, capability_features: list[str]) -> dict[str, Any]:
    profiles = _hardware_profiles(cohort, capability_features)
    per_feature = []
    for column in capability_features:
        states = profiles[["config_hardware", column]].copy()
        states["missing"] = states[column].isna()
        per_feature.append({
            "capability": column,
            "hardware_profiles_missing": int(states["missing"].sum()),
            "hardware_profiles_present": int((~states["missing"]).sum()),
            "missing_by_hardware": states.set_index("config_hardware")["missing"].to_dict(),
        })
    pattern_columns = [column for column in capability_features if column in profiles]
    patterns = profiles.set_index("config_hardware")[pattern_columns].isna().astype(int)
    return {
        "hardware_profiles": int(len(profiles)),
        "distinct_missingness_patterns": int(patterns.astype(str).agg("|".join, axis=1).nunique()),
        "patterns": patterns.reset_index().to_dict("records"),
        "per_feature": per_feature,
        "risk": "Capability missingness can act as a vendor/generation identity proxy. Fold-local indicators preserve missingness honestly but do not make this proxy causal or transferable.",
    }


def _best_capability_tree(results: dict[str, Any]) -> tuple[str | None, str | None]:
    candidates: list[tuple[float, str, str]] = []
    for representation in (
        "B_identity_plus_capabilities", "C_capabilities_only", "D_capabilities_plus_matched_precision"
    ):
        for model_name in ("random_forest", "catboost"):
            model = results.get(representation, {}).get("models", {}).get(model_name, {})
            summary = model.get("summary") or {}
            r2 = summary.get("pooled_oof", {}).get("r2")
            if r2 is not None and np.isfinite(r2):
                candidates.append((float(r2), representation, model_name))
    return (max(candidates)[1], max(candidates)[2]) if candidates else (None, None)


def reproduce_historical_tabfm_result() -> dict[str, Any]:
    """Run the exact selected full-context TabFM helper on the legacy source."""

    available, reason = model_available("tabfm")
    if not available:
        return {
            "attempted": True, "available": False, "reason": reason,
            "command_contract": ".venv-tabfm/bin/python scripts/run_hardware_validation.py --include-tabfm --historical-only",
        }
    from modeling.tail_diagnostics import tabfm_oof_diagnostics

    _joined, historical, source = _source_frame(HISTORICAL_SOURCE_DIR)
    features = list(_prior_tabfm_contract().get("feature_columns", []))
    if not features:
        raise RuntimeError("The recorded historical TabFM feature contract is unavailable.")
    result = tabfm_oof_diagnostics(
        historical, features, TARGET, 4096, SEED, n_splits=N_SPLITS,
        context_cap=None, context_strategy="random",
    )
    return {
        "attempted": True, "available": True,
        "source": source,
        "cohort_rows": int(result["preparation"]["usable_rows"]),
        "configs": int(prepare_model_frame(historical, features, TARGET, 4096, SEED)[0]["config_id"].nunique(dropna=False)),
        "features": features,
        "protocol": "Exact modeling.tail_diagnostics.tabfm_oof_diagnostics full-fold-context protocol: GroupKFold by config_id, max_rows=4096, seed=42, raw target, no context cap.",
        "recorded_reference": _prior_tabfm_contract().get("selected_full_context_result", {}),
        "result": result,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            **{
                package: (importlib.metadata.version(package) if importlib.util.find_spec(module) else None)
                for package, module in (("tabfm", "tabfm"), ("pandas", "pandas"), ("numpy", "numpy"),
                                        ("scikit-learn", "sklearn"), ("torch", "torch"))
            },
        },
        "command_contract": ".venv-tabfm/bin/python scripts/run_hardware_validation.py --include-tabfm --historical-only",
    }


def run_primary_tabfm_ablation(context_cap: int | None = None) -> dict[str, Any]:
    """Run TabFM A/B/C/D on the exact full Epoch-resolved cohort and folds.

    ``context_cap=None`` means all outer-training rows are context.  A positive
    cap is a documented, deterministic training-context budget shared by every
    representation; it is never chosen using validation targets.
    """

    available, reason = model_available("tabfm")
    if not available:
        return {
            "study": "primary_epoch_resolved_tabfm_ablation", "available": False,
            "reason": reason, "context_cap": context_cap,
        }
    cohort, metadata = primary_epoch_resolved_cohort()
    folds = deterministic_grouped_folds(cohort, N_SPLITS)
    results, _predictions = evaluate_fixed_folds(
        cohort, metadata["representations"], ["tabfm"], folds,
        tabfm_context_cap=context_cap,
    )
    return {
        "study": "primary_epoch_resolved_tabfm_ablation", "available": True,
        "target": TARGET, "primary_cohort": metadata,
        "context_cap": context_cap,
        "context_policy": "Full fold-local context" if context_cap is None else f"Deterministic maximum {context_cap} training rows per outer fold",
        "results": results,
        "claim_boundary": "TabFM results are a companion controlled ablation. They use the same rows, target, features other than hardware representation, and grouped folds as the conventional benchmark.",
    }


def run_controlled_validation(
    *,
    include_tabfm: bool = False,
    reproduce_historical_tabfm: bool = False,
) -> dict[str, Any]:
    """Run reconciliation, fixed-fold ladder/ablation, transfer, and diagnostics."""

    reconciliation, historical, _resolved = reconcile_cohorts()
    cohort, metadata = primary_epoch_resolved_cohort()
    # Build one fold partition once; all conventional models and representations reuse it.
    folds = deterministic_grouped_folds(cohort, N_SPLITS)
    models = [
        "global_median", "workload_median", "nearest_configuration",
        "ridge", "random_forest", "catboost",
    ]
    if include_tabfm:
        models.append("tabfm")
    results, predictions = evaluate_fixed_folds(cohort, metadata["representations"], models, folds)
    paired = paired_error_comparison(cohort, folds, predictions, "random_forest")
    best_representation, best_tree = _best_capability_tree(results)
    importance = (
        held_out_permutation_importance(
            cohort, metadata["representations"][best_representation], folds, best_tree
        )
        if best_representation and best_tree else {"available": False, "reason": "No successful capability tree model."}
    )
    transfer = held_out_hardware_transfer(cohort, metadata)
    historical_reproduction: dict[str, Any] = {
        "attempted": bool(reproduce_historical_tabfm),
        "available": False,
        "reason": "Not requested in this run; use the TabFM environment and --reproduce-historical-tabfm.",
    }
    if reproduce_historical_tabfm:
        historical_reproduction = reproduce_historical_tabfm_result()
    validation = {
        "study": "controlled_epoch_hardware_validation",
        "target": TARGET,
        "scope": "Epoch physical hardware descriptors only; no Epoch model/benchmark, cluster, datacenter, or Artificial Analysis fields.",
        "reconciliation": reconciliation,
        "primary_cohort": metadata,
        "protocol": {
            "models": models,
            "random_seed": SEED,
            "folds": [{
                "fold": index + 1, "train_rows": int(len(train)), "validation_rows": int(len(valid)),
                "train_configs": int(cohort.iloc[train]["config_id"].nunique(dropna=False)),
                "validation_configs": int(cohort.iloc[valid]["config_id"].nunique(dropna=False)),
            } for index, (train, valid) in enumerate(folds)],
            "hyperparameters": {"random_forest": RF_SETTINGS, "ridge_alpha": RIDGE_ALPHA,
                                "catboost": {"iterations": 100, "depth": 6, "learning_rate": 0.05}},
            "relative_error_caution": "Raw APE is reported but unstable in the throughput lower tail; median absolute log-ratio is its scale-stable companion.",
        },
        "results": results,
        "paired_identity_comparison": paired,
        "feature_importance": importance,
        "missingness_proxy_diagnostic": missingness_proxy_diagnostic(cohort, metadata["capability_feature_columns"]),
        "held_out_hardware_transfer": transfer,
        "historical_tabfm_reproduction": historical_reproduction,
        "software_versions": {
            package: (importlib.metadata.version(package) if importlib.util.find_spec(module) else None)
            for package, module in (("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn"),
                                    ("catboost", "catboost"), ("tabfm", "tabfm"))
        },
    }
    return validation


def write_controlled_validation_outputs(validation: dict[str, Any]) -> None:
    """Persist compact machine-readable artifacts in the approved derived location."""

    write_validation_json(RECONCILIATION_ROOT / "cohort_reconciliation.json", validation["reconciliation"])
    write_validation_json(VALIDATION_ROOT / "controlled_hardware_validation.json", validation)
    write_validation_json(VALIDATION_ROOT / "held_out_hardware_transfer.json", validation["held_out_hardware_transfer"])
    write_validation_json(VALIDATION_ROOT / "feature_importance.json", validation["feature_importance"])
