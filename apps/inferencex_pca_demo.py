from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_DIR = "inferencex-pca-data"
DEFAULT_JSON_DUMP_DIR = "inferencex-dump-2026-06-29"
REQUIRED_CSV_FILES = ("benchmark_results.csv", "configs.csv")
REQUIRED_JSON_FILES = ("benchmark_results.json", "configs.json")
ANALYSIS_UNIT_OPTIONS = (
    "Raw benchmark rows",
    "Latest row per config/workload/concurrency",
    "Median aggregate per config/workload/concurrency",
    "One row per config",
)
METRIC_NAME_TERMS = (
    "metric",
    "metrics_",
    "throughput",
    "latency",
    "token",
    "tok",
    "qps",
    "rps",
    "score",
    "time",
    "tput",
    "ttft",
    "tpot",
    "itl",
    "e2el",
    "intvty",
)
METADATA_NAME_TERMS = (
    "url",
    "log",
    "date",
    "timestamp",
    "path",
    "raw",
    "error",
    "note",
    "description",
    "image",
)
CONFIG_NAME_TERMS = (
    "model",
    "hardware",
    "framework",
    "precision",
    "benchmark_type",
    "isl",
    "osl",
    "conc",
    "workers",
    "disagg",
    "is_multinode",
    "spec_method",
    "gpu",
    "prefill",
    "decode",
    "tp",
    "ep",
    "dp_attention",
    "num_workers",
)
DEFAULT_CATEGORICAL_FEATURES = (
    "benchmark_type",
    "config_hardware",
    "config_framework",
    "config_model",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
)


st.set_page_config(page_title="InferenceX PCA Demo", layout="wide")


def read_records_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]

    if isinstance(raw, dict):
        for key in ("rows", "data", "items", "results"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in raw.values()):
            return list(raw.values())

    raise ValueError(f"{path.name} is not a JSON record list or record mapping")


def normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.json_normalize(records, sep="_")
    frame.columns = [str(column).replace(".", "_") for column in frame.columns]
    return frame


def prefix_config_columns(configs: pd.DataFrame) -> pd.DataFrame:
    if "id" in configs.columns:
        renamed = configs.rename(columns={"id": "config_id"}).copy()
    elif "config_id" in configs.columns:
        renamed = configs.copy()
    else:
        return configs
    renamed.columns = [
        column
        if column == "config_id" or column.startswith("config_")
        else f"config_{column}"
        for column in renamed.columns
    ]
    return renamed


def join_benchmarks_configs(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
) -> pd.DataFrame:
    if "id" in benchmarks.columns:
        benchmarks = benchmarks.rename(columns={"id": "benchmark_id"})

    if "config_id" not in benchmarks.columns or "config_id" not in configs.columns:
        return benchmarks.copy()

    overlapping = [
        column
        for column in configs.columns
        if column != "config_id" and column in benchmarks.columns
    ]
    configs_for_join = configs.drop(columns=overlapping)
    return benchmarks.merge(configs_for_join, on="config_id", how="left", validate="many_to_one")


def resolve_data_dir(data_dir_text: str) -> Path:
    data_dir = Path(data_dir_text).expanduser()
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    return data_dir


def data_source_status(data_dir_text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_dir = resolve_data_dir(data_dir_text)
    json_dir = resolve_data_dir(DEFAULT_JSON_DUMP_DIR)
    rows = []
    for mode, directory, files in (
        ("CSV", csv_dir, REQUIRED_CSV_FILES),
        ("JSON fallback", json_dir, REQUIRED_JSON_FILES),
    ):
        for file_name in files:
            path = directory / file_name
            rows.append(
                {
                    "mode": mode,
                    "file": file_name,
                    "found": path.exists(),
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else np.nan,
                }
            )
    status = pd.DataFrame(rows)
    csv_ready = bool(status[status["mode"] == "CSV"]["found"].all())
    json_ready = bool(status[status["mode"] == "JSON fallback"]["found"].all())
    active_mode = "CSV" if csv_ready else "JSON fallback" if json_ready else "missing"
    active_dir = csv_dir if active_mode == "CSV" else json_dir if active_mode == "JSON fallback" else csv_dir
    return status, {
        "csv_ready": csv_ready,
        "json_ready": json_ready,
        "active_mode": active_mode,
        "active_dir": active_dir,
    }


@st.cache_data(show_spinner="Loading benchmark/config data")
def load_joined_data(data_dir_text: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    status, source_info = data_source_status(data_dir_text)
    active_mode = source_info["active_mode"]

    if active_mode == "CSV":
        data_dir = source_info["active_dir"]
        benchmarks = pd.read_csv(data_dir / "benchmark_results.csv", low_memory=False)
        configs = prefix_config_columns(pd.read_csv(data_dir / "configs.csv", low_memory=False))
        joined = join_benchmarks_configs(benchmarks, configs)
        return benchmarks, configs, joined, source_info

    if active_mode == "JSON fallback":
        dump_dir = source_info["active_dir"]
        benchmarks = normalize_records(read_records_json(dump_dir / "benchmark_results.json"))
        configs = prefix_config_columns(normalize_records(read_records_json(dump_dir / "configs.json")))
        joined = join_benchmarks_configs(benchmarks, configs)
        return benchmarks, configs, joined, source_info

    missing = status[~status["found"]][["mode", "file", "path"]]
    raise FileNotFoundError(
        "Missing required CSV files and JSON fallback files:\n"
        + missing.to_string(index=False)
    )


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() > 1
    ]


def is_metric_column(col: str) -> bool:
    lowered = col.lower()
    return lowered.startswith("metrics_") or any(term in lowered for term in METRIC_NAME_TERMS)


def is_metadata_column(col: str) -> bool:
    lowered = col.lower()
    if lowered in {"id", "config_id", "benchmark_id", "workflow_run_id", "server_log_id"}:
        return True
    if lowered.endswith("_id"):
        return True
    return any(term in lowered for term in METADATA_NAME_TERMS)


def is_config_column(col: str) -> bool:
    lowered = col.lower()
    if is_metric_column(lowered) or is_metadata_column(lowered):
        return False
    return any(term in lowered for term in CONFIG_NAME_TERMS)


def metric_like_numeric_columns(frame: pd.DataFrame) -> list[str]:
    numeric = numeric_columns(frame)
    metric_like = [column for column in numeric if is_metric_column(column)]
    return prioritized_metric_columns(metric_like or numeric)


def metric_priority(column: str) -> tuple[int, int, str]:
    lowered = column.lower()
    statistic_rank = 0
    if "_p99_9" in lowered:
        statistic_rank = 3
    elif "_p99_" in lowered or lowered.endswith("_p99"):
        statistic_rank = 0
    elif "mean" in lowered:
        statistic_rank = 1
    elif "median" in lowered:
        statistic_rank = 2
    elif "p99" in lowered:
        statistic_rank = 3
    else:
        statistic_rank = 4

    metric_rank = 20
    for rank, term in enumerate(("itl", "tpot", "ttft", "e2el", "throughput", "tput")):
        if term in lowered:
            metric_rank = rank
            break
    return metric_rank, statistic_rank, lowered


def prioritized_metric_columns(columns: list[str]) -> list[str]:
    return sorted(columns, key=metric_priority)


def config_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in numeric_columns(frame) if is_config_column(column)]


def config_categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in categorical_columns(frame) if is_config_column(column)]


def config_workload_grouping_keys(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("config_id", "benchmark_type", "isl", "osl", "conc")
        if column in frame.columns
    ]


def timestamp_sort_column(frame: pd.DataFrame) -> str | None:
    candidates = (
        "date",
        "created_at",
        "createdAt",
        "run_created_at",
        "workflow_run_created_at",
        "started_at",
        "updated_at",
        "timestamp",
    )
    return next((column for column in candidates if column in frame.columns), None)


def safe_hashable_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)
    return value


def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.map(safe_hashable_value).nunique(dropna=True))


def categorical_columns(frame: pd.DataFrame, max_cardinality: int = 80) -> list[str]:
    categorical: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, CategoricalDtype)
        ):
            sample = series.dropna().head(100)
            if sample.map(lambda value: isinstance(value, (dict, list, set, tuple))).any():
                continue
            unique_count = safe_nunique(series)
            if 1 < unique_count <= max_cardinality:
                categorical.append(column)
    return categorical


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    clean_max = max(1, min(max_rows, len(frame)))
    if len(frame) <= clean_max:
        return frame.copy()
    return frame.sample(n=clean_max, random_state=seed).copy()


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def mode_or_first(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    counts = non_null.map(safe_hashable_value).value_counts()
    if counts.empty:
        return non_null.iloc[0]
    top_value = counts.index[0]
    for value in non_null:
        if safe_hashable_value(value) == top_value:
            return value
    return non_null.iloc[0]


def numeric_group_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    if safe_nunique(non_null) <= 1:
        return non_null.iloc[0]
    return pd.to_numeric(non_null, errors="coerce").median()


def aggregate_analysis_frame(frame: pd.DataFrame, grouping_keys: list[str]) -> pd.DataFrame:
    if not grouping_keys:
        result = frame.copy()
        result["row_count_in_group"] = 1
        return result

    source = frame.drop(columns=["row_count_in_group"], errors="ignore").copy()
    group = source.groupby(grouping_keys, dropna=False, sort=False)
    agg_map: dict[str, Any] = {}
    for column in source.columns:
        if column in grouping_keys:
            continue
        if pd.api.types.is_bool_dtype(source[column]):
            agg_map[column] = mode_or_first
        elif pd.api.types.is_numeric_dtype(source[column]):
            agg_map[column] = "median" if is_metric_column(column) else numeric_group_value
        elif is_metadata_column(column) or column in {"image", "error"}:
            agg_map[column] = first_non_null
        else:
            agg_map[column] = mode_or_first

    aggregated = group.agg(agg_map).reset_index()
    row_counts = group.size().rename("row_count_in_group").reset_index()
    return aggregated.merge(row_counts, on=grouping_keys, how="left")


def build_analysis_frame(
    joined: pd.DataFrame,
    analysis_unit: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "analysis_unit": analysis_unit,
        "raw_row_count": len(joined),
        "analysis_row_count": len(joined),
        "grouping_keys": [],
        "timestamp_column": "",
        "warning": "",
    }

    if analysis_unit == "Raw benchmark rows":
        result = joined.copy()
        result["row_count_in_group"] = 1
        return result, metadata

    if analysis_unit == "One row per config":
        grouping_keys = ["config_id"] if "config_id" in joined.columns else []
        if not grouping_keys:
            metadata["warning"] = "config_id is unavailable; falling back to raw rows."
            result = joined.copy()
            result["row_count_in_group"] = 1
            return result, metadata
        result = aggregate_analysis_frame(joined, grouping_keys)
        metadata.update(
            {
                "analysis_row_count": len(result),
                "grouping_keys": grouping_keys,
            }
        )
        return result, metadata

    grouping_keys = config_workload_grouping_keys(joined)
    if not grouping_keys:
        metadata["warning"] = "No config/workload grouping keys are available; falling back to raw rows."
        result = joined.copy()
        result["row_count_in_group"] = 1
        return result, metadata

    if analysis_unit == "Median aggregate per config/workload/concurrency":
        result = aggregate_analysis_frame(joined, grouping_keys)
        metadata.update(
            {
                "analysis_row_count": len(result),
                "grouping_keys": grouping_keys,
            }
        )
        return result, metadata

    source = joined.drop(columns=["row_count_in_group"], errors="ignore").copy()
    source["_original_order"] = np.arange(len(source))
    timestamp_column = timestamp_sort_column(source)
    sort_columns = grouping_keys + ["_original_order"]
    if timestamp_column:
        source["_analysis_timestamp"] = pd.to_datetime(source[timestamp_column], errors="coerce")
        sort_columns = grouping_keys + ["_analysis_timestamp", "_original_order"]
        metadata["timestamp_column"] = timestamp_column
    else:
        metadata["warning"] = (
            "No reliable timestamp field was found; latest-row selection falls back to "
            "the last row after stable sorting."
        )

    group_sizes = source.groupby(grouping_keys, dropna=False, sort=False).size().rename("row_count_in_group")
    latest = (
        source.sort_values(sort_columns, kind="mergesort")
        .groupby(grouping_keys, dropna=False, sort=False)
        .tail(1)
        .drop(columns=["_original_order", "_analysis_timestamp"], errors="ignore")
    )
    latest = latest.merge(group_sizes.reset_index(), on=grouping_keys, how="left")
    metadata.update(
        {
            "analysis_row_count": len(latest),
            "grouping_keys": grouping_keys,
        }
    )
    return latest.reset_index(drop=True), metadata


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=30,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def coerce_model_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str | None = None,
) -> pd.DataFrame:
    columns = feature_columns + ([target_column] if target_column else [])
    columns = [column for column in columns if column and column in frame.columns]
    model_frame = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    return model_frame.dropna(axis=0, how="all", subset=feature_columns)


def split_features(frame: pd.DataFrame, selected_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in selected_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in selected_columns if column not in numeric]
    return numeric, categorical


def clean_feature_label(label: str) -> str:
    for prefix in ("num__", "cat__"):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def unique_preserve_order(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        unique.append(column)
    return unique


def normalize_feature_groups(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str], list[str]]:
    numeric_unique = unique_preserve_order(numeric_features)
    categorical_unique = unique_preserve_order(categorical_features)
    numeric_set = set(numeric_unique)
    overlap = [column for column in categorical_unique if column in numeric_set]
    categorical_unique = [column for column in categorical_unique if column not in numeric_set]
    return numeric_unique, categorical_unique, overlap


def source_feature_for_encoded(
    encoded_feature: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    if encoded_feature in numeric_features:
        return encoded_feature

    for column in sorted(categorical_features, key=len, reverse=True):
        if encoded_feature == column or encoded_feature.startswith(f"{column}_"):
            return column

    return encoded_feature


def build_pca_loading_details(
    pca: PCA,
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_idx, encoded_feature in enumerate(feature_names):
        source_feature = source_feature_for_encoded(
            encoded_feature,
            numeric_features,
            categorical_features,
        )
        weighted_contribution = float(
            np.sum((pca.components_[:, feature_idx] ** 2) * pca.explained_variance_ratio_)
        )
        row: dict[str, Any] = {
            "encoded_feature": encoded_feature,
            "source_feature": source_feature,
            "weighted_contribution": weighted_contribution,
        }
        for component_idx in range(len(pca.components_)):
            row[f"PC{component_idx + 1}_loading"] = pca.components_[component_idx, feature_idx]
        rows.append(row)

    details = pd.DataFrame(rows)
    total = details["weighted_contribution"].sum()
    details["contribution_share"] = (
        details["weighted_contribution"] / total if total > 0 else 0.0
    )
    return details.sort_values("weighted_contribution", ascending=False)


def original_feature_contributions(loading_details: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        loading_details.groupby("source_feature", as_index=False)["weighted_contribution"]
        .sum()
        .sort_values("weighted_contribution", ascending=False)
    )
    total = grouped["weighted_contribution"].sum()
    grouped["contribution_share"] = (
        grouped["weighted_contribution"] / total if total > 0 else 0.0
    )
    return grouped


def format_loading_list(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "_No loadings available._"
    return "\n".join(
        f"- `{row.encoded_feature}`: {row.loading:+.3f}"
        for row in rows.itertuples(index=False)
    )


def interpret_component(component_loadings: pd.DataFrame) -> str:
    group_strength = (
        component_loadings.assign(abs_loading=component_loadings["loading"].abs())
        .groupby("source_feature", as_index=False)["abs_loading"]
        .sum()
        .sort_values("abs_loading", ascending=False)
        .head(3)
    )
    if group_strength.empty:
        return "This component has no dominant feature group."

    groups = ", ".join(group_strength["source_feature"].tolist())
    return (
        "This component mainly separates runs along "
        f"{groups}. Check the positive and negative loadings to see which values "
        "sit on opposite sides of the axis."
    )


def build_component_interpretations(
    pca: PCA,
    loading_details: pd.DataFrame,
) -> pd.DataFrame:
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    rows: list[dict[str, Any]] = []
    for component_idx in range(min(4, len(pca.components_))):
        component_name = f"PC{component_idx + 1}"
        component_loadings = loading_details[
            ["encoded_feature", "source_feature", f"{component_name}_loading"]
        ].rename(columns={f"{component_name}_loading": "loading"})
        positive = (
            component_loadings[component_loadings["loading"] > 0]
            .sort_values("loading", ascending=False)
            .head(8)
        )
        negative = (
            component_loadings[component_loadings["loading"] < 0]
            .sort_values("loading", ascending=True)
            .head(8)
        )
        absolute = (
            component_loadings.assign(abs_loading=component_loadings["loading"].abs())
            .sort_values("abs_loading", ascending=False)
            .head(8)
        )
        dominant_groups = (
            component_loadings.assign(abs_loading=component_loadings["loading"].abs())
            .groupby("source_feature", as_index=False)["abs_loading"]
            .sum()
            .sort_values("abs_loading", ascending=False)
            .head(3)["source_feature"]
            .tolist()
        )
        rows.append(
            {
                "component": component_name,
                "explained_variance_ratio": pca.explained_variance_ratio_[component_idx],
                "cumulative_explained_variance": cumulative_variance[component_idx],
                "dominant_feature_groups": ", ".join(dominant_groups),
                "interpretation": interpret_component(component_loadings),
                "top_positive": "; ".join(
                    f"{row.encoded_feature} ({row.loading:+.3f})"
                    for row in positive.itertuples(index=False)
                ),
                "top_negative": "; ".join(
                    f"{row.encoded_feature} ({row.loading:+.3f})"
                    for row in negative.itertuples(index=False)
                ),
                "top_absolute": "; ".join(
                    f"{row.encoded_feature} ({row.loading:+.3f})"
                    for row in absolute.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_pc_target_correlations(
    coords: np.ndarray,
    row_index: pd.Index,
    joined: pd.DataFrame,
    target_metric: str,
) -> pd.DataFrame:
    target_values = pd.to_numeric(joined.loc[row_index, target_metric], errors="coerce")
    pc_count = min(4, coords.shape[1])
    pc_frame = pd.DataFrame(
        coords[:, :pc_count],
        columns=[f"PC{idx + 1}" for idx in range(pc_count)],
        index=row_index,
    )
    correlation_rows: list[dict[str, Any]] = []
    for component_name in pc_frame.columns:
        pair = pd.concat(
            [pc_frame[component_name], target_values.rename(target_metric)],
            axis=1,
        ).dropna()
        if len(pair) < 3 or pair[component_name].std() == 0 or pair[target_metric].std() == 0:
            correlation = np.nan
        else:
            correlation = pair[component_name].corr(pair[target_metric])
        correlation_rows.append(
            {
                "component": component_name,
                "target_metric": target_metric,
                "correlation": correlation,
                "abs_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
                "rows": len(pair),
            }
        )
    return pd.DataFrame(correlation_rows).sort_values(
        "abs_correlation",
        ascending=False,
        na_position="last",
    )


def compact_list(values: list[str], max_items: int = 3) -> str:
    if not values:
        return "none identified"
    return ", ".join(values[:max_items])


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame.empty:
        return "_No rows available._"

    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_findings_markdown(
    dataset_summary: dict[str, Any],
    pca_analysis: dict[str, Any] | None,
    target_analysis: dict[str, Any] | None,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# InferenceX PCA Findings Summary",
        "",
        f"Generated: {timestamp}",
        "",
        "## Dataset Summary",
        "",
        f"- Total benchmark rows loaded: {dataset_summary.get('benchmark_rows', 0):,}",
        f"- Joined rows loaded: {dataset_summary.get('joined_rows', 0):,}",
        f"- Analysis unit: {dataset_summary.get('analysis_unit', 'Raw benchmark rows')}",
        f"- Raw rows: {dataset_summary.get('raw_row_count', dataset_summary.get('joined_rows', 0)):,}",
        f"- Analysis rows: {dataset_summary.get('analysis_row_count', dataset_summary.get('joined_rows', 0)):,}",
        f"- Grouping keys: {', '.join(dataset_summary.get('grouping_keys', [])) or 'none'}",
        "",
    ]

    if not pca_analysis:
        lines.extend(["## PCA Summary", "", "PCA has not produced results yet."])
        return "\n".join(lines)

    source_contributions = pca_analysis["source_contributions"]
    encoded_contributions = pca_analysis["encoded_contributions"]
    component_interpretations = pca_analysis["component_interpretations"]
    pc_target_correlations = pca_analysis["pc_target_correlations"]

    lines.extend(
        [
            "## PCA Summary",
            "",
            f"- Sampled rows used: {pca_analysis['sampled_rows']:,}",
            f"- PCA input feature count: {pca_analysis['input_feature_count']:,}",
            f"- Selected PC correlation target: `{pca_analysis['target_metric']}`",
            "",
            "### Top Original Feature Groups",
            "",
            dataframe_to_markdown(source_contributions, 10),
            "",
            "### Top Encoded Feature Contributions",
            "",
            dataframe_to_markdown(encoded_contributions, 10),
            "",
            "## PC1-PC4 Interpretations",
            "",
            dataframe_to_markdown(component_interpretations, 4),
            "",
            "## PC vs Target Correlations",
            "",
            dataframe_to_markdown(pc_target_correlations, 4),
            "",
        ]
    )

    target_importance = (
        target_analysis.get("importance_frame", pd.DataFrame())
        if target_analysis
        else pd.DataFrame()
    )
    target_metric = (
        target_analysis.get("target_metric", pca_analysis["target_metric"])
        if target_analysis
        else pca_analysis["target_metric"]
    )
    lines.extend(
        [
            "## Target-Aware Feature Importance",
            "",
            f"Selected target metric: `{target_metric}`",
            "",
            dataframe_to_markdown(target_importance, 10),
            "",
            "## Final Implications",
            "",
            "Use PCA to understand how benchmark configurations cluster and vary. Use the "
            "supervised target-aware layer to identify which setup features predict the "
            "selected performance metric. For inference infrastructure and datacenter asset "
            "analysis, features that appear in both lists deserve deeper engineering and "
            "economic investigation because they are both structurally common sources of "
            "variation and performance-relevant in the selected model.",
            "",
            "## Limitations",
            "",
            "- PCA identifies variance structure, not feature value.",
            "- Target-aware importance is predictive, not causal.",
            "- PC-target correlation is descriptive and can be confounded.",
            "- Results depend on the selected target metric and sampled rows.",
            "- Repeated benchmark rows can overweight frequently tested configurations. "
            "Aggregated analysis reduces this bias.",
            "- This app reads local JSON dumps only and intentionally skips giant log/sample files.",
        ]
    )
    return "\n".join(lines)


PLAIN_ENGLISH_MEANINGS = {
    "isl": "Input sequence length in tokens.",
    "osl": "Output sequence length in tokens.",
    "conc": "Concurrency / concurrent requests.",
    "benchmark_type": "Workload or benchmark test type.",
    "image": "Container image used for the benchmark run.",
    "config_model": "Model key used by the benchmark configuration.",
    "config_hardware": "GPU or system target.",
    "config_framework": "Inference serving framework.",
    "config_precision": "Numerical precision.",
    "config_spec_method": "Speculative decoding / speculative method.",
    "config_disagg": "Whether prefill and decode are separated.",
    "config_is_multinode": "Whether the run spans multiple nodes.",
    "config_prefill_tp": "Prefill tensor parallelism.",
    "config_prefill_ep": "Prefill expert parallelism.",
    "config_prefill_dp_attention": "Prefill DP-attention flag.",
    "config_prefill_num_workers": "Prefill worker count.",
    "config_decode_tp": "Decode tensor parallelism.",
    "config_decode_ep": "Decode expert parallelism.",
    "config_decode_dp_attention": "Decode DP-attention flag.",
    "config_decode_num_workers": "Decode worker count.",
    "config_num_prefill_gpu": "GPUs allocated to prefill.",
    "config_num_decode_gpu": "GPUs allocated to decode.",
}


def source_table_for_column(column: str) -> str:
    if column.startswith("config_"):
        return "configs"
    if column.startswith("metrics_"):
        return "benchmark_results.metrics"
    return "benchmark_results"


def feature_family_for_column(column: str) -> str:
    lowered = column.lower()
    if column in {"isl", "osl", "conc", "benchmark_type"}:
        return "Workload shape"
    if is_metric_column(column):
        if any(term in lowered for term in ("ttft", "tpot", "itl", "e2el", "latency")):
            return "Outcome metrics: latency"
        if any(term in lowered for term in ("throughput", "tput", "qps", "rps", "tok")):
            return "Outcome metrics: throughput"
        if any(term in lowered for term in ("joules", "energy")):
            return "Outcome metrics: energy"
        if "power" in lowered:
            return "Telemetry: power"
        if "util" in lowered:
            return "Telemetry: utilization"
        if "mem" in lowered:
            return "Telemetry: memory"
        return "Outcome metrics"
    if is_metadata_column(column) or column in {"image", "error"}:
        return "Provenance/audit"
    if column.startswith("config_"):
        return "Configuration"
    return "Metadata"


def role_for_column(column: str) -> str:
    if is_metric_column(column):
        return "outcome metric"
    if column in {"isl", "osl", "conc", "benchmark_type"}:
        return "workload"
    if is_metadata_column(column) or column in {"image", "error"}:
        return "provenance"
    if is_config_column(column):
        return "input/config"
    return "metadata"


def modeling_use_for_column(column: str, frame: pd.DataFrame) -> str:
    unique_count = safe_nunique(frame[column])
    if is_metadata_column(column) or column in {"image", "error"}:
        return "exclude"
    if is_metric_column(column):
        return "target candidate / color overlay"
    if unique_count <= 1:
        return "exclude"
    if unique_count > 80 and not pd.api.types.is_numeric_dtype(frame[column]):
        return "exclude"
    if is_config_column(column) or column in {"isl", "osl", "conc", "benchmark_type"}:
        return "PCA input"
    return "color overlay"


def metric_kind_direction_use(column: str) -> tuple[str, str, str]:
    lowered = column.lower()
    if any(term in lowered for term in ("ttft", "tpot", "itl", "e2el", "latency")):
        return "latency", "lower_is_better", "target"
    if any(term in lowered for term in ("throughput", "tput", "qps", "rps", "intvty", "interactivity")):
        return "throughput/interactivity", "higher_is_better", "target"
    if any(term in lowered for term in ("joules", "energy")):
        return "energy", "lower_is_better", "target"
    if "power" in lowered:
        return "power", "diagnostic", "diagnostic/cost input"
    if "util" in lowered:
        return "utilization", "diagnostic", "diagnostic"
    if "mem" in lowered:
        return "memory", "diagnostic", "diagnostic"
    if "temp" in lowered:
        return "temperature", "diagnostic", "diagnostic"
    if "score" in lowered:
        return "score", "unknown", "overlay"
    return "metric", "unknown", "overlay"


def meaning_for_column(column: str) -> str:
    if column in PLAIN_ENGLISH_MEANINGS:
        return PLAIN_ENGLISH_MEANINGS[column]
    lowered = column.lower()
    if column.startswith("metrics_"):
        stat = column.removeprefix("metrics_")
        if "ttft" in lowered:
            return f"{stat}: time to first token."
        if "tpot" in lowered:
            return f"{stat}: time per output token."
        if "itl" in lowered:
            return f"{stat}: inter-token latency."
        if "e2el" in lowered:
            return f"{stat}: end-to-end latency."
        if "intvty" in lowered or "interactivity" in lowered:
            return f"{stat}: interactivity metric."
        if "tput" in lowered or "throughput" in lowered:
            return f"{stat}: throughput, tokens/sec/GPU when applicable."
        if "power" in lowered:
            return f"{stat}: power telemetry."
        if "joules" in lowered:
            return f"{stat}: energy per token."
        if "temp" in lowered:
            return f"{stat}: temperature telemetry."
        if "util" in lowered:
            return f"{stat}: GPU utilization."
        if "mem" in lowered:
            return f"{stat}: memory used."
        return f"{stat}: benchmark output metric."
    if column == "workflow_run_id":
        return "Provenance link to the workflow run that produced the row."
    if column == "server_log_id":
        return "Provenance link to raw server logs, which this app intentionally skips."
    if column == "config_id":
        return "Foreign key joining benchmark_results to configs."
    if column == "benchmark_id":
        return "Benchmark result row identifier."
    if column == "date":
        return "Benchmark date/timestamp."
    if column == "error":
        return "Failure/error field; non-null values indicate failed or suspect rows."
    return "Inferred field from the joined benchmark/config dump."


def example_values(series: pd.Series, limit: int = 3) -> str:
    values = []
    for value in series.dropna().head(50):
        normalized = safe_hashable_value(value)
        if normalized not in values:
            values.append(normalized)
        if len(values) >= limit:
            break
    return ", ".join(str(value) for value in values)


def build_feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    row_count = max(len(frame), 1)
    for column in frame.columns:
        non_null = int(frame[column].notna().sum())
        rows.append(
            {
                "column": column,
                "source table": source_table_for_column(column),
                "feature family": feature_family_for_column(column),
                "role": role_for_column(column),
                "dtype": str(frame[column].dtype),
                "non_null_count": non_null,
                "missing_pct": 100 * (1 - non_null / row_count),
                "unique_count": safe_nunique(frame[column]),
                "example_values": example_values(frame[column]),
                "modeling_use": modeling_use_for_column(column, frame),
                "plain_english_meaning": meaning_for_column(column),
            }
        )
    return pd.DataFrame(rows)


def build_numeric_distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in numeric_columns(frame):
        series = pd.to_numeric(frame[column], errors="coerce").astype(float)
        kind, direction, recommended_use = (
            metric_kind_direction_use(column) if is_metric_column(column) else ("input", "", "")
        )
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.median(),
                "p75": series.quantile(0.75),
                "max": series.max(),
                "missing_pct": 100 * series.isna().mean(),
                "metric_type": kind,
                "direction": direction,
                "recommended_use": recommended_use,
            }
        )
    return pd.DataFrame(rows)


def build_categorical_top_values(frame: pd.DataFrame, max_columns: int = 60) -> pd.DataFrame:
    rows = []
    for column in categorical_columns(frame)[:max_columns]:
        counts = frame[column].dropna().map(safe_hashable_value).value_counts().head(8)
        for value, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "share_pct": 100 * count / max(len(frame), 1),
                }
            )
    return pd.DataFrame(rows)


def build_metric_target_guide(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in metric_like_numeric_columns(frame):
        kind, direction, recommended_use = metric_kind_direction_use(column)
        rows.append(
            {
                "metric": column,
                "metric_type": kind,
                "direction": direction,
                "recommended_use": recommended_use,
                "interpretation": meaning_for_column(column),
                "non_null_count": int(frame[column].notna().sum()),
                "missing_pct": 100 * frame[column].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def build_data_quality_report(frame: pd.DataFrame, feature_dictionary: pd.DataFrame) -> dict[str, Any]:
    key_columns = config_workload_grouping_keys(frame)
    if key_columns and all(column in frame.columns for column in key_columns):
        duplicate_rows = int(frame.duplicated(subset=key_columns, keep=False).sum())
    else:
        duplicate_rows = 0

    metric_cols = metric_like_numeric_columns(frame)
    missing_metrics = pd.DataFrame(
        {
            "column": metric_cols,
            "missing_count": [int(frame[column].isna().sum()) for column in metric_cols],
            "missing_pct": [100 * frame[column].isna().mean() for column in metric_cols],
        }
    ).sort_values("missing_pct", ascending=False)

    failed_rows = int(frame["error"].notna().sum()) if "error" in frame.columns else 0
    high_cardinality = feature_dictionary[
        (feature_dictionary["unique_count"] > 80)
        & (feature_dictionary["role"].isin(["metadata", "provenance"]))
    ][["column", "unique_count", "role"]]
    single_value = feature_dictionary[feature_dictionary["unique_count"] <= 1][
        ["column", "unique_count", "role"]
    ]
    excluded = feature_dictionary[feature_dictionary["modeling_use"] == "exclude"][
        ["column", "role", "plain_english_meaning"]
    ]

    suspicious_rows: list[dict[str, Any]] = []
    for column in metric_cols:
        series = pd.to_numeric(frame[column], errors="coerce")
        negative_count = int((series < 0).sum())
        if negative_count:
            suspicious_rows.append(
                {"check": f"negative values in {column}", "row_count": negative_count}
            )
    for column in ("conc", "isl", "osl"):
        if column in frame.columns:
            zero_count = int((pd.to_numeric(frame[column], errors="coerce") <= 0).sum())
            if zero_count:
                suspicious_rows.append(
                    {"check": f"non-positive values in {column}", "row_count": zero_count}
                )

    return {
        "duplicate_rows": duplicate_rows,
        "missing_metrics": missing_metrics,
        "failed_rows": failed_rows,
        "high_cardinality": high_cardinality,
        "single_value": single_value,
        "excluded": excluded,
        "suspicious": pd.DataFrame(suspicious_rows),
    }


def repeated_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouping_keys = config_workload_grouping_keys(frame)
    if not grouping_keys:
        return pd.DataFrame()
    return (
        frame.groupby(grouping_keys, dropna=False, sort=False)
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values("row_count", ascending=False)
    )


def build_data_quality_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# InferenceX PCA Demo Data Quality Report",
            "",
            f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            f"- Duplicate rows by config/workflow/workload key: {report['duplicate_rows']:,}",
            f"- Failed rows where `error` is non-null: {report['failed_rows']:,}",
            "",
            "## Missing Metrics",
            "",
            dataframe_to_markdown(report["missing_metrics"], 20),
            "",
            "## High-Cardinality Provenance/Metadata Columns",
            "",
            dataframe_to_markdown(report["high_cardinality"], 20),
            "",
            "## One-Value Columns",
            "",
            dataframe_to_markdown(report["single_value"], 20),
            "",
            "## Excluded From PCA/Modeling By Default",
            "",
            dataframe_to_markdown(report["excluded"], 30),
            "",
            "## Suspicious Values",
            "",
            dataframe_to_markdown(report["suspicious"], 20),
        ]
    )


@st.cache_data(show_spinner=False)
def load_optional_small_tables(data_dir_text: str, max_mb: float = 10.0) -> dict[str, pd.DataFrame]:
    data_dir = resolve_data_dir(data_dir_text)
    json_dir = resolve_data_dir(DEFAULT_JSON_DUMP_DIR)
    tables: dict[str, pd.DataFrame] = {}
    for table_name in (
        "availability",
        "run_stats",
        "eval_results",
        "changelog_entries",
        "workflow_runs",
    ):
        csv_path = data_dir / f"{table_name}.csv"
        json_path = json_dir / f"{table_name}.json"
        if csv_path.exists() and csv_path.stat().st_size <= max_mb * 1024 * 1024:
            tables[table_name] = pd.read_csv(csv_path, low_memory=False)
            continue
        if not json_path.exists() or json_path.stat().st_size > max_mb * 1024 * 1024:
            continue
        try:
            tables[table_name] = normalize_records(read_records_json(json_path))
        except ValueError:
            pass
    return tables


def render_data_preview(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    joined: pd.DataFrame,
) -> None:
    metric_cols = metric_like_numeric_columns(joined)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Benchmark rows", f"{len(benchmarks):,}")
    col_b.metric("Config rows", f"{len(configs):,}")
    col_c.metric("Joined rows", f"{len(joined):,}")
    col_d.metric("Metric-like numeric columns", f"{len(metric_cols):,}")

    st.subheader("Columns")
    column_summary = pd.DataFrame(
        {
            "column": joined.columns,
            "dtype": [str(joined[column].dtype) for column in joined.columns],
            "non_null": [int(joined[column].notna().sum()) for column in joined.columns],
            "unique": [safe_nunique(joined[column]) for column in joined.columns],
        }
    )
    st.dataframe(column_summary, use_container_width=True, height=360)

    st.subheader("Sample Rows")
    st.dataframe(joined.head(100), use_container_width=True, height=420)


def render_data_understanding(
    joined: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    data_dir: str,
    max_rows: int,
    seed: int,
) -> None:
    st.header("Data Understanding")
    st.caption(
        "Use this tab to understand table relationships, column meanings, missingness, "
        "cardinality, metric direction, and which fields are safe for PCA or supervised modeling."
    )

    sample = sample_frame(analysis_frame, max_rows, seed)
    feature_dictionary = build_feature_dictionary(joined)
    numeric_summary = build_numeric_distribution_summary(sample)
    categorical_top_values = build_categorical_top_values(sample)
    metric_target_guide = build_metric_target_guide(joined)
    quality_report = build_data_quality_report(joined, feature_dictionary)
    quality_markdown = build_data_quality_markdown(quality_report)
    repeat_summary = repeated_group_summary(joined)

    st.subheader("Dataset Map")
    st.markdown(
        """
        The local data folder contains CSV exports for team sharing, with JSON dump fallback
        for local development:

        | File | Status in this app | Purpose |
        | --- | --- | --- |
        | `benchmark_results.csv` / `.json` | Loaded | Raw benchmark rows, workload shape, metrics, provenance ids. |
        | `configs.csv` / `.json` | Loaded | Model/hardware/framework/precision/setup fields. |
        | `workflow_runs.csv` / `.json` | Optional small side table | Run provenance and workflow metadata. |
        | `eval_results.csv` / `.json` | Optional small side table | Evaluation results, if later joined for eval analysis. |
        | `availability.csv` / `.json` | Optional small side table | Model/date availability metadata. |
        | `run_stats.csv` / `.json` | Optional small side table | Run-level stats. |
        | `changelog_entries.csv` / `.json` | Optional small side table | Human-readable changelog notes. |
        | `server_logs.json` | Intentionally skipped | Huge raw logs; use only for audits outside this app. |
        | `eval_samples.json` | Intentionally skipped | Huge raw eval samples; not needed for PCA. |

        Core join: `benchmark_results.config_id = configs.id`.
        `workflow_run_id` and `server_log_id` are provenance/audit links, not modeling inputs.
        """
    )

    st.subheader("Analysis Unit")
    st.markdown(
        "Repeated benchmark rows can overweight frequently tested configurations. "
        "Aggregated analysis reduces this bias by making each selected group contribute once."
    )
    unit_cols = st.columns(4)
    unit_cols[0].metric("Selected unit", analysis_metadata["analysis_unit"])
    unit_cols[1].metric("Raw rows", f"{analysis_metadata['raw_row_count']:,}")
    unit_cols[2].metric("Analysis rows", f"{analysis_metadata['analysis_row_count']:,}")
    unit_cols[3].metric(
        "Grouping keys",
        ", ".join(analysis_metadata["grouping_keys"]) or "none",
    )
    if analysis_metadata.get("warning"):
        st.warning(analysis_metadata["warning"])

    if not repeat_summary.empty:
        repeated_only = repeat_summary[repeat_summary["row_count"] > 1]
        st.markdown("**Top repeated config/workload/concurrency groups**")
        st.dataframe(repeated_only.head(30), use_container_width=True, hide_index=True)
        st.caption(
            f"{len(repeated_only):,} groups have repeated raw rows out of "
            f"{len(repeat_summary):,} total config/workload/concurrency groups."
        )

    if st.checkbox("Optionally load small side tables", value=False):
        optional_tables = load_optional_small_tables(data_dir)
        if optional_tables:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                        for name, frame in optional_tables.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No optional side tables were loaded within the size guard.")

    st.subheader("Feature Dictionary")
    family_options = [
        "Inputs/configuration",
        "Workload shape",
        "Outcome metrics",
        "Provenance/audit",
        "Excluded/high-cardinality",
        "Reliability/eval",
    ]
    selected_families = st.multiselect(
        "Column family filters",
        options=family_options,
        default=family_options[:5],
    )
    dictionary_view = feature_dictionary.copy()
    family_mask = pd.Series(False, index=dictionary_view.index)
    if "Inputs/configuration" in selected_families:
        family_mask |= dictionary_view["role"].eq("input/config")
    if "Workload shape" in selected_families:
        family_mask |= dictionary_view["role"].eq("workload")
    if "Outcome metrics" in selected_families:
        family_mask |= dictionary_view["role"].eq("outcome metric")
    if "Provenance/audit" in selected_families:
        family_mask |= dictionary_view["role"].isin(["provenance", "metadata"])
    if "Excluded/high-cardinality" in selected_families:
        family_mask |= dictionary_view["modeling_use"].eq("exclude")
    if "Reliability/eval" in selected_families:
        family_mask |= dictionary_view["role"].isin(["reliability", "eval"])
    if selected_families:
        dictionary_view = dictionary_view[family_mask]
    st.dataframe(dictionary_view, use_container_width=True, height=480, hide_index=True)

    st.subheader("Distribution Summary")
    st.caption(f"Distribution summaries use up to {len(sample):,} sampled rows.")
    dist_left, dist_right = st.columns(2)
    with dist_left:
        st.markdown("**Numeric columns**")
        st.dataframe(numeric_summary, use_container_width=True, height=420, hide_index=True)
    with dist_right:
        st.markdown("**Categorical top values**")
        st.dataframe(categorical_top_values, use_container_width=True, height=420, hide_index=True)

    st.subheader("Coverage Matrix")
    coverage = joined.copy()
    filters = st.columns(5)
    if "config_model" in coverage.columns:
        model_options = ["All"] + sorted(str(value) for value in coverage["config_model"].dropna().unique())
        selected_model = filters[0].selectbox("Model", model_options)
        if selected_model != "All":
            coverage = coverage[coverage["config_model"].astype(str) == selected_model]
    if "config_precision" in coverage.columns:
        precision_options = ["All"] + sorted(
            str(value) for value in coverage["config_precision"].dropna().unique()
        )
        selected_precision = filters[1].selectbox("Precision", precision_options)
        if selected_precision != "All":
            coverage = coverage[coverage["config_precision"].astype(str) == selected_precision]
    for col_idx, column in enumerate(("isl", "osl")):
        if column in coverage.columns:
            options = ["All"] + sorted(str(value) for value in coverage[column].dropna().unique())
            selected = filters[col_idx + 2].selectbox(column.upper(), options)
            if selected != "All":
                coverage = coverage[coverage[column].astype(str) == selected]
    if "date" in coverage.columns:
        date_options = ["All"] + sorted(str(value)[:10] for value in coverage["date"].dropna().unique())
        selected_date = filters[4].selectbox("Date", unique_preserve_order(date_options))
        if selected_date != "All":
            coverage = coverage[coverage["date"].astype(str).str.startswith(selected_date)]

    matrix_cols = st.columns(2)
    if {"config_hardware", "config_framework"}.issubset(coverage.columns):
        hw_framework = coverage.pivot_table(
            index="config_hardware",
            columns="config_framework",
            values="config_id" if "config_id" in coverage.columns else coverage.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        matrix_cols[0].markdown("**config_hardware x config_framework**")
        matrix_cols[0].dataframe(hw_framework, use_container_width=True)
        matrix_cols[0].plotly_chart(
            px.imshow(hw_framework, text_auto=True, aspect="auto"),
            use_container_width=True,
        )
    if {"config_model", "config_hardware"}.issubset(coverage.columns):
        model_hw = coverage.pivot_table(
            index="config_model",
            columns="config_hardware",
            values="config_id" if "config_id" in coverage.columns else coverage.columns[0],
            aggfunc="count",
            fill_value=0,
        )
        matrix_cols[1].markdown("**config_model x config_hardware**")
        matrix_cols[1].dataframe(model_hw, use_container_width=True)
        matrix_cols[1].plotly_chart(
            px.imshow(model_hw, text_auto=True, aspect="auto"),
            use_container_width=True,
        )

    st.subheader("Workload Shape Summary")
    st.caption(
        "ISL, OSL, and concurrency define request shape. They matter because prefill cost, "
        "decode cost, batching pressure, and latency/throughput tradeoffs change with workload."
    )
    workload_cols = st.columns(3)
    for idx, column in enumerate(("isl", "osl", "conc")):
        if column in joined.columns:
            counts = (
                joined[column]
                .dropna()
                .map(safe_hashable_value)
                .value_counts()
                .head(30)
                .rename_axis(column)
                .reset_index(name="row_count")
            )
            workload_cols[idx].dataframe(counts, use_container_width=True, hide_index=True)
            workload_cols[idx].plotly_chart(
                px.bar(counts, x=column, y="row_count", title=column.upper()),
                use_container_width=True,
            )
    if {"isl", "osl"}.issubset(joined.columns):
        seq_pairs = (
            joined.assign(sequence_pair=joined["isl"].astype(str) + " / " + joined["osl"].astype(str))
            ["sequence_pair"]
            .value_counts()
            .head(30)
            .rename_axis("isl / osl")
            .reset_index(name="row_count")
        )
        st.markdown("**Sequence pair counts**")
        st.dataframe(seq_pairs, use_container_width=True, hide_index=True)

    st.subheader("Metric Target Guide")
    st.dataframe(metric_target_guide, use_container_width=True, height=420, hide_index=True)

    st.subheader("Data Quality Checks")
    q_cols = st.columns(3)
    q_cols[0].metric("Duplicate key rows", f"{quality_report['duplicate_rows']:,}")
    q_cols[1].metric("Failed rows", f"{quality_report['failed_rows']:,}")
    q_cols[2].metric("Excluded columns", f"{len(quality_report['excluded']):,}")
    st.markdown("**Missing metrics**")
    st.dataframe(quality_report["missing_metrics"].head(30), use_container_width=True, hide_index=True)
    dq_cols = st.columns(2)
    dq_cols[0].markdown("**High-cardinality metadata/provenance**")
    dq_cols[0].dataframe(quality_report["high_cardinality"], use_container_width=True, hide_index=True)
    dq_cols[1].markdown("**One-unique-value columns**")
    dq_cols[1].dataframe(quality_report["single_value"], use_container_width=True, hide_index=True)
    st.markdown("**Suspicious values**")
    if quality_report["suspicious"].empty:
        st.success("No negative metric or non-positive workload values found in the loaded rows.")
    else:
        st.dataframe(quality_report["suspicious"], use_container_width=True, hide_index=True)
    st.markdown("**Excluded from PCA/modeling by default**")
    st.dataframe(quality_report["excluded"], use_container_width=True, hide_index=True)

    st.subheader("Caveats")
    st.markdown(
        """
        - Raw `benchmark_results` may include historical repeated runs.
        - The production dashboard may select the newest run per line rather than all raw rows.
        - PCA should use configuration inputs only.
        - Supervised targets should be outcome metrics only.
        - Correlation and feature importance are descriptive, not causal.
        - Do not use ids, dates, images, URLs, logs, or error fields as model predictors by default.
        """
    )

    st.subheader("Downloads")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "data_dictionary.csv",
        data=feature_dictionary.to_csv(index=False),
        file_name="data_dictionary.csv",
        mime="text/csv",
        key="download_data_dictionary",
    )
    download_cols[1].download_button(
        "numeric_distribution_summary.csv",
        data=numeric_summary.to_csv(index=False),
        file_name="numeric_distribution_summary.csv",
        mime="text/csv",
        key="download_numeric_distribution_summary",
    )
    download_cols[2].download_button(
        "categorical_top_values.csv",
        data=categorical_top_values.to_csv(index=False),
        file_name="categorical_top_values.csv",
        mime="text/csv",
        key="download_categorical_top_values",
    )
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "metric_target_guide.csv",
        data=metric_target_guide.to_csv(index=False),
        file_name="metric_target_guide.csv",
        mime="text/csv",
        key="download_metric_target_guide",
    )
    download_cols[1].download_button(
        "data_quality_report.md",
        data=quality_markdown,
        file_name="data_quality_report.md",
        mime="text/markdown",
        key="download_data_quality_report",
    )


def render_pca_explorer(
    joined: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    metric_cols = metric_like_numeric_columns(joined)
    numeric_candidates = numeric_columns(joined)
    categorical_candidates = categorical_columns(joined)

    config_categorical = config_categorical_columns(joined)
    default_categorical = [
        column for column in DEFAULT_CATEGORICAL_FEATURES if column in config_categorical
    ]
    default_numeric = [
        column for column in config_numeric_columns(joined) if column not in default_categorical
    ][:12]
    color_options = unique_preserve_order(
        [""] + metric_cols + config_categorical + categorical_candidates + numeric_candidates
    )
    default_color = metric_cols[0] if metric_cols else "config_hardware"
    default_color_index = color_options.index(default_color) if default_color in color_options else 0

    st.caption(
        "PCA should use setup/configuration features. Outcome metrics can be used to color "
        "the scatter plot, but should not be PCA inputs."
    )
    st.info(
        f"PCA is using `{analysis_metadata['analysis_unit']}` "
        f"({analysis_metadata['analysis_row_count']:,} analysis rows from "
        f"{analysis_metadata['raw_row_count']:,} raw rows)."
    )

    col_a, col_b = st.columns(2)
    selected_numeric = col_a.multiselect(
        "Numeric features",
        options=numeric_candidates,
        default=default_numeric,
        key="pca_numeric_features",
    )
    selected_categorical = col_b.multiselect(
        "Categorical features",
        options=categorical_candidates,
        default=default_categorical,
        key="pca_categorical_features",
    )
    color_by = st.selectbox(
        "Color scatter by",
        options=color_options,
        index=default_color_index,
    )

    selected_numeric, selected_categorical, overlap = normalize_feature_groups(
        selected_numeric,
        selected_categorical,
    )
    if overlap:
        st.warning(
            "Some PCA features were selected as both numeric and categorical; keeping them "
            f"numeric only: {', '.join(overlap)}"
        )

    feature_columns = selected_numeric + selected_categorical
    selected_metrics = [column for column in feature_columns if is_metric_column(column)]
    if selected_metrics:
        st.warning(
            "Metric/outcome columns are selected as PCA inputs. This can make PCA summarize "
            f"performance outputs instead of setup features: {', '.join(selected_metrics)}"
        )
    if not feature_columns:
        st.info("Select at least one numeric or categorical feature to run PCA.")
        return

    work = sample_frame(joined, max_rows, seed)
    work = coerce_model_frame(work, feature_columns)
    if len(work) < 3:
        st.warning("Not enough usable rows after dropping empty feature rows.")
        return

    numeric_features, categorical_features = split_features(work, feature_columns)
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    try:
        matrix = preprocessor.fit_transform(work[feature_columns])
    except Exception as exc:  # Defensive UI boundary for unexpected dump shape.
        st.error(f"Could not preprocess selected features: {exc}")
        return

    if matrix.shape[1] < 2:
        st.warning("PCA needs at least two encoded feature dimensions.")
        return

    pca = PCA(n_components=min(5, matrix.shape[1], len(work)), random_state=seed)
    coords = pca.fit_transform(matrix)

    explained = pd.DataFrame(
        {
            "component": [f"PC{idx + 1}" for idx in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    st.subheader("Explained Variance")
    st.dataframe(explained, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            explained,
            x="component",
            y="explained_variance_ratio",
            text="explained_variance_ratio",
        ).update_traces(texttemplate="%{text:.1%}", textposition="outside"),
        use_container_width=True,
    )

    plot_frame = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
    if color_by and color_by in joined.columns:
        plot_frame[color_by] = joined.loc[work.index, color_by]

    st.subheader("PC1 vs PC2")
    st.plotly_chart(
        px.scatter(
            plot_frame,
            x="PC1",
            y="PC2",
            color=color_by if color_by in plot_frame.columns else None,
            opacity=0.72,
            render_mode="webgl",
        ),
        use_container_width=True,
    )

    feature_names = [clean_feature_label(name) for name in preprocessor.get_feature_names_out()]
    loading_details = build_pca_loading_details(
        pca,
        feature_names,
        numeric_features,
        categorical_features,
    )

    st.caption(
        "PC1 and PC2 are synthetic axes. The useful PCA output is the "
        "loading/contribution table, which maps those axes back to original features."
    )

    st.subheader("Feature Contribution Summary")
    st.dataframe(
        loading_details.head(30),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(
        px.bar(
            loading_details.head(30).sort_values("weighted_contribution"),
            x="weighted_contribution",
            y="encoded_feature",
            color="source_feature",
            orientation="h",
            hover_data=["contribution_share"],
        ),
        use_container_width=True,
    )

    st.subheader("Top Original Feature Groups by Variance Contribution")
    source_contributions = original_feature_contributions(loading_details)
    st.dataframe(
        source_contributions.head(30),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(
        px.bar(
            source_contributions.head(30).sort_values("weighted_contribution"),
            x="weighted_contribution",
            y="source_feature",
            orientation="h",
            hover_data=["contribution_share"],
        ),
        use_container_width=True,
    )

    st.subheader("Component Interpretation Cards")
    component_interpretations = build_component_interpretations(pca, loading_details)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    for row_start in range(0, min(4, len(pca.components_)), 2):
        card_columns = st.columns(2)
        for card_column, component_idx in zip(
            card_columns,
            range(row_start, min(row_start + 2, min(4, len(pca.components_)))),
        ):
            component_name = f"PC{component_idx + 1}"
            component_loadings = loading_details[
                ["encoded_feature", "source_feature", f"{component_name}_loading"]
            ].rename(columns={f"{component_name}_loading": "loading"})
            positive = (
                component_loadings[component_loadings["loading"] > 0]
                .sort_values("loading", ascending=False)
                .head(8)
            )
            negative = (
                component_loadings[component_loadings["loading"] < 0]
                .sort_values("loading", ascending=True)
                .head(8)
            )
            absolute = (
                component_loadings.assign(abs_loading=component_loadings["loading"].abs())
                .sort_values("abs_loading", ascending=False)
                .head(8)
            )

            with card_column.container(border=True):
                st.markdown(f"#### {component_name}")
                st.metric(
                    "Explained variance",
                    f"{pca.explained_variance_ratio_[component_idx]:.1%}",
                )
                st.metric(
                    "Cumulative variance",
                    f"{cumulative_variance[component_idx]:.1%}",
                )
                st.write(interpret_component(component_loadings))
                col_pos, col_neg, col_abs = st.columns(3)
                col_pos.markdown("**Top positive**")
                col_pos.markdown(format_loading_list(positive))
                col_neg.markdown("**Top negative**")
                col_neg.markdown(format_loading_list(negative))
                col_abs.markdown("**Top absolute**")
                col_abs.markdown(format_loading_list(absolute))

    st.subheader("PC vs Target Correlation")
    st.caption(
        "This checks whether a high-variance PCA component is related to a performance "
        "outcome. Correlation is descriptive, not causal."
    )
    target_metric = ""
    correlation_frame = pd.DataFrame()
    if metric_cols:
        target_metric = st.selectbox(
            "Performance metric for PC correlation",
            options=metric_cols,
            key="pca_target_correlation_metric",
        )
        correlation_frame = compute_pc_target_correlations(
            coords,
            work.index,
            joined,
            target_metric,
        )
        st.dataframe(correlation_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No metric-like numeric columns are available for target correlation.")

    st.session_state["pca_analysis"] = {
        "sampled_rows": len(work),
        "input_feature_count": len(feature_columns),
        "input_features": feature_columns,
        "target_metric": target_metric,
        "analysis_unit": analysis_metadata["analysis_unit"],
        "raw_row_count": analysis_metadata["raw_row_count"],
        "analysis_row_count": analysis_metadata["analysis_row_count"],
        "grouping_keys": analysis_metadata["grouping_keys"],
        "encoded_contributions": loading_details,
        "source_contributions": source_contributions,
        "component_interpretations": component_interpretations,
        "pc_target_correlations": correlation_frame,
        "explained_variance": explained,
    }

    component_options = [f"PC{idx + 1}" for idx in range(len(pca.components_))]
    selected_component = st.selectbox("Loadings component", options=component_options)
    component_idx = component_options.index(selected_component)
    loading_frame = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "loading": pca.components_[component_idx],
        }
    )
    loading_frame["abs_loading"] = loading_frame["loading"].abs()
    loading_frame["weighted_abs_loading"] = (
        loading_frame["abs_loading"] * pca.explained_variance_ratio_[component_idx]
    )
    loading_frame = loading_frame.sort_values("abs_loading", ascending=False).head(30)

    st.subheader("Top PCA Loadings")
    st.dataframe(loading_frame, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            loading_frame.sort_values("abs_loading"),
            x="abs_loading",
            y="encoded_feature",
            orientation="h",
            hover_data=["loading", "weighted_abs_loading"],
        ),
        use_container_width=True,
    )


def render_target_feature_value(
    joined: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    metric_cols = metric_like_numeric_columns(joined)
    numeric_candidates = numeric_columns(joined)
    categorical_candidates = categorical_columns(joined)

    if not metric_cols:
        st.warning("No numeric metric-like target columns were detected.")
        return

    st.caption(
        "This supervised layer estimates which setup/configuration features predict the "
        "selected performance target."
    )
    st.info(
        f"Target modeling is using `{analysis_metadata['analysis_unit']}` "
        f"({analysis_metadata['analysis_row_count']:,} analysis rows from "
        f"{analysis_metadata['raw_row_count']:,} raw rows)."
    )

    target = st.selectbox("Target metric", options=metric_cols, key="target_metric")
    include_other_metrics = st.checkbox("Allow other metric-like columns as predictors", value=False)

    default_categorical = [
        column for column in DEFAULT_CATEGORICAL_FEATURES if column in config_categorical_columns(joined)
    ]
    default_numeric = [
        column
        for column in config_numeric_columns(joined)
        if column != target
        and column not in default_categorical
        and (
            include_other_metrics
            or not is_metric_column(column)
        )
    ][:12]
    if not default_numeric:
        default_numeric = [
            column
            for column in numeric_candidates
            if column != target and not is_metric_column(column) and not is_metadata_column(column)
        ][:12]

    col_a, col_b = st.columns(2)
    selected_numeric = col_a.multiselect(
        "Numeric predictors",
        options=[column for column in numeric_candidates if column != target],
        default=default_numeric,
        key="target_numeric_features",
    )
    selected_categorical = col_b.multiselect(
        "Categorical predictors",
        options=categorical_candidates,
        default=default_categorical,
        key="target_categorical_features",
    )
    split_options = ["Random split"]
    if "config_id" in joined.columns:
        split_options.insert(0, "Grouped split by config_id")
    split_mode = st.selectbox("Train/test split mode", options=split_options)
    if split_mode == "Random split":
        st.warning(
            "Random splits can overestimate performance when repeated configurations or "
            "near-identical config/workload rows appear in both train and test."
        )
    n_estimators = st.slider("Random forest trees", min_value=50, max_value=400, value=150, step=50)

    selected_numeric, selected_categorical, overlap = normalize_feature_groups(
        selected_numeric,
        selected_categorical,
    )
    if overlap:
        st.warning(
            "Some predictors were selected as both numeric and categorical; keeping them "
            f"numeric only: {', '.join(overlap)}"
        )

    feature_columns = selected_numeric + selected_categorical
    if not feature_columns:
        st.info("Select at least one predictor to train the model.")
        return

    work_columns = unique_preserve_order(
        feature_columns
        + [target]
        + (["config_id"] if "config_id" in joined.columns else [])
    )
    work = sample_frame(joined, max_rows, seed)
    work = work[[column for column in work_columns if column in work.columns]].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    work = work.dropna(axis=0, how="all", subset=feature_columns)
    work = work.dropna(axis=0, subset=[target])
    if len(work) < 20:
        st.warning("Not enough rows with a non-null target value to train/test a model.")
        return

    X = work.loc[:, feature_columns]
    removed_duplicate_columns: list[str] = []
    if not X.columns.is_unique:
        removed_duplicate_columns = X.columns[X.columns.duplicated()].tolist()
        X = X.loc[:, ~X.columns.duplicated()]
        feature_columns = list(X.columns)
        selected_numeric = [column for column in selected_numeric if column in feature_columns]
        selected_categorical = [column for column in selected_categorical if column in feature_columns]

    if removed_duplicate_columns:
        st.warning(
            "Removed duplicate predictor columns before training: "
            f"{', '.join(unique_preserve_order(removed_duplicate_columns))}"
        )

    if not X.columns.is_unique:
        st.error("Predictor columns are still not unique after de-duplication.")
        return

    numeric_features, categorical_features = split_features(work, feature_columns)
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    y = work[target]
    if split_mode == "Grouped split by config_id":
        groups = work.loc[X.index, "config_id"].fillna("__missing_config_id__")
        if groups.nunique(dropna=False) < 2:
            st.warning("Grouped split needs at least two config_id groups; falling back to random split.")
            split_mode = "Random split"
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=seed,
            )
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=seed,
        )

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except Exception as exc:
        st.error(f"Could not train model: {exc}")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Train rows", f"{len(X_train):,}")
    col_b.metric("Test R2", f"{r2_score(y_test, predictions):.3f}")
    col_c.metric("Test MAE", f"{mean_absolute_error(y_test, predictions):.3f}")

    with st.spinner("Computing permutation importance"):
        importance = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
            scoring="r2",
        )

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    st.session_state["target_analysis"] = {
        "target_metric": target,
        "sampled_rows": len(work),
        "predictor_count": len(feature_columns),
        "split_mode": split_mode,
        "analysis_unit": analysis_metadata["analysis_unit"],
        "raw_row_count": analysis_metadata["raw_row_count"],
        "analysis_row_count": analysis_metadata["analysis_row_count"],
        "grouping_keys": analysis_metadata["grouping_keys"],
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "importance_frame": importance_frame,
    }

    st.subheader("Permutation Importance")
    st.dataframe(importance_frame, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            importance_frame.head(30).sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
        ),
        use_container_width=True,
    )


def render_findings(
    joined: pd.DataFrame,
    benchmarks: pd.DataFrame,
    analysis_metadata: dict[str, Any],
) -> None:
    st.header("Findings")
    st.caption(
        "A compact executive summary of the PCA structure, target-aware predictors, "
        "and where the two views agree."
    )

    pca_analysis = st.session_state.get("pca_analysis")
    target_analysis = st.session_state.get("target_analysis")
    dataset_summary = {
        "benchmark_rows": len(benchmarks),
        "joined_rows": len(joined),
        "analysis_unit": analysis_metadata["analysis_unit"],
        "raw_row_count": analysis_metadata["raw_row_count"],
        "analysis_row_count": analysis_metadata["analysis_row_count"],
        "grouping_keys": analysis_metadata["grouping_keys"],
    }

    if not pca_analysis:
        st.info("Run the PCA Explorer settings once to generate Findings.")
        return

    source_contributions = pca_analysis["source_contributions"]
    encoded_contributions = pca_analysis["encoded_contributions"]
    component_interpretations = pca_analysis["component_interpretations"]
    pc_target_correlations = pca_analysis["pc_target_correlations"]
    target_importance = (
        target_analysis.get("importance_frame", pd.DataFrame())
        if target_analysis
        else pd.DataFrame()
    )
    selected_target = (
        target_analysis.get("target_metric")
        if target_analysis
        else pca_analysis.get("target_metric", "")
    )
    split_mode = target_analysis.get("split_mode", "not run") if target_analysis else "not run"

    top_pca_groups = source_contributions.head(10)["source_feature"].tolist()
    top_target_predictors = (
        target_importance.head(10)["feature"].tolist()
        if not target_importance.empty
        else []
    )
    overlap = [feature for feature in top_pca_groups if feature in set(top_target_predictors)]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Analysis unit", analysis_metadata["analysis_unit"])
    col_b.metric("Raw rows", f"{analysis_metadata['raw_row_count']:,}")
    col_c.metric("Analysis rows", f"{analysis_metadata['analysis_row_count']:,}")
    col_d.metric("Target metric", selected_target or "Not selected")
    st.caption(
        "Grouping keys: "
        f"{', '.join(analysis_metadata['grouping_keys']) or 'none'} | "
        f"Split mode: {split_mode}"
    )
    st.write(
        f"These findings are based on **{analysis_metadata['analysis_unit']}**, not blindly "
        "on every raw benchmark row."
    )

    structural_text = compact_list(top_pca_groups)
    predictor_text = compact_list(top_target_predictors)
    overlap_text = compact_list(overlap)
    st.subheader("Executive Summary")
    st.write(
        "PCA shows that the largest structural variation in the benchmark configuration "
        f"space for `{analysis_metadata['analysis_unit']}` is driven by {structural_text}. "
        f"For the selected target "
        f"`{selected_target}`, the strongest supervised predictors are {predictor_text}. "
        f"The overlap between these lists is {overlap_text}, which highlights setup "
        "features that are both structurally important and performance-relevant."
    )

    st.subheader("Key Findings")
    st.markdown(
        "\n".join(
            [
                f"- **Main structural drivers:** {structural_text}.",
                f"- **Main performance predictors:** {predictor_text}.",
                f"- **Where PCA and target-aware importance overlap:** {overlap_text}.",
                (
                    "- **Implication for inference/datacenter valuation:** features that "
                    "drive both configuration variance and target performance deserve "
                    "closer review when comparing deployment recipes, GPU fleets, and "
                    "datacenter asset quality."
                ),
            ]
        )
    )

    st.info(
        "Recommended Interpretation: PCA identifies variance structure. Target-aware "
        "modeling identifies predictors of the chosen performance metric. Correlation is "
        "descriptive, not causal. Do not claim PCA proves feature value. Repeated benchmark "
        "rows can overweight frequently tested configurations; aggregated analysis reduces "
        "this bias."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top PCA Variance Contributors")
        st.dataframe(source_contributions.head(10), use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("Top Target-Aware Predictors")
        if target_importance.empty:
            st.info("Target-aware permutation importance is not available yet.")
        else:
            st.dataframe(target_importance.head(10), use_container_width=True, hide_index=True)

    st.subheader("PC1-PC4 Interpretations")
    st.dataframe(component_interpretations, use_container_width=True, hide_index=True)

    st.subheader("PC vs Target Correlation")
    st.dataframe(pc_target_correlations, use_container_width=True, hide_index=True)

    st.subheader("Overlap")
    if overlap:
        st.success(
            "Top PCA contributors that also appear as top target-aware predictors: "
            f"{', '.join(overlap)}"
        )
    else:
        st.warning(
            "No overlap in the current top-10 lists. That can mean the biggest sources "
            "of setup variation are not the strongest predictors of the selected target."
        )

    st.subheader("Downloads")
    report_markdown = build_findings_markdown(dataset_summary, pca_analysis, target_analysis)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "findings_summary.md",
        data=report_markdown,
        file_name="findings_summary.md",
        mime="text/markdown",
        key="download_findings_summary",
    )
    download_cols[1].download_button(
        "pca_original_feature_contributions.csv",
        data=source_contributions.to_csv(index=False),
        file_name="pca_original_feature_contributions.csv",
        mime="text/csv",
        key="download_pca_original_contributions",
    )
    download_cols[2].download_button(
        "pca_encoded_feature_contributions.csv",
        data=encoded_contributions.to_csv(index=False),
        file_name="pca_encoded_feature_contributions.csv",
        mime="text/csv",
        key="download_pca_encoded_contributions",
    )

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "pca_component_interpretations.csv",
        data=component_interpretations.to_csv(index=False),
        file_name="pca_component_interpretations.csv",
        mime="text/csv",
        key="download_pca_component_interpretations",
    )
    download_cols[1].download_button(
        "pc_target_correlations.csv",
        data=pc_target_correlations.to_csv(index=False),
        file_name="pc_target_correlations.csv",
        mime="text/csv",
        key="download_pc_target_correlations",
    )
    if not target_importance.empty:
        download_cols[2].download_button(
            "target_permutation_importance.csv",
            data=target_importance.to_csv(index=False),
            file_name="target_permutation_importance.csv",
            mime="text/csv",
            key="download_target_permutation_importance",
        )

    st.subheader("Deployment Readiness")
    st.markdown(
        """
        - Do not commit `inferencex-dump-2026-06-29` or any giant JSON dump to GitHub.
        - Keep local exports, virtualenvs, Python caches, and Streamlit secrets out of git.
        - Keep `requirements-streamlit.txt` as the sidecar dependency list.
        - Local deployment is straightforward when the dump folder is available on disk.
        - Cloud deployment requires a safe data access plan; do not upload huge dump files
          directly to GitHub.
        - Do not commit secrets. For team use, each teammate should download the dump
          locally or the deployment should mount/fetch the dump from an approved storage
          location.
        - Run locally with:

        ```bash
        python3 -m venv .venv-streamlit
        source .venv-streamlit/bin/activate
        python3 -m pip install -r requirements-streamlit.txt
        python3 -m streamlit run apps/inferencex_pca_demo.py
        ```
        """
    )


def render_notes() -> None:
    st.markdown(
        """
        This demo reads only `benchmark_results.json` and `configs.json` by default.
        It intentionally skips `server_logs.json` and `eval_samples.json`, which can be
        many gigabytes and are not needed for this PCA workflow.

        PCA is unsupervised: it finds feature directions that explain variance in the
        selected columns. A large loading means a transformed feature contributes strongly
        to a principal component, but it does not mean the feature predicts a chosen
        performance target.

        The target-aware tab is supervised: it trains a `RandomForestRegressor` for one
        selected numeric metric and then uses permutation importance on the test split.
        That ranking is a baseline estimate of which selected predictors matter for the
        chosen target, not a causal claim.

        The app samples rows before PCA and model training to keep local memory use
        reasonable. Increase the sidebar row limit if your machine has headroom.
        """
    )


def main() -> None:
    st.title("InferenceX PCA Demo")

    with st.sidebar:
        st.header("Data")
        data_dir = st.text_input("Data directory", value=DEFAULT_DATA_DIR)
        analysis_unit = st.selectbox(
            "Analysis Unit",
            options=ANALYSIS_UNIT_OPTIONS,
            index=2,
            help=(
                "Choose whether PCA/modeling use raw rows, latest rows, median aggregates, "
                "or one row per config."
            ),
        )
        max_rows = st.number_input(
            "Max rows for PCA/modeling",
            min_value=500,
            max_value=100_000,
            value=20_000,
            step=500,
        )
        seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)
        st.caption("CSV mode loads benchmark_results.csv and configs.csv; JSON fallback remains supported.")

    st.subheader("Data Location")
    file_status, source_probe = data_source_status(data_dir)
    status_cols = st.columns(3)
    status_cols[0].metric("Data directory", str(resolve_data_dir(data_dir)))
    status_cols[1].metric("Active source", source_probe["active_mode"])
    status_cols[2].metric(
        "Required files",
        "found" if source_probe["active_mode"] != "missing" else "missing",
    )
    if source_probe["active_mode"] == "JSON fallback":
        st.warning(
            "CSV files were not found in the selected data directory. Using JSON fallback from "
            f"`{DEFAULT_JSON_DUMP_DIR}`."
        )
    if source_probe["active_mode"] == "missing":
        st.error(
            "Required data files are missing. For team CSV mode, place "
            "`benchmark_results.csv` and `configs.csv` under `inferencex-pca-data`, "
            "or update the sidebar path. For JSON fallback, keep "
            f"`{DEFAULT_JSON_DUMP_DIR}/benchmark_results.json` and `configs.json` available."
        )
    with st.expander("Required file status", expanded=source_probe["active_mode"] == "missing"):
        st.dataframe(file_status, use_container_width=True, hide_index=True)

    try:
        benchmarks, configs, joined, source_info = load_joined_data(data_dir)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return
    st.caption(
        f"Loaded in {source_info['active_mode']} mode from `{source_info['active_dir']}`."
    )

    analysis_frame, analysis_metadata = build_analysis_frame(joined, analysis_unit)
    with st.sidebar:
        st.metric("Raw rows", f"{len(joined):,}")
        st.metric("Analysis rows", f"{len(analysis_frame):,}")
        if analysis_metadata["grouping_keys"]:
            st.caption("Grouping keys: " + ", ".join(analysis_metadata["grouping_keys"]))
        if analysis_metadata.get("timestamp_column"):
            st.caption(f"Latest-row timestamp: {analysis_metadata['timestamp_column']}")
        if analysis_metadata.get("warning"):
            st.warning(analysis_metadata["warning"])

    default_split_label = (
        "Grouped split by config_id" if "config_id" in analysis_frame.columns else "Random split"
    )
    with st.container(border=True):
        st.markdown("### Project Status")
        status_a, status_b, status_c, status_d = st.columns(4)
        status_a.success("Data loaded")
        status_b.info(f"Analysis unit: {analysis_metadata['analysis_unit']}")
        status_c.success("PCA inputs exclude outcome metrics by default")
        status_d.info(f"Target split shown in UI; default is {default_split_label}")
        st.caption(
            "Caveat: this is descriptive analysis for structure and prediction, not causal proof."
        )

    tabs = st.tabs(
        [
            "Data Preview",
            "Data Understanding",
            "PCA Explorer",
            "Target-Aware Feature Value",
            "Findings",
            "Notes",
        ]
    )
    with tabs[0]:
        render_data_preview(benchmarks, configs, joined)
    with tabs[1]:
        render_data_understanding(
            joined,
            analysis_frame,
            analysis_metadata,
            data_dir,
            int(max_rows),
            int(seed),
        )
    with tabs[2]:
        render_pca_explorer(analysis_frame, analysis_metadata, int(max_rows), int(seed))
    with tabs[3]:
        render_target_feature_value(analysis_frame, analysis_metadata, int(max_rows), int(seed))
    with tabs[4]:
        render_findings(analysis_frame, benchmarks, analysis_metadata)
    with tabs[5]:
        render_notes()


if __name__ == "__main__":
    main()
