from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import subprocess
import tempfile
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
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling.comparison import evaluate_models, missingness_report
from modeling.energy_measurements import (
    CONFIG_FIELDS as ENERGY_CONFIG_FIELDS,
    ENERGY_TARGET,
    POWER_METRIC as ENERGY_POWER_METRIC,
    THROUGHPUT_METRIC as ENERGY_THROUGHPUT_METRIC,
    WORKLOAD_FIELDS as ENERGY_WORKLOAD_FIELDS,
    available_control_values,
    energy_model_availability,
    energy_support_summary,
    exact_observed_lookup,
    mark_dominated_comparisons,
    nearest_measured_configurations,
)
from modeling.research_summary import build_research_summary
from modeling.pca_target_analysis import (
    ENERGY_TARGET as PCA_ENERGY_TARGET,
    LATENCY_TARGET as PCA_LATENCY_TARGET,
    OUTPUT_TARGET as PCA_OUTPUT_TARGET,
    PCA_FEATURES as TARGET_PCA_FEATURES,
    fit_shared_pca as fit_target_shared_pca,
    target_overlay as build_target_overlay,
)


ACTIVE_DUMP_VERSION = "db-dump/2026-07-20"
ACTIVE_DUMP_RELEASE = "InferenceX database snapshot 2026-07-20"
DEFAULT_DATA_DIR = os.environ.get(
    "INFERENCEX_DATA_DIR",
    "/tmp/inferencex-dump-comparison/db-dump-2026-07-20",
)
ROLLBACK_DATA_DIR = "inferencex-pca-data"
DEFAULT_JSON_DUMP_DIR = os.environ.get("INFERENCEX_JSON_DUMP_DIR", "")
REQUIRED_CSV_FILES = ("benchmark_results.csv", "configs.csv")
REQUIRED_RAW_CSV_FILES = ("benchmark_results_raw.csv", "configs.csv")
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
DEFAULT_GROUPED_FOLDS = 5
DEFAULT_PERMUTATION_REPEATS = 5
DEFAULT_PCA_STABILITY_RUNS = 5
DEFAULT_PCA_STABILITY_FRACTION = 0.8
RESEARCH_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PCA_TARGET_ARTIFACT_PATH = RESEARCH_ARTIFACT_DIR / "pca-db-dump-2026-07-20.json"
MAIN_TAB_LABELS = ("Overview", "Data Understanding", "PCA", "Model Results")
REMOVED_TOP_LEVEL_SECTION_LABELS = (
    "Data Preview",
    "Target-Aware Feature Value",
    "Model Comparison",
    "Findings",
    "Sales Pitch Visuals",
    "Research Results",
    "Notes",
)


st.set_page_config(
    page_title="InferenceX Benchmark Research",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def format_compact_count(value: int | float) -> str:
    """Format dashboard counts without changing the stored value."""
    value = int(value)
    return f"{value / 1_000:.1f}K" if abs(value) >= 10_000 else f"{value:,}"


def format_overview_r2(value: float) -> str:
    return f"{value:.3f}"


def format_overview_mae(value: float) -> str:
    return f"{value:.1f}"


def format_overview_percentage(value: float) -> str:
    return f"{value:.1%}"


def render_dashboard_shell() -> None:
    """Apply a small, theme-neutral visual foundation for the dashboard."""
    st.markdown(
        """
        <style>
        .main .block-container { max-width: 1200px; padding-top: 1.4rem; padding-bottom: 2.25rem; }
        .dashboard-title { margin: 0; font-size: 2rem; font-weight: 650; letter-spacing: -0.025em; }
        .dashboard-subtitle { margin: 0.3rem 0 0.55rem; color: rgba(127, 127, 127, 1); font-size: 1rem; }
        .status-badge { display: inline-block; padding: 0.16rem 0.55rem; border: 1px solid rgba(127, 127, 127, 0.35); border-radius: 999px; background: rgba(127, 127, 127, 0.10); font-size: 0.76rem; font-weight: 600; }
        .dashboard-card { min-height: 102px; padding: 0.8rem 0.9rem; border: 1px solid rgba(127, 127, 127, 0.26); border-radius: 0.6rem; background: rgba(127, 127, 127, 0.07); }
        .dashboard-card-label { color: rgba(127, 127, 127, 1); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.035em; }
        .dashboard-card-value { margin-top: 0.3rem; font-size: 1.38rem; font-weight: 650; line-height: 1.15; }
        .dashboard-card-detail { margin-top: 0.35rem; color: rgba(127, 127, 127, 1); font-size: 0.82rem; line-height: 1.3; }
        .dashboard-surface { padding: 0.95rem 1rem; border: 1px solid rgba(127, 127, 127, 0.26); border-radius: 0.65rem; background: rgba(127, 127, 127, 0.055); }
        .dashboard-takeaway { margin: 0.75rem 0 0.2rem; padding: 0.75rem 0.9rem; border-left: 3px solid rgba(127, 127, 127, 0.65); background: rgba(127, 127, 127, 0.08); border-radius: 0 0.45rem 0.45rem 0; }
        [data-baseweb="tab-list"] { gap: 0.35rem; margin-bottom: 1rem; }
        [data-baseweb="tab"] { height: 2.45rem; padding: 0 0.8rem; }
        [data-baseweb="tab"][aria-selected="true"] { font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compact_card(label: str, value: str, detail: str = "") -> None:
    """Render one escaped, compact display card."""
    escaped_detail = (
        f'<div class="dashboard-card-detail">{html.escape(detail)}</div>' if detail else ""
    )
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-card-label">{html.escape(label)}</div>'
        f'<div class="dashboard-card-value">{html.escape(value)}</div>{escaped_detail}</div>',
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, description: str) -> None:
    st.markdown(f"### {html.escape(title)}")
    st.caption(description)


def stable_json_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)
    try:
        hash(value)
    except TypeError:
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)
    return value


def read_records_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            rows = []
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if isinstance(value, dict):
                    rows.append(value)
            if rows:
                return rows
            raise

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
    for column in frame.columns:
        if pd.api.types.is_object_dtype(frame[column]):
            sample = frame[column].dropna().head(100)
            if sample.map(lambda value: isinstance(value, (dict, list, set, tuple))).any():
                frame[column] = frame[column].map(stable_json_value)
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


def has_required_files(directory: Path, files: tuple[str, ...]) -> bool:
    return all((directory / file_name).exists() for file_name in files)


def json_fallback_candidates(data_dir_text: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = [
        ("selected data directory", resolve_data_dir(data_dir_text)),
    ]
    if DEFAULT_JSON_DUMP_DIR:
        candidates.append(("INFERENCEX_JSON_DUMP_DIR", resolve_data_dir(DEFAULT_JSON_DUMP_DIR)))
    for dump_dir in sorted(Path.cwd().glob("inferencex-dump-*")):
        if dump_dir.is_dir():
            candidates.append((f"local dump: {dump_dir.name}", dump_dir))
    dump_dir_env = os.environ.get("DUMP_DIR")
    if dump_dir_env:
        candidates.append(("DUMP_DIR", resolve_data_dir(dump_dir_env)))

    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, directory in candidates:
        resolved = directory.resolve() if directory.exists() else directory
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((label, directory))
    return unique


def data_source_status(data_dir_text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_dir = resolve_data_dir(data_dir_text)
    rows = []
    for file_name in REQUIRED_CSV_FILES:
        path = csv_dir / file_name
        rows.append(
            {
                "mode": "CSV",
                "candidate": "selected data directory",
                "file": file_name,
                "found": path.exists(),
                "path": str(path),
                "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else np.nan,
            }
        )
    for file_name in REQUIRED_RAW_CSV_FILES:
        path = csv_dir / file_name
        rows.append(
            {
                "mode": "Raw CSV",
                "candidate": "selected data directory",
                "file": file_name,
                "found": path.exists(),
                "path": str(path),
                "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else np.nan,
            }
        )

    json_candidates = json_fallback_candidates(data_dir_text)
    for label, directory in json_candidates:
        for file_name in REQUIRED_JSON_FILES:
            path = directory / file_name
            rows.append(
                {
                    "mode": "JSON fallback",
                    "candidate": label,
                    "file": file_name,
                    "found": path.exists(),
                    "path": str(path),
                    "size_mb": path.stat().st_size / (1024 * 1024) if path.exists() else np.nan,
                }
            )
    status = pd.DataFrame(rows)
    csv_ready = bool(status[status["mode"] == "CSV"]["found"].all())
    raw_csv_ready = bool(status[status["mode"] == "Raw CSV"]["found"].all())
    json_ready_dir: Path | None = None
    json_ready_label = ""
    for label, directory in json_candidates:
        if has_required_files(directory, REQUIRED_JSON_FILES):
            json_ready_dir = directory
            json_ready_label = label
            break
    json_ready = json_ready_dir is not None
    active_mode = (
        "CSV"
        if csv_ready
        else "Raw CSV"
        if raw_csv_ready
        else "JSON fallback"
        if json_ready
        else "missing"
    )
    active_dir = (
        csv_dir
        if active_mode in {"CSV", "Raw CSV"}
        else json_ready_dir
        if active_mode == "JSON fallback" and json_ready_dir
        else csv_dir
    )
    return status, {
        "csv_ready": csv_ready,
        "raw_csv_ready": raw_csv_ready,
        "json_ready": json_ready,
        "active_mode": active_mode,
        "active_dir": active_dir,
        "active_candidate": "selected data directory" if active_mode in {"CSV", "Raw CSV"} else json_ready_label,
    }


def flatten_metrics_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand the official dump's JSON metrics column without changing row identity."""
    if "metrics" not in frame.columns:
        return frame

    def parse_metrics(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if pd.isna(value) or value == "":
            return {}
        try:
            parsed = json.loads(str(value))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    metrics = pd.json_normalize(frame["metrics"].map(parse_metrics)).add_prefix("metrics_")
    metrics.index = frame.index
    return pd.concat([frame.drop(columns=["metrics"]), metrics], axis=1)


@st.cache_data(show_spinner="Loading benchmark/config data")
def load_joined_data(
    data_dir_text: str,
    dataset_fingerprint: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    status, source_info = data_source_status(data_dir_text)
    active_mode = source_info["active_mode"]

    if active_mode == "CSV":
        data_dir = source_info["active_dir"]
        benchmarks = pd.read_csv(data_dir / "benchmark_results.csv", low_memory=False)
        configs = prefix_config_columns(pd.read_csv(data_dir / "configs.csv", low_memory=False))
        joined = join_benchmarks_configs(benchmarks, configs)
        return benchmarks, configs, joined, source_info

    if active_mode == "Raw CSV":
        data_dir = source_info["active_dir"]
        benchmarks = flatten_metrics_column(
            pd.read_csv(data_dir / "benchmark_results_raw.csv", low_memory=False)
        )
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
    return stable_json_value(value)


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


@st.cache_data(show_spinner=False)
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


def file_snapshot(path: Path, sample_bytes: int = 1_048_576) -> dict[str, Any]:
    """Return non-row-level metadata and a bounded content fingerprint for one source file."""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "content_sample_sha256": digest.hexdigest(),
    }


def build_dataset_manifest(source_info: dict[str, Any]) -> dict[str, Any]:
    active_dir = Path(source_info["active_dir"])
    required_files = (
        REQUIRED_CSV_FILES
        if source_info["active_mode"] == "CSV"
        else REQUIRED_RAW_CSV_FILES
        if source_info["active_mode"] == "Raw CSV"
        else REQUIRED_JSON_FILES
    )
    files = [file_snapshot(active_dir / file_name) for file_name in required_files]
    payload = {
        "active_mode": source_info["active_mode"],
        "active_dir": str(active_dir.resolve()),
        "files": files,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "fingerprint": fingerprint}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def analysis_signature(
    analysis_metadata: dict[str, Any],
    analysis_kind: str,
    controls: dict[str, Any],
) -> str:
    payload = {
        "dataset_fingerprint": analysis_metadata.get("dataset_fingerprint", ""),
        "analysis_unit": analysis_metadata.get("analysis_unit", ""),
        "analysis_row_count": analysis_metadata.get("analysis_row_count", 0),
        "analysis_kind": analysis_kind,
        "controls": controls,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def current_artifact(key: str, signature: str) -> dict[str, Any] | None:
    artifact = st.session_state.get(key)
    if artifact and artifact.get("analysis_signature") != signature:
        st.session_state.pop(key, None)
        return None
    return artifact


def signed_export(frame: pd.DataFrame, signature: str) -> pd.DataFrame:
    exported = frame.copy()
    exported.insert(0, "analysis_signature", signature)
    return exported


def default_pca_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    default_categorical = [
        column for column in DEFAULT_CATEGORICAL_FEATURES if column in config_categorical_columns(frame)
    ]
    default_numeric = [
        column for column in config_numeric_columns(frame) if column not in default_categorical
    ][:12]
    numeric, categorical, _ = normalize_feature_groups(default_numeric, default_categorical)
    return numeric, categorical


def default_target_features(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    default_categorical = [
        column for column in DEFAULT_CATEGORICAL_FEATURES if column in config_categorical_columns(frame)
    ]
    default_numeric = [
        column
        for column in config_numeric_columns(frame)
        if column != target and column not in default_categorical and not is_metric_column(column)
    ][:12]
    numeric, categorical, _ = normalize_feature_groups(default_numeric, default_categorical)
    return numeric, categorical


def pca_controls_from_state(frame: pd.DataFrame, max_rows: int, seed: int) -> dict[str, Any]:
    default_numeric, default_categorical = default_pca_features(frame)
    numeric, categorical, _ = normalize_feature_groups(
        list(st.session_state.get("pca_numeric_features", default_numeric)),
        list(st.session_state.get("pca_categorical_features", default_categorical)),
    )
    metric_cols = metric_like_numeric_columns(frame)
    target = st.session_state.get("pca_target_correlation_metric", metric_cols[0] if metric_cols else "")
    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target_metric": target,
        "max_rows": int(max_rows),
        "seed": int(seed),
        "stability_runs": int(st.session_state.get("pca_stability_runs", DEFAULT_PCA_STABILITY_RUNS)),
    }


def target_controls_from_state(frame: pd.DataFrame, max_rows: int, seed: int) -> dict[str, Any]:
    metric_cols = metric_like_numeric_columns(frame)
    target = st.session_state.get("target_metric", metric_cols[0] if metric_cols else "")
    default_numeric, default_categorical = default_target_features(frame, target)
    numeric, categorical, _ = normalize_feature_groups(
        list(st.session_state.get("target_numeric_features", default_numeric)),
        list(st.session_state.get("target_categorical_features", default_categorical)),
    )
    return {
        "target": target,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "max_rows": int(max_rows),
        "seed": int(seed),
        "split_mode": st.session_state.get(
            "target_split_mode", "Grouped cross-validation by config_id"
        ),
        "n_estimators": int(st.session_state.get("target_n_estimators", 150)),
        "include_other_metrics": bool(st.session_state.get("include_other_metrics", False)),
        "permutation_repeats": DEFAULT_PERMUTATION_REPEATS,
    }


def fit_pca_analysis(
    frame: pd.DataFrame,
    feature_columns: list[str],
    max_rows: int,
    seed: int,
    target_metric: str = "",
) -> tuple[dict[str, Any] | None, str]:
    work = sample_frame(frame, max_rows, seed)
    work = coerce_model_frame(work, feature_columns)
    if len(work) < 3:
        return None, "Not enough usable rows after dropping empty feature rows."

    numeric_features, categorical_features = split_features(work, feature_columns)
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    try:
        matrix = preprocessor.fit_transform(work[feature_columns])
    except Exception as exc:
        return None, f"Could not preprocess selected features: {exc}"
    if matrix.shape[1] < 2:
        return None, "PCA needs at least two encoded feature dimensions."

    pca = PCA(n_components=min(5, matrix.shape[1], len(work)), random_state=seed)
    coords = pca.fit_transform(matrix)
    explained = pd.DataFrame(
        {
            "component": [f"PC{idx + 1}" for idx in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    feature_names = [clean_feature_label(name) for name in preprocessor.get_feature_names_out()]
    loading_details = build_pca_loading_details(
        pca, feature_names, numeric_features, categorical_features
    )
    component_interpretations = build_component_interpretations(pca, loading_details)
    correlation_frame = (
        compute_pc_target_correlations(coords, work.index, frame, target_metric)
        if target_metric and target_metric in frame.columns
        else pd.DataFrame()
    )
    return {
        "sampled_rows": len(work),
        "input_feature_count": len(feature_columns),
        "input_features": feature_columns,
        "target_metric": target_metric,
        "encoded_contributions": loading_details,
        "source_contributions": original_feature_contributions(loading_details),
        "component_interpretations": component_interpretations,
        "pc_target_correlations": correlation_frame,
        "explained_variance": explained,
        "pc_scores": pd.DataFrame(
            coords[:, : min(5, coords.shape[1])],
            columns=[f"PC{idx + 1}" for idx in range(min(5, coords.shape[1]))],
            index=work.index,
        ),
        "component_group_vectors": {
            f"PC{idx + 1}": component_group_loadings(loading_details, f"PC{idx + 1}")
            .set_index("source_feature")["signed_loading"]
            .to_dict()
            for idx in range(len(pca.components_))
        },
    }, ""


def pca_stability_summary(
    frame: pd.DataFrame,
    feature_columns: list[str],
    max_rows: int,
    seed: int,
    runs: int,
) -> tuple[dict[str, Any], str]:
    run_count = max(2, int(runs))
    sample_rows = min(max_rows, max(3, int(round(len(frame) * DEFAULT_PCA_STABILITY_FRACTION))))
    runs_data: list[dict[str, Any]] = []
    for run_idx in range(run_count):
        result, error = fit_pca_analysis(
            frame, feature_columns, sample_rows, seed + run_idx, ""
        )
        if result is None:
            return {}, error
        runs_data.append(result)

    components = [f"PC{idx + 1}" for idx in range(min(5, len(runs_data[0]["explained_variance"])))]
    explained_rows = []
    component_rows = []
    base_vectors = runs_data[0]["component_group_vectors"]
    for component in components:
        values = [
            float(run["explained_variance"].loc[
                run["explained_variance"]["component"] == component,
                "explained_variance_ratio",
            ].iloc[0])
            for run in runs_data
        ]
        similarities = []
        base = pd.Series(base_vectors.get(component, {}), dtype=float)
        for run in runs_data[1:]:
            candidate = pd.Series(run["component_group_vectors"].get(component, {}), dtype=float)
            vector = pd.concat([base, candidate], axis=1).fillna(0.0)
            left, right = vector.iloc[:, 0].to_numpy(), vector.iloc[:, 1].to_numpy()
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarities.append(abs(float(np.dot(left, right) / denominator)) if denominator else np.nan)
        explained_rows.append(
            {
                "component": component,
                "mean_explained_variance": float(np.mean(values)),
                "std_explained_variance": float(np.std(values, ddof=0)),
                "min_explained_variance": float(np.min(values)),
                "max_explained_variance": float(np.max(values)),
            }
        )
        component_rows.append(
            {
                "component": component,
                "mean_sign_aligned_loading_similarity": float(np.nanmean(similarities)) if similarities else 1.0,
                "min_sign_aligned_loading_similarity": float(np.nanmin(similarities)) if similarities else 1.0,
            }
        )

    top_counts: dict[str, int] = {}
    for run in runs_data:
        for feature in run["source_contributions"].head(10)["source_feature"]:
            top_counts[feature] = top_counts.get(feature, 0) + 1
    feature_frequency = pd.DataFrame(
        [
            {"source_feature": feature, "top_driver_runs": count, "top_driver_frequency": count / run_count}
            for feature, count in top_counts.items()
        ]
    ).sort_values(["top_driver_frequency", "source_feature"], ascending=[False, True])
    component_stability = pd.DataFrame(component_rows)
    warnings = []
    unstable_components = component_stability[
        component_stability["min_sign_aligned_loading_similarity"] < 0.8
    ]["component"].tolist()
    if unstable_components:
        warnings.append(
            "Loading patterns were unstable for " + ", ".join(unstable_components) + "."
        )
    unstable_features = feature_frequency[
        feature_frequency["top_driver_frequency"] < 0.6
    ]["source_feature"].head(5).tolist()
    if unstable_features:
        warnings.append(
            "Some top-driver appearances were inconsistent: " + ", ".join(unstable_features) + "."
        )
    return {
        "runs": run_count,
        "sample_rows": sample_rows,
        "sample_fraction": sample_rows / max(len(frame), 1),
        "explained_variance": pd.DataFrame(explained_rows),
        "component_similarity": component_stability,
        "top_driver_frequency": feature_frequency,
        "warnings": warnings,
    }, ""


def grouped_rf_evaluation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    max_rows: int,
    seed: int,
    n_estimators: int,
    split_mode: str = "Grouped cross-validation by config_id",
    n_splits: int = DEFAULT_GROUPED_FOLDS,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
) -> tuple[dict[str, Any] | None, str]:
    work_columns = unique_preserve_order(
        feature_columns + [target] + (["config_id"] if "config_id" in frame.columns else [])
    )
    work = sample_frame(frame, max_rows, seed)
    work = work[[column for column in work_columns if column in work.columns]].replace(
        [np.inf, -np.inf], np.nan
    )
    work = work.dropna(axis=0, how="all", subset=feature_columns).dropna(axis=0, subset=[target])
    if len(work) < 20:
        return None, "Not enough rows with a non-null target value to train/test a model."

    X = work.loc[:, feature_columns]
    if not X.columns.is_unique:
        X = X.loc[:, ~X.columns.duplicated()]
        feature_columns = list(X.columns)
    numeric_features, categorical_features = split_features(work, feature_columns)
    y = work[target]
    warnings: list[str] = []
    groups: pd.Series | None = None
    if split_mode == "Grouped cross-validation by config_id" and "config_id" in work.columns:
        groups = work.loc[X.index, "config_id"].fillna("__missing_config_id__")
        group_count = int(groups.nunique(dropna=False))
        if group_count >= 2:
            fold_count = min(max(2, n_splits), group_count)
            splitter = GroupKFold(n_splits=fold_count)
            split_iter = splitter.split(X, y, groups=groups)
            evaluation_mode = "Grouped cross-validation by config_id"
        else:
            warnings.append("Fewer than two config_id groups; using random K-fold fallback.")
            groups = None
    else:
        if split_mode != "Grouped cross-validation by config_id":
            warnings.append("Random K-fold fallback selected; groups may overlap across folds.")
        else:
            warnings.append("config_id is unavailable; using random K-fold fallback.")

    if groups is None:
        fold_count = min(max(2, n_splits), len(work))
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
        evaluation_mode = "Random K-fold fallback"

    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, validation_idx) in enumerate(split_iter, start=1):
        X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
        y_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]
        model = Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=seed + fold_idx,
                        n_jobs=1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)
            importance = permutation_importance(
                model,
                X_validation,
                y_validation,
                n_repeats=permutation_repeats,
                random_state=seed + fold_idx,
                n_jobs=1,
                scoring="r2",
            )
        except Exception as exc:
            return None, f"Could not complete fold {fold_idx}: {exc}"
        train_groups = set(groups.iloc[train_idx]) if groups is not None else set()
        validation_groups = set(groups.iloc[validation_idx]) if groups is not None else set()
        overlap = len(train_groups.intersection(validation_groups))
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "validation_rows": len(validation_idx),
                "train_groups": len(train_groups) if groups is not None else np.nan,
                "validation_groups": len(validation_groups) if groups is not None else np.nan,
                "group_overlap": overlap if groups is not None else np.nan,
                "r2": r2_score(y_validation, predictions),
                "mae": mean_absolute_error(y_validation, predictions),
            }
        )
        for feature, mean, std in zip(
            feature_columns, importance.importances_mean, importance.importances_std
        ):
            importance_rows.append(
                {
                    "fold": fold_idx,
                    "feature": feature,
                    "importance_mean": mean,
                    "importance_std": std,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    importance_by_fold = pd.DataFrame(importance_rows)
    importance_frame = (
        importance_by_fold.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            folds=("fold", "nunique"),
        )
        .fillna({"importance_std": 0.0})
        .sort_values("importance_mean", ascending=False)
    )
    metric_summary = {
        metric: {
            "mean": float(fold_metrics[metric].mean()),
            "std": float(fold_metrics[metric].std(ddof=0)),
            "min": float(fold_metrics[metric].min()),
            "max": float(fold_metrics[metric].max()),
        }
        for metric in ("r2", "mae")
    }
    if groups is not None and int(fold_metrics["group_overlap"].max()) != 0:
        return None, "Grouped cross-validation produced overlapping config_id groups."
    return {
        "target_metric": target,
        "sampled_rows": len(work),
        "predictor_count": len(feature_columns),
        "split_mode": evaluation_mode,
        "fold_count": fold_count,
        "fold_metrics": fold_metrics,
        "metric_summary": metric_summary,
        "r2": metric_summary["r2"]["mean"],
        "mae": metric_summary["mae"]["mean"],
        "importance_frame": importance_frame,
        "importance_by_fold": importance_by_fold,
        "warnings": warnings,
    }, ""


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


READABLE_FEATURE_LABELS = {
    "config_disagg": "Disaggregated serving",
    "config_is_multinode": "Multinode serving",
    "config_decode_tp": "Decode tensor parallelism",
    "config_prefill_tp": "Prefill tensor parallelism",
    "config_prefill_ep": "Prefill expert parallelism",
    "config_decode_ep": "Decode expert parallelism",
    "config_num_prefill_gpu": "Prefill GPU allocation",
    "config_num_decode_gpu": "Decode GPU allocation",
    "config_decode_num_workers": "Decode worker count",
    "config_prefill_num_workers": "Prefill worker count",
    "config_decode_dp_attention": "Decode attention parallelism",
    "config_prefill_dp_attention": "Prefill attention parallelism",
    "config_framework": "Serving framework",
    "config_hardware": "Hardware platform",
    "config_model": "Model family",
    "config_precision": "Numeric precision",
    "config_spec_method": "Speculative decoding method",
    "isl": "Input sequence length",
    "osl": "Output sequence length",
    "conc": "Concurrency",
    "benchmark_type": "Workload type",
}


def readable_feature_label(feature: str) -> str:
    if feature in READABLE_FEATURE_LABELS:
        return READABLE_FEATURE_LABELS[feature]
    cleaned = feature
    for prefix in ("config_", "metrics_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    parts = cleaned.split("_")
    acronyms = {
        "p50",
        "p90",
        "p95",
        "p99",
        "itl",
        "ttft",
        "tpot",
        "e2el",
        "qps",
        "rps",
        "gpu",
        "tp",
        "ep",
        "dp",
    }
    return " ".join(part.upper() if part.lower() in acronyms else part.title() for part in parts)


def readable_feature_list(features: list[str], max_items: int = 3) -> str:
    labels = [readable_feature_label(feature) for feature in features if feature]
    if not labels:
        return "mixed infrastructure choices"
    return ", ".join(labels[:max_items])


def sales_feature_category(feature: str) -> str:
    serving_topology = {
        "config_disagg",
        "config_is_multinode",
        "config_prefill_tp",
        "config_prefill_ep",
        "config_prefill_dp_attention",
        "config_prefill_num_workers",
        "config_decode_tp",
        "config_decode_ep",
        "config_decode_dp_attention",
        "config_decode_num_workers",
        "config_num_prefill_gpu",
        "config_num_decode_gpu",
    }
    workload_shape = {"isl", "osl", "conc", "benchmark_type"}
    model_hardware_software = {
        "config_model",
        "config_hardware",
        "config_framework",
        "config_precision",
        "config_spec_method",
    }
    if feature in serving_topology:
        return "Serving topology / parallelism"
    if feature in workload_shape:
        return "Workload shape"
    if feature in model_hardware_software:
        return "Model / hardware / software"
    return "Other"


def infer_sales_axis_name(
    component: str,
    group_loadings: pd.DataFrame,
    used_names: set[str],
) -> str:
    groups = set(group_loadings.head(6)["source_feature"].tolist())
    positive = set(
        group_loadings[group_loadings["signed_loading"] > 0]
        .sort_values("signed_loading", ascending=False)
        .head(4)["source_feature"]
        .tolist()
    )
    negative = set(
        group_loadings[group_loadings["signed_loading"] < 0]
        .sort_values("signed_loading", ascending=True)
        .head(4)["source_feature"]
        .tolist()
    )

    if component == "PC1" and groups.intersection({"config_disagg", "config_is_multinode"}):
        return "Disaggregated / multinode serving"
    if (
        component == "PC2"
        and {"config_num_prefill_gpu", "config_prefill_tp", "config_prefill_ep"}.issubset(positive)
        and negative.intersection({"config_is_multinode", "config_disagg"})
    ):
        return "Prefill scaling vs distributed serving"
    if component == "PC3" and {"config_prefill_tp", "config_decode_tp"}.issubset(groups):
        return "Prefill/decode scaling mix"
    if {"isl", "osl"}.issubset(groups) and (
        ("isl" in positive and "osl" in negative)
        or ("osl" in positive and "isl" in negative)
    ):
        return "Input vs output workload shape"
    if groups.intersection({"config_prefill_tp", "config_decode_tp"}) and groups.intersection(
        {
            "config_prefill_ep",
            "config_decode_ep",
            "config_prefill_dp_attention",
            "config_decode_dp_attention",
            "conc",
        }
    ):
        return "Prefill/decode parallelism tradeoff"
    if groups.intersection(
        {
            "config_prefill_tp",
            "config_num_prefill_gpu",
            "config_prefill_ep",
        }
    ):
        name = "Prefill GPU and tensor-parallel scaling"
        if name not in used_names:
            return name
    if groups.intersection({"config_decode_tp", "config_num_decode_gpu"}):
        name = "Decode GPU and tensor-parallel scaling"
        if name not in used_names:
            return name
    if groups.intersection({"config_prefill_ep", "config_decode_ep"}):
        return "Expert-parallel serving"
    if groups.intersection({"isl", "osl", "conc"}):
        return "Workload shape"
    return "Configuration mix"


def unique_sales_axis_name(axis_name: str, feature_groups: list[str], used_names: set[str]) -> str:
    if axis_name not in used_names:
        return axis_name
    groups = set(feature_groups)
    alternatives = []
    if groups.intersection({"config_prefill_tp", "config_prefill_ep", "config_num_prefill_gpu"}):
        alternatives.append("Prefill-side scaling mix")
    if groups.intersection({"config_decode_tp", "config_decode_ep", "config_num_decode_gpu"}):
        alternatives.append("Decode-side scaling mix")
    if groups.intersection({"isl", "osl", "conc"}):
        alternatives.append("Workload pressure mix")
    if groups.intersection({"config_disagg", "config_is_multinode"}):
        alternatives.append("Serving topology mix")
    alternatives.append("Configuration mix")
    for alternative in alternatives:
        if alternative not in used_names:
            return alternative
    return f"{axis_name} ({len(used_names) + 1})"


def component_group_loadings(
    loading_details: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    loading_col = f"{component}_loading"
    if loading_col not in loading_details.columns:
        return pd.DataFrame()
    grouped = (
        loading_details.assign(abs_loading=loading_details[loading_col].abs())
        .groupby("source_feature", as_index=False)
        .agg(
            signed_loading=(loading_col, "sum"),
            absolute_loading=("abs_loading", "sum"),
        )
    )
    return grouped.sort_values("absolute_loading", ascending=False)


def build_sales_component_cards(
    loading_details: pd.DataFrame,
    explained: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for row in explained.head(4).itertuples(index=False):
        component = row.component
        group_loadings = component_group_loadings(loading_details, component)
        dominant_groups = group_loadings.head(3)["source_feature"].tolist()
        axis_name = infer_sales_axis_name(component, group_loadings, used_names)
        axis_name = unique_sales_axis_name(axis_name, dominant_groups, used_names)
        used_names.add(axis_name)
        positive = (
            group_loadings[group_loadings["signed_loading"] > 0]
            .sort_values("signed_loading", ascending=False)
            .head(3)["source_feature"]
            .tolist()
        )
        negative = (
            group_loadings[group_loadings["signed_loading"] < 0]
            .sort_values("signed_loading", ascending=True)
            .head(3)["source_feature"]
            .tolist()
        )
        rows.append(
            {
                "component": component,
                "axis_name": axis_name,
                "explained_variance_ratio": row.explained_variance_ratio,
                "cumulative_explained_variance": row.cumulative_explained_variance,
                "top_positive_drivers": readable_feature_list(positive),
                "top_negative_drivers": readable_feature_list(negative),
                "raw_top_positive_drivers": ", ".join(positive),
                "raw_top_negative_drivers": ", ".join(negative),
                "dominant_feature_groups": readable_feature_list(dominant_groups),
                "raw_dominant_feature_groups": ", ".join(dominant_groups),
                "interpretation": (
                    f"{component} mostly separates benchmark configurations by "
                    f"{readable_feature_list(dominant_groups)}, so it is best "
                    f"read as a {axis_name.lower()} axis."
                ),
            }
        )
    return pd.DataFrame(rows)


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
        f"- Dataset fingerprint: `{dataset_summary.get('dataset_fingerprint', '')}`",
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
            f"- Analysis signature: `{pca_analysis.get('analysis_signature', '')}`",
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
            f"- RF analysis signature: `{target_analysis.get('analysis_signature', '') if target_analysis else ''}`",
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
            "- CSV is the preferred source; JSON fallback intentionally skips giant log/sample files.",
        ]
    )
    return "\n".join(lines)


def build_structure_vs_performance(
    source_contributions: pd.DataFrame,
    target_importance: pd.DataFrame,
) -> pd.DataFrame:
    pca_top = source_contributions.head(10).reset_index(drop=True).copy()
    pca_top["PCA rank"] = pca_top.index + 1
    pca_lookup = pca_top.set_index("raw_feature_name").to_dict(orient="index")

    if target_importance.empty:
        rows = []
        for _, row in pca_top.iterrows():
            rows.append(
                {
                    "Feature": row["readable_feature_label"],
                    "raw_feature_name": row["raw_feature_name"],
                    "PCA rank": row["PCA rank"],
                    "PCA contribution": row["weighted_contribution"],
                    "PCA contribution share": row["contribution_share"],
                    "Target rank": np.nan,
                    "Target importance": np.nan,
                    "Interpretation": "Structural only",
                    "Why it matters": (
                        "Major configuration differentiator; not yet shown as a top "
                        "predictor for this metric."
                    ),
                }
            )
        return pd.DataFrame(rows)

    target_top = target_importance.head(10).reset_index(drop=True).copy()
    target_top["Target rank"] = target_top.index + 1
    target_lookup = target_top.set_index("feature").to_dict(orient="index")
    features = unique_preserve_order(
        pca_top["raw_feature_name"].tolist() + target_top["feature"].tolist()
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        pca_row = pca_lookup.get(feature, {})
        target_row = target_lookup.get(feature, {})
        in_pca = bool(pca_row)
        in_target = bool(target_row)
        if in_pca and in_target:
            interpretation = "Structural and predictive"
            why = "Shapes configuration space and predicts the selected performance outcome."
        elif in_pca:
            interpretation = "Structural only"
            why = (
                "Major configuration differentiator; not yet shown as a top predictor "
                "for this metric."
            )
        else:
            interpretation = "Predictive only"
            why = "Not a major structural axis, but important for predicting this target."
        rows.append(
            {
                "Feature": readable_feature_label(feature),
                "raw_feature_name": feature,
                "PCA rank": pca_row.get("PCA rank", np.nan),
                "PCA contribution": pca_row.get("weighted_contribution", np.nan),
                "PCA contribution share": pca_row.get("contribution_share", np.nan),
                "Target rank": target_row.get("Target rank", np.nan),
                "Target importance": target_row.get("importance_mean", np.nan),
                "Interpretation": interpretation,
                "Why it matters": why,
            }
        )
    return pd.DataFrame(rows)


def default_sales_target_metric(metric_cols: list[str]) -> str:
    for column in metric_cols:
        if column.lower() == "metrics_p99_itl":
            return column
    return metric_cols[0] if metric_cols else ""


def compute_target_importance_for_sales(
    joined: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
    target: str,
) -> tuple[dict[str, Any] | None, str]:
    if not target or target not in joined.columns:
        return None, "No selected target metric is available for target-aware modeling."

    default_numeric, default_categorical = default_target_features(joined, target)
    feature_columns = unique_preserve_order(default_numeric + default_categorical)
    if not feature_columns:
        return None, "No setup/configuration predictors were available for the target model."
    result, error = grouped_rf_evaluation(
        joined,
        feature_columns,
        target,
        max_rows,
        seed,
        n_estimators=150,
        permutation_repeats=3,
    )
    if result is None:
        return None, error
    result.update(
        {
            "analysis_unit": analysis_metadata["analysis_unit"],
            "raw_row_count": analysis_metadata["raw_row_count"],
            "analysis_row_count": analysis_metadata["analysis_row_count"],
            "grouping_keys": analysis_metadata["grouping_keys"],
            "auto_generated": True,
        }
    )
    return result, ""


def build_sales_pitch_markdown(
    dataset_summary: dict[str, Any],
    source_contributions: pd.DataFrame,
    component_cards: pd.DataFrame,
    selected_target: str,
    bridge: pd.DataFrame,
    pca_signature: str = "",
    target_signature: str = "",
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    top_groups = source_contributions.head(3)["raw_feature_name"].tolist()
    structural_predictive = bridge[
        bridge["Interpretation"] == "Structural and predictive"
    ]["Feature"].tolist()
    overlap_text = compact_list(structural_predictive) if structural_predictive else ""
    lines = [
        "# InferenceX PCA Sales Pitch Summary",
        "",
        f"Generated: {timestamp}",
        "",
        "## Top 3 Sales Takeaways",
        "",
        (
            "- Serving architecture dominates the benchmark landscape: the top structural "
            "drivers are "
            f"{readable_feature_list(top_groups)}."
        ),
        (
            "- The first four PCA axes compress thousands of config/workload rows into "
            "readable infrastructure dimensions."
        ),
        (
            "- Overlap features are the strongest candidates for infrastructure valuation: "
            f"{overlap_text}."
            if overlap_text
            else (
                "- Performance overlays separate configuration structure from outcome "
                "prediction; run the target-aware model to identify which structural "
                "drivers also predict performance."
            )
        ),
        "",
        "## Dataset Summary",
        "",
        f"- Analysis unit: {dataset_summary.get('analysis_unit', 'unknown')}",
        f"- Raw rows: {dataset_summary.get('raw_row_count', 0):,}",
        f"- Analysis rows: {dataset_summary.get('analysis_row_count', 0):,}",
        f"- Grouping keys: {', '.join(dataset_summary.get('grouping_keys', [])) or 'none'}",
        f"- PCA analysis signature: `{pca_signature}`",
        f"- Target analysis signature: `{target_signature}`",
        "",
        "## Top PCA Feature Groups",
        "",
        dataframe_to_markdown(source_contributions.head(10), 10),
        "",
        "## PC1-PC4 Names",
        "",
        dataframe_to_markdown(
            component_cards[
                [
                    "component",
                    "axis_name",
                    "explained_variance_ratio",
                    "cumulative_explained_variance",
                    "dominant_feature_groups",
                ]
            ],
            4,
        ),
        "",
        "## Selected Target Metric",
        "",
        f"`{selected_target or 'Not selected'}`",
        "",
        "## Structural vs Predictive Overlap",
        "",
        dataframe_to_markdown(bridge, 20),
        "",
        "## Limitations",
        "",
        "- PCA means structure, not value.",
        "- Loading means feature contribution to a synthetic PC axis.",
        "- Correlation and permutation importance are descriptive, not causal proof.",
        "- p99 ITL means bad-case inter-token delay; TTFT is time until first token; TPOT is time per output token.",
        "- Results depend on the selected analysis unit, sampled rows, and target metric.",
    ]
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
                    "value": str(value),
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
def load_optional_small_tables(
    data_dir_text: str,
    dataset_fingerprint: str = "",
    max_mb: float = 10.0,
) -> dict[str, pd.DataFrame]:
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
    st.dataframe(column_summary, width="stretch", height=360)

    st.subheader("Sample Rows")
    st.dataframe(joined.head(100), width="stretch", height=420)


def render_data_understanding(
    joined: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    data_dir: str,
    max_rows: int,
    seed: int,
) -> None:
    render_section_intro("Data Understanding", "Coverage, target availability, and the configuration and workload mix behind the active analysis unit.")
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
        st.dataframe(repeated_only.head(30), width="stretch", hide_index=True)
        st.caption(
            f"{len(repeated_only):,} groups have repeated raw rows out of "
            f"{len(repeat_summary):,} total config/workload/concurrency groups."
        )

    if st.checkbox("Optionally load small side tables", value=False):
        optional_tables = load_optional_small_tables(
            data_dir, analysis_metadata.get("dataset_fingerprint", "")
        )
        if optional_tables:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                        for name, frame in optional_tables.items()
                    ]
                ),
                width="stretch",
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
    st.dataframe(dictionary_view, width="stretch", height=480, hide_index=True)

    st.subheader("Distribution Summary")
    st.caption(f"Distribution summaries use up to {len(sample):,} sampled rows.")
    dist_left, dist_right = st.columns(2)
    with dist_left:
        st.markdown("**Numeric columns**")
        st.dataframe(numeric_summary, width="stretch", height=420, hide_index=True)
    with dist_right:
        st.markdown("**Categorical top values**")
        st.dataframe(categorical_top_values, width="stretch", height=420, hide_index=True)

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
        matrix_cols[0].dataframe(hw_framework, width="stretch")
        matrix_cols[0].plotly_chart(
            px.imshow(hw_framework, text_auto=True, aspect="auto"),
            width="stretch",
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
        matrix_cols[1].dataframe(model_hw, width="stretch")
        matrix_cols[1].plotly_chart(
            px.imshow(model_hw, text_auto=True, aspect="auto"),
            width="stretch",
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
            workload_cols[idx].dataframe(counts, width="stretch", hide_index=True)
            workload_cols[idx].plotly_chart(
                px.bar(counts, x=column, y="row_count", title=column.upper()),
                width="stretch",
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
        st.dataframe(seq_pairs, width="stretch", hide_index=True)

    st.subheader("Metric Target Guide")
    st.dataframe(metric_target_guide, width="stretch", height=420, hide_index=True)

    st.subheader("Data Quality Checks")
    q_cols = st.columns(3)
    q_cols[0].metric("Duplicate key rows", f"{quality_report['duplicate_rows']:,}")
    q_cols[1].metric("Failed rows", f"{quality_report['failed_rows']:,}")
    q_cols[2].metric("Excluded columns", f"{len(quality_report['excluded']):,}")
    st.markdown("**Missing metrics**")
    st.dataframe(quality_report["missing_metrics"].head(30), width="stretch", hide_index=True)
    dq_cols = st.columns(2)
    dq_cols[0].markdown("**High-cardinality metadata/provenance**")
    dq_cols[0].dataframe(quality_report["high_cardinality"], width="stretch", hide_index=True)
    dq_cols[1].markdown("**One-unique-value columns**")
    dq_cols[1].dataframe(quality_report["single_value"], width="stretch", hide_index=True)
    st.markdown("**Suspicious values**")
    if quality_report["suspicious"].empty:
        st.success("No negative metric or non-positive workload values found in the loaded rows.")
    else:
        st.dataframe(quality_report["suspicious"], width="stretch", hide_index=True)
    st.markdown("**Excluded from PCA/modeling by default**")
    st.dataframe(quality_report["excluded"], width="stretch", hide_index=True)

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
    default_numeric, default_categorical = default_pca_features(joined)
    color_options = unique_preserve_order(
        [""] + metric_cols + config_categorical_columns(joined) + categorical_candidates + numeric_candidates
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
    target_metric = ""
    if metric_cols:
        target_metric = st.selectbox(
            "Performance metric for PC correlation",
            options=metric_cols,
            key="pca_target_correlation_metric",
        )
    stability_runs = st.slider(
        "PCA stability runs",
        min_value=2,
        max_value=10,
        value=DEFAULT_PCA_STABILITY_RUNS,
        key="pca_stability_runs",
        help="Repeated deterministic 80% samples of the current analysis frame.",
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
    controls = pca_controls_from_state(joined, max_rows, seed)
    controls["target_metric"] = target_metric
    controls["stability_runs"] = stability_runs
    signature = analysis_signature(analysis_metadata, "pca", controls)
    pca_analysis = current_artifact("pca_analysis", signature)
    if pca_analysis is None:
        with st.spinner("Fitting PCA and deterministic stability samples"):
            pca_analysis, error = fit_pca_analysis(
                joined, feature_columns, max_rows, seed, target_metric
            )
            if pca_analysis is not None:
                stability, stability_error = pca_stability_summary(
                    joined, feature_columns, max_rows, seed, stability_runs
                )
                if stability_error:
                    error = stability_error
                else:
                    pca_analysis["stability"] = stability
        if pca_analysis is None or error:
            st.error(error)
            return
        pca_analysis.update(
            {
                "analysis_signature": signature,
                "controls": controls,
                "analysis_unit": analysis_metadata["analysis_unit"],
                "raw_row_count": analysis_metadata["raw_row_count"],
                "analysis_row_count": analysis_metadata["analysis_row_count"],
                "grouping_keys": analysis_metadata["grouping_keys"],
            }
        )
        st.session_state["pca_analysis"] = pca_analysis

    explained = pca_analysis["explained_variance"]
    loading_details = pca_analysis["encoded_contributions"]
    source_contributions = pca_analysis["source_contributions"]
    component_interpretations = pca_analysis["component_interpretations"]
    st.subheader("Explained Variance")
    st.dataframe(explained, width="stretch", hide_index=True)
    st.plotly_chart(
        px.bar(
            explained,
            x="component",
            y="explained_variance_ratio",
            text="explained_variance_ratio",
        ).update_traces(texttemplate="%{text:.1%}", textposition="outside"),
        width="stretch",
    )
    plot_frame = pca_analysis["pc_scores"][["PC1", "PC2"]].copy()
    if color_by and color_by in joined.columns:
        plot_frame[color_by] = joined.reindex(plot_frame.index)[color_by]

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
        width="stretch",
    )

    st.caption(
        "PC1 and PC2 are synthetic axes. The useful PCA output is the "
        "loading/contribution table, which maps those axes back to original features."
    )

    st.subheader("Feature Contribution Summary")
    st.dataframe(
        loading_details.head(30),
        width="stretch",
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
        width="stretch",
    )

    st.subheader("Top Original Feature Groups by Variance Contribution")
    source_contributions = original_feature_contributions(loading_details)
    st.dataframe(
        source_contributions.head(30),
        width="stretch",
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
        width="stretch",
    )

    st.subheader("Component Interpretation Cards")
    for row_start in range(0, min(4, len(component_interpretations)), 2):
        card_columns = st.columns(2)
        for card_column, row in zip(
            card_columns,
            component_interpretations.iloc[row_start : row_start + 2].itertuples(index=False),
        ):
            with card_column.container(border=True):
                st.markdown(f"#### {row.component}")
                st.metric("Explained variance", f"{row.explained_variance_ratio:.1%}")
                st.metric("Cumulative variance", f"{row.cumulative_explained_variance:.1%}")
                st.write(row.interpretation)
                st.caption(f"Top absolute loadings: {row.top_absolute}")

    st.subheader("PCA Stability and Sensitivity")
    stability = pca_analysis["stability"]
    st.caption(
        f"{stability['runs']} deterministic runs, each sampled at {stability['sample_rows']:,} "
        f"rows ({stability['sample_fraction']:.0%} of this analysis frame)."
    )
    st.dataframe(stability["explained_variance"], width="stretch", hide_index=True)
    st.dataframe(stability["component_similarity"], width="stretch", hide_index=True)
    st.dataframe(stability["top_driver_frequency"].head(15), width="stretch", hide_index=True)
    if stability["warnings"]:
        for warning in stability["warnings"]:
            st.warning(warning)
    else:
        st.success("Top-driver and sign-aligned loading checks were stable across sampled runs.")

    st.subheader("PC vs Target Correlation")
    if target_metric:
        st.dataframe(pca_analysis["pc_target_correlations"], width="stretch", hide_index=True)
    else:
        st.info("No metric-like numeric columns are available for target correlation.")

    component_options = explained["component"].tolist()
    selected_component = st.selectbox("Loadings component", options=component_options)
    loading_frame = loading_details[["encoded_feature", f"{selected_component}_loading"]].rename(
        columns={f"{selected_component}_loading": "loading"}
    )
    loading_frame["abs_loading"] = loading_frame["loading"].abs()
    loading_frame["weighted_abs_loading"] = (
        loading_frame["abs_loading"]
        * explained.loc[explained["component"] == selected_component, "explained_variance_ratio"].iloc[0]
    )
    loading_frame = loading_frame.sort_values("abs_loading", ascending=False).head(30)

    st.subheader("Top PCA Loadings")
    st.dataframe(loading_frame, width="stretch", hide_index=True)
    st.plotly_chart(
        px.bar(
            loading_frame.sort_values("abs_loading"),
            x="abs_loading",
            y="encoded_feature",
            orientation="h",
            hover_data=["loading", "weighted_abs_loading"],
        ),
        width="stretch",
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
    include_other_metrics = st.checkbox(
        "Allow other metric-like columns as predictors", value=False, key="include_other_metrics"
    )
    default_numeric, default_categorical = default_target_features(joined, target)
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
    split_options = ["Random K-fold fallback"]
    if "config_id" in joined.columns:
        split_options.insert(0, "Grouped cross-validation by config_id")
    split_mode = st.selectbox("Evaluation mode", options=split_options, key="target_split_mode")
    if split_mode == "Random K-fold fallback":
        st.warning(
            "Random K-fold fallback can mix repeated configurations across validation folds."
        )
    n_estimators = st.slider(
        "Random forest trees", min_value=50, max_value=400, value=150, step=50,
        key="target_n_estimators",
    )

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
    controls = target_controls_from_state(joined, max_rows, seed)
    controls.update(
        {
            "target": target,
            "numeric_features": selected_numeric,
            "categorical_features": selected_categorical,
            "split_mode": split_mode,
            "n_estimators": n_estimators,
            "include_other_metrics": include_other_metrics,
        }
    )
    signature = analysis_signature(analysis_metadata, "rf", controls)
    target_analysis = current_artifact("target_analysis", signature)
    if target_analysis is None:
        with st.spinner("Running deterministic cross-validation and held-out permutation importance"):
            target_analysis, error = grouped_rf_evaluation(
                joined,
                feature_columns,
                target,
                max_rows,
                seed,
                n_estimators,
                split_mode=split_mode,
            )
        if target_analysis is None:
            st.error(error)
            return
        target_analysis.update(
            {
                "analysis_signature": signature,
                "controls": controls,
                "analysis_unit": analysis_metadata["analysis_unit"],
                "raw_row_count": analysis_metadata["raw_row_count"],
                "analysis_row_count": analysis_metadata["analysis_row_count"],
                "grouping_keys": analysis_metadata["grouping_keys"],
            }
        )
        st.session_state["target_analysis"] = target_analysis

    summary = target_analysis["metric_summary"]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Folds", str(target_analysis["fold_count"]))
    col_b.metric("CV R2", f"{summary['r2']['mean']:.3f} ± {summary['r2']['std']:.3f}")
    col_c.metric("CV MAE", f"{summary['mae']['mean']:.3f} ± {summary['mae']['std']:.3f}")
    col_d.metric("R2 range", f"{summary['r2']['min']:.3f} to {summary['r2']['max']:.3f}")
    st.caption(
        f"MAE range: {summary['mae']['min']:.3f} to {summary['mae']['max']:.3f}. "
        f"Evaluation: {target_analysis['split_mode']}."
    )
    for warning in target_analysis["warnings"]:
        st.warning(warning)
    st.subheader("Per-Fold Metrics")
    st.dataframe(target_analysis["fold_metrics"], width="stretch", hide_index=True)
    st.subheader("Cross-Validated Permutation Importance")
    importance_frame = target_analysis["importance_frame"]
    st.dataframe(importance_frame, width="stretch", hide_index=True)
    st.plotly_chart(
        px.bar(
            importance_frame.head(30).sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
        ),
        width="stretch",
    )


def run_tabfm_comparison_subprocess(
    data_dir: str,
    target: str,
    folds: int,
    max_rows: int,
    seed: int,
    context_cap: int,
    analysis_unit: str,
) -> tuple[dict[str, Any] | None, str]:
    """Keep TabFM imports and checkpoint loading out of the Streamlit process."""
    interpreter = Path(".venv-tabfm/bin/python")
    script = Path("scripts/model_comparison.py")
    if not interpreter.exists():
        return None, "TabFM environment is unavailable at .venv-tabfm/bin/python."
    if not script.exists():
        return None, "The TabFM comparison script is unavailable."
    temporary = tempfile.NamedTemporaryFile(prefix="inferencex-tabfm-", suffix=".json", delete=False)
    temporary.close()
    command = [
        str(interpreter), str(script), "--data-dir", data_dir, "--target", target,
        "--models", "tabfm", "--folds", str(folds), "--max-rows", str(max_rows),
        "--seed", str(seed), "--tabfm-max-context", str(context_cap),
        "--analysis-unit", analysis_unit, "--output", temporary.name,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            return None, f"TabFM subprocess failed: {message[-1000:]}"
        artifact = json.loads(Path(temporary.name).read_text(encoding="utf-8"))
        return artifact["comparison"]["models"]["tabfm"], ""
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError) as exc:
        return None, f"TabFM subprocess failed: {type(exc).__name__}: {exc}"
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def render_model_comparison(
    joined: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    data_dir: str,
    max_rows: int,
    seed: int,
) -> None:
    """Compact, explicit comparison UI. TabFM remains research-only and subprocess-bound."""
    metric_cols = metric_like_numeric_columns(joined)
    if not metric_cols:
        st.warning("No numeric metric-like target columns were detected.")
        return
    target = st.selectbox("Comparison target", metric_cols, key="comparison_target")
    numeric, categorical = default_target_features(joined, target)
    features = numeric + categorical
    model_options = ["random_forest", "catboost", "tabfm"]
    models = st.multiselect("Models", model_options, default=["random_forest", "catboost"], key="comparison_models")
    folds = st.slider("Grouped folds", min_value=1, max_value=5, value=5, key="comparison_folds")
    context_cap = st.number_input("TabFM context-row cap", min_value=16, max_value=2_000, value=128, step=16, key="comparison_context_cap")
    st.caption(
        "Fold-local policy: missing targets are excluded; numeric predictors use training-fold medians plus indicators; categoricals use `__MISSING__`. TabFM is optional research-only CPU inference through `.venv-tabfm/bin/python` and is never imported at normal startup."
    )
    availability = {
        "random_forest": "available",
        "catboost": "available" if importlib.util.find_spec("catboost") else "unavailable in this environment",
        "tabfm": "research-only subprocess" if Path(".venv-tabfm/bin/python").exists() else "TabFM environment unavailable",
    }
    st.dataframe(pd.DataFrame([{"model": name, "availability": availability[name]} for name in model_options]), width="stretch", hide_index=True)
    controls = {"target": target, "features": features, "models": models, "folds": folds, "max_rows": max_rows, "seed": seed, "context_cap": int(context_cap)}
    signature = analysis_signature(analysis_metadata, "model-comparison", controls)
    comparison = current_artifact("model_comparison", signature)
    if st.button("Run model comparison", type="primary", disabled=not models):
        with st.spinner("Running grouped, fold-safe model comparison"):
            requested_standard = [name for name in models if name != "tabfm"]
            try:
                comparison_result = evaluate_models(joined, features, target, requested_standard, max_rows, seed, folds)
                if "tabfm" in models:
                    tabfm, error = run_tabfm_comparison_subprocess(data_dir, target, folds, max_rows, seed, int(context_cap), analysis_metadata["analysis_unit"])
                    if tabfm is None:
                        comparison_result["models"]["tabfm"] = {"available": False, "availability": "subprocess", "folds": [], "metrics": {}, "importance": [], "runtime_seconds": 0.0, "error": error}
                    else:
                        comparison_result["models"]["tabfm"] = tabfm
                comparison = {"analysis_signature": signature, "controls": controls, "result": comparison_result, "missingness": missingness_report(joined, features, metric_cols)}
                st.session_state["model_comparison"] = comparison
            except (ValueError, AssertionError) as exc:
                st.error(str(exc))
                return
    if not comparison:
        st.info("Choose models and click Run model comparison. Results are invalidated whenever target, features, folds, row cap, seed, context cap, or model selection changes.")
        return
    result = comparison["result"]
    st.caption(f"Dataset/control signature: `{comparison['analysis_signature']}`. Features: {', '.join(features)}")
    summary_rows = []
    for name, model in result["models"].items():
        metrics = model.get("metrics", {})
        summary_rows.append({"model": name, "available": model.get("available"), "r2_mean": metrics.get("r2", {}).get("mean"), "r2_std": metrics.get("r2", {}).get("std"), "mae_mean": metrics.get("mae", {}).get("mean"), "mae_std": metrics.get("mae", {}).get("std"), "runtime_seconds": model.get("runtime_seconds"), "error": model.get("error", "")})
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    for name, model in result["models"].items():
        if model.get("folds"):
            st.subheader(f"{name} folds")
            st.dataframe(pd.DataFrame(model["folds"]), width="stretch", hide_index=True)
        if model.get("error"):
            st.warning(f"{name}: {model['error']}")
    missing = comparison["missingness"]
    st.subheader("Compact missingness audit")
    st.caption(f"Usable rows by target: {missing['usable_rows_per_target']}. Aggregate-only summaries; target values are never imputed.")
    st.dataframe(pd.DataFrame(missing["columns"])[["column", "role", "missing_count", "missing_percentage", "complete_case_count", "likely_structural_or_not_applicable"]], width="stretch", hide_index=True)


def render_findings(
    joined: pd.DataFrame,
    benchmarks: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    st.header("Findings")
    st.caption(
        "A compact executive summary of the PCA structure, target-aware predictors, "
        "and where the two views agree."
    )

    pca_signature = analysis_signature(
        analysis_metadata, "pca", pca_controls_from_state(joined, max_rows, seed)
    )
    target_signature = analysis_signature(
        analysis_metadata, "rf", target_controls_from_state(joined, max_rows, seed)
    )
    pca_analysis = current_artifact("pca_analysis", pca_signature)
    target_analysis = current_artifact("target_analysis", target_signature)
    dataset_summary = {
        "benchmark_rows": len(benchmarks),
        "joined_rows": len(joined),
        "analysis_unit": analysis_metadata["analysis_unit"],
        "raw_row_count": analysis_metadata["raw_row_count"],
        "analysis_row_count": analysis_metadata["analysis_row_count"],
        "grouping_keys": analysis_metadata["grouping_keys"],
        "dataset_fingerprint": analysis_metadata.get("dataset_fingerprint", ""),
    }

    if not pca_analysis:
        st.info("Run PCA Explorer with the current controls to generate Findings.")
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
        st.dataframe(source_contributions.head(10), width="stretch", hide_index=True)
    with col_right:
        st.subheader("Top Target-Aware Predictors")
        if target_importance.empty:
            st.info("Target-aware permutation importance is not available yet.")
        else:
            st.dataframe(target_importance.head(10), width="stretch", hide_index=True)

    st.subheader("PC1-PC4 Interpretations")
    st.dataframe(component_interpretations, width="stretch", hide_index=True)

    stability = pca_analysis.get("stability", {})
    if stability:
        st.subheader("PCA Stability")
        st.caption(
            f"{stability['runs']} deterministic samples of {stability['sample_rows']:,} rows; "
            "component similarities are sign-aligned."
        )
        st.dataframe(stability["component_similarity"], width="stretch", hide_index=True)
        for warning in stability["warnings"]:
            st.warning(warning)

    st.subheader("PC vs Target Correlation")
    st.dataframe(pc_target_correlations, width="stretch", hide_index=True)

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
        data=signed_export(source_contributions, pca_signature).to_csv(index=False),
        file_name="pca_original_feature_contributions.csv",
        mime="text/csv",
        key="download_pca_original_contributions",
    )
    download_cols[2].download_button(
        "pca_encoded_feature_contributions.csv",
        data=signed_export(encoded_contributions, pca_signature).to_csv(index=False),
        file_name="pca_encoded_feature_contributions.csv",
        mime="text/csv",
        key="download_pca_encoded_contributions",
    )

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "pca_component_interpretations.csv",
        data=signed_export(component_interpretations, pca_signature).to_csv(index=False),
        file_name="pca_component_interpretations.csv",
        mime="text/csv",
        key="download_pca_component_interpretations",
    )
    download_cols[1].download_button(
        "pc_target_correlations.csv",
        data=signed_export(pc_target_correlations, pca_signature).to_csv(index=False),
        file_name="pc_target_correlations.csv",
        mime="text/csv",
        key="download_pc_target_correlations",
    )
    if not target_importance.empty:
        download_cols[2].download_button(
            "target_permutation_importance.csv",
            data=signed_export(target_importance, target_signature).to_csv(index=False),
            file_name="target_permutation_importance.csv",
            mime="text/csv",
            key="download_target_permutation_importance",
        )

    st.subheader("Deployment Readiness")
    st.markdown(
        """
        - Do not commit `inferencex-dump-*` or any giant database/JSON dump to GitHub.
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


def render_sales_pitch_visuals(
    joined: pd.DataFrame,
    benchmarks: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    st.header("Sales Pitch Visuals")
    st.caption(
        "A plain-English storytelling layer for PCA results. PCA = structure, not value. "
        "Loading = feature contribution to a synthetic PC axis."
    )

    pca_signature = analysis_signature(
        analysis_metadata, "pca", pca_controls_from_state(joined, max_rows, seed)
    )
    pca_analysis = current_artifact("pca_analysis", pca_signature)
    target_controls = target_controls_from_state(joined, max_rows, seed)
    target_analysis = current_artifact(
        "target_analysis", analysis_signature(analysis_metadata, "rf", target_controls)
    )
    if not pca_analysis:
        st.info("Run PCA Explorer once to generate the sales visuals.")
        return

    required_pca_keys = {
        "source_contributions",
        "encoded_contributions",
        "explained_variance",
        "component_interpretations",
        "pc_scores",
    }
    missing_pca_keys = sorted(required_pca_keys - set(pca_analysis.keys()))
    if missing_pca_keys:
        st.info(
            "Run PCA Explorer again to refresh the saved PCA artifacts required for "
            f"Sales Pitch Visuals: {', '.join(missing_pca_keys)}."
        )
        return

    source_contributions = pca_analysis["source_contributions"].copy()
    source_contributions["raw_feature_name"] = source_contributions["source_feature"]
    source_contributions["readable_feature_label"] = source_contributions["raw_feature_name"].map(
        readable_feature_label
    )
    encoded_contributions = pca_analysis["encoded_contributions"]
    explained = pca_analysis["explained_variance"]
    component_interpretations = pca_analysis["component_interpretations"]
    pc_scores = pca_analysis["pc_scores"].copy()
    component_cards = build_sales_component_cards(encoded_contributions, explained)
    source_contributions["category"] = source_contributions["raw_feature_name"].map(
        sales_feature_category
    )

    metric_cols = metric_like_numeric_columns(joined)
    initial_target = (
        target_analysis.get("target_metric")
        if target_analysis
        else default_sales_target_metric(metric_cols)
    )
    if initial_target not in metric_cols and metric_cols:
        initial_target = default_sales_target_metric(metric_cols)
    selected_target = initial_target

    first_five_variance = explained["cumulative_explained_variance"].iloc[
        min(4, len(explained) - 1)
    ]
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Raw rows", f"{analysis_metadata['raw_row_count']:,}")
    kpi_cols[1].metric("Analysis rows", f"{analysis_metadata['analysis_row_count']:,}")
    kpi_cols[2].metric("Analysis unit", analysis_metadata["analysis_unit"])
    kpi_cols[3].metric("First 5 PCs", f"{first_five_variance:.1%}")
    kpi_cols[4].metric("Target metric", selected_target or "Not selected")

    st.subheader("Inference Value Is Not Just Hardware")
    st.write(
        "The biggest structural differences in the benchmark space come from serving "
        "topology and parallelism, not just model or hardware labels."
    )
    top_features = source_contributions.head(10).copy()
    if not top_features.empty and (
        top_features["category"] == "Serving topology / parallelism"
    ).all():
        st.success("Top 10 structural drivers are all serving topology / parallelism features.")
    if {"config_model", "config_hardware"}.isdisjoint(set(top_features["raw_feature_name"])):
        st.info("Model and hardware labels are not the dominant PCA drivers in this run.")
    st.plotly_chart(
        px.bar(
            top_features.sort_values("weighted_contribution"),
            x="weighted_contribution",
            y="readable_feature_label",
            color="category",
            text="contribution_share",
            orientation="h",
            labels={
                "weighted_contribution": "Weighted PCA variance contribution",
                "readable_feature_label": "Feature group",
                "category": "Plain-English category",
                "contribution_share": "Contribution share",
            },
            title="Top structural drivers in the configuration space",
        ).update_traces(texttemplate="%{text:.1%}", textposition="outside"),
        width="stretch",
    )
    st.caption("PCA contribution means variance structure, not causal value.")

    st.subheader("Four Axes of Inference Configuration")
    st.write(
        "PC1-PC4 turn the high-dimensional setup space into four readable configuration axes."
    )
    for row_start in range(0, len(component_cards), 2):
        card_cols = st.columns(2)
        for card_col, card in zip(
            card_cols,
            component_cards.iloc[row_start : row_start + 2].itertuples(index=False),
        ):
            with card_col.container(border=True):
                st.markdown(f"#### {card.component}: {card.axis_name}")
                var_cols = st.columns(2)
                var_cols[0].metric("Explained variance", f"{card.explained_variance_ratio:.1%}")
                var_cols[1].metric(
                    "Cumulative variance",
                    f"{card.cumulative_explained_variance:.1%}",
                )
                st.write(card.interpretation)
                st.markdown(f"**High side of axis:** {card.top_positive_drivers}")
                st.markdown(f"**Low side of axis:** {card.top_negative_drivers}")
    with st.expander("Technical detail: PC1-PC4 loadings and encoded features"):
        st.dataframe(component_cards, width="stretch", hide_index=True)
        st.dataframe(component_interpretations, width="stretch", hide_index=True)

    st.subheader("Configuration Map Colored by Performance")
    st.write(
        "This map shows configuration similarity. Color shows the selected performance outcome."
    )
    st.caption("Each point is one aggregated config/workload/concurrency row.")
    st.caption("Distance means configuration similarity; color means selected performance outcome.")
    if not metric_cols or not {"PC1", "PC2"}.issubset(pc_scores.columns):
        st.info("No metric-like numeric columns or PC1/PC2 scores are available for the map.")
        map_points = pc_scores
    else:
        color_index = metric_cols.index(initial_target) if initial_target in metric_cols else 0
        color_metric = st.selectbox(
            "Performance color metric",
            options=metric_cols,
            index=color_index,
            key="sales_map_color_metric",
            help=(
                "p99 ITL = bad-case inter-token delay. TTFT = time until first token. "
                "TPOT = time per output token."
            ),
        )
        selected_target = color_metric
        hover_fields = [
            column
            for column in (
                "config_model",
                "config_hardware",
                "config_framework",
                "config_precision",
                "benchmark_type",
                "isl",
                "osl",
                "conc",
            )
            if column in joined.columns
        ]
        row_data = joined.reindex(pc_scores.index)
        map_points = pc_scores.join(row_data[[color_metric] + hover_fields], how="left")
        map_points["selected_metric_raw_name"] = color_metric
        map_points["selected_metric_readable_label"] = readable_feature_label(color_metric)
        map_points[color_metric] = pd.to_numeric(map_points[color_metric], errors="coerce")
        metric_values = map_points[color_metric].dropna()
        if metric_values.empty:
            clipped_metric = color_metric
            p5 = p95 = np.nan
        else:
            p5 = float(metric_values.quantile(0.05))
            p95 = float(metric_values.quantile(0.95))
            clipped_metric = f"{color_metric}_clipped"
            map_points[clipped_metric] = map_points[color_metric].clip(lower=p5, upper=p95)
            st.caption(
                f"Color scale clipped to p5-p95: {p5:.3g} to {p95:.3g} for "
                f"{readable_feature_label(color_metric)}."
            )
        hover_labels: list[str] = []
        for column in hover_fields:
            label = readable_feature_label(column)
            map_points[label] = row_data[column]
            hover_labels.append(label)
        axis_name_lookup = component_cards.set_index("component")["axis_name"].to_dict()
        fig = px.scatter(
            map_points,
            x="PC1",
            y="PC2",
            color=clipped_metric,
            opacity=0.78,
            render_mode="webgl",
            hover_data=hover_labels,
            color_continuous_scale="Viridis",
            labels={
                "PC1": f"PC1: {axis_name_lookup.get('PC1', 'Configuration mix')}",
                "PC2": f"PC2: {axis_name_lookup.get('PC2', 'Configuration mix')}",
                clipped_metric: readable_feature_label(color_metric),
            },
            title="Configuration similarity map with performance overlay",
        ).update_traces(marker={"size": 7, "line": {"width": 0}})
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.12,
            text=f"High PC1: {axis_name_lookup.get('PC1', 'Configuration mix')}",
            showarrow=False,
            bgcolor="rgba(249,250,251,0.94)",
            bordercolor="rgba(51,65,85,0.45)",
            font={"color": "#111827", "size": 12},
            xanchor="right",
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=f"High PC2: {axis_name_lookup.get('PC2', 'Configuration mix')}",
            showarrow=False,
            bgcolor="rgba(249,250,251,0.94)",
            bordercolor="rgba(51,65,85,0.45)",
            font={"color": "#111827", "size": 12},
            xanchor="left",
        )
        if not metric_values.empty:
            highest_idx = map_points[color_metric].idxmax()
            if highest_idx in map_points.index:
                x_offset = -55 if map_points.loc[highest_idx, "PC1"] > map_points["PC1"].median() else 55
                y_offset = 45 if map_points.loc[highest_idx, "PC2"] > map_points["PC2"].median() else -45
                fig.add_annotation(
                    x=map_points.loc[highest_idx, "PC1"],
                    y=map_points.loc[highest_idx, "PC2"],
                    text=f"Higher {readable_feature_label(color_metric)}",
                    showarrow=True,
                    arrowhead=2,
                    ax=x_offset,
                    ay=y_offset,
                    bgcolor="rgba(249,250,251,0.94)",
                    bordercolor="rgba(51,65,85,0.45)",
                    font={"color": "#111827", "size": 12},
                )
        st.plotly_chart(
            fig,
            width="stretch",
        )
        st.warning("The color metric is not used to build the PCA axes.")

    sales_numeric, sales_categorical = default_target_features(joined, selected_target)
    sales_controls = {
        "target": selected_target,
        "numeric_features": sales_numeric,
        "categorical_features": sales_categorical,
        "max_rows": int(max_rows),
        "seed": int(seed),
        "n_estimators": 150,
        "permutation_repeats": 3,
        "split_mode": "Grouped cross-validation by config_id",
    }
    sales_signature = analysis_signature(analysis_metadata, "sales_rf", sales_controls)
    sales_view_signature = analysis_signature(
        analysis_metadata,
        "sales_view",
        {"pca_signature": pca_signature, "target": selected_target},
    )
    sales_target_analysis = current_artifact("sales_target_analysis", sales_signature)
    target_error = ""
    if sales_target_analysis:
        target_importance = sales_target_analysis.get("importance_frame", pd.DataFrame())
    elif selected_target:
        with st.spinner(
            f"Computing target-aware predictors for {readable_feature_label(selected_target)}"
        ):
            computed_target_analysis, target_error = compute_target_importance_for_sales(
                joined,
                analysis_metadata,
                max_rows,
                seed,
                selected_target,
        )
        if computed_target_analysis:
            computed_target_analysis.update(
                {
                    "analysis_signature": sales_signature,
                    "controls": sales_controls,
                }
            )
            st.session_state["sales_target_analysis"] = computed_target_analysis
            sales_target_analysis = computed_target_analysis
            target_importance = computed_target_analysis["importance_frame"]
            st.success(
                "Auto-computed target-aware predictors using "
                f"{computed_target_analysis['split_mode']}."
            )
        else:
            target_importance = pd.DataFrame()
    else:
        target_importance = pd.DataFrame()

    st.subheader("Structure vs Performance")
    st.write(
        "Overlap features are the strongest candidates for infrastructure valuation because "
        "they both shape configuration space and predict the selected outcome."
    )
    if target_importance.empty and target_error:
        st.warning(target_error)
    elif target_importance.empty:
        st.info("Target-aware predictors are unavailable for the selected metric.")
    bridge = build_structure_vs_performance(source_contributions, target_importance)
    bridge_display = bridge[
        [
            "Feature",
            "PCA rank",
            "PCA contribution",
            "Target rank",
            "Target importance",
            "Why it matters",
        ]
    ].copy()
    bridge_display["PCA rank"] = bridge_display["PCA rank"].map(
        lambda value: "" if pd.isna(value) else f"{int(value)}"
    )
    bridge_display["PCA contribution"] = bridge_display["PCA contribution"].map(
        lambda value: "" if pd.isna(value) else f"{value:.4f}"
    )
    bridge_display["Target rank"] = bridge_display["Target rank"].map(
        lambda value: "Not top 10" if pd.isna(value) else f"{int(value)}"
    )
    bridge_display["Target importance"] = bridge_display["Target importance"].map(
        lambda value: "Not a top predictor" if pd.isna(value) else f"{value:.4f}"
    )
    st.dataframe(
        bridge_display,
        width="stretch",
        hide_index=True,
    )
    with st.expander("Technical detail: encoded PCA contribution table"):
        st.dataframe(encoded_contributions.head(50), width="stretch", hide_index=True)

    structural_predictive = bridge[
        bridge["Interpretation"] == "Structural and predictive"
    ]["Feature"].tolist()
    top_structural = source_contributions.head(3)["raw_feature_name"].tolist()
    top_takeaways = [
        (
            "Serving architecture dominates the benchmark landscape: the top structural "
            f"drivers are {readable_feature_list(top_structural)}."
        ),
        (
            "The first four PCA axes compress thousands of config/workload rows into "
            "readable infrastructure dimensions."
        ),
        (
            "Overlap features are the strongest candidates for infrastructure valuation: "
            f"{compact_list(structural_predictive)}."
            if structural_predictive
            else (
                "Performance overlays separate configuration structure from outcome "
                "prediction; run the target-aware model to identify which structural "
                "drivers also predict performance."
            )
        ),
    ]
    st.subheader("Top 3 Sales Takeaways")
    st.markdown("\n".join(f"- {takeaway}" for takeaway in top_takeaways))

    dataset_summary = {
        "analysis_unit": analysis_metadata["analysis_unit"],
        "raw_row_count": analysis_metadata["raw_row_count"],
        "analysis_row_count": analysis_metadata["analysis_row_count"],
        "grouping_keys": analysis_metadata["grouping_keys"],
        "benchmark_rows": len(benchmarks),
    }
    sales_markdown = build_sales_pitch_markdown(
        dataset_summary,
        source_contributions,
        component_cards,
        selected_target,
        bridge,
        pca_signature,
        sales_signature,
    )

    st.subheader("Downloads")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "sales_pca_feature_contributions.csv",
        data=signed_export(source_contributions, pca_signature).to_csv(index=False),
        file_name="sales_pca_feature_contributions.csv",
        mime="text/csv",
        key="download_sales_pca_feature_contributions",
    )
    download_cols[1].download_button(
        "sales_component_cards.csv",
        data=signed_export(component_cards, pca_signature).to_csv(index=False),
        file_name="sales_component_cards.csv",
        mime="text/csv",
        key="download_sales_component_cards",
    )
    download_cols[2].download_button(
        "sales_pc_map_points.csv",
        data=signed_export(map_points.reset_index(drop=True), sales_view_signature).to_csv(index=False),
        file_name="sales_pc_map_points.csv",
        mime="text/csv",
        key="download_sales_pc_map_points",
    )
    download_cols = st.columns(2)
    if not target_importance.empty:
        download_cols[0].download_button(
            "sales_structure_vs_performance.csv",
            data=signed_export(bridge, sales_signature).to_csv(index=False),
            file_name="sales_structure_vs_performance.csv",
            mime="text/csv",
            key="download_sales_structure_vs_performance",
        )
    download_cols[1].download_button(
        "sales_pitch_summary.md",
        data=sales_markdown,
        file_name="sales_pitch_summary.md",
        mime="text/markdown",
        key="download_sales_pitch_summary",
    )


def render_notes() -> None:
    st.markdown(
        """
        This demo prefers `benchmark_results.csv` and `configs.csv`; JSON is a local fallback.
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


def render_research_results() -> None:
    """Show completed aggregate-only research; never fit or import a model runtime."""
    st.subheader("Research Results")
    st.warning(
        "Research-only model diagnostics. This section reads aggregate JSON artifacts only; "
        "it does not fit models or provide production predictions."
    )
    try:
        summary = build_research_summary(RESEARCH_ARTIFACT_DIR)
    except Exception as exc:
        st.info(f"Research artifacts are unavailable: {exc}")
        return

    point = summary["selected_throughput_point_model"]
    metrics = point.get("metrics") or {}
    st.markdown("### Throughput point model")
    if metrics:
        columns = st.columns(3)
        columns[0].metric("Model", "Full-context TabFM")
        columns[1].metric("Grouped R2", f"{metrics['r2']:.6f} +/- {metrics['r2_std']:.6f}")
        columns[2].metric("MAE", f"{metrics['mae']:.6f}")
    else:
        st.info("The full-context throughput aggregate artifact is not available.")

    uncertainty = summary["selected_uncertainty_method"]
    st.markdown("### Experimental uncertainty")
    st.write("Selected research method: conditional-scale split conformal.")
    interval_rows = []
    for level, values in uncertainty.get("intervals", {}).items():
        interval_rows.append({
            "Nominal coverage": f"{float(level):.0%}",
            "Empirical coverage": f"{values['empirical_coverage']:.2%}",
            "Average width": f"{values['average_interval_width']:.3f}",
        })
    if interval_rows:
        st.dataframe(pd.DataFrame(interval_rows), width="stretch", hide_index=True)
    else:
        st.info("The aggregate uncertainty artifact is not available.")
    split_metrics = uncertainty.get("uncertainty_evaluation_point_model") or {}
    if split_metrics:
        st.caption(
            "Important: the leakage-safe uncertainty evaluation used only about half of "
            f"each outer-training fold as TabFM context (R2 {split_metrics['r2']:.6f}). "
            "Its intervals are research-grade and are not calibrated around the final "
            "full-context point model."
        )

    st.markdown("### Scope decision")
    st.write(summary["latency_recommendation"]["decision"])
    st.write(summary["vae_crvae"]["decision"])
    st.caption(summary["next_step"])
    unavailable = [name for name, state in summary["artifact_status"].items() if state != "available"]
    if unavailable:
        st.caption("Missing aggregate artifacts: " + ", ".join(unavailable))


def compact_value_chart(frame: pd.DataFrame, column: str, title: str) -> None:
    """Render one small categorical/workload distribution when the field exists."""
    if column not in frame.columns:
        return
    counts = (
        frame[column]
        .dropna()
        .map(safe_hashable_value)
        .value_counts()
        .head(12)
        .rename_axis(column)
        .reset_index(name="rows")
    )
    if not counts.empty:
        st.plotly_chart(px.bar(counts, x=column, y="rows", title=title), width="stretch")


def research_summary_or_none() -> tuple[dict[str, Any] | None, str]:
    """Read completed aggregate artifacts without touching a model runtime."""
    try:
        return build_research_summary(RESEARCH_ARTIFACT_DIR), ""
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_pca_target_artifact(path_text: str = str(PCA_TARGET_ARTIFACT_PATH)) -> dict[str, Any]:
    artifact = json.loads(Path(path_text).read_text(encoding="utf-8"))
    if artifact.get("dump", {}).get("version") != ACTIVE_DUMP_VERSION:
        raise ValueError("PCA artifact dump version does not match the active July snapshot.")
    if artifact.get("shared_basis", {}).get("feature_order") != list(TARGET_PCA_FEATURES):
        raise ValueError("PCA artifact feature order does not match the frozen shared basis.")
    return artifact


def render_overview(
    benchmarks: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    source_info: dict[str, Any],
    research_summary: dict[str, Any] | None,
) -> None:
    point_metrics = (
        research_summary or {}
    ).get("selected_throughput_point_model", {}).get("metrics") or {}
    uncertainty = (research_summary or {}).get("selected_uncertainty_method", {})
    configuration_count = benchmarks["config_id"].nunique() if "config_id" in benchmarks else 0

    st.markdown("#### Dataset summary")
    dataset_cards = st.columns(4)
    with dataset_cards[0]:
        render_compact_card("Benchmark rows", format_compact_count(analysis_metadata["raw_row_count"]), "Loaded")
    with dataset_cards[1]:
        render_compact_card("Analysis rows", format_compact_count(analysis_metadata["analysis_row_count"]), "Active analysis unit")
    with dataset_cards[2]:
        render_compact_card("Configurations", format_compact_count(configuration_count), "Unique configurations")
    with dataset_cards[3]:
        render_compact_card("Data source", source_info["active_mode"], "Dataset available")

    st.markdown("#### Throughput model")
    with st.container(border=True):
        throughput_cols = st.columns([1.8, 0.8, 0.85, 1.05, 1.35])
        with throughput_cols[0]:
            render_compact_card("Selected model", "Full-context TabFM" if point_metrics else "Artifact unavailable")
        with throughput_cols[1]:
            render_compact_card("R²", format_overview_r2(point_metrics["r2"]) if point_metrics else "—")
        with throughput_cols[2]:
            render_compact_card("Fold stability", f"±{format_overview_r2(point_metrics['r2_std'])}" if point_metrics else "—")
        with throughput_cols[3]:
            render_compact_card("MAE", f"{format_overview_mae(point_metrics['mae'])} tok/s/GPU" if point_metrics else "—")
        with throughput_cols[4]:
            render_compact_card("Evaluation", "Grouped by configuration")
        st.caption("The model explains about 96% of held-out throughput variation across unseen configuration groups." if point_metrics else "The selected throughput aggregate artifact is unavailable.")

    st.markdown("#### Research decisions")
    decision_cols = st.columns(3)
    with decision_cols[0]:
        render_compact_card(
            "Uncertainty",
            "Experimental" if uncertainty else "Unavailable",
            "Conditional-scale conformal · Not production-calibrated" if uncertainty else "Aggregate artifact unavailable",
        )
    with decision_cols[1]:
        render_compact_card("Latency", "Research paused", "Tail specialization did not improve median TPOT")
    with decision_cols[2]:
        render_compact_card("Generative modeling", "Not pursued", "VAE and CRVAE were not justified by the results")
    st.markdown(
        '<div class="dashboard-takeaway"><strong>Key takeaway</strong><br>'
        'Throughput prediction is the strongest validated result. Uncertainty remains research-grade, while latency modeling is paused.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Model details"):
        if point_metrics:
            detail_cols = st.columns(2)
            detail_cols[0].caption(f"Exact full-context R²: {point_metrics['r2']:.6f} +/- {point_metrics['r2_std']:.6f}")
            detail_cols[1].caption(f"Exact full-context MAE: {point_metrics['mae']:.6f} tok/s/GPU")
        else:
            st.info("The selected throughput aggregate artifact is unavailable.")
    st.caption("This dashboard is descriptive: it summarizes benchmark structure and completed research results, not causal effects.")


def render_data_source_details(
    file_status: pd.DataFrame,
    source_info: dict[str, Any],
    analysis_metadata: dict[str, Any],
) -> None:
    """Keep exact data provenance details available without dominating the overview."""
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.caption("Selected analysis unit")
        st.write(analysis_metadata["analysis_unit"])
        st.caption("Exact row counts")
        st.write(
            f"{analysis_metadata['raw_row_count']:,} raw benchmark rows · "
            f"{analysis_metadata['analysis_row_count']:,} analysis rows"
        )
    with detail_cols[1]:
        st.caption("Resolved source")
        st.write(f"{source_info['active_mode']} · {source_info.get('active_candidate') or 'selected data directory'}")
        st.caption("Required-file status")
        st.write("Available")
    st.caption("Local data directory")
    st.code(str(source_info["active_dir"]), language=None)
    st.dataframe(file_status, width="stretch", hide_index=True)


def render_data_understanding_dashboard(
    joined: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    render_section_intro(
        "Data Understanding",
        "Coverage, target availability, and the configuration and workload mix behind the active analysis unit.",
    )
    feature_dictionary = build_feature_dictionary(joined)
    metrics = metric_like_numeric_columns(joined)
    target_availability = pd.DataFrame(
        {
            "target": metrics,
            "available rows": [int(joined[column].notna().sum()) for column in metrics],
            "missing": [f"{joined[column].isna().mean():.1%}" for column in metrics],
        }
    ).head(12)

    counts = st.columns(4)
    for column, label, value in zip(
        counts,
        ("Raw rows", "Analysis rows", "Configurations", "Available targets"),
        (format_compact_count(len(joined)), format_compact_count(len(analysis_frame)), format_compact_count(joined["config_id"].nunique()) if "config_id" in joined else "n/a", f"{len(metrics):,}"),
    ):
        with column:
            render_compact_card(label, value)

    st.subheader("Target availability")
    st.dataframe(target_availability, width="stretch", hide_index=True)

    distribution_cols = st.columns(2)
    throughput = next((column for column in metrics if "tput" in column.lower() or "throughput" in column.lower()), None)
    latency = next((column for column in metrics if any(term in column.lower() for term in ("itl", "tpot", "ttft", "e2el"))), None)
    if throughput:
        distribution_cols[0].plotly_chart(px.histogram(joined, x=throughput, nbins=30, title="Throughput distribution"), width="stretch")
    if latency:
        distribution_cols[1].plotly_chart(px.histogram(joined, x=latency, nbins=30, title="Latency distribution"), width="stretch")

    st.subheader("Configuration and workload mix")
    config_cols = st.columns(2)
    for target, column, title in zip(config_cols, ("config_hardware", "config_model"), ("Hardware", "Model")):
        with target:
            compact_value_chart(joined, column, title)
    workload_cols = st.columns(3)
    for target, column, title in zip(workload_cols, ("conc", "isl", "osl"), ("Concurrency", "Input length", "Output length")):
        with target:
            compact_value_chart(joined, column, title)

    full_missingness = pd.DataFrame(
        {
            "column": joined.columns,
            "missing rows": [int(joined[column].isna().sum()) for column in joined.columns],
            "missing": [f"{joined[column].isna().mean():.1%}" for column in joined.columns],
        }
    ).sort_values("missing rows", ascending=False)
    with st.expander("Column schema"):
        st.dataframe(feature_dictionary, width="stretch", height=420, hide_index=True)
    with st.expander("Raw data preview"):
        st.dataframe(joined.head(50), width="stretch", height=360, hide_index=True)
    with st.expander("Full missingness table"):
        st.dataframe(full_missingness, width="stretch", height=420, hide_index=True)
    with st.expander("Developer metadata"):
        st.caption(f"Sample cap: {max_rows:,}; random seed: {seed}; grouping keys: {', '.join(analysis_metadata['grouping_keys']) or 'none'}.")
        if analysis_metadata.get("warning"):
            st.caption(analysis_metadata["warning"])


def render_pca_dashboard(
    joined: pd.DataFrame,
    analysis_metadata: dict[str, Any],
    max_rows: int,
    seed: int,
) -> None:
    render_section_intro(
        "PCA",
        "Updated full-dataset PCA using the cumulative July 20 snapshot.",
    )
    st.info(
        "The PCA basis is fit once on configuration and workload variables from the full eligible "
        "dataset in the cumulative July 20 snapshot. Median TPOT, throughput per GPU, and joules "
        "per output token are not PCA inputs. They are separate outcome overlays used to interpret "
        "how the same configuration space relates to latency, throughput, and observed energy."
    )
    try:
        artifact = load_pca_target_artifact()
    except Exception as exc:
        st.error(f"The cumulative-snapshot PCA artifact is unavailable: {exc}")
        st.code(
            "PYTHONPATH=. .venv-streamlit/bin/python scripts/build_july_pca_artifact.py",
            language="bash",
        )
        return
    active_fingerprint = analysis_metadata.get("dataset_manifest", {}).get("fingerprint")
    artifact_fingerprint = artifact.get("dump", {}).get("manifest", {}).get("fingerprint")
    if active_fingerprint != artifact_fingerprint:
        st.error(
            "The loaded data does not match the cumulative-snapshot PCA artifact. "
            "No saved basis or target overlay was applied."
        )
        return

    basis = artifact["shared_basis"]
    explained = pd.DataFrame(basis["explained_variance"]).head(10)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Full eligible rows", f"{basis['full_eligible_row_count']:,}")
    summary_cols[1].metric("PCA source features", f"{len(basis['feature_order'])}")
    summary_cols[2].metric("PC1–PC5 variance", f"{basis['updated_snapshot_first_five_cumulative']:.2%}")
    summary_cols[3].metric("Components for 90%", str(basis["component_thresholds"]["90%"]))
    st.caption(
        f"Eligible source observations span {basis['eligible_source_raw_date_range'][0]} through "
        f"{basis['eligible_source_raw_date_range'][1]}. All {basis['previous_snapshot_eligible_groups_retained']:,} "
        f"previous eligible groups are retained alongside {basis['new_eligible_groups_vs_previous_snapshot']:,} new groups. "
        f"The {basis['excluded_rows']:,} agentic-trace aggregates are excluded because they do not share ISL/OSL semantics."
    )
    st.plotly_chart(
        px.bar(
            explained,
            x="component",
            y="explained_variance_ratio",
            text="explained_variance_ratio",
            title="Updated full-dataset explained variance",
        ).update_traces(texttemplate="%{text:.1%}", textposition="outside"),
        width="stretch",
    )
    with st.expander("Shared-basis methodology and updated-snapshot/June stability"):
        st.write("Feature order: " + ", ".join(basis["feature_order"]))
        st.caption("No metrics, latency, throughput, power, or energy fields are PCA inputs.")
        st.dataframe(pd.DataFrame(basis["basis_comparison"]["components"]), width="stretch", hide_index=True)
        st.write(
            "Five-dimensional principal angles (degrees): "
            + ", ".join(f"{value:.2f}" for value in basis["basis_comparison"]["principal_angles_degrees"])
        )

    signature = analysis_signature(
        analysis_metadata,
        "cumulative_snapshot_target_pca",
        {"features": list(TARGET_PCA_FEATURES), "seed": seed, "dump": ACTIVE_DUMP_VERSION},
    )
    projection = current_artifact("cumulative_target_pca_projection", signature)
    if projection is None and st.button("Build interactive target projections", type="primary"):
        with st.spinner("Projecting all three outcomes into the shared full-dataset basis"):
            fitted = fit_target_shared_pca(joined, seed=seed)
            projection = {
                "analysis_signature": signature,
                PCA_LATENCY_TARGET: build_target_overlay(fitted, PCA_LATENCY_TARGET),
                PCA_OUTPUT_TARGET: build_target_overlay(fitted, PCA_OUTPUT_TARGET),
                PCA_ENERGY_TARGET: build_target_overlay(fitted, PCA_ENERGY_TARGET),
            }
            st.session_state["cumulative_target_pca_projection"] = projection
    if projection is None:
        st.info(
            "The aggregate artifact is loaded. Build interactive projections to render row-level scatter views; no supervised model is loaded or run."
        )

    def render_target_mode(target: str, projection_data: dict[str, Any] | None) -> None:
        metadata = artifact["targets"][target]
        is_energy = target == PCA_ENERGY_TARGET
        is_latency = target == PCA_LATENCY_TARGET
        title = (
            "Median TPOT"
            if is_latency
            else "Joules per output token"
            if is_energy
            else "Throughput per GPU"
        )
        subtitle = (
            "Latency-focused descriptive overlay on the shared configuration PCA."
            if is_latency
            else "Observed-energy descriptive overlay on the measured subset."
            if is_energy
            else "Final supervised target shown as a descriptive overlay on the shared PCA."
        )
        st.markdown(f"#### {title}")
        st.caption(subtitle)
        st.caption(
            f"{metadata['display_name']} · {metadata['transformation']} · {metadata['unit']} · "
            f"{metadata['direction']}. The target is a color/association overlay, not a PCA input."
        )
        cards = st.columns(4)
        cards[0].metric("Usable aggregate rows", f"{metadata['usable_rows']:,}")
        cards[1].metric("Configurations", f"{metadata['unique_configurations']:,}")
        cards[2].metric("Target transform", "Raw / identity" if not is_energy else "Observed raw")
        cards[3].metric("Direction", metadata["direction"].capitalize())
        if is_energy:
            st.warning(
                f"Measured-only support: {metadata['raw_measured_rows']:,} raw rows, "
                f"{metadata['usable_rows']:,} aggregate groups, {metadata['configuration_coverage']:.2%} configuration coverage, "
                f"{metadata['date_range'][0]} through {metadata['date_range'][1]}. PCA is descriptive and does not predict energy."
            )
        elif is_latency:
            st.info(
                f"Primary latency outcome studied in this PCA: raw median time per output token. "
                f"The valid target spans {metadata['date_range'][0]} through {metadata['date_range'][1]}; "
                "it is not the final selected supervised target and PCA does not predict latency."
            )
            st.caption(
                "Workload support: "
                + "; ".join(
                    f"{field}={', '.join(map(str, values))}"
                    for field, values in metadata["workload_support"].items()
                    if field != "conc"
                )
            )
        else:
            validation = metadata["historical_validation_context"]
            st.info(
                "Target selection context only: raw throughput per GPU was the final 4,096-row, "
                f"three-fold grouped TabFM target (R² {validation['r2_mean']:.6f} ± {validation['r2_std']:.6f}; "
                f"MAE {validation['mae']:.6f} tokens/s/GPU). No model was retrained."
            )

        distribution = pd.DataFrame(
            [{"statistic": key, "value": value} for key, value in metadata["raw_distribution"].items()]
        )
        associations = pd.DataFrame(metadata["associations"])
        view_cols = st.columns(2)
        with view_cols[0]:
            st.markdown("**Target distribution summary**")
            st.dataframe(distribution, width="stretch", hide_index=True)
        with view_cols[1]:
            st.markdown("**PC–target associations**")
            st.dataframe(associations, width="stretch", hide_index=True)

        if projection_data is not None:
            plot_frame = projection_data["frame"].copy()
            color_column = target
            color_title = metadata["display_name"]
            if is_latency:
                use_log_color = st.checkbox(
                    "Use optional log1p color scale (display only)",
                    value=False,
                    key="median_tpot_log_color",
                    help="The stored target and all summary statistics remain raw seconds/output token.",
                )
                if use_log_color:
                    color_column = "_median_tpot_log1p_display"
                    plot_frame[color_column] = np.log1p(plot_frame[target])
                    color_title = "log1p median TPOT (display only)"
            scatter_cols = st.columns(2)
            scatter_cols[0].plotly_chart(
                px.scatter(
                    plot_frame,
                    x="PC1",
                    y="PC2",
                    color=color_column,
                    opacity=0.68,
                    render_mode="webgl",
                    title=f"PC1 vs PC2 · {color_title}",
                ),
                width="stretch",
            )
            scatter_cols[1].plotly_chart(
                px.scatter(
                    plot_frame,
                    x="PC1",
                    y="PC3",
                    color=color_column,
                    opacity=0.68,
                    render_mode="webgl",
                    title=f"PC1 vs PC3 · {color_title}",
                ),
                width="stretch",
            )
            if is_energy and "date" in plot_frame:
                dated = plot_frame.copy()
                dated["measurement_month"] = pd.to_datetime(
                    dated["date"], errors="coerce"
                ).dt.to_period("M").astype(str)
                st.plotly_chart(
                    px.scatter(
                        dated,
                        x="PC1",
                        y="PC2",
                        color="measurement_month",
                        opacity=0.65,
                        render_mode="webgl",
                        title="Observed-energy cohort by measurement month",
                    ),
                    width="stretch",
                )

        bins = pd.DataFrame(metadata["component_bins"])
        st.markdown("**Target values by component quantile**")
        st.dataframe(bins, width="stretch", hide_index=True)
        strongest = associations.iloc[associations[["pearson", "spearman"]].abs().max(axis=1).argmax()]["component"]
        source_loadings = pd.DataFrame(basis["source_loadings_first_five"])
        st.markdown(f"**Top configuration/workload loadings for {strongest}**")
        st.dataframe(
            source_loadings.loc[source_loadings["component"].eq(strongest)].head(10),
            width="stretch",
            hide_index=True,
        )
        encoded_loadings = pd.DataFrame(basis["encoded_loadings_first_five"])
        component_loadings = encoded_loadings.loc[encoded_loadings["component"].eq(strongest)]
        loading_cols = st.columns(2)
        with loading_cols[0]:
            st.markdown("**Top positive loadings**")
            st.dataframe(
                component_loadings.sort_values("loading", ascending=False).head(8)[
                    ["encoded_feature", "source_feature", "loading"]
                ],
                width="stretch",
                hide_index=True,
            )
        with loading_cols[1]:
            st.markdown("**Top negative loadings**")
            st.dataframe(
                component_loadings.sort_values("loading", ascending=True).head(8)[
                    ["encoded_feature", "source_feature", "loading"]
                ],
                width="stretch",
                hide_index=True,
            )
        if is_energy:
            with st.expander("Measured support, subgroup summaries, and time checks"):
                support_rows = [
                    {"dimension": key, "observed values": ", ".join(map(str, values))}
                    for key, values in {**metadata["workload_support"], **metadata["configuration_support"]}.items()
                ]
                st.dataframe(pd.DataFrame(support_rows), width="stretch", hide_index=True)
                st.dataframe(pd.DataFrame(metadata["temporal_summary"]), width="stretch", hide_index=True)
                for warning in metadata["sparse_group_warnings"]:
                    st.caption("Sparse support: " + warning)
                for dimension, rows in metadata["subgroup_summaries"].items():
                    st.markdown(f"**{dimension}**")
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption("Subgroup values are shown only as observed summaries; sparse groups are flagged and no generalization is claimed.")

    latency_tab, output_tab, energy_tab = st.tabs(
        ("Median TPOT", "Throughput per GPU", "Joules per output token")
    )
    with latency_tab:
        render_target_mode(PCA_LATENCY_TARGET, projection.get(PCA_LATENCY_TARGET) if projection else None)
    with output_tab:
        render_target_mode(PCA_OUTPUT_TARGET, projection.get(PCA_OUTPUT_TARGET) if projection else None)
    with energy_tab:
        render_target_mode(PCA_ENERGY_TARGET, projection.get(PCA_ENERGY_TARGET) if projection else None)


def render_model_results_dashboard(research_summary: dict[str, Any] | None, error: str = "") -> None:
    render_section_intro("Model Results", "Completed aggregate research artifacts only. No model is fit, imported, or run from this dashboard.")
    st.caption(
        "The supervised results are preserved historical experiments on the June snapshot. "
        "They are target-selection context only and are not applied to cumulative-snapshot rows."
    )
    if research_summary is None:
        st.info(f"Research artifacts are unavailable{': ' + error if error else '.'}")
        return
    point = research_summary["selected_throughput_point_model"]
    metrics = point.get("metrics") or {}
    uncertainty = research_summary["selected_uncertainty_method"]
    st.markdown("#### Selected throughput result")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_compact_card("Selected model", "Full-context TabFM" if metrics else "Artifact unavailable")
    with metric_cols[1]:
        render_compact_card("Grouped configuration R²", format_overview_r2(metrics["r2"]) if metrics else "—", f"Fold stability ±{format_overview_r2(metrics['r2_std'])}" if metrics else "")
    with metric_cols[2]:
        render_compact_card("MAE", f"{format_overview_mae(metrics['mae'])} tok/s/GPU" if metrics else "—")

    st.markdown("#### Experimental uncertainty")
    st.caption("Research only — not production-calibrated around the selected full-context point model.")
    interval_rows = [
        {"Nominal coverage": format_overview_percentage(float(level)), "Empirical coverage": format_overview_percentage(values['empirical_coverage']), "Average width": f"{values['average_interval_width']:.1f}"}
        for level, values in uncertainty.get("intervals", {}).items()
    ]
    if interval_rows:
        st.dataframe(pd.DataFrame(interval_rows), width="stretch", hide_index=True)
    else:
        st.info("The aggregate uncertainty artifact is not available.")
    split_metrics = uncertainty.get("uncertainty_evaluation_point_model") or {}
    if split_metrics:
        st.info(f"Uncertainty-split R²: {format_overview_r2(split_metrics['r2'])}. This reduced-context evaluation is separate from the full-context R² above.")

    st.markdown("#### Research decisions")
    decision_cols = st.columns(2)
    with decision_cols[0]:
        render_compact_card("Latency", "Research paused", "Tail specialization did not improve the median-TPOT baseline.")
    with decision_cols[1]:
        render_compact_card("Generative modeling", "Not pursued", "VAE and CRVAE were not justified by the results.")
    with st.expander("Detailed fold results"):
        st.caption("The selected aggregate conclusion does not expose row-level predictions or residuals. Fold-level model outputs remain outside this dashboard.")
    with st.expander("Subgroup coverage"):
        detail_rows = []
        for level, values in uncertainty.get("intervals", {}).items():
            worst = values.get("worst_subgroup_undercoverage") or {}
            detail_rows.append({"coverage": f"{float(level):.0%}", "minimum subgroup rows": values.get("subgroup_minimum_support", "n/a"), "worst supported subgroup": f"{worst.get('feature', 'n/a')} = {worst.get('value', 'n/a')}"})
        st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
    with st.expander("Methodology"):
        st.write("Throughput uses grouped evaluation by config_id. Conditional-scale split conformal was selected as one consistent method across 50%, 80%, and 95% coverage levels.")
        st.write("The uncertainty evaluation uses about half of each outer-training fold as context, so it is research-grade only and cannot replace the full-context point-model result.")
    unavailable = [name for name, state in research_summary["artifact_status"].items() if state != "available"]
    if unavailable:
        st.caption("Missing aggregate artifacts: " + ", ".join(unavailable))


def _energy_control(field: str, options: list[Any], disabled: bool = False) -> Any:
    label = field.replace("config_", "").replace("_", " ").upper() if field in {"isl", "osl"} else field.replace("config_", "").replace("_", " ").title()
    key = f"energy_measurement_{field}"
    if pd.api.types.is_numeric_dtype(pd.Series(options)):
        return st.select_slider(label, options=options, value=options[0], disabled=disabled, key=key)
    return st.selectbox(label, options=options, disabled=disabled, key=key)


def render_energy_support_panel(joined: pd.DataFrame) -> None:
    support = energy_support_summary(joined)
    st.markdown("#### Measurement support")
    cards = st.columns(4)
    for column, label, value in zip(
        cards,
        ("Usable raw rows", "Measured configurations", "Configuration coverage", "Measurement dates"),
        (
            f"{support['usable_raw_rows']:,}",
            f"{support['measured_configurations']:,} / {support['all_configurations']:,}",
            f"{support['configuration_coverage']:.1%}",
            f"{support['date_start']} – {support['date_end']}",
        ),
    ):
        with column:
            render_compact_card(label, value)
    st.warning(support["measurement_warning"] + " Energy prediction remains disabled.")
    support_cols = st.columns(2)
    support_cols[0].markdown("**Observed workloads**")
    support_cols[0].dataframe(pd.DataFrame(support["observed_workloads"]), width="stretch", hide_index=True)
    coverage_rows = [
        {"dimension": field.replace("config_", ""), "measured values": len(values), "values": ", ".join(values)}
        for field, values in support["coverage"].items()
    ]
    support_cols[1].markdown("**Measured configuration coverage**")
    support_cols[1].dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True)


def render_energy_measurements_dashboard(joined: pd.DataFrame) -> None:
    st.divider()
    render_section_intro(
        "Energy Measurements",
        "Observed benchmark measurements only. Exact matches are aggregated with the median; nearby rows are comparisons, never predictions.",
    )
    if ENERGY_TARGET not in joined or joined[ENERGY_TARGET].notna().sum() == 0:
        st.info("Observed energy measurements are unavailable in the loaded dataset.")
        return
    render_energy_support_panel(joined)
    options = available_control_values(joined)
    missing_controls = [field for field in ENERGY_WORKLOAD_FIELDS + ENERGY_CONFIG_FIELDS if not options.get(field)]
    if missing_controls:
        st.error("Energy controls are unavailable for: " + ", ".join(missing_controls))
        return

    with st.form("energy_measurements_form"):
        st.markdown("#### Configure an observed workload")
        workload_columns = st.columns(4)
        selection: dict[str, Any] = {}
        for column, field in zip(workload_columns, ENERGY_WORKLOAD_FIELDS):
            with column:
                selection[field] = _energy_control(field, options[field], disabled=len(options[field]) == 1)

        st.markdown("**Hardware and software**")
        software_fields = ENERGY_CONFIG_FIELDS[:7]
        for start in range(0, len(software_fields), 4):
            columns = st.columns(4)
            for column, field in zip(columns, software_fields[start : start + 4]):
                with column:
                    selection[field] = _energy_control(field, options[field])

        topology_fields = (
            "config_prefill_num_workers", "config_decode_num_workers",
            "config_num_prefill_gpu", "config_num_decode_gpu",
        )
        st.markdown("**Serving topology**")
        topology_columns = st.columns(4)
        for column, field in zip(topology_columns, topology_fields):
            with column:
                selection[field] = _energy_control(field, options[field])

        advanced_fields = [field for field in ENERGY_CONFIG_FIELDS if field not in software_fields + topology_fields]
        with st.expander("Advanced parallelism"):
            for start in range(0, len(advanced_fields), 4):
                columns = st.columns(4)
                for column, field in zip(columns, advanced_fields[start : start + 4]):
                    with column:
                        selection[field] = _energy_control(field, options[field])

        electricity_price = st.number_input(
            "Electricity price (USD per kWh)", min_value=0.0, max_value=5.0,
            value=0.12, step=0.01, format="%.2f",
            help="Used only for a mathematical cost conversion of an observed energy value.",
        )
        submitted = st.form_submit_button("Find observed measurement", type="primary")
    if submitted:
        st.session_state["energy_measurement_query"] = {
            "selection": selection,
            "electricity_price": float(electricity_price),
        }
    query = st.session_state.get("energy_measurement_query")
    if not query:
        st.info("Choose a measured-support configuration and select Find observed measurement.")
        return

    result = exact_observed_lookup(joined, query["selection"], query["electricity_price"])
    st.markdown(f'<span class="status-badge">{html.escape(result["label"])}</span>', unsafe_allow_html=True)
    if result["status"] == "observed":
        primary = st.columns(4)
        primary[0].metric("Joules per output token", f"{result['joules_per_output_token']:.4f}")
        primary[1].metric("Observed range", f"{result['minimum']:.4f} – {result['maximum']:.4f}")
        primary[2].metric("Observations", f"{result['match_count']:,}")
        primary[3].metric("Tokens per kWh", f"{result['tokens_per_kwh']:,.0f}")
        derived = st.columns(3)
        derived[0].metric("kWh / 1M output tokens", f"{result['kwh_per_million_output_tokens']:.4f}")
        derived[1].metric("Electricity cost / 1M output tokens", f"${result['electricity_cost_per_million_output_tokens']:.4f}")
        derived[2].metric("Observed date range", f"{result['date_start']} – {result['date_end']}")
        st.caption("Electricity cost is a mathematical conversion of the observed median, not a separately measured field.")
        details = pd.DataFrame([{
            "config_id": ", ".join(result["config_ids"]),
            "median throughput / GPU": result["throughput_per_gpu_median"],
            "median average power (W)": result["average_power_w_median"],
        }])
        st.dataframe(details, width="stretch", hide_index=True)
    elif result["status"] == "unsupported":
        st.error("Unsupported measured selection: " + ", ".join(result["unsupported_fields"]))
    else:
        st.info("No exact measured energy result. The rows below are observed comparisons, not predictions.")

    comparisons = nearest_measured_configurations(joined, query["selection"], limit=5)
    comparisons = mark_dominated_comparisons(comparisons)
    if comparisons.empty:
        st.info("No measured comparison rows are available.")
    else:
        st.markdown("#### Nearby observed configurations")
        st.caption("Distance combines categorical mismatches and range-normalized numeric differences. Identical workloads are used whenever available.")
        display_columns = [
            column for column in (
                "config_id", "differing_fields", ENERGY_TARGET, ENERGY_THROUGHPUT_METRIC,
                ENERGY_POWER_METRIC, "benchmark_type", "isl", "osl", "conc", "distance",
                "same_workload", "dominated_in_comparison",
            ) if column in comparisons
        ]
        st.dataframe(comparisons[display_columns], width="stretch", hide_index=True)
        plot = comparisons.dropna(subset=[ENERGY_TARGET, ENERGY_THROUGHPUT_METRIC])
        if not plot.empty:
            st.plotly_chart(
                px.scatter(
                    plot, x=ENERGY_THROUGHPUT_METRIC, y=ENERGY_TARGET,
                    color="dominated_in_comparison", hover_name="config_id",
                    hover_data=["differing_fields", "distance"],
                    title="Observed energy versus throughput within the comparison set",
                ),
                width="stretch",
            )
            st.caption("Dominance is evaluated only within these same-workload comparison rows; this is not a global Pareto frontier.")

    model_state = energy_model_availability()
    with st.expander("Future estimator integration"):
        st.write(model_state.reason)
        st.code("EnergyModelProvider.predict(selection)", language="python")
        st.caption("No model artifact is loaded and no modeled, extrapolated, or imputed energy value is produced.")


def main() -> None:
    render_dashboard_shell()
    st.markdown('<h1 class="dashboard-title">InferenceX Benchmark Research</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dashboard-subtitle">Configuration analysis and validated model results from the InferenceX benchmark dataset.</p>'
        '<span class="status-badge">Research dashboard</span>',
        unsafe_allow_html=True,
    )

    data_dir = st.session_state.get("data_dir_control", DEFAULT_DATA_DIR)
    file_status, source_probe = data_source_status(data_dir)

    with st.sidebar:
        st.header("Controls")
        analysis_unit = st.selectbox("Analysis unit", options=ANALYSIS_UNIT_OPTIONS, index=2)
        with st.expander("Advanced settings"):
            max_rows = st.number_input("Maximum rows", min_value=500, max_value=100_000, value=20_000, step=500)
            seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)
            data_dir = st.text_input("Data directory", value=data_dir, key="data_dir_control")
            st.caption("The official raw CSV export, flattened CSV, and JSON fallback are supported.")
            fallback_status = "available" if source_probe["json_ready"] else "not found"
            st.caption(f"Source check: {source_probe['active_mode']}. JSON fallback: {fallback_status}.")

    research_summary, research_error = research_summary_or_none()
    tabs = st.tabs(MAIN_TAB_LABELS)
    if source_probe["active_mode"] == "missing":
        with tabs[0]:
            st.info("Benchmark data is unavailable. Update the data directory in Advanced settings.")
            with st.expander("Data source details"):
                st.dataframe(file_status, width="stretch", hide_index=True)
        with tabs[1]:
            st.info("Data Understanding is available after benchmark data loads.")
        with tabs[2]:
            st.info("PCA is available after benchmark data loads.")
        with tabs[3]:
            render_model_results_dashboard(research_summary, research_error)
        return

    dataset_manifest = build_dataset_manifest(source_probe)
    try:
        benchmarks, _configs, joined, source_info = load_joined_data(data_dir, dataset_manifest["fingerprint"])
        analysis_frame, analysis_metadata = build_analysis_frame(joined, analysis_unit)
        pca_frame, pca_metadata = build_analysis_frame(
            joined, "Median aggregate per config/workload/concurrency"
        )
    except Exception as exc:
        st.error(f"Could not load benchmark data: {exc}")
        return
    analysis_metadata["dataset_fingerprint"] = dataset_manifest["fingerprint"]
    analysis_metadata["dataset_manifest"] = dataset_manifest
    pca_metadata["dataset_fingerprint"] = dataset_manifest["fingerprint"]
    pca_metadata["dataset_manifest"] = dataset_manifest

    with tabs[0]:
        render_overview(benchmarks, analysis_metadata, source_info, research_summary)
        with st.expander("Data source details"):
            render_data_source_details(file_status, source_info, analysis_metadata)
    with tabs[1]:
        render_data_understanding_dashboard(joined, analysis_frame, analysis_metadata, int(max_rows), int(seed))
    with tabs[2]:
        render_pca_dashboard(pca_frame, pca_metadata, int(max_rows), int(seed))
    with tabs[3]:
        render_model_results_dashboard(research_summary, research_error)
        render_energy_measurements_dashboard(joined)


if __name__ == "__main__":
    main()
