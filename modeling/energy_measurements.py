"""Observed-only energy lookup and comparison utilities.

No function in this module fits or loads a predictive model. A future model may
implement ``EnergyModelProvider`` only after the target definition is documented
and broader grouped validation is approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


ENERGY_TARGET = "metrics_joules_per_output_token"
THROUGHPUT_METRIC = "metrics_tput_per_gpu"
POWER_METRIC = "metrics_avg_power_w"
WORKLOAD_FIELDS = ("benchmark_type", "isl", "osl", "conc")
CONFIG_FIELDS = (
    "config_model",
    "config_hardware",
    "config_framework",
    "config_precision",
    "config_spec_method",
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
)
MATCH_FIELDS = WORKLOAD_FIELDS + CONFIG_FIELDS
ENERGY_MODELING_ENABLED = False


class EnergyModelProvider(Protocol):
    """Reserved interface for a separately validated future artifact."""

    def predict(self, selection: dict[str, Any]) -> float: ...


@dataclass(frozen=True)
class EnergyModelAvailability:
    enabled: bool = False
    reason: str = (
        "Energy prediction is blocked pending target documentation and broader measured support."
    )


def energy_model_availability() -> EnergyModelAvailability:
    return EnergyModelAvailability()


def measured_energy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with a real, positive observed energy target."""
    if ENERGY_TARGET not in frame:
        return frame.iloc[0:0].copy()
    target = pd.to_numeric(frame[ENERGY_TARGET], errors="coerce")
    return frame.loc[target.notna() & np.isfinite(target) & target.gt(0)].copy()


def available_control_values(frame: pd.DataFrame) -> dict[str, list[Any]]:
    measured = measured_energy_rows(frame)
    values: dict[str, list[Any]] = {}
    for field in MATCH_FIELDS:
        if field not in measured:
            continue
        unique = measured[field].drop_duplicates().dropna().tolist()
        if pd.api.types.is_numeric_dtype(measured[field]):
            unique = sorted(unique)
        else:
            unique = sorted(unique, key=lambda value: str(value))
        values[field] = unique
    return values


def unsupported_selection_fields(
    frame: pd.DataFrame, selection: dict[str, Any]
) -> list[str]:
    available = available_control_values(frame)
    return [
        field
        for field, value in selection.items()
        if field in MATCH_FIELDS and (field not in available or not _contains(available[field], value))
    ]


def _contains(values: list[Any], selected: Any) -> bool:
    return any(_equal(value, selected) for value in values)


def _equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _selection_mask(frame: pd.DataFrame, selection: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for field in MATCH_FIELDS:
        if field not in frame or field not in selection:
            return pd.Series(False, index=frame.index)
        value = selection[field]
        mask &= frame[field].isna() if pd.isna(value) else frame[field].eq(value)
    return mask


def observed_energy_conversions(joules_per_output_token: float, price_per_kwh: float) -> dict[str, float]:
    if joules_per_output_token <= 0:
        raise ValueError("Observed joules per output token must be positive.")
    if price_per_kwh < 0:
        raise ValueError("Electricity price must be non-negative.")
    kwh_per_million = joules_per_output_token / 3.6
    return {
        "tokens_per_kwh": 3_600_000 / joules_per_output_token,
        "kwh_per_million_output_tokens": kwh_per_million,
        "electricity_cost_per_million_output_tokens": kwh_per_million * price_per_kwh,
    }


def exact_observed_lookup(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    price_per_kwh: float = 0.12,
) -> dict[str, Any]:
    measured = measured_energy_rows(frame)
    unsupported = unsupported_selection_fields(measured, selection)
    if unsupported:
        return {
            "status": "unsupported",
            "label": "Unsupported measured workload or configuration",
            "unsupported_fields": unsupported,
            "is_prediction": False,
            "match_count": 0,
        }
    matches = measured.loc[_selection_mask(measured, selection)].copy()
    if matches.empty:
        return {
            "status": "no_exact_match",
            "label": "No exact measured energy result",
            "unsupported_fields": [],
            "is_prediction": False,
            "match_count": 0,
        }
    target = pd.to_numeric(matches[ENERGY_TARGET], errors="coerce")
    median = float(target.median())
    dates = pd.to_datetime(matches.get("date"), errors="coerce", utc=True)
    result = {
        "status": "observed",
        "label": "Observed energy measurement",
        "is_prediction": False,
        "match_count": int(len(matches)),
        "config_ids": sorted(matches["config_id"].dropna().astype(str).unique().tolist())
        if "config_id" in matches else [],
        "joules_per_output_token": median,
        "minimum": float(target.min()),
        "maximum": float(target.max()),
        "date_start": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().date().isoformat() if dates.notna().any() else None,
        "throughput_per_gpu_median": _median_or_none(matches, THROUGHPUT_METRIC),
        "average_power_w_median": _median_or_none(matches, POWER_METRIC),
    }
    return {**result, **observed_energy_conversions(median, price_per_kwh)}


def _median_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").median()
    return None if pd.isna(value) else float(value)


def aggregate_measured_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    measured = measured_energy_rows(frame)
    grouping = [field for field in ("config_id",) + MATCH_FIELDS if field in measured]
    aggregations: dict[str, Any] = {ENERGY_TARGET: "median"}
    for column in (THROUGHPUT_METRIC, POWER_METRIC):
        if column in measured:
            aggregations[column] = "median"
    if "date" in measured:
        aggregations["date"] = ["min", "max"]
    grouped = measured.groupby(grouping, dropna=False, sort=False).agg(aggregations).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    rename = {
        f"{ENERGY_TARGET}_median": ENERGY_TARGET,
        f"{THROUGHPUT_METRIC}_median": THROUGHPUT_METRIC,
        f"{POWER_METRIC}_median": POWER_METRIC,
        "date_min": "date_start",
        "date_max": "date_end",
    }
    return grouped.rename(columns=rename)


def nearest_measured_configurations(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    limit: int = 5,
) -> pd.DataFrame:
    """Return observed comparison rows, prioritizing identical workloads."""
    rows = aggregate_measured_configurations(frame)
    if rows.empty:
        return rows
    same_workload = pd.Series(True, index=rows.index)
    for field in WORKLOAD_FIELDS:
        if field in rows and field in selection:
            same_workload &= rows[field].eq(selection[field])
    pool = rows.loc[same_workload].copy() if same_workload.any() else rows.copy()
    numeric_fields = [
        field for field in MATCH_FIELDS
        if field in pool and field in selection
        and pd.api.types.is_numeric_dtype(pool[field])
        and not pd.api.types.is_bool_dtype(pool[field])
    ]
    ranges = {
        field: float(pd.to_numeric(rows[field], errors="coerce").max() - pd.to_numeric(rows[field], errors="coerce").min())
        for field in numeric_fields
    }
    distances: list[float] = []
    differences: list[str] = []
    for _, row in pool.iterrows():
        distance = 0.0
        differing: list[str] = []
        for field in MATCH_FIELDS:
            if field not in pool or field not in selection or _equal(row[field], selection[field]):
                continue
            differing.append(f"{field}: {selection[field]} → {row[field]}")
            if field in numeric_fields and ranges[field] > 0:
                distance += abs(float(row[field]) - float(selection[field])) / ranges[field]
            else:
                distance += 1.0
        distances.append(distance)
        differences.append("; ".join(differing) or "none")
    pool["distance"] = distances
    pool["differing_fields"] = differences
    pool["comparison_only"] = True
    pool["same_workload"] = same_workload.loc[pool.index].astype(bool)
    return pool.sort_values(["distance", ENERGY_TARGET, "config_id"], kind="mergesort").head(limit).reset_index(drop=True)


def mark_dominated_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark rows dominated within this same-workload comparison set only."""
    result = frame.copy()
    result["dominated_in_comparison"] = False
    if frame.empty or THROUGHPUT_METRIC not in frame:
        return result
    for index, row in result.iterrows():
        energy = row.get(ENERGY_TARGET)
        throughput = row.get(THROUGHPUT_METRIC)
        if pd.isna(energy) or pd.isna(throughput):
            continue
        peers = result[result.get("same_workload", True).astype(bool)] if "same_workload" in result else result
        dominates = (
            peers[ENERGY_TARGET].lt(energy)
            & peers[THROUGHPUT_METRIC].gt(throughput)
        )
        result.at[index, "dominated_in_comparison"] = bool(dominates.any())
    return result


def energy_support_summary(frame: pd.DataFrame) -> dict[str, Any]:
    measured = measured_energy_rows(frame)
    all_configs = int(frame["config_id"].nunique()) if "config_id" in frame else 0
    measured_configs = int(measured["config_id"].nunique()) if "config_id" in measured else 0
    dates = pd.to_datetime(measured.get("date"), errors="coerce", utc=True)
    coverage = {}
    for field in ("config_hardware", "config_framework", "config_model"):
        coverage[field] = sorted(measured[field].dropna().astype(str).unique().tolist()) if field in measured else []
    workloads = (
        measured[[field for field in ("benchmark_type", "isl", "osl") if field in measured]]
        .drop_duplicates().sort_values([field for field in ("benchmark_type", "isl", "osl") if field in measured])
        .to_dict("records")
    )
    return {
        "usable_raw_rows": int(len(measured)),
        "measured_configurations": measured_configs,
        "all_configurations": all_configs,
        "configuration_coverage": measured_configs / all_configs if all_configs else 0.0,
        "observed_workloads": workloads,
        "date_start": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().date().isoformat() if dates.notna().any() else None,
        "coverage": coverage,
        "measurement_warning": "The energy measurement procedure and system boundary are undocumented.",
    }
