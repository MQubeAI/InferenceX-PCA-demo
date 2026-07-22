#!/usr/bin/env python3
"""Build the versioned July PCA artifact without fitting supervised models."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps import inferencex_pca_demo as app
from modeling.energy_measurements import energy_support_summary, measured_energy_rows
from modeling.pca_target_analysis import (
    ENERGY_TARGET,
    ENERGY_TARGET_LABEL,
    ENERGY_TARGET_UNIT,
    OUTPUT_TARGET,
    OUTPUT_TARGET_LABEL,
    OUTPUT_TARGET_UNIT,
    PCA_FEATURES,
    SHARED_COHORT_FILTERS,
    compare_bases,
    component_thresholds,
    explained_variance_table,
    fit_shared_pca,
    loading_table,
    preprocessing_state,
    source_loading_table,
    sparse_group_warnings,
    target_overlay,
    validate_pca_feature_schema,
)


EXPECTED_COUNTS = {
    "raw_rows": 81_851,
    "aggregate_rows": 8_239,
    "configurations": 1_368,
    "energy_raw_rows": 5_175,
    "energy_aggregate_groups": 2_766,
    "energy_configurations": 305,
}
OFFICIAL_RELEASE_URL = (
    "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20"
)


def json_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [{key: json_value(item) for key, item in row.items()} for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_aggregate(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _, probe = app.data_source_status(data_dir)
    if probe["active_mode"] == "missing":
        raise FileNotFoundError(f"No supported data source found in {data_dir}")
    manifest = app.build_dataset_manifest(probe)
    _benchmarks, _configs, joined, _source = app.load_joined_data(data_dir, manifest["fingerprint"])
    aggregate, metadata = app.build_analysis_frame(
        joined, "Median aggregate per config/workload/concurrency"
    )
    return joined, aggregate, {"manifest": manifest, "analysis": metadata}


def observed_values(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame:
        return []
    series = frame[column].dropna().drop_duplicates()
    values = [json_value(value) for value in series.tolist()]
    return sorted(values) if pd.api.types.is_numeric_dtype(series) else sorted(values, key=str)


def supported_group_summary(frame: pd.DataFrame, target: str, column: str, minimum: int = 30) -> list[dict[str, Any]]:
    values = pd.to_numeric(frame[target], errors="coerce")
    work = frame.loc[values.notna(), [column]].copy()
    work[target] = values.loc[values.notna()]
    grouped = work.groupby(column, dropna=False)[target].agg(["count", "median", "mean"]).reset_index()
    grouped["sufficient_support"] = grouped["count"].ge(minimum)
    return json_value(grouped)


def build_artifact(july_dir: str, june_dir: str) -> dict[str, Any]:
    validate_pca_feature_schema(PCA_FEATURES)
    july_raw, july_aggregate, july_metadata = load_aggregate(july_dir)
    june_raw, june_aggregate, june_metadata = load_aggregate(june_dir)
    energy_raw = measured_energy_rows(july_raw)
    energy_aggregate = july_aggregate.loc[
        pd.to_numeric(july_aggregate.get(ENERGY_TARGET), errors="coerce").gt(0)
    ].copy()
    actual_counts = {
        "raw_rows": len(july_raw),
        "aggregate_rows": len(july_aggregate),
        "configurations": int(july_raw["config_id"].nunique()),
        "energy_raw_rows": len(energy_raw),
        "energy_aggregate_groups": len(energy_aggregate),
        "energy_configurations": int(energy_raw["config_id"].nunique()),
    }
    if actual_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"July count gate failed: expected {EXPECTED_COUNTS}, got {actual_counts}")

    july_pca = fit_shared_pca(july_aggregate)
    june_pca = fit_shared_pca(june_aggregate)
    output = target_overlay(july_pca, OUTPUT_TARGET)
    energy = target_overlay(july_pca, ENERGY_TARGET)
    explained = explained_variance_table(july_pca)
    old_explained = explained_variance_table(june_pca)
    comparison = compare_bases(june_pca, july_pca)
    energy_support = energy_support_summary(july_raw)
    dates = pd.to_datetime(energy_raw.get("date"), errors="coerce")
    temporal = energy_raw.assign(
        month=dates.dt.to_period("M").astype(str),
        _energy=pd.to_numeric(energy_raw[ENERGY_TARGET], errors="coerce"),
    ).groupby("month", dropna=False)["_energy"].agg(["count", "median", "mean"]).reset_index()
    log_energy = np.log1p(pd.to_numeric(energy["frame"][ENERGY_TARGET], errors="coerce"))

    return {
        "schema_version": "pca-target-overlays-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "dump": {
            "version": app.ACTIVE_DUMP_VERSION,
            "release": app.ACTIVE_DUMP_RELEASE,
            "official_release_url": OFFICIAL_RELEASE_URL,
            "active_path": str(Path(july_dir).resolve()),
            "manifest": july_metadata["manifest"],
            "rollback_path": str(Path(june_dir).resolve()),
            "rollback_manifest": june_metadata["manifest"],
        },
        "counts": actual_counts,
        "analysis_unit": "median by config_id, benchmark_type, isl, osl, conc",
        "shared_basis": {
            "decision": "One shared July basis for direct target-overlay comparability",
            "cohort_filters": SHARED_COHORT_FILTERS,
            "cohort_rows": len(july_pca.cohort),
            "excluded_rows": len(july_aggregate) - len(july_pca.cohort),
            "exclusion_reason": "agentic_traces rows do not share the ISL/OSL workload semantics used by both target cohorts",
            "feature_order": list(PCA_FEATURES),
            "target_metrics_in_inputs": [],
            "preprocessing": preprocessing_state(july_pca),
            "explained_variance": json_value(explained),
            "component_thresholds": component_thresholds(july_pca),
            "encoded_loadings_first_five": json_value(loading_table(july_pca)),
            "source_loadings_first_five": json_value(source_loading_table(july_pca)),
            "june_first_five_cumulative": float(old_explained.iloc[4]["cumulative_explained_variance"]),
            "july_first_five_cumulative": float(explained.iloc[4]["cumulative_explained_variance"]),
            "basis_comparison": json_value(comparison),
        },
        "targets": {
            OUTPUT_TARGET: {
                "display_name": OUTPUT_TARGET_LABEL,
                "raw_target": OUTPUT_TARGET,
                "transformation": "identity",
                "inverse_transformation": "identity",
                "unit": OUTPUT_TARGET_UNIT,
                "direction": "higher is better",
                "cohort_filters": SHARED_COHORT_FILTERS,
                "aggregation_unit": "median by config_id, benchmark_type, isl, osl, conc",
                "usable_rows": output["usable_rows"],
                "unique_configurations": output["unique_configurations"],
                "workload_support": {
                    key: observed_values(output["frame"], key)
                    for key in ("benchmark_type", "isl", "osl", "conc")
                },
                "raw_distribution": output["distribution"],
                "transformed_distribution": output["distribution"],
                "associations": json_value(output["associations"]),
                "component_bins": json_value(output["component_bins"]),
                "historical_validation_context": {
                    "rows": 4096,
                    "folds": 3,
                    "grouping": "config_id",
                    "model": "TabFM",
                    "r2_mean": 0.961979,
                    "r2_std": 0.008605,
                    "mae": 338.540384,
                    "mae_unit": OUTPUT_TARGET_UNIT,
                    "note": "Target-selection context only; no model was retrained for this refresh.",
                },
            },
            ENERGY_TARGET: {
                "display_name": ENERGY_TARGET_LABEL,
                "raw_target": ENERGY_TARGET,
                "transformation": "identity (log1p used only for distribution visualization)",
                "inverse_transformation": "identity",
                "unit": ENERGY_TARGET_UNIT,
                "direction": "lower is better",
                "cohort_filters": {**SHARED_COHORT_FILTERS, "target": "observed positive values only"},
                "aggregation_unit": "median by config_id, benchmark_type, isl, osl, conc",
                "usable_rows": energy["usable_rows"],
                "raw_measured_rows": len(energy_raw),
                "unique_configurations": energy["unique_configurations"],
                "configuration_coverage": energy_support["configuration_coverage"],
                "date_range": [energy_support["date_start"], energy_support["date_end"]],
                "workload_support": {
                    key: observed_values(energy["frame"], key)
                    for key in ("benchmark_type", "isl", "osl", "conc")
                },
                "configuration_support": {
                    key: observed_values(energy["frame"], key)
                    for key in (
                        "config_hardware",
                        "config_framework",
                        "config_model",
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
                    )
                },
                "raw_distribution": energy["distribution"],
                "log1p_distribution": {
                    key: float(value)
                    for key, value in log_energy.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict().items()
                },
                "associations": json_value(energy["associations"]),
                "component_bins": json_value(energy["component_bins"]),
                "temporal_summary": json_value(temporal),
                "subgroup_summaries": {
                    key: supported_group_summary(energy["frame"], ENERGY_TARGET, key)
                    for key in ("isl", "config_hardware", "config_framework", "config_model")
                },
                "sparse_group_warnings": sparse_group_warnings(energy["frame"], ENERGY_TARGET),
                "interpretation_guard": "Observed association only; PCA does not predict energy.",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--july-data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--june-data-dir", default=app.ROLLBACK_DATA_DIR)
    parser.add_argument("--output", default="artifacts/pca-db-dump-2026-07-20.json")
    args = parser.parse_args()
    artifact = build_artifact(args.july_data_dir, args.june_data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_value(artifact), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print(json.dumps(artifact["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
