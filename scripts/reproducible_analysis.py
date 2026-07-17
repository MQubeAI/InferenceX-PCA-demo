#!/usr/bin/env python3
"""Run the documented default analysis without the Streamlit interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps import inferencex_pca_demo as app


def records(frame: Any) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--output", default="artifacts/reproducible-results.json")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-runs", type=int, default=app.DEFAULT_PCA_STABILITY_RUNS)
    args = parser.parse_args()

    _, source_info = app.data_source_status(args.data_dir)
    if source_info["active_mode"] == "missing":
        raise SystemExit("Required CSV files and JSON fallback files are unavailable.")
    manifest = app.build_dataset_manifest(source_info)
    benchmarks, _, joined, _ = app.load_joined_data(args.data_dir, manifest["fingerprint"])
    analysis_rows: dict[str, int] = {}
    frames: dict[str, tuple[Any, dict[str, Any]]] = {}
    for unit in app.ANALYSIS_UNIT_OPTIONS:
        frame, metadata = app.build_analysis_frame(joined, unit)
        analysis_rows[unit] = len(frame)
        frames[unit] = (frame, metadata)

    default_unit = "Median aggregate per config/workload/concurrency"
    analysis_frame, metadata = frames[default_unit]
    metadata["dataset_fingerprint"] = manifest["fingerprint"]
    numeric_features, categorical_features = app.default_pca_features(analysis_frame)
    selected_features = numeric_features + categorical_features
    target = app.default_sales_target_metric(app.metric_like_numeric_columns(analysis_frame))
    pca_result, pca_error = app.fit_pca_analysis(
        analysis_frame, selected_features, args.max_rows, args.seed, target
    )
    if pca_result is None:
        raise SystemExit(pca_error)
    stability, stability_error = app.pca_stability_summary(
        analysis_frame, selected_features, args.max_rows, args.seed, args.stability_runs
    )
    if stability_error:
        raise SystemExit(stability_error)
    target_numeric, target_categorical = app.default_target_features(analysis_frame, target)
    target_features = target_numeric + target_categorical
    rf_result, rf_error = app.grouped_rf_evaluation(
        analysis_frame,
        target_features,
        target,
        args.max_rows,
        args.seed,
        n_estimators=150,
    )
    if rf_result is None:
        raise SystemExit(rf_error)

    pca_controls = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_metric": target,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "stability_runs": args.stability_runs,
    }
    rf_controls = {
        "target": target,
        "numeric_features": target_numeric,
        "categorical_features": target_categorical,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "split_mode": "Grouped cross-validation by config_id",
        "n_estimators": 150,
        "permutation_repeats": app.DEFAULT_PERMUTATION_REPEATS,
    }
    artifact = {
        "generation_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": app.git_commit(),
        "dataset": manifest,
        "row_counts": {
            "benchmark_rows": len(benchmarks),
            "joined_rows": len(joined),
            "analysis_units": analysis_rows,
        },
        "default_analysis": {
            "analysis_unit": default_unit,
            "analysis_rows": len(analysis_frame),
            "sample_size_limit": args.max_rows,
            "seed": args.seed,
            "pca_features": selected_features,
            "target": target,
            "rf_features": target_features,
            "pca_signature": app.analysis_signature(metadata, "pca", pca_controls),
            "rf_signature": app.analysis_signature(metadata, "rf", rf_controls),
        },
        "pca": {
            "sampled_rows": pca_result["sampled_rows"],
            "explained_variance": records(pca_result["explained_variance"]),
            "top_feature_groups": records(pca_result["source_contributions"].head(15)),
            "stability": {
                "runs": stability["runs"],
                "sample_rows": stability["sample_rows"],
                "sample_fraction": stability["sample_fraction"],
                "explained_variance": records(stability["explained_variance"]),
                "component_similarity": records(stability["component_similarity"]),
                "top_driver_frequency": records(stability["top_driver_frequency"]),
                "warnings": stability["warnings"],
            },
        },
        "grouped_random_forest": {
            "evaluation_mode": rf_result["split_mode"],
            "sampled_rows": rf_result["sampled_rows"],
            "fold_count": rf_result["fold_count"],
            "fold_metrics": records(rf_result["fold_metrics"]),
            "aggregate_metrics": rf_result["metric_summary"],
            "permutation_importance": records(rf_result["importance_frame"]),
            "warnings": rf_result["warnings"],
        },
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("streamlit", "pandas", "numpy", "scikit-learn", "plotly")
        },
        "warnings": stability["warnings"] + rf_result["warnings"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({artifact['dataset']['fingerprint'][:12]})")


if __name__ == "__main__":
    main()
