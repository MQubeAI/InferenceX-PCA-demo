"""Build the fixed PCA protocol for a versioned candidate snapshot.

Unlike the historical July builder, this command has no July count gate and does
not overwrite any committed artifact.  Candidate manifests supply provenance and
the deterministic cohort ordering comes from ``shared_basis_cohort``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from apps import inferencex_pca_demo as app  # noqa: E402
from modeling.pca_target_analysis import (  # noqa: E402
    ENERGY_TARGET,
    LATENCY_TARGET,
    OUTPUT_TARGET,
    PCA_FEATURES,
    SHARED_COHORT_FILTERS,
    component_thresholds,
    explained_variance_table,
    fit_shared_pca,
    loading_table,
    preprocessing_state,
    source_loading_table,
    target_overlay,
    validate_pca_feature_schema,
)
from modeling.representation_analysis import canonical_representation_data  # noqa: E402


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [{key: _json_value(item) for key, item in row.items()} for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def load_candidate_aggregate(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _status, source = app.data_source_status(str(data_dir))
    if source["active_mode"] != "Raw CSV":
        raise RuntimeError("Candidate PCA requires raw benchmark_results_raw.csv and configs.csv.")
    manifest = app.build_dataset_manifest(source)
    _benchmarks, _configs, joined, _source = app.load_joined_data(str(data_dir), manifest["fingerprint"])
    aggregate, metadata = app.build_analysis_frame(joined, "Median aggregate per config/workload/concurrency")
    return joined, aggregate, {"manifest": manifest, "analysis": metadata}


def build_snapshot_pca_artifact(
    *,
    data_dir: str | Path,
    snapshot_manifest: dict[str, Any],
) -> dict[str, Any]:
    validate_pca_feature_schema(PCA_FEATURES)
    raw, aggregate, metadata = load_candidate_aggregate(data_dir)
    result = fit_shared_pca(aggregate)
    representation = canonical_representation_data(aggregate, enforce_snapshot_counts=False)
    state = preprocessing_state(result)
    basis_hash = hashlib.sha256(json.dumps(_json_value(state), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target_results = {target: target_overlay(result, target) for target in (LATENCY_TARGET, OUTPUT_TARGET, ENERGY_TARGET)}
    dates = pd.to_datetime(result.cohort.get("date"), errors="coerce")
    return {
        "schema_version": "pca-target-overlays-v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "dataset_id": snapshot_manifest["dataset_id"],
            "source_release": snapshot_manifest["source"]["release"],
            "files": [
                {"name": name, "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
                for name, item in sorted(snapshot_manifest["files"].items())
            ],
        },
        "counts": {
            "raw_rows": len(raw),
            "aggregate_rows": len(aggregate),
            "configurations": int(raw["config_id"].nunique()),
            "cohort_rows": len(result.cohort),
            "eligible_configurations": int(result.cohort["config_id"].nunique()),
        },
        "analysis_unit": "median by config_id, benchmark_type, isl, osl, conc",
        "shared_basis": {
            "decision": "Fixed PCA feature schema and deterministic canonical cohort ordering",
            "cohort_filters": SHARED_COHORT_FILTERS,
            "feature_order": list(PCA_FEATURES),
            "target_metrics_in_inputs": [],
            "target_overlays": [LATENCY_TARGET, OUTPUT_TARGET, ENERGY_TARGET],
            "basis_sha256": basis_hash,
            "semantic_identity": representation.semantic_identity,
            "aggregate_representative_date_range": [
                dates.min().date().isoformat() if dates.notna().any() else None,
                dates.max().date().isoformat() if dates.notna().any() else None,
            ],
            "preprocessing": state,
            "explained_variance": _json_value(explained_variance_table(result)),
            "component_thresholds": component_thresholds(result),
            "encoded_loadings_first_five": _json_value(loading_table(result)),
            "source_loadings_first_five": _json_value(source_loading_table(result)),
        },
        "targets": {
            target: {
                "usable_rows": values["usable_rows"],
                "unique_configurations": values["unique_configurations"],
                "distribution": values["distribution"],
                "associations": _json_value(values["associations"]),
                "component_bins": _json_value(values["component_bins"]),
            }
            for target, values in target_results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
    artifact = build_snapshot_pca_artifact(data_dir=args.data_dir, snapshot_manifest=manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(artifact), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} for {artifact['dataset']['source_release']}")


if __name__ == "__main__":
    main()
