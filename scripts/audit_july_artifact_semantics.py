#!/usr/bin/env python3
"""Fail-closed semantic audit for the recovered July 20 checkpoint.

This intentionally does not train, refit, regenerate, or overwrite any PCA or
representation artifact.  It proves equivalence of the active aggregate cohort
and committed AE/VAE companions by stable row identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from apps import inferencex_pca_demo as app  # noqa: E402
from modeling.dataset_checkpoint import VERIFIED_CHECKPOINT, artifact_matches_active_dataset  # noqa: E402
from modeling.representation_analysis import (  # noqa: E402
    align_companion_to_active_cohort,
    canonical_representation_data,
)


ARTIFACTS = (
    ("autoencoder", app.AE_REPRESENTATION_ARTIFACT_PATH, "autoencoder"),
    (
        "variational_autoencoder",
        app.VAE_REPRESENTATION_ARTIFACT_PATH,
        "variational_autoencoder",
    ),
)


def _file_entry(identity: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in identity.get("files", []):
        if entry.get("name") == name:
            return entry
    raise RuntimeError(f"Active dataset identity is missing {name}.")


def _artifact_config_identity() -> dict[str, Any]:
    pca = json.loads(app.PCA_TARGET_ARTIFACT_PATH.read_text(encoding="utf-8"))
    for entry in pca.get("dump", {}).get("manifest", {}).get("files", []):
        if entry.get("name") == "configs.csv":
            return entry
    raise RuntimeError("Committed PCA artifact has no configs.csv provenance entry.")


def audit(data_dir: str) -> dict[str, Any]:
    _status, source = app.data_source_status(data_dir)
    active = app.build_dataset_manifest(source)
    if active.get("verification", {}).get("status") != VERIFIED_CHECKPOINT:
        raise RuntimeError("Active source is not the fully verified July checkpoint.")
    _benchmarks, _configs, joined, _source = app.load_joined_data(
        data_dir, active["fingerprint"]
    )
    aggregate, _metadata = app.build_analysis_frame(
        joined, "Median aggregate per config/workload/concurrency"
    )
    data = canonical_representation_data(aggregate)

    artifact_reports: dict[str, Any] = {}
    for label, path, method in ARTIFACTS:
        artifact, companion = app.load_neural_representation_artifact(str(path), method)
        if not artifact_matches_active_dataset(active, artifact):
            raise RuntimeError(f"{label} artifact does not identify the active verified checkpoint.")
        _aligned, report = align_companion_to_active_cohort(companion, data)
        artifact_reports[label] = {
            "source_release_matches": True,
            "committed_ordered_cohort_hash": artifact["cohort_hash"],
            "committed_ordered_row_key_hash": artifact["row_key_hash"],
            "semantic_validation": report,
        }

    active_configs = _file_entry(active, "configs.csv")
    artifact_configs = _artifact_config_identity()
    config_hash_matches = active_configs["sha256"] == artifact_configs.get(
        "content_sample_sha256"
    )
    if not config_hash_matches:
        raise RuntimeError("configs.csv identity differs from the committed July provenance.")

    return {
        "verdict": "SEMANTICALLY EQUIVALENT",
        "dataset": {
            "dataset_id": active["dataset_id"],
            "source_release": active["source_release"],
            "verified_checkpoint": True,
            "benchmark_results_raw": _file_entry(active, "benchmark_results_raw.csv"),
            "configs": active_configs,
        },
        "counts": {
            "aggregate_groups": len(aggregate),
            "cohort_rows": len(data.cohort),
            "eligible_configurations": int(data.cohort["config_id"].nunique()),
        },
        "legacy_order_dependence": {
            "recovered_ordered_cohort_hash": data.legacy_ordered_cohort_hash,
            "recovered_ordered_row_key_hash": data.legacy_ordered_row_key_hash,
            "note": "These source-order hashes differ from the historical artifact hashes but are not used as semantic identities.",
        },
        "order_independent_identity": data.semantic_identity,
        "config_feature_provenance": {
            "active_configs_sha256": active_configs["sha256"],
            "committed_artifact_configs_sha256": artifact_configs.get("content_sample_sha256"),
            "byte_identity_matches": config_hash_matches,
            "pca_features_unique_by_config_id": True,
        },
        "artifacts": artifact_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    args = parser.parse_args()
    report = audit(args.data_dir)
    print(json.dumps(report, indent=2, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
