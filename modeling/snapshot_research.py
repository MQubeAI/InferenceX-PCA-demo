"""Fixed-protocol snapshot research planning and candidate/active comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from modeling.snapshot_refresh import SnapshotRefreshError, snapshot_artifact_names


def selected_tabfm_contract(repository_root: str | Path) -> dict[str, Any]:
    """Describe the actual selected protocol without claiming a serving checkpoint."""

    root = Path(repository_root)
    conclusion = (root / "docs" / "model-research-conclusion.md").read_text(encoding="utf-8")
    if "metrics_tput_per_gpu" not in conclusion or "TabFM" not in conclusion:
        raise SnapshotRefreshError("The selected supervised model contract is not documented.")
    persisted = list((root / "artifacts").glob("*tabfm*.pt")) + list((root / "artifacts").glob("*tabfm*.ckpt"))
    return {
        "target": "metrics_tput_per_gpu",
        "transformation": "raw identity",
        "validation": "three grouped config_id folds; full-context TabFM",
        "status": "evaluation_only" if not persisted else "persisted_checkpoint",
        "persisted_checkpoints": [str(path) for path in persisted],
        "runner_requirement": (
            "A dedicated approved TabFM runner/environment is required; scripts/run_tabfm_mac.sh is macOS-specific "
            "and is not suitable for standard GitHub-hosted runners."
            if not persisted
            else None
        ),
    }


def fixed_protocol_plan(snapshot_manifest: dict[str, Any], artifact_directory: str | Path) -> dict[str, Any]:
    """Produce snapshot-named fixed research outputs, never a model search plan."""

    date = snapshot_manifest["snapshot_date"]
    names = snapshot_artifact_names(date)
    output = Path(artifact_directory)
    return {
        "dataset_id": snapshot_manifest["dataset_id"],
        "source_release": snapshot_manifest["source"]["release"],
        "artifact_directory": str(output),
        "artifact_names": names,
        "invariants": {
            "pca_features": "frozen 19 configuration/workload fields",
            "pca_targets": "outcome metrics excluded from fitting",
            "row_order": "canonical row identity ordering",
            "ae_vae": "fixed selected final architecture and seeds 42, 123, 2026",
            "validation": "grouped config_id folds with zero overlap",
            "stage4": "methodological validation, not model search",
            "tabfm": "selected raw throughput protocol only",
        },
    }


def _metric_summary(artifact: dict[str, Any], path: tuple[str, ...]) -> dict[str, float] | None:
    value: Any = artifact
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if not isinstance(value, dict):
        return None
    result = {}
    for key in ("mean", "standard_deviation", "minimum", "maximum"):
        if isinstance(value.get(key), (int, float)) and math.isfinite(float(value[key])):
            result[key] = float(value[key])
    return result or None


def compare_research_artifacts(active_paths: dict[str, Path], candidate_paths: dict[str, Path]) -> dict[str, Any]:
    """Compare only recorded metrics and report unavailable evidence explicitly."""

    result: dict[str, Any] = {}
    for key in ("pca", "ae", "vae", "comparison", "stage4", "tabfm", "uncertainty"):
        candidate_path = candidate_paths.get(key)
        active_path = active_paths.get(key)
        if not candidate_path or not candidate_path.is_file():
            result[key] = {"status": "FAIL", "reason": "candidate artifact missing"}
            continue
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        active = json.loads(active_path.read_text(encoding="utf-8")) if active_path and active_path.is_file() else None
        if key == "pca":
            candidate_value = candidate.get("shared_basis", {}).get("explained_variance", [])
            active_value = active.get("shared_basis", {}).get("explained_variance", []) if active else []
            result[key] = {"status": "PASS", "candidate_explained_variance": candidate_value[:5], "active_explained_variance": active_value[:5]}
        elif key in {"ae", "vae"}:
            result[key] = {
                "status": "PASS",
                "candidate_validation_mse": _metric_summary(candidate, ("summary", "validation_mse")),
                "active_validation_mse": _metric_summary(active, ("summary", "validation_mse")) if active else None,
                "candidate_validation_mae": _metric_summary(candidate, ("summary", "validation_mae")),
                "active_validation_mae": _metric_summary(active, ("summary", "validation_mae")) if active else None,
            }
        else:
            result[key] = {"status": "PASS", "candidate_path": str(candidate_path), "active_path": str(active_path) if active_path else None}
    return result
