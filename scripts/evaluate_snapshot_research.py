"""Validate fixed-protocol candidate artifacts and write explicit promotion gates.

This command evaluates metadata and aggregate results only.  It never retrains
or silently substitutes a historical artifact for a candidate one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.representation_analysis import validate_final_representation_artifact  # noqa: E402
from modeling.research_summary import read_aggregate_artifact  # noqa: E402
from modeling.snapshot_refresh import SnapshotRefreshError, snapshot_artifact_names  # noqa: E402
from modeling.snapshot_research import compare_research_artifacts  # noqa: E402
from scripts.build_snapshot_model_summary import build_snapshot_model_summary  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_paths(directory: Path, snapshot_date: str) -> dict[str, Path]:
    names = snapshot_artifact_names(snapshot_date)
    beta = directory / f"representation-vae-beta-diagnostic-db-dump-{snapshot_date}.json"
    return {
        "pca": directory / names["pca"],
        "ae": directory / names["ae_json"],
        "vae": directory / names["vae_json"],
        "vae_beta": beta,
        "comparison": directory / names["comparison"],
        "stage4": directory / names["stage4"],
        "tabfm": directory / names["tabfm_evaluation"],
        "uncertainty": directory / names["uncertainty"],
        "model_summary": directory / names["model_summary"],
    }


def evaluate_candidate_research(manifest: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    """Return PASS/FAIL gates for every fixed candidate research artifact."""

    source_release = manifest["source"]["release"]
    paths = _candidate_paths(artifact_dir, manifest["snapshot_date"])
    gates: dict[str, dict[str, str]] = {}
    try:
        pca = _read(paths["pca"])
        if (pca.get("dump", {}).get("version") or pca.get("dataset", {}).get("source_release")) != source_release:
            raise SnapshotRefreshError("PCA source release does not match candidate manifest.")
        if pca.get("shared_basis", {}).get("feature_order") is None:
            raise SnapshotRefreshError("PCA artifact lacks its frozen feature schema.")
        gates["pca"] = {"status": "PASS"}
    except (OSError, ValueError, SnapshotRefreshError, json.JSONDecodeError) as exc:
        gates["pca"] = {"status": "FAIL", "reason": str(exc)}

    cohort_hash: str | None = None
    for key, method in (("ae", "autoencoder"), ("vae", "variational_autoencoder")):
        try:
            artifact = _read(paths[key])
            validate_final_representation_artifact(
                artifact,
                artifact_path=paths[key],
                expected_method=method,
                expected_cohort_hash=cohort_hash,
                expected_source_dump=source_release,
            )
            cohort_hash = artifact["cohort_hash"]
            gates[key] = {"status": "PASS"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            gates[key] = {"status": "FAIL", "reason": str(exc)}

    for key in ("comparison", "stage4"):
        try:
            artifact = _read(paths[key])
            if artifact.get("source_dump") != source_release:
                raise SnapshotRefreshError(f"{key} artifact source release does not match candidate manifest.")
            if cohort_hash and artifact.get("cohort_hash") not in (None, cohort_hash):
                raise SnapshotRefreshError(f"{key} artifact cohort hash does not match AE/VAE.")
            for split in artifact.get("split_definitions", []):
                if split.get("group_overlap") != 0:
                    raise SnapshotRefreshError(f"{key} artifact has grouped configuration leakage.")
            gates[key] = {"status": "PASS"}
        except (OSError, ValueError, SnapshotRefreshError, json.JSONDecodeError) as exc:
            gates[key] = {"status": "FAIL", "reason": str(exc)}

    try:
        beta = _read(paths["vae_beta"])
        if beta.get("source_dump") != source_release or beta.get("selection_status") != "selected":
            raise SnapshotRefreshError("VAE beta diagnostic does not record the inherited selected protocol.")
        gates["vae_beta"] = {"status": "PASS"}
    except (OSError, ValueError, SnapshotRefreshError, json.JSONDecodeError) as exc:
        gates["vae_beta"] = {"status": "FAIL", "reason": str(exc)}

    for key in ("tabfm", "uncertainty"):
        try:
            read_aggregate_artifact(paths[key])
            gates[key] = {"status": "PASS"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            gates[key] = {"status": "FAIL", "reason": str(exc)}
    try:
        summary = build_snapshot_model_summary(
            snapshot_manifest=manifest, tabfm_path=paths["tabfm"], uncertainty_path=paths["uncertainty"]
        )
        written = _read(paths["model_summary"])
        if written != summary:
            raise SnapshotRefreshError("Model summary does not faithfully project the candidate aggregate artifacts.")
        gates["model_summary"] = {"status": "PASS"}
    except (OSError, ValueError, SnapshotRefreshError, json.JSONDecodeError) as exc:
        gates["model_summary"] = {"status": "FAIL", "reason": str(exc)}
    active = {
        "pca": ROOT / "artifacts" / "pca-db-dump-2026-07-20.json",
        "ae": ROOT / "artifacts" / "representation-ae-final-db-dump-2026-07-20.json",
        "vae": ROOT / "artifacts" / "representation-vae-final-db-dump-2026-07-20.json",
        "comparison": ROOT / "artifacts" / "representation-comparison-final-db-dump-2026-07-20.json",
        "stage4": ROOT / "artifacts" / "representation-validation-stage4-db-dump-2026-07-20.json",
        "tabfm": ROOT / "artifacts" / "model-diagnostics-4096.json",
        "uncertainty": ROOT / "artifacts" / "throughput-uncertainty-4096-seed-42.json",
    }
    comparison = compare_research_artifacts(active, paths)
    try:
        candidate_pca, active_pca = _read(paths["pca"]), _read(active["pca"])
        comparison["cohort_counts"] = {
            "candidate": candidate_pca.get("counts", {}),
            "active": active_pca.get("counts", {}),
        }
    except (OSError, json.JSONDecodeError):
        comparison["cohort_counts"] = {"status": "UNAVAILABLE"}
    return {
        "gates": gates,
        "paths": {key: str(path) for key, path in paths.items()},
        "candidate_vs_active": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate_candidate_research(_read(args.snapshot_manifest), args.artifact_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if all(gate["status"] == "PASS" for gate in result["gates"].values()) else 2
    except (OSError, ValueError, SnapshotRefreshError, json.JSONDecodeError) as exc:
        print(f"Candidate research evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
