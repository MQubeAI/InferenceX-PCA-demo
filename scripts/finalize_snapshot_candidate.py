"""Prepare immutable candidate-release metadata without changing active pointers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.snapshot_refresh import SnapshotRefreshError, sha256_file, snapshot_artifact_names


def _artifact_bundle(artifact_dir: Path, output: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    names = snapshot_artifact_names(output.stem.removeprefix("inferencex-research-artifacts-"))
    required = [names[key] for key in ("pca", "ae_json", "ae_embeddings", "ae_weights", "vae_json", "vae_embeddings", "vae_weights", "comparison", "stage4", "tabfm_evaluation", "uncertainty", "model_summary")]
    beta = f"representation-vae-beta-diagnostic-db-dump-{output.stem.removeprefix('inferencex-research-artifacts-')}.json"
    required.append(beta)
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise SnapshotRefreshError("Candidate research bundle is missing: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(required):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (artifact_dir / name).read_bytes())
    descriptors = {
        name: {"filename": name, "size_bytes": (artifact_dir / name).stat().st_size, "sha256": sha256_file(artifact_dir / name)}
        for name in required
    }
    return descriptors, {"filename": output.name, "size_bytes": output.stat().st_size, "sha256": sha256_file(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-data-bundle", type=Path, required=True)
    parser.add_argument("--candidate-data-url", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--artifact-url", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-artifact-entry", type=Path, required=True)
    parser.add_argument("--output-artifact-bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "candidate":
            raise SnapshotRefreshError("Only a candidate manifest may be finalized for review.")
        date = manifest["snapshot_date"]
        expected_data_name = f"inferencex-dashboard-data-{date}.zip"
        if args.candidate_data_bundle.name != expected_data_name:
            raise SnapshotRefreshError("Candidate data bundle filename does not match snapshot date.")
        if sha256_file(args.candidate_data_bundle) != manifest["bundle"]["sha256"]:
            raise SnapshotRefreshError("Candidate data bundle hash differs from its manifest.")
        descriptors, research_bundle = _artifact_bundle(args.artifact_dir, args.output_artifact_bundle)
        if research_bundle["filename"] not in args.artifact_url:
            raise SnapshotRefreshError("Candidate artifact URL does not identify the generated bundle filename.")
        manifest["bundle"]["download_url"] = args.candidate_data_url
        manifest["generation"]["candidate_release"] = {
            "data_bundle_url": args.candidate_data_url,
            "research_bundle_url": args.artifact_url,
            "research_bundle": research_bundle,
        }
        artifact_names = snapshot_artifact_names(date)
        entry = {
            "dataset_id": manifest["dataset_id"],
            "status": "candidate",
            "source_release": manifest["source"]["release"],
            "storage": "release_bundle",
            "bundle": {**research_bundle, "download_url": args.artifact_url},
            "artifacts": {
                "pca": descriptors[artifact_names["pca"]],
                "ae": descriptors[artifact_names["ae_json"]],
                "vae": descriptors[artifact_names["vae_json"]],
                "vae_beta": descriptors[beta],
                "comparison": descriptors[artifact_names["comparison"]],
                "stage4": descriptors[artifact_names["stage4"]],
                "tabfm": descriptors[artifact_names["tabfm_evaluation"]],
                "uncertainty": descriptors[artifact_names["uncertainty"]],
                "model_summary": descriptors[artifact_names["model_summary"]],
            },
        }
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        args.output_artifact_entry.parent.mkdir(parents=True, exist_ok=True)
        args.output_artifact_entry.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "CANDIDATE_FINALIZED_NOT_PROMOTED", "manifest": str(args.output_manifest), "artifact_entry": str(args.output_artifact_entry), "research_bundle": str(args.output_artifact_bundle)}, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, SnapshotRefreshError) as exc:
        print(f"Candidate finalization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
