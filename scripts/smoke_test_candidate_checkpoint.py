"""Exercise a candidate bundle in an isolated disposable checkout.

The generated clone receives temporary promoted pointers only inside its own
worktree.  The real repository's active manifest and artifact index remain
unchanged, so this supplies a pre-promotion clean-clone gate without any
implicit promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.snapshot_refresh import active_manifest_from_versioned  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-artifact-entry", type=Path, required=True)
    parser.add_argument("--data-bundle", type=Path, required=True)
    parser.add_argument("--artifact-bundle", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    artifact_entry = json.loads(args.candidate_artifact_entry.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate" or artifact_entry.get("status") != "candidate":
        raise SystemExit("Candidate smoke test requires candidate-only inputs.")
    if not args.data_bundle.is_file() or not args.artifact_bundle.is_file():
        raise SystemExit("Candidate smoke test requires finalized local bundles.")
    with tempfile.TemporaryDirectory(prefix="inferencex-candidate-smoke-") as temporary:
        clone = Path(temporary) / "clone"
        subprocess.run(["git", "clone", "--no-local", str(ROOT), str(clone)], check=True, capture_output=True, text=True)
        manifest["status"] = "promoted"
        manifest["bundle"]["download_url"] = args.data_bundle.resolve().as_uri()
        artifact_entry["status"] = "promoted"
        artifact_entry["bundle"]["download_url"] = args.artifact_bundle.resolve().as_uri()
        date = manifest["snapshot_date"]
        (clone / "data" / "manifests").mkdir(parents=True, exist_ok=True)
        (clone / "data" / "manifests" / f"{date}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (clone / "data" / "artifact-index.json").parent.mkdir(parents=True, exist_ok=True)
        index_path = clone / "data" / "artifact-index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["active_dataset_id"] = manifest["dataset_id"]
        index["snapshots"][manifest["dataset_id"]] = artifact_entry
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        (clone / "data-manifest.json").write_text(
            json.dumps(active_manifest_from_versioned(manifest), indent=2) + "\n", encoding="utf-8"
        )
        environment = dict(os.environ)
        environment.pop("INFERENCEX_DATA_DIR", None)
        subprocess.run([sys.executable, "scripts/clean_clone_smoke_test.py"], cwd=clone, env=environment, check=True)
    print("Candidate clean-clone smoke test passed without promoting the real active pointer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
