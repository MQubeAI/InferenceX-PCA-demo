"""Apply an explicit reviewed active-checkpoint pointer change or rollback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.snapshot_refresh import SnapshotRefreshError, active_manifest_from_versioned


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True, help="promoted YYYY-MM-DD versioned manifest")
    parser.add_argument("--apply", action="store_true", help="write pointers; otherwise show the intended change")
    parser.add_argument("--manifests-dir", type=Path, default=REPOSITORY_ROOT / "data" / "manifests")
    parser.add_argument("--active-manifest", type=Path, default=REPOSITORY_ROOT / "data-manifest.json")
    parser.add_argument("--artifact-index", type=Path, default=REPOSITORY_ROOT / "data" / "artifact-index.json")
    args = parser.parse_args()
    try:
        source_path = args.manifests_dir / f"{args.snapshot_date}.json"
        source = _read(source_path)
        active = active_manifest_from_versioned(source)
        index = _read(args.artifact_index)
        entry = index.get("snapshots", {}).get(source["dataset_id"])
        if not isinstance(entry, dict) or entry.get("status") != "promoted":
            raise SnapshotRefreshError("Artifact index has no promoted entry for the requested snapshot.")
        if entry.get("source_release") != source["source"]["release"]:
            raise SnapshotRefreshError("Artifact index source release differs from the requested manifest.")
        if args.apply:
            args.active_manifest.write_text(json.dumps(active, indent=2) + "\n", encoding="utf-8")
            index["active_dataset_id"] = source["dataset_id"]
            args.artifact_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "APPLIED" if args.apply else "REVIEW_ONLY", "dataset_id": source["dataset_id"], "source_release": source["source"]["release"]}, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, SnapshotRefreshError) as exc:
        print(f"Promotion refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
