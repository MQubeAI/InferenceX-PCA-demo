"""Non-promoting July source replay fixture for the candidate ingestion pipeline.

``--plan`` is a cheap release-metadata assertion.  ``--execute`` performs the
same official-download, checksum, streaming-restore, deterministic-export, and
candidate-bundle path as a future release, then proves the resulting PCA cohort
identity equals the active July checkpoint.  It never rewrites the historical
July release, manifest, or research artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_snapshot_pca_artifact import load_candidate_aggregate  # noqa: E402
from modeling.representation_analysis import canonical_representation_data  # noqa: E402
from modeling.snapshot_refresh import discover_new_release, fetch_official_releases, load_active_manifest  # noqa: E402


JULY_RELEASE = "db-dump/2026-07-20"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--upstream-parts-dir", type=Path, help="re-verify an already-downloaded official July source")
    parser.add_argument("--active-data-dir", type=Path, default=REPOSITORY_ROOT / ".data" / "inferencex-db-dump-2026-07-20")
    args = parser.parse_args()
    if args.plan == args.execute:
        parser.error("choose exactly one of --plan or --execute")
    active = load_active_manifest(REPOSITORY_ROOT)
    releases = fetch_official_releases()
    available = {release.tag for release in releases}
    if JULY_RELEASE not in available:
        raise SystemExit("Official July replay source is no longer discoverable from the release API.")
    if args.plan:
        print(json.dumps({"status": "JULY_REPLAY_PLAN_READY", "source_release": JULY_RELEASE, "active_release": active["source"]["release"], "active_matches_replay": active["source"]["release"] == JULY_RELEASE, "no_historical_artifact_retrain": True}, indent=2))
        return 0
    if not args.output_root:
        parser.error("--execute requires --output-root")
    command = [
        sys.executable,
        "scripts/build_snapshot_candidate.py",
        "--source-release", JULY_RELEASE,
        "--output-root", str(args.output_root),
        "--previous-manifest", str(REPOSITORY_ROOT / "data-manifest.json"),
        "--active-data-dir", str(args.active_data_dir),
    ]
    if args.database_url:
        command.extend(["--database-url", args.database_url])
    if args.upstream_parts_dir:
        command.extend(["--upstream-parts-dir", str(args.upstream_parts_dir)])
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    _raw_active, aggregate_active, _active_metadata = load_candidate_aggregate(args.active_data_dir)
    _raw_candidate, aggregate_candidate, _candidate_metadata = load_candidate_aggregate(args.output_root / "checkpoint")
    active_identity = canonical_representation_data(aggregate_active, enforce_snapshot_counts=False).semantic_identity
    candidate_identity = canonical_representation_data(aggregate_candidate, enforce_snapshot_counts=False).semantic_identity
    if active_identity != candidate_identity:
        raise SystemExit("July replay candidate is not semantically identical to the active July cohort.")
    print(json.dumps({"status": "JULY_REPLAY_SEMANTICALLY_EQUIVALENT", "semantic_identity": active_identity, "historical_artifacts_retrained": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
