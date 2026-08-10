"""Build a non-promoted, versioned checkpoint from an official DB dump release.

This is deliberately a candidate-only command.  It never writes
``data-manifest.json``, never touches a prior release, and never writes a full
decompressed PostgreSQL archive.  It starts a disposable local PostgreSQL 17
cluster when no explicit isolated database URL is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.snapshot_ingestion import (  # noqa: E402
    download_and_verify_upstream_parts,
    export_deterministic_tables,
    isolated_postgres17,
    require_free_disk,
    selective_stream_restore,
    verify_upstream_parts,
)
from modeling.dataset_checkpoint import load_data_manifest  # noqa: E402
from modeling.snapshot_refresh import (  # noqa: E402
    OfficialRelease,
    SnapshotRefreshError,
    audit_candidate_against_active,
    candidate_manifest,
    create_checkpoint_bundle,
    fetch_official_releases,
    refresh_report_markdown,
)


def _select_release(tag: str) -> OfficialRelease:
    releases = fetch_official_releases()
    for release in releases:
        if release.tag == tag:
            return release
    raise SnapshotRefreshError(f"Official release was not found or is not eligible: {tag}")


def _run_identifier() -> str:
    return os.environ.get("GITHUB_RUN_ID") or f"local-{int(time.time())}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", required=True, help="official db-dump/YYYY-MM-DD tag")
    parser.add_argument("--output-root", type=Path, required=True, help="scratch candidate directory outside Git")
    parser.add_argument("--previous-manifest", type=Path, default=REPOSITORY_ROOT / "data-manifest.json")
    parser.add_argument("--active-data-dir", type=Path, required=True, help="verified active CSV checkpoint")
    parser.add_argument("--database-url", help="empty isolated PostgreSQL 17 database URL")
    parser.add_argument("--minimum-free-gib", type=int, default=60)
    parser.add_argument("--bundle-url", help="immutable candidate release URL after upload")
    parser.add_argument("--upstream-parts-dir", type=Path, help="already-downloaded official parts; re-verify before restore")
    parser.add_argument("--dry-run", action="store_true", help="validate release metadata only; do not download or restore")
    parser.add_argument("--keep-upstream-parts", action="store_true", help="retain verified compressed parts under output root")
    args = parser.parse_args()

    try:
        release = _select_release(args.source_release)
        parent = load_data_manifest(args.previous_manifest)
        if args.dry_run:
            print(json.dumps({
                "status": "CANDIDATE_PLAN_READY",
                "source_release": release.tag,
                "snapshot_date": release.snapshot_date,
                "parent_active_checkpoint": parent["dataset_id"],
                "streaming_restore": True,
                "requires_postgresql_major": 17,
            }, indent=2))
            return 0
        if args.database_url and "inferencex" not in args.database_url.lower():
            raise SnapshotRefreshError("Refusing database URL without an explicit inferencex-isolated database name.")
        args.output_root.mkdir(parents=True, exist_ok=True)
        disk = require_free_disk(args.output_root, args.minimum_free_gib)
        upstream_dir = args.output_root / "upstream"
        if args.upstream_parts_dir:
            parts = verify_upstream_parts(release, args.upstream_parts_dir)
            downloaded_here = False
        else:
            parts = download_and_verify_upstream_parts(release, upstream_dir)
            downloaded_here = True
        temporary_database = (
            nullcontext({"database_url": args.database_url, "postgres_version": 17, "managed": False})
            if args.database_url
            else isolated_postgres17(
                args.output_root,
                database_name=f"inferencex_candidate_{release.snapshot_date.replace('-', '_')}",
            )
        )
        with temporary_database as database:
            database_url = database["database_url"]
            restore = selective_stream_restore(parts, database_url=database_url, workdir=args.output_root)
            checkpoint_dir = args.output_root / "checkpoint"
            exported = export_deterministic_tables(database_url, checkpoint_dir)
        bundle_path = args.output_root / f"inferencex-dashboard-data-{release.snapshot_date}.zip"
        bundle = create_checkpoint_bundle(checkpoint_dir, bundle_path)
        bundle["download_url"] = args.bundle_url
        manifest = candidate_manifest(
            checkpoint_dir=checkpoint_dir,
            source_release=release,
            parent_manifest=parent,
            bundle=bundle,
            generation={
                "workflow_run": _run_identifier(),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "disk": disk,
                "streaming_restore": restore,
                "export": exported,
                "temporary_postgresql": {key: value for key, value in database.items() if key != "database_url"},
            },
        )
        manifest_path = args.output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        audit = audit_candidate_against_active(args.active_data_dir, checkpoint_dir)
        audit_path = args.output_root / "candidate-vs-active.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        # Research completion is intentionally not fabricated here; the heavy runner
        # replaces this placeholder with fixed-protocol results before a PR is opened.
        policy = {
            "promotion_status": "FAIL",
            "hard_failures": ["mandatory research gate incomplete: pca, ae, vae, comparison, stage4, tabfm, uncertainty", "clean-clone validation not passed"],
        }
        report_path = args.output_root / f"db-dump-{release.snapshot_date}.md"
        report_path.write_text(refresh_report_markdown(manifest, audit, policy), encoding="utf-8")
        if downloaded_here and not args.keep_upstream_parts:
            shutil.rmtree(upstream_dir)
        print(json.dumps({
            "status": "CANDIDATE_INGESTED_NOT_PROMOTED",
            "manifest": str(manifest_path),
            "audit": str(audit_path),
            "report": str(report_path),
            "bundle": str(bundle_path),
        }, indent=2))
        return 0
    except SnapshotRefreshError as exc:
        print(f"Candidate ingestion failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
