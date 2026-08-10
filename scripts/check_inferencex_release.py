"""Cheap official-release discovery; it never downloads a dump."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.dataset_checkpoint import load_data_manifest
from modeling.snapshot_refresh import discover_new_release, fetch_official_releases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "data-manifest.json")
    parser.add_argument("--allow-prerelease", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load_data_manifest(args.manifest)
    releases = fetch_official_releases(allow_prerelease=args.allow_prerelease)
    result = discover_new_release(releases, manifest["source"]["release"], allow_prerelease=args.allow_prerelease)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if result["status"] == "INVALID_RELEASE_METADATA" else 0


if __name__ == "__main__":
    raise SystemExit(main())
