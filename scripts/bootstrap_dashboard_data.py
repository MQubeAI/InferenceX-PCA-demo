#!/usr/bin/env python3
"""Install or validate the repository's pinned dashboard CSV checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.dataset_checkpoint import (  # noqa: E402
    DatasetBootstrapError,
    DatasetManifestError,
    bootstrap_dataset,
    default_data_dir,
    load_data_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install or verify the frozen InferenceX dashboard dataset."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data-manifest.json",
        help="committed frozen-checkpoint manifest",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="advanced override for the checkpoint destination",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify only; never attempt to download",
    )
    args = parser.parse_args()
    try:
        manifest = load_data_manifest(args.manifest)
        installed = bootstrap_dataset(
            manifest_path=args.manifest,
            data_dir=args.data_dir,
            offline=args.offline,
        )
    except (DatasetManifestError, DatasetBootstrapError) as exc:
        print(f"Dataset bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(f"Verified dataset checkpoint: {manifest['dataset_id']}")
    print(f"Location: {installed or default_data_dir(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
