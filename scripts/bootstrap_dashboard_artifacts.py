"""Install or validate the promoted research artifacts for the active checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.artifact_checkpoint import ArtifactIndexError, bootstrap_active_artifacts
from modeling.dataset_checkpoint import DatasetBootstrapError, load_data_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="verify only; never download")
    args = parser.parse_args()
    try:
        paths = bootstrap_active_artifacts(load_data_manifest(), offline=args.offline)
    except (ArtifactIndexError, DatasetBootstrapError) as exc:
        print(f"Artifact bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print("Verified active research artifacts: " + ", ".join(sorted(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
