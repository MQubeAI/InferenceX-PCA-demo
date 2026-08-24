#!/usr/bin/env python3
"""Build the audited Epoch-hardware enrichment and prototype artifacts.

The command verifies raw Epoch SHA-256 identities before and after processing.
It writes only under ``data/derived/integration`` unless an explicit derived
output directory is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from apps import inferencex_pca_demo as app
from modeling.integration import INTEGRATION_ROOT, run_integration_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=str(INTEGRATION_ROOT))
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Build audits, mappings, capabilities, and view but skip RF ablation/holdout studies.",
    )
    args = parser.parse_args()
    result = run_integration_pipeline(
        args.data_dir,
        Path(args.output_root),
        run_experiments=not args.skip_experiments,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
