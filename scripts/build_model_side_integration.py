"""Build frozen Artificial Analysis model-side integration artifacts; no network."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modeling.model_side_integration import build_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    result = build_artifacts(args.data_dir)
    print(json.dumps({"snapshot_verified": result["verification"]["manifest_consistent"], "view_rows": result["view_rows"], "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
