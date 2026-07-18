"""Create the final model-research summary from aggregate JSON artifacts only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.research_summary import (
    apply_subgroup_support_to_uncertainty_artifact,
    build_research_summary,
    conclusion_markdown,
    read_aggregate_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--summary-output", default="artifacts/model-research-summary.json")
    parser.add_argument("--conclusion-output", default="docs/model-research-conclusion.md")
    parser.add_argument(
        "--refresh-uncertainty-artifact",
        action="store_true",
        help="Update only completed aggregate subgroup reporting with the default 20-row support gate.",
    )
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    if args.refresh_uncertainty_artifact:
        path = artifact_dir / "throughput-uncertainty-4096-seed-42.json"
        if path.exists():
            upgraded = apply_subgroup_support_to_uncertainty_artifact(read_aggregate_artifact(path))
            path.write_text(json.dumps(upgraded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = build_research_summary(artifact_dir)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.conclusion_output).write_text(conclusion_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
