"""Run leakage-safe TabFM throughput uncertainty diagnostics (aggregate-only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import inferencex_pca_demo as app
from modeling.throughput_uncertainty import (
    DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
    evaluate_throughput_uncertainty,
)


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--subgroup-minimum-support",
        type=int,
        default=DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
        help=(
            "Minimum rows required for a subgroup to be eligible for the headline "
            "worst-undercoverage ranking. All subgroup rows remain in the artifact."
        ),
    )
    parser.add_argument("--output", default="artifacts/throughput-uncertainty-4096-seed-42.json")
    args = parser.parse_args()

    source = app.data_source_status(args.data_dir)[1]
    manifest = app.build_dataset_manifest(source)
    _, _, joined, _ = app.load_joined_data(args.data_dir, manifest["fingerprint"])
    frame, _ = app.build_analysis_frame(joined, "Median aggregate per config/workload/concurrency")
    target = "metrics_tput_per_gpu"
    numeric, categorical = app.default_target_features(frame, target)
    print(
        f"experiment=throughput_uncertainty target={target} seed={args.seed} "
        f"folds={args.folds} expected_tabfm_calls={args.folds} completed-work=0.00",
        flush=True,
    )
    result = evaluate_throughput_uncertainty(
        frame,
        numeric + categorical,
        target,
        args.max_rows,
        args.seed,
        args.folds,
        args.subgroup_minimum_support,
    )
    artifact = {
        "aggregate_only": True,
        "generation_timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": manifest,
        "controls": {
            "target": target,
            "max_rows": args.max_rows,
            "seed": args.seed,
            "folds": args.folds,
            "subgroup_minimum_support": args.subgroup_minimum_support,
            "point_model": "tabfm",
            "uncertainty_models": ["global_split_conformal", "catboost_conditional_scale", "catboost_conditional_residual_quantiles"],
        },
        "result": result,
    }
    output = Path(args.output)
    write_atomic(output, artifact)
    print(
        f"experiment=throughput_uncertainty checkpoint={output} "
        f"tabfm_calls={result['tabfm']['invocation_count']} completed-work=1.00",
        flush=True,
    )


if __name__ == "__main__":
    main()
