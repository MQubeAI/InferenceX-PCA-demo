"""Run aggregate-only throughput residual or leakage-safe TPOT tail diagnostics."""
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
from modeling.tail_diagnostics import tabfm_oof_diagnostics, threshold_comparison, two_stage_latency


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("throughput", "median-tpot"), required=True)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR); parser.add_argument("--max-rows", type=int, default=4096); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--folds", type=int, default=3); parser.add_argument("--output", required=True)
    args = parser.parse_args(); output = Path(args.output)
    source = app.data_source_status(args.data_dir)[1]; manifest = app.build_dataset_manifest(source); _, _, joined, _ = app.load_joined_data(args.data_dir, manifest["fingerprint"]); frame, _ = app.build_analysis_frame(joined, "Median aggregate per config/workload/concurrency")
    target = "metrics_tput_per_gpu" if args.kind == "throughput" else "metrics_median_tpot"
    numeric, categorical = app.default_target_features(frame, target); features = numeric + categorical
    print(f"experiment={args.kind} target={target} seed={args.seed} context=full completed-work=0.00", flush=True)
    global_result = tabfm_oof_diagnostics(frame, features, target, args.max_rows, args.seed, args.folds, context_cap=None)
    if args.kind == "throughput":
        artifact = {"aggregate_only": True, "generation_timestamp_utc": datetime.now(UTC).isoformat(), "dataset": manifest, "controls": {"target": target, "max_rows": args.max_rows, "seed": args.seed, "folds": args.folds, "model": "tabfm", "context": "full_fold_local"}, "result": global_result}
    else:
        thresholds = threshold_comparison(frame, features, target, args.max_rows, args.seed, global_result, args.folds)
        # p95 is selected by the documented prevalence/support rule, before any
        # final model score is considered.
        segmented = two_stage_latency(frame, features, target, args.max_rows, args.seed, "p95", args.folds, context_cap=None)
        artifact = {"aggregate_only": True, "generation_timestamp_utc": datetime.now(UTC).isoformat(), "dataset": manifest, "controls": {"target": target, "max_rows": args.max_rows, "seed": args.seed, "folds": args.folds, "model": "tabfm_plus_tail_models", "context": "full_fold_local"}, "global_tabfm": global_result, "tail_thresholds": thresholds, "two_stage": segmented}
    write_atomic(output, artifact)
    print(f"experiment={args.kind} checkpoint={output} completed-work=1.00", flush=True)


if __name__ == "__main__":
    main()
