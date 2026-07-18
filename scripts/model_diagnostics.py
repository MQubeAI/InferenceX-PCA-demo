"""Bounded, resumable aggregate-only grouped model diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from apps import inferencex_pca_demo as app
from modeling.comparison import evaluate_models, prepare_model_frame
from modeling.diagnostics import known_config_folds

TARGETS = ("metrics_p99_itl", "metrics_median_itl", "metrics_mean_itl", "metrics_median_ttft", "metrics_median_tpot", "metrics_mean_e2el", "metrics_tput_per_gpu")


def artifact_seed_is_compatible(artifact: dict, seed: int) -> bool:
    """Reject --resume when it would mix experiments from different samples."""
    prior = artifact.get("controls", {}).get("seed")
    return prior is None or int(prior) == int(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace an artifact; an interruption cannot leave a half JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def experiment_signature(key: str, controls: dict[str, Any], features: list[str]) -> str:
    value = json.dumps({"key": key, "controls": controls, "features": features}, sort_keys=True, default=str)
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(folds: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in ("r2", "mae"):
        values = [float(row[name]) for row in folds if row.get(name) is not None]
        result[f"{name}_mean"] = sum(values) / len(values) if values else None
    return result


def _print(message: str) -> None:
    print(message, flush=True)


def run_resumable_evaluation(frame: Any, features: list[str], target: str, models: list[str], max_rows: int, seed: int, folds: int, context_cap: int | None, strategy: str, key: str, existing_experiment: dict[str, Any] | None, artifact: dict[str, Any], output: Path, target_transform: str = "raw") -> dict[str, Any]:
    controls = {"target": target, "seed": seed, "max_rows": max_rows, "folds": folds, "models": models, "context_cap": context_cap, "strategy": strategy, "target_transform": target_transform}
    signature = experiment_signature(key, controls, features)
    if existing_experiment and existing_experiment.get("status") == "complete" and existing_experiment.get("signature") == signature:
        _print(f"experiment={key} completed-work=1.00 reused=complete")
        return existing_experiment["result"]
    partial_models = dict((existing_experiment or {}).get("partial_models", {})) if (existing_experiment or {}).get("signature") == signature else {}
    total_work = max(1, len(models) * max(1, folds))

    def checkpoint(model: str, completed: list[dict[str, Any]]) -> None:
        partial_models[model] = completed
        fraction = sum(len(value) for value in partial_models.values()) / total_work
        latest = completed[-1]
        _print(f"experiment={key} target={target} seed={seed} context={context_cap or 'full'} fold-complete={latest['fold']} r2={latest.get('r2')} mae={latest.get('mae')} elapsed_seconds={latest['runtime_seconds']:.2f} checkpoint={output} completed-work={fraction:.2f}")
        artifact["experiments"][key] = {"status": "partial", "signature": signature, "controls": controls, "partial_models": partial_models, "aggregate_fold_metrics": {name: _metrics(rows) for name, rows in partial_models.items()}, "aggregate_only": True}
        artifact["generation_timestamp_utc"] = datetime.now(UTC).isoformat()
        atomic_write_json(output, artifact)

    for model in models:
        _print(f"experiment={key} target={target} seed={seed} context={context_cap or 'full'} model={model} fold-start=next completed-work={sum(len(v) for v in partial_models.values()) / total_work:.2f}")
    result = evaluate_models(frame, features, target, models, max_rows, seed, folds, context_cap, target_transform, strategy, completed_folds=partial_models, on_fold_complete=checkpoint)
    artifact["experiments"][key] = {"status": "complete", "signature": signature, "controls": controls, "result": result, "aggregate_only": True}
    atomic_write_json(output, artifact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR); parser.add_argument("--max-rows", type=int, default=1024); parser.add_argument("--folds", type=int, default=3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=",".join(TARGETS)); parser.add_argument("--models", default="random_forest,catboost"); parser.add_argument("--tabfm-context-strategies", default="random"); parser.add_argument("--tabfm-context-sizes", default="512"); parser.add_argument("--include-log1p", action="store_true"); parser.add_argument("--known-config", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--refresh", action="store_true"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); output = Path(args.output)
    artifact = json.loads(output.read_text()) if args.resume and output.exists() else {"experiments": {}}
    if not artifact_seed_is_compatible(artifact, args.seed):
        raise ValueError(f"Refusing to resume {output}: artifact seed differs from requested seed {args.seed}.")
    source = app.data_source_status(args.data_dir)[1]; manifest = app.build_dataset_manifest(source); _, _, joined, _ = app.load_joined_data(args.data_dir, manifest["fingerprint"]); frame, meta = app.build_analysis_frame(joined, "Median aggregate per config/workload/concurrency")
    models = [x for x in args.models.split(",") if x]; strategies = [x for x in args.tabfm_context_strategies.split(",") if x]; sizes = [int(x) for x in args.tabfm_context_sizes.split(",") if x]
    for target in [x for x in args.targets.split(",") if x]:
        if target not in frame or frame[target].notna().sum() < 20:
            artifact.setdefault("skipped", {})[target] = "unavailable or fewer than 20 usable rows"; continue
        numeric, categorical = app.default_target_features(frame, target); features = numeric + categorical
        transforms = ["raw"] + (["log1p"] if args.include_log1p and (frame[target].dropna() >= 0).all() else [])
        for transform in transforms:
            baseline_models = [model for model in models if model != "tabfm"]
            if baseline_models:
                key = f"unseen:{target}:{transform}:baselines"
                if args.refresh: artifact["experiments"].pop(key, None)
                run_resumable_evaluation(frame, features, target, baseline_models, args.max_rows, args.seed, args.folds, None, "random", key, artifact["experiments"].get(key), artifact, output, transform)
            if "tabfm" in models:
                for strategy in strategies:
                    for size in sizes:
                        key = f"unseen:{target}:{transform}:tabfm:{strategy}:{size}"
                        if args.refresh: artifact["experiments"].pop(key, None)
                        run_resumable_evaluation(frame, features, target, ["tabfm"], args.max_rows, args.seed, args.folds, size, strategy, key, artifact["experiments"].get(key), artifact, output, transform)
        if args.known_config:
            work, _ = prepare_model_frame(frame, features, target, args.max_rows, args.seed)
            known_folds = known_config_folds(work, args.folds, args.seed)
            key = f"known_config:{target}:raw:baselines"
            if args.refresh: artifact["experiments"].pop(key, None)
            if key not in artifact["experiments"]:
                result = evaluate_models(frame, features, target, [model for model in models if model != "tabfm"], args.max_rows, args.seed, args.folds, folds_override=known_folds, evaluation_label="known_config_interpolation")
                artifact["experiments"][key] = {"status": "complete", "result": result, "aggregate_only": True}
                atomic_write_json(output, artifact)
    artifact.update({"generation_timestamp_utc": datetime.now(UTC).isoformat(), "dataset": manifest, "controls": {"max_rows": args.max_rows, "folds": args.folds, "seed": args.seed, "models": models, "targets": args.targets, "strategies": strategies, "sizes": sizes}, "aggregate_only": True})
    atomic_write_json(output, artifact)
    _print(f"Wrote {output} with {len(artifact['experiments'])} aggregate experiments")


if __name__ == "__main__":
    main()
