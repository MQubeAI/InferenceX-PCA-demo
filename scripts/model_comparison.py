"""Run aggregate-only, grouped RF/CatBoost/optional-TabFM comparisons."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps import inferencex_pca_demo as app
from modeling.comparison import evaluate_models, missingness_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--target", required=True)
    parser.add_argument("--models", default="random_forest,catboost")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--tabfm-max-context", type=int, default=None)
    parser.add_argument("--tabfm-results", help="Aggregate-only TabFM artifact from the dedicated environment to merge after signature checks.")
    parser.add_argument("--analysis-unit", default="Median aggregate per config/workload/concurrency", choices=app.ANALYSIS_UNIT_OPTIONS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    permitted = {"random_forest", "catboost", "tabfm"}
    unsupported = set(models) - permitted
    if unsupported:
        raise SystemExit(f"Unsupported models: {', '.join(sorted(unsupported))}")
    source_info = app.data_source_status(args.data_dir)[1]
    if source_info["active_mode"] == "missing":
        raise SystemExit("Required CSV files and JSON fallback files are unavailable.")
    manifest = app.build_dataset_manifest(source_info)
    _, _, joined, _ = app.load_joined_data(args.data_dir, manifest["fingerprint"])
    frame, analysis_metadata = app.build_analysis_frame(joined, args.analysis_unit)
    analysis_metadata["dataset_fingerprint"] = manifest["fingerprint"]
    if args.target not in frame:
        raise SystemExit(f"Target is absent from the selected analysis frame: {args.target}")
    numeric, categorical = app.default_target_features(frame, args.target)
    features = numeric + categorical
    comparison = evaluate_models(frame, features, args.target, models, args.max_rows, args.seed, args.folds, args.tabfm_max_context)
    if args.tabfm_results:
        tabfm_artifact = json.loads(Path(args.tabfm_results).read_text(encoding="utf-8"))
        tabfm_controls = tabfm_artifact.get("controls", {})
        required_controls = {"target": args.target, "features": features, "folds": args.folds, "seed": args.seed, "max_rows": args.max_rows}
        mismatch = [key for key, value in required_controls.items() if tabfm_controls.get(key) != value]
        if tabfm_artifact.get("dataset", {}).get("fingerprint") != manifest["fingerprint"]:
            mismatch.append("dataset fingerprint")
        tabfm_model = tabfm_artifact.get("comparison", {}).get("models", {}).get("tabfm")
        if not tabfm_model:
            mismatch.append("TabFM model result")
        if mismatch:
            raise SystemExit("Cannot merge TabFM artifact; mismatch: " + ", ".join(mismatch))
        comparison["models"]["tabfm"] = tabfm_model
    report = missingness_report(frame, features, app.metric_like_numeric_columns(frame))
    controls = {"target": args.target, "features": features, "models": models + (["tabfm"] if args.tabfm_results else []), "folds": args.folds, "seed": args.seed, "max_rows": args.max_rows, "tabfm_max_context": args.tabfm_max_context}
    artifact: dict[str, Any] = {
        "generation_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": app.git_commit(),
        "dataset": manifest,
        "analysis": {"unit": args.analysis_unit, "rows": len(frame), "signature": app.analysis_signature(analysis_metadata, "model-comparison", controls)},
        "controls": controls,
        "missingness": report,
        "comparison": comparison,
        "package_versions": {package: importlib.metadata.version(package) for package in ("pandas", "numpy", "scikit-learn")},
    }
    for optional in ("catboost", "tabfm", "torch"):
        try:
            artifact["package_versions"][optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            continue
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {output}; models: {', '.join(models)}")


if __name__ == "__main__":
    main()
