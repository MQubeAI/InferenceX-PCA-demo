"""Build a snapshot-versioned, aggregate-only supervised-model dashboard summary.

The dedicated TabFM runner must first write the approved aggregate evaluation and
uncertainty artifacts.  This command never trains, imports TabFM, or reads raw
rows; it validates those two artifacts and projects their recorded aggregates
into the small JSON contract consumed by the dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.research_summary import (  # noqa: E402
    DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
    _subgroup_report,
    read_aggregate_artifact,
)


class SummaryError(ValueError):
    """Raised when a runner output cannot support the dashboard's fixed summary."""


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SummaryError(f"{label} must be a finite numeric aggregate.")
    return float(value)


def _tabfm_metrics(artifact: dict[str, Any]) -> dict[str, float]:
    """Read the approved grouped full-context metrics from known aggregate layouts."""

    candidates: list[dict[str, Any]] = []
    result_metrics = artifact.get("result", {}).get("metrics")
    if isinstance(result_metrics, dict):
        candidates.append(result_metrics)
    comparison_metrics = artifact.get("comparison", {}).get("models", {}).get("tabfm", {}).get("metrics")
    if isinstance(comparison_metrics, dict):
        candidates.append(comparison_metrics)
    for experiment in artifact.get("experiments", {}).values():
        if experiment.get("target") == "metrics_tput_per_gpu":
            metrics = experiment.get("models", {}).get("tabfm", {}).get("metrics")
            if isinstance(metrics, dict):
                candidates.append(metrics)
    for metrics in candidates:
        if {"r2_mean", "r2_std", "mae_mean"}.issubset(metrics):
            return {
                "r2": _finite(metrics["r2_mean"], "TabFM r2_mean"),
                "r2_std": _finite(metrics["r2_std"], "TabFM r2_std"),
                "mae": _finite(metrics["mae_mean"], "TabFM mae_mean"),
            }
        r2, mae = metrics.get("r2"), metrics.get("mae")
        if isinstance(r2, dict) and isinstance(mae, dict) and {"mean", "std"}.issubset(r2) and "mean" in mae:
            return {
                "r2": _finite(r2["mean"], "TabFM r2.mean"),
                "r2_std": _finite(r2["std"], "TabFM r2.std"),
                "mae": _finite(mae["mean"], "TabFM mae.mean"),
            }
    raise SummaryError(
        "TabFM artifact lacks finite grouped full-context r2/r2_std/mae aggregates. "
        "The external runner must emit the approved aggregate schema."
    )


def _uncertainty_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise SummaryError("Uncertainty artifact lacks result.")
    intervals = result.get("intervals", {}).get("conditional_scale")
    if not isinstance(intervals, dict) or not intervals:
        raise SummaryError("Uncertainty artifact lacks conditional-scale aggregate intervals.")
    selected: dict[str, dict[str, Any]] = {}
    for level, interval in sorted(intervals.items(), key=lambda item: float(item[0])):
        if not isinstance(interval, dict):
            raise SummaryError(f"Uncertainty interval {level} is invalid.")
        report = _subgroup_report(interval, DEFAULT_SUBGROUP_MINIMUM_SUPPORT)
        for metric in ("nominal_coverage", "empirical_coverage", "average_interval_width"):
            _finite(report.get(metric), f"Uncertainty {level} {metric}")
        selected[str(level)] = {
            key: report[key]
            for key in (
                "nominal_coverage",
                "empirical_coverage",
                "average_interval_width",
                "median_interval_width",
                "interval_score",
                "subgroup_minimum_support",
                "subgroup_groups_excluded_from_worst_ranking",
                "worst_subgroup_undercoverage",
            )
            if key in report
        }
    point = result.get("point_model")
    if point is not None:
        if not isinstance(point, dict):
            raise SummaryError("Uncertainty point model aggregate is invalid.")
        for metric in ("r2", "mae"):
            if metric in point:
                _finite(point[metric], f"Uncertainty point model {metric}")
    return {
        "method": "conditional-scale split conformal",
        "status": "research_only_not_production_calibrated",
        "intervals": selected,
        "uncertainty_evaluation_point_model": point,
        "point_model_context": "about half of each outer-training fold was TabFM context",
        "selection_reason": "One consistent conditional method across coverage levels; conditional quantiles only narrowly won at 50%.",
    }


def build_snapshot_model_summary(
    *, snapshot_manifest: dict[str, Any], tabfm_path: Path, uncertainty_path: Path
) -> dict[str, Any]:
    tabfm = read_aggregate_artifact(tabfm_path)
    uncertainty = read_aggregate_artifact(uncertainty_path)
    return {
        "aggregate_only": True,
        "schema_version": 1,
        "dataset_id": snapshot_manifest["dataset_id"],
        "source_release": snapshot_manifest["source"]["release"],
        "artifact_status": {
            "full_context_throughput": "available",
            "throughput_uncertainty": "available",
            "median_tpot_tail": "not rerun: frozen research decision",
            "throughput_residuals": "not rerun: selected protocol does not require a new search",
        },
        "selected_throughput_point_model": {
            "model": "TabFM",
            "target": "metrics_tput_per_gpu",
            "context": "full fold-local context",
            "metrics": _tabfm_metrics(tabfm),
            "decision": "selected research point model",
        },
        "selected_uncertainty_method": _uncertainty_summary(uncertainty),
        "residual_evidence": {},
        "latency_recommendation": {
            "decision": "Do not continue latency modeling, segmentation, or residual modeling now.",
            "two_stage_tpot_summary": {},
            "rejection_reason": "The selected refresh protocol preserves the prior no-search decision.",
        },
        "vae_crvae": {"decision": "Do not implement VAE or CRVAE."},
        "next_step": "No further expensive model runs are currently required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--tabfm", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
        summary = build_snapshot_model_summary(
            snapshot_manifest=manifest, tabfm_path=args.tabfm, uncertainty_path=args.uncertainty
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Snapshot model summary failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
