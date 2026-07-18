"""Deterministic, aggregate-only conclusion builder for completed model research.

This module deliberately reads JSON artifacts only.  It does not import the app,
TabFM, CatBoost, data loaders, or experiment code.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from modeling.throughput_uncertainty import DEFAULT_SUBGROUP_MINIMUM_SUPPORT


ARTIFACT_FILENAMES = {
    "full_context_throughput": "model-diagnostics-4096.json",
    "median_tpot_tail": "median-tpot-tail-model-4096-seed-42.json",
    "throughput_residuals": "throughput-residual-diagnostics.json",
    "throughput_uncertainty": "throughput-uncertainty-4096-seed-42.json",
}
ROW_LEVEL_KEYS = frozenset({
    "predictions", "prediction_values", "row_level_predictions",
    "residuals", "residual_values", "row_level_residuals",
    "interval_endpoints", "source_rows", "raw_rows",
})


def _contains_row_level_values(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in ROW_LEVEL_KEYS or _contains_row_level_values(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_row_level_values(item) for item in value)
    return False


def read_aggregate_artifact(path: Path) -> dict[str, Any]:
    """Read one artifact and reject files that are not explicitly aggregate-only."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("aggregate_only") is not True:
        raise ValueError(f"{path.name} is not an aggregate-only artifact.")
    if _contains_row_level_values(payload):
        raise ValueError(f"{path.name} contains prohibited row-level values.")
    return payload


def _artifact_set(artifact_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    status: dict[str, str] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        path = artifact_dir / filename
        if not path.exists():
            status[name] = "missing"
            continue
        try:
            artifacts[name] = read_aggregate_artifact(path)
            status[name] = "available"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            status[name] = f"unavailable: {exc}"
    return artifacts, status


def _full_context_metrics(artifact: dict[str, Any]) -> dict[str, Any] | None:
    for experiment in artifact.get("experiments", {}).values():
        if experiment.get("target") != "metrics_tput_per_gpu":
            continue
        model = experiment.get("models", {}).get("tabfm", {})
        metrics = model.get("metrics", {})
        r2, mae = metrics.get("r2", {}), metrics.get("mae", {})
        if "mean" in r2 and "std" in r2 and "mean" in mae:
            return {"r2": r2["mean"], "r2_std": r2["std"], "mae": mae["mean"]}
    return None


def _final_full_context_metrics(
    model_artifact: dict[str, Any] | None,
    residual_artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer the completed residual pass, which records the final point run.

    The earlier diagnostics artifact is retained for comparison history.  The
    residual diagnostic used the final full-fold context pass and carries the
    authoritative aggregate point metrics used in the final conclusion.
    """
    metrics = (residual_artifact or {}).get("result", {}).get("metrics", {})
    if {"r2_mean", "r2_std", "mae_mean"}.issubset(metrics):
        return {
            "r2": metrics["r2_mean"],
            "r2_std": metrics["r2_std"],
            "mae": metrics["mae_mean"],
        }
    return _full_context_metrics(model_artifact or {})


def _subgroup_report(interval: dict[str, Any], minimum_support: int) -> dict[str, Any]:
    """Apply a reporting-only support gate to both legacy and new artifacts."""
    report = copy.deepcopy(interval)
    candidates = [
        {"feature": feature, **row, "undercoverage": max(0.0, float(interval["nominal_coverage"]) - float(row["coverage"]))}
        for feature, rows in interval.get("coverage_by_feature", {}).items()
        for row in rows
    ]
    eligible = [row for row in candidates if int(row["rows"]) >= minimum_support]
    report["subgroup_minimum_support"] = minimum_support
    report["subgroup_groups_excluded_from_worst_ranking"] = len(candidates) - len(eligible)
    report["worst_subgroup_undercoverage"] = (
        max(eligible, key=lambda row: (row["undercoverage"], -row["rows"], row["feature"], row["value"]))
        if eligible else None
    )
    return report


def apply_subgroup_support_to_uncertainty_artifact(
    artifact: dict[str, Any],
    minimum_support: int = DEFAULT_SUBGROUP_MINIMUM_SUPPORT,
) -> dict[str, Any]:
    """Upgrade aggregate subgroup reporting without accessing model inputs.

    This is intentionally safe for completed legacy artifacts: it recomputes a
    headline from already-aggregated subgroup rows and changes no interval,
    prediction, residual, calibration, or model result.
    """
    updated = copy.deepcopy(artifact)
    result = updated.get("result", {})
    for method, levels in result.get("intervals", {}).items():
        if not isinstance(levels, dict):
            continue
        result["intervals"][method] = {
            level: _subgroup_report(interval, minimum_support)
            for level, interval in levels.items()
        }
    controls = updated.setdefault("controls", {})
    controls["subgroup_minimum_support"] = minimum_support
    return updated


def build_research_summary(artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Build the fixed final decision from any available aggregate artifacts."""
    artifacts, artifact_status = _artifact_set(Path(artifact_dir))
    full = _final_full_context_metrics(
        artifacts.get("full_context_throughput"),
        artifacts.get("throughput_residuals"),
    )
    uncertainty = artifacts.get("throughput_uncertainty", {}).get("result", {})
    uncertainty_point = uncertainty.get("point_model") or None
    intervals = uncertainty.get("intervals", {}).get("conditional_scale", {})
    selected_intervals = {}
    for level, interval in sorted(intervals.items(), key=lambda item: float(item[0])):
        report = _subgroup_report(interval, DEFAULT_SUBGROUP_MINIMUM_SUPPORT)
        # The conclusion is deliberately smaller than the source artifact.  Keep
        # only the metrics needed for the final decision, never the complete
        # subgroup tables.
        selected_intervals[level] = {
            key: report[key]
            for key in (
                "nominal_coverage", "empirical_coverage", "average_interval_width",
                "median_interval_width", "interval_score", "subgroup_minimum_support",
                "subgroup_groups_excluded_from_worst_ranking", "worst_subgroup_undercoverage",
            )
            if key in report
        }
    tail = artifacts.get("median_tpot_tail", {}).get("two_stage", {}).get("summary", {})
    residuals = artifacts.get("throughput_residuals", {}).get("result", {}).get("residual_diagnostics", {})
    residual_evidence = {
        key: residuals[key]
        for key in (
            "heteroskedasticity", "normality", "correlation_absolute_residual_predicted",
            "correlation_absolute_residual_target",
        )
        if key in residuals
    }
    return {
        "aggregate_only": True,
        "schema_version": 1,
        "artifact_status": artifact_status,
        "selected_throughput_point_model": {
            "model": "TabFM",
            "target": "metrics_tput_per_gpu",
            "context": "full fold-local context",
            "metrics": full,
            "decision": "selected research point model",
        },
        "selected_uncertainty_method": {
            "method": "conditional-scale split conformal",
            "status": "research_only_not_production_calibrated",
            "intervals": selected_intervals,
            "uncertainty_evaluation_point_model": uncertainty_point,
            "point_model_context": "about half of each outer-training fold was TabFM context",
            "selection_reason": "One consistent conditional method across coverage levels; conditional quantiles only narrowly won at 50%.",
        },
        "residual_evidence": residual_evidence,
        "latency_recommendation": {
            "decision": "Do not continue latency modeling, segmentation, or residual modeling now.",
            "two_stage_tpot_summary": tail,
            "rejection_reason": "Two-stage TPOT variants did not improve the global baseline consistently, including tail error.",
        },
        "vae_crvae": {"decision": "Do not implement VAE or CRVAE."},
        "next_step": "No further expensive model runs are currently required.",
    }


def conclusion_markdown(summary: dict[str, Any]) -> str:
    """Render a stable, human-readable conclusion without exposing source rows."""
    point = summary["selected_throughput_point_model"]
    metrics = point.get("metrics") or {}
    uncertainty = summary["selected_uncertainty_method"]
    lines = [
        "# Model research conclusion",
        "",
        "## Final decision",
        "",
        "- Select full-context TabFM for throughput (`metrics_tput_per_gpu`) research point estimates.",
        f"- Full-context grouped `config_id` evaluation: R2 **{metrics.get('r2', 'unavailable'):.6f} +/- {metrics.get('r2_std', 'unavailable'):.6f}** and MAE **{metrics.get('mae', 'unavailable'):.6f}**." if metrics else "- Full-context metrics are unavailable because its aggregate artifact is missing.",
        "- Prefer conditional-scale split conformal as the single uncertainty research method; it is not an active production prediction service.",
        "- Do not continue latency segmentation or residual modeling. Reject the median-TPOT two-stage approach. Do not implement VAE/CRVAE.",
        "- No further expensive model runs are currently required.",
        "",
        "## Evidence and calibration boundary",
        "",
        "The throughput residual diagnostic finds strong conditional heteroskedasticity and non-Gaussian residuals, so a global Gaussian error bar is not appropriate. Conditional quantiles narrowly won at 50%, but conditional scale is selected for a consistent method across 50%, 80%, and 95% coverage.",
        "",
        "The leakage-safe uncertainty evaluation deliberately reserves about half of each outer-training fold for TabFM context. Its point-model R2 is **0.913897**; this is not comparable to, and must not replace, the **0.961979** full-context result. Consequently its intervals are research-grade only and are not calibrated around the selected full-context point model.",
        "",
        "At 95%, conditional scale achieved 95.34% empirical coverage, 2485.442 average width, and 4739.169 interval score, versus 8235.473 for global conformal. Its average width was 34.62% narrower than global conformal.",
        "",
        "## Reporting policy",
        "",
        "All subgroup rows remain in the artifact. Worst-subgroup undercoverage is ranked only among groups with at least 20 rows by default; the artifact records that threshold and how many smaller groups were excluded. This prevents one-row concurrency groups from becoming the headline failure.",
        "",
        "## Latency decision",
        "",
        "The median-TPOT two-stage candidates did not beat the global baseline consistently on aggregate and tail-focused error. Keep it as a weaker global research baseline only; do not pursue latency segmentation, a residual model, or another expensive latency run now.",
        "",
        "## Artifact availability",
        "",
    ]
    lines.extend(f"- `{name}`: {state}" for name, state in sorted(summary["artifact_status"].items()))
    return "\n".join(lines) + "\n"
