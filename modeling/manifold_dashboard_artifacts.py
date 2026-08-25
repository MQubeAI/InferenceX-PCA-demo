"""Dashboard-safe validation and access for the promoted Stage 5 artifacts.

This module intentionally reads only completed JSON/Parquet artifacts.  It does
not import UMAP, t-SNE, or the Stage 5 execution layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


STAGE5_STRUCTURAL_SHA256 = "33441444687c6fdf30dcca5885c31a30b155e3cb634f73d3e7d6c0efc5757fce"
STAGE5_PROJECTION_SHA256 = "9f877cee2e5492a7ab4f208b8554a5063fc874baec1dba75fa150ab2202a7f39"
STAGE5_SOURCE_DUMP = "db-dump/2026-07-20"
STAGE5_COHORT_ROWS = 8_063
STAGE5_CONFIGURATIONS = 1_354
STAGE5_FEATURE_ORDER = (
    "isl",
    "osl",
    "conc",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_dp_attention",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_dp_attention",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_hardware",
    "config_framework",
    "config_model",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
)
STAGE5_STRUCTURAL_RUN_COUNTS = {
    "umap_2": 27,
    "tsne_2": 9,
    "workload_only_umap_2": 3,
    "configuration_only_umap_2": 3,
    "mixed_distance_umap_2": 3,
}
STAGE5_QUANTITATIVE_RUN_COUNTS = {"umap_15": 27, "grouped_held_out": 27}
STAGE5_PROJECTION_METHOD_COUNTS = {
    "umap-2": 27,
    "tsne-2": 9,
    "workload_only": 3,
    "configuration_only": 3,
    "mixed-umap-2": 3,
}
STAGE5_PROJECTION_ROWS = STAGE5_COHORT_ROWS * sum(STAGE5_PROJECTION_METHOD_COUNTS.values())
STAGE5_PROJECTION_COLUMNS = {
    "row_id",
    "method",
    "parameter_id",
    "seed",
    "n_neighbors",
    "min_dist",
    "perplexity",
    "x",
    "y",
}
_UMAP_NEIGHBORS = (15, 50, 100)
_UMAP_MIN_DIST = (0.0, 0.1, 0.5)
_STAGE5_SEEDS = (42, 123, 2026)
_TSNE_PERPLEXITIES = (15, 30, 50)
_HELD_OUT_PARTITION_SEEDS = (17, 29, 43)
_HELD_OUT_FOLDS = (0, 1, 2)


class Stage5ArtifactError(ValueError):
    """The promoted Stage 5 dashboard artifact set is incompatible or unsafe."""


@dataclass(frozen=True)
class Stage5DashboardArtifacts:
    """Validated completed Stage 5 artifacts for dashboard rendering only."""

    final: dict[str, Any]
    structural: dict[str, Any]
    projections: pd.DataFrame
    structural_sha256: str
    projection_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage5ArtifactError(f"Stage 5 {label} artifact is unreadable.") from exc
    if not isinstance(payload, dict):
        raise Stage5ArtifactError(f"Stage 5 {label} artifact must be a JSON object.")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage5ArtifactError(message)


def _validate_identity(artifact: dict[str, Any], *, label: str) -> None:
    _require(
        artifact.get("schema_version") == "manifold-analysis-stage5-results-v1",
        f"Stage 5 {label} schema is incompatible.",
    )
    _require(
        artifact.get("source_dump") == STAGE5_SOURCE_DUMP,
        f"Stage 5 {label} snapshot is incompatible.",
    )
    _require(
        artifact.get("cohort_rows") == STAGE5_COHORT_ROWS,
        f"Stage 5 {label} cohort row count is incompatible.",
    )
    _require(
        artifact.get("configurations") == STAGE5_CONFIGURATIONS,
        f"Stage 5 {label} configuration count is incompatible.",
    )
    _require(
        artifact.get("feature_order") == list(STAGE5_FEATURE_ORDER),
        f"Stage 5 {label} structural feature identity is incompatible.",
    )


def _validate_projections(projections: pd.DataFrame) -> None:
    _require(
        set(projections.columns) == STAGE5_PROJECTION_COLUMNS,
        "Stage 5 projection columns are incompatible.",
    )
    _require(len(projections) == STAGE5_PROJECTION_ROWS, "Stage 5 projection row count is incomplete.")
    _require(projections["row_id"].notna().all(), "Stage 5 projections contain missing row IDs.")
    _require(
        not projections.duplicated(["row_id", "parameter_id"]).any(),
        "Stage 5 projections contain duplicate row/parameter identities.",
    )
    parameter_counts = projections.groupby("method", dropna=False)["parameter_id"].nunique().to_dict()
    _require(
        parameter_counts == STAGE5_PROJECTION_METHOD_COUNTS,
        "Stage 5 projections contain an incomplete or unexpected method set.",
    )
    _require(
        projections["parameter_id"].nunique() == 45,
        "Stage 5 projections must contain exactly 45 frozen parameter IDs.",
    )
    per_projection = projections.groupby("parameter_id")["row_id"].nunique()
    _require(
        len(per_projection) == 45 and per_projection.eq(STAGE5_COHORT_ROWS).all(),
        "Every Stage 5 projection must contain every canonical row ID exactly once.",
    )
    reference_ids: set[str] | None = None
    for _parameter_id, rows in projections.groupby("parameter_id", sort=False):
        row_ids = set(rows["row_id"].astype(str))
        if reference_ids is None:
            reference_ids = row_ids
        else:
            _require(
                row_ids == reference_ids,
                "Stage 5 projection parameter IDs do not share the same semantic row identities.",
            )
    _require(
        not projections["method"].eq("umap-15").any(),
        "Stage 5 projection companion must not expose UMAP-15 coordinates.",
    )


def _run_parameter_ids(runs: Any, *, label: str, expected_count: int) -> set[str]:
    _require(isinstance(runs, list) and len(runs) == expected_count, f"Stage 5 {label} runs are incomplete.")
    parameter_ids = {str(run.get("parameter_id")) for run in runs if isinstance(run, dict)}
    _require(
        len(parameter_ids) == expected_count and "None" not in parameter_ids,
        f"Stage 5 {label} parameter identities are incomplete.",
    )
    return parameter_ids


def _validate_umap_grid(runs: Any, *, label: str, n_components: int) -> set[str]:
    parameter_ids = _run_parameter_ids(runs, label=label, expected_count=27)
    expected = {
        (n_neighbors, min_dist, seed)
        for n_neighbors in _UMAP_NEIGHBORS
        for min_dist in _UMAP_MIN_DIST
        for seed in _STAGE5_SEEDS
    }
    actual: set[tuple[int, float, int]] = set()
    for run in runs:
        config = run.get("config")
        _require(isinstance(config, dict), f"Stage 5 {label} configuration is missing.")
        _require(config.get("n_components") == n_components, f"Stage 5 {label} dimensionality is incompatible.")
        _require(config.get("metric") == "euclidean", f"Stage 5 {label} metric is incompatible.")
        actual.add((config.get("n_neighbors"), config.get("min_dist"), config.get("seed")))
    _require(actual == expected, f"Stage 5 {label} parameter grid is incompatible.")
    return parameter_ids


def _validate_tsne_grid(runs: Any) -> set[str]:
    parameter_ids = _run_parameter_ids(runs, label="t-SNE", expected_count=9)
    expected = {(perplexity, seed) for perplexity in _TSNE_PERPLEXITIES for seed in _STAGE5_SEEDS}
    actual: set[tuple[int, int]] = set()
    for run in runs:
        config = run.get("config")
        _require(isinstance(config, dict), "Stage 5 t-SNE configuration is missing.")
        _require(config.get("n_components") == 2, "Stage 5 t-SNE dimensionality is incompatible.")
        _require(config.get("init") == "pca", "Stage 5 t-SNE initialization is incompatible.")
        _require(config.get("learning_rate") == "auto", "Stage 5 t-SNE learning rate is incompatible.")
        _require(config.get("max_iter") == 1000, "Stage 5 t-SNE iteration limit is incompatible.")
        _require(config.get("early_exaggeration") == 12.0, "Stage 5 t-SNE early exaggeration is incompatible.")
        _require(config.get("metric") == "euclidean", "Stage 5 t-SNE metric is incompatible.")
        _require(config.get("method") == "barnes_hut", "Stage 5 t-SNE method is incompatible.")
        actual.add((config.get("perplexity"), config.get("seed")))
    _require(actual == expected, "Stage 5 t-SNE parameter grid is incompatible.")
    return parameter_ids


def _validate_seed_runs(runs: Any, *, label: str, parameter_prefix: str) -> set[str]:
    parameter_ids = _run_parameter_ids(runs, label=label, expected_count=3)
    seeds = {run.get("seed") for run in runs}
    _require(seeds == set(_STAGE5_SEEDS), f"Stage 5 {label} seed set is incompatible.")
    expected = {f"{parameter_prefix}|min_dist=0.1|n_neighbors=15|seed={seed}" for seed in _STAGE5_SEEDS}
    _require(parameter_ids == expected, f"Stage 5 {label} parameter identities are incompatible.")
    return parameter_ids


def _validate_grouped_held_out(runs: Any) -> None:
    _require(isinstance(runs, list) and len(runs) == 27, "Stage 5 grouped held-out runs are incomplete.")
    expected = {
        (partition_seed, fold, seed)
        for partition_seed in _HELD_OUT_PARTITION_SEEDS
        for fold in _HELD_OUT_FOLDS
        for seed in _STAGE5_SEEDS
    }
    actual = {
        (run.get("partition_seed"), run.get("fold"), run.get("seed"))
        for run in runs
        if isinstance(run, dict)
    }
    _require(actual == expected, "Stage 5 grouped held-out identities are incompatible.")
    _require(
        all(run.get("group_overlap") == 0 for run in runs),
        "Stage 5 grouped held-out validation has configuration overlap.",
    )


def _validate_frozen_run_identity(structural: dict[str, Any], final: dict[str, Any], projections: pd.DataFrame) -> None:
    """Verify the exact frozen grid without importing research execution code."""

    umap_ids = _validate_umap_grid(structural.get("umap_2", {}).get("runs"), label="UMAP-2", n_components=2)
    tsne_ids = _validate_tsne_grid(structural.get("tsne_2", {}).get("runs"))
    ablations = structural.get("ablations", {})
    workload_ids = _validate_seed_runs(
        ablations.get("workload_only", {}).get("runs"),
        label="workload-only UMAP-2",
        parameter_prefix="workload_only",
    )
    configuration_ids = _validate_seed_runs(
        ablations.get("configuration_only", {}).get("runs"),
        label="configuration-only UMAP-2",
        parameter_prefix="configuration_only",
    )
    mixed_ids = _validate_seed_runs(
        structural.get("mixed_distance", {}).get("runs"),
        label="mixed-distance UMAP-2",
        parameter_prefix="mixed_umap2",
    )
    umap_15 = final.get("umap_15", {})
    _validate_umap_grid(umap_15.get("runs"), label="UMAP-15", n_components=15)
    _validate_grouped_held_out(umap_15.get("grouped_held_out"))

    expected_by_method = {
        "umap-2": umap_ids,
        "tsne-2": tsne_ids,
        "workload_only": workload_ids,
        "configuration_only": configuration_ids,
        "mixed-umap-2": mixed_ids,
    }
    for method, expected_ids in expected_by_method.items():
        observed_ids = set(projections.loc[projections["method"].eq(method), "parameter_id"].astype(str))
        _require(
            observed_ids == expected_ids,
            f"Stage 5 projection parameter IDs do not match frozen {method} JSON runs.",
        )


def load_stage5_dashboard_artifacts(
    *,
    final_path: str | Path,
    structural_path: str | Path,
    projection_path: str | Path,
    canonical_row_ids: Sequence[str] | None = None,
    active_source_dump: str | None = None,
) -> Stage5DashboardArtifacts:
    """Load promoted Stage 5 results and fail closed on artifact-contract drift."""

    final_file = Path(final_path)
    structural_file = Path(structural_path)
    projection_file = Path(projection_path)
    structural = _read_json(structural_file, "structural")
    final = _read_json(final_file, "final")
    _validate_identity(structural, label="structural")
    _validate_identity(final, label="final")
    if active_source_dump is not None:
        _require(
            active_source_dump == STAGE5_SOURCE_DUMP,
            "Active dashboard snapshot is not the Stage 5 July source release.",
        )
    _require(structural.get("phase") == "structural_freeze", "Stage 5 structural phase is incompatible.")
    _require(structural.get("target_metrics_in_inputs") == [], "Stage 5 structural artifact reports outcome leakage.")
    _require("outcome_overlays" not in structural, "Stage 5 structural artifact must not contain outcome overlays.")
    structural_sha256 = _sha256_file(structural_file)
    _require(
        structural_sha256 == STAGE5_STRUCTURAL_SHA256,
        "Stage 5 structural artifact SHA-256 is incompatible.",
    )
    _require(
        structural.get("run_counts") == STAGE5_STRUCTURAL_RUN_COUNTS,
        "Stage 5 structural frozen run counts are incomplete.",
    )
    _require(final.get("phase") == "final", "Stage 5 final phase is incompatible.")
    _require(
        final.get("target_metrics_in_inputs") == [],
        "Stage 5 final artifact reports structural outcome leakage.",
    )
    _require(
        final.get("outcome_overlays", {}).get("descriptive_only") is True,
        "Stage 5 final outcome overlays must be explicitly descriptive only.",
    )
    _require(
        final.get("structural_artifact", {}).get("sha256") == structural_sha256,
        "Stage 5 final artifact does not reference the frozen structural SHA-256.",
    )
    _require(
        final.get("umap_15", {}).get("run_counts") == STAGE5_QUANTITATIVE_RUN_COUNTS,
        "Stage 5 quantitative frozen run counts are incomplete.",
    )
    _require(
        final.get("umap_15", {}).get("promotion_status") == "not_promoted_human_review_required",
        "Stage 5 final artifact incorrectly promotes UMAP-15.",
    )
    _require(
        final.get("cross_method", {}).get("tsne_included") is False,
        "Stage 5 final artifact incorrectly includes t-SNE in consensus.",
    )
    projection_sha256 = _sha256_file(projection_file)
    _require(
        projection_sha256 == STAGE5_PROJECTION_SHA256,
        "Stage 5 projection artifact SHA-256 is incompatible.",
    )
    for artifact, label in ((structural, "structural"), (final, "final")):
        reference = artifact.get("projection_artifact", {})
        _require(
            reference.get("sha256") == projection_sha256,
            f"Stage 5 {label} artifact projection SHA-256 is incompatible.",
        )
        _require(
            reference.get("rows") == STAGE5_PROJECTION_ROWS,
            f"Stage 5 {label} artifact projection row count is incompatible.",
        )
    try:
        projections = pd.read_parquet(projection_file)
    except (ImportError, OSError, ValueError) as exc:
        raise Stage5ArtifactError("Stage 5 projection companion is unreadable.") from exc
    _validate_projections(projections)
    _validate_frozen_run_identity(structural, final, projections)
    projections = projections.copy()
    projections["row_id"] = projections["row_id"].astype(str)
    if canonical_row_ids is not None:
        expected = pd.Index([str(value) for value in canonical_row_ids])
        _require(expected.is_unique, "Active canonical cohort has duplicate row identities.")
        observed = pd.Index(projections.loc[projections["parameter_id"].eq(projections["parameter_id"].iloc[0]), "row_id"])
        _require(
            len(expected) == STAGE5_COHORT_ROWS and expected.difference(observed).empty and observed.difference(expected).empty,
            "Stage 5 projection row identities do not match the active canonical cohort.",
        )
    return Stage5DashboardArtifacts(
        final=final,
        structural=structural,
        projections=projections,
        structural_sha256=structural_sha256,
        projection_sha256=projection_sha256,
    )


def aligned_projection_frame(
    artifacts: Stage5DashboardArtifacts,
    *,
    parameter_id: str,
    canonical_row_ids: Sequence[str],
    canonical_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Return one projection joined to canonical metadata exclusively by ``row_id``."""

    row_ids = [str(value) for value in canonical_row_ids]
    _require(len(row_ids) == len(canonical_cohort), "Canonical metadata and row IDs are misaligned.")
    _require(len(row_ids) == len(set(row_ids)), "Canonical metadata row IDs are not unique.")
    coordinates = artifacts.projections.loc[
        artifacts.projections["parameter_id"].eq(parameter_id)
    ].copy()
    _require(len(coordinates) == STAGE5_COHORT_ROWS, "Requested Stage 5 projection is unavailable.")
    _require(
        set(coordinates["row_id"]) == set(row_ids),
        "Requested Stage 5 projection does not match canonical row identities.",
    )
    metadata = canonical_cohort.copy()
    metadata.insert(0, "row_id", row_ids)
    _require(not metadata["row_id"].duplicated().any(), "Canonical metadata contains duplicate row IDs.")
    aligned = coordinates.set_index("row_id").loc[row_ids].reset_index().merge(
        metadata,
        on="row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    _require(len(aligned) == STAGE5_COHORT_ROWS, "Stage 5 projection metadata join is incomplete.")
    return aligned
