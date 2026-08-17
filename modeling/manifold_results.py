"""Execution layer for the frozen Stage 5 manifold/neighborhood protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import gc
import tempfile
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from modeling.manifold_analysis import (
    STAGE5_SCHEMA_VERSION,
    STAGE5_SEEDS,
    TSNE_2_GRID,
    UMAP_15_GRID,
    UMAP_2_GRID,
    UMAP_CANONICAL_DISPLAY,
    TSNE_CANONICAL_DISPLAY,
    UMAPConfig,
    canonical_stage5_data,
    fit_stage5_tsne,
    fit_stage5_umap,
    grouped_umap_transform,
)
from modeling.neighborhood_analysis import (
    cross_method_neighbor_agreement,
    exact_knn_indices,
    grouped_bootstrap,
    local_rank_agreement,
    neighbor_jaccard,
    neighbor_recall,
    source_mixed_knn_indices,
    materialize_source_mixed_distance_matrix,
    mixed_distance_memory_bytes,
    within_workload_knn_indices,
    workload_neighborhood_purity,
)
from modeling.pca_target_analysis import CATEGORICAL_FEATURES, PCA_FEATURES
from modeling.representation_analysis import (
    BOOLEAN_FEATURES,
    OUTCOME_TARGETS,
    GROUP_COLUMN,
    align_companion_to_active_cohort,
    grouped_partition_definitions,
    load_final_representation_artifact,
    make_preprocessor_for_features,
    software_versions,
)


RESULT_SCHEMA_VERSION = "manifold-analysis-stage5-results-v1"
STRUCTURAL_ARTIFACT = "manifold-analysis-stage5-structural-db-dump-2026-07-20.json"
FINAL_ARTIFACT = "manifold-analysis-stage5-db-dump-2026-07-20.json"
PROJECTION_ARTIFACT = "manifold-projections-stage5-db-dump-2026-07-20.parquet"
K_VALUES = (10, 30, 50)
ALL_K_VALUES = (*K_VALUES, 100)
PARTITION_SEEDS = (17, 29, 43)
CACHE_DIRECTORY = ".manifold-analysis-stage5-cache"


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def stable_artifact_hash(artifact: dict[str, Any]) -> str:
    """Hash canonical serialized content, independent of JSON pretty printing."""

    return hashlib.sha256(
        json.dumps(_json(artifact), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, artifact: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json(artifact), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_structural_freeze(path: str | Path) -> str:
    """Validate an outcome-free structural boundary and return its byte hash."""

    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("phase") != "structural_freeze":
        raise ValueError("Outcome analysis requires a structural_freeze artifact.")
    if artifact.get("target_metrics_in_inputs"):
        raise ValueError("Structural freeze reports outcome inputs.")
    if "outcome_overlays" in artifact:
        raise ValueError("Structural freeze must not contain outcome overlays.")
    return hashlib.sha256(artifact_path.read_bytes()).hexdigest()


def _parameter_id(kind: str, **params: Any) -> str:
    """Return a stable run identifier without colliding with t-SNE's ``method``.

    ``TSNEConfig`` intentionally includes the frozen Barnes-Hut ``method``
    parameter.  Naming this positional label ``kind`` leaves that scientific
    parameter intact in the identifier.
    """

    return kind + "|" + "|".join(f"{key}={params[key]}" for key in sorted(params))


def _nested_graphs(
    values: np.ndarray, *, maximum_k: int = max(ALL_K_VALUES)
) -> dict[int, np.ndarray]:
    """Compute a maximum-k exact graph once, then retain its nested prefixes."""

    largest = exact_knn_indices(values, maximum_k)
    return {k: largest[:, :k] for k in ALL_K_VALUES}


class _ReferenceTrustworthiness:
    """Exact scikit-learn-equivalent trustworthiness with reusable source ranks.

    Scikit-learn recalculates a complete original-space rank matrix for every
    requested k. Stage 5 uses the same formula, but builds the rank matrix in
    bounded blocks in a temporary memory-mapped file and reuses it for the
    frozen k values. It is never written to a research artifact.
    """

    def __init__(
        self,
        reference: np.ndarray | None = None,
        *,
        distances: np.ndarray | None = None,
        temporary_directory: Path | None = None,
        chunk_size: int = 128,
    ) -> None:
        if (reference is None) == (distances is None):
            raise ValueError("Provide exactly one reference matrix or distance matrix.")
        if chunk_size < 1:
            raise ValueError("trustworthiness chunk_size must be positive.")
        if distances is None:
            source = np.asarray(reference, dtype=float)
        else:
            source = np.asarray(distances)
            if source.ndim != 2 or source.shape[0] != source.shape[1]:
                raise ValueError("Precomputed trustworthiness distances must be square.")
        self.rows = len(source)
        if self.rows < 3:
            raise ValueError("Trustworthiness needs at least three rows.")
        rank_dtype = np.uint16 if self.rows <= np.iinfo(np.uint16).max else np.uint32
        if temporary_directory is not None:
            temporary_directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix="trustworthiness-ranks-",
            suffix=".dat",
            dir=str(temporary_directory) if temporary_directory else None,
        )
        os.close(descriptor)
        self._rank_path = Path(raw_path)
        self.ranks = np.memmap(
            self._rank_path,
            dtype=rank_dtype,
            mode="w+",
            shape=(self.rows, self.rows),
        )
        rank_values = np.arange(1, self.rows + 1, dtype=rank_dtype)
        for start in range(0, self.rows, chunk_size):
            stop = min(self.rows, start + chunk_size)
            if distances is None:
                block = pairwise_distances(source[start:stop], source, metric="euclidean")
            else:
                # Copy only one bounded block so the precomputed matrix remains
                # valid for the subsequent UMAP fit.
                block = np.array(source[start:stop], copy=True)
            block[np.arange(stop - start), np.arange(start, stop)] = np.inf
            ordered = np.argsort(block, axis=1)
            ranks = np.empty((stop - start, self.rows), dtype=rank_dtype)
            ranks[np.arange(stop - start)[:, None], ordered] = rank_values
            self.ranks[start:stop] = ranks
        self.ranks.flush()

    def close(self) -> None:
        """Release the temporary rank map after the corresponding phase."""

        if getattr(self, "ranks", None) is not None:
            self.ranks.flush()
            mmap = getattr(self.ranks, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self.ranks = None
        if getattr(self, "_rank_path", None) is not None:
            self._rank_path.unlink(missing_ok=True)

    def scores(self, candidate_graphs: dict[int, np.ndarray]) -> dict[int, float]:
        if self.ranks is None:
            raise RuntimeError("Trustworthiness rank map has already been closed.")
        values = {}
        rows = np.arange(self.rows)[:, None]
        for k in ALL_K_VALUES:
            if k >= self.rows / 2:
                raise ValueError("trustworthiness requires k to be smaller than half the row count.")
            candidates = candidate_graphs[k]
            ranks = self.ranks[rows, candidates].astype(np.int64, copy=False) - k
            penalty = int(ranks[ranks > 0].sum())
            values[k] = float(
                1.0
                - penalty
                * (2.0 / (self.rows * k * (2.0 * self.rows - 3.0 * k - 1.0)))
            )
        return values


def _metrics(
    candidate: np.ndarray,
    graphs: dict[int, np.ndarray],
    *,
    groups: pd.Series,
    trustworthiness_reference: _ReferenceTrustworthiness,
) -> dict[str, Any]:
    candidate_graphs = _nested_graphs(candidate)
    rows = []
    trust = trustworthiness_reference.scores(candidate_graphs)
    for k in ALL_K_VALUES:
        recall = neighbor_recall(graphs[k], candidate_graphs[k])
        jaccard = neighbor_jaccard(graphs[k], candidate_graphs[k])
        rank = local_rank_agreement(graphs[k], candidate_graphs[k])
        rows.append(
            {
                "k": k,
                "trustworthiness": trust[k],
                "neighbor_recall": recall["mean"],
                "neighbor_recall_grouped_bootstrap": _boot(recall["per_row"], groups),
                "neighbor_jaccard": jaccard["mean"],
                "neighbor_jaccard_grouped_bootstrap": _boot(jaccard["per_row"], groups),
                "local_rank_agreement": rank["mean"],
                "rank_eligible_rows": rank["eligible_rows"],
                "rank_insufficient_shared_rows": rank["insufficient_shared_rows"],
                "local_rank_grouped_bootstrap": (
                    _boot(rank["per_row"], groups) if rank["eligible_rows"] else None
                ),
            }
        )
    return {"by_k": rows, "candidate_graphs": candidate_graphs}


def _compact_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return metrics["by_k"]


def _projection_rows(
    row_ids: list[str], coordinates: np.ndarray, *, method: str, parameter_id: str,
    seed: int, n_neighbors: int | None = None, min_dist: float | None = None,
    perplexity: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "method": method,
            "parameter_id": parameter_id,
            "seed": seed,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "perplexity": perplexity,
            "x": coordinates[:, 0].astype(np.float32),
            "y": coordinates[:, 1].astype(np.float32),
        }
    )


def _boot(values: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    result = grouped_bootstrap(
        values,
        groups.astype(str).to_numpy(),
        return_samples=False,
    )
    return {
        "mean": float(np.nanmean(values)),
        "percentile_interval": result["percentile_interval"],
        "replicates": result["replicates"],
        "seed": result["seed"],
    }


def _cache_identity(
    data: Any,
    *,
    method: str,
    parameters: dict[str, Any],
    encoded_feature_names: list[str] | tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete scientific identity of one resumable fit."""

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_schema_version": STAGE5_SCHEMA_VERSION,
        "source_dump": "db-dump/2026-07-20",
        "cohort_hash": data.cohort_hash,
        "row_key_hash": data.row_key_hash,
        "method": method,
        "parameters": _json(parameters),
        "encoded_feature_names": list(encoded_feature_names),
        "extra": _json(extra or {}),
    }


def _cached_coordinates(
    cache_dir: Path,
    identity: dict[str, Any],
    fitter: Callable[[], tuple[np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any], str, str]:
    """Load a validated coordinate cache or atomically write one after fitting."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as cached:
                cached_identity = str(cached["identity"].item())
                coordinates = np.asarray(cached["coordinates"], dtype=float)
                metadata = json.loads(str(cached["metadata"].item()))
        except Exception as exc:  # pragma: no cover - filesystem corruption path
            raise RuntimeError(f"Stage 5 cache entry is unreadable: {path.name}") from exc
        if cached_identity != canonical_identity:
            raise RuntimeError(f"Stage 5 cache identity mismatch: {path.name}")
        return coordinates, metadata, "validated_cache", key
    coordinates, metadata = fitter()
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or not np.isfinite(coordinates).all():
        raise RuntimeError("Stage 5 fit returned invalid cached coordinates.")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identity=np.asarray(canonical_identity),
            coordinates=coordinates,
            metadata=np.asarray(json.dumps(_json(metadata), sort_keys=True)),
        )
    temporary.replace(path)
    return coordinates.astype(float), metadata, "fresh_fit", key


def _graph_seed_stability(graphs_by_run: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    """Neighbor-based seed stability without coordinate-axis comparison."""

    pairs = []
    for left, right in combinations(graphs_by_run, 2):
        for k in K_VALUES:
            pairs.append(
                {
                    "left_run": left,
                    "right_run": right,
                    "k": k,
                    "neighbor_recall": neighbor_recall(
                        graphs_by_run[left][k], graphs_by_run[right][k]
                    )["mean"],
                    "neighbor_jaccard": neighbor_jaccard(
                        graphs_by_run[left][k], graphs_by_run[right][k]
                    )["mean"],
                }
            )
    return {
        "pairs": pairs,
        "mean_jaccard": float(np.mean([item["neighbor_jaccard"] for item in pairs]))
        if pairs
        else None,
    }


def _mixed_disagreement(
    encoded_graphs: dict[int, np.ndarray],
    mixed_graphs: dict[int, np.ndarray],
    data: Any,
) -> list[dict[str, Any]]:
    """Describe, without outcomes, where source-balanced neighbors differ."""

    workload = data.cohort[["isl", "osl", "conc"]].astype(str).agg("|".join, axis=1)
    configurations = data.cohort[GROUP_COLUMN].astype(str)
    rows = []
    for k in K_VALUES:
        agreement = neighbor_jaccard(encoded_graphs[k], mixed_graphs[k])
        per_row = agreement["per_row"]
        by_workload = pd.DataFrame({"group": workload, "value": per_row}).groupby("group")["value"].mean()
        by_configuration = pd.DataFrame({"group": configurations, "value": per_row}).groupby("group")["value"].mean()
        rows.append(
            {
                "k": k,
                "neighbor_recall": neighbor_recall(encoded_graphs[k], mixed_graphs[k])["mean"],
                "neighbor_jaccard": agreement["mean"],
                "per_row_jaccard": {
                    "minimum": float(np.min(per_row)),
                    "q25": float(np.quantile(per_row, 0.25)),
                    "median": float(np.median(per_row)),
                    "q75": float(np.quantile(per_row, 0.75)),
                    "maximum": float(np.max(per_row)),
                    "fraction_below_half_overlap": float(np.mean(per_row < 0.5)),
                },
                "mean_jaccard_by_workload_cell": {
                    "cells": int(len(by_workload)),
                    "minimum": float(by_workload.min()),
                    "median": float(by_workload.median()),
                    "maximum": float(by_workload.max()),
                },
                "mean_jaccard_by_configuration": {
                    "configurations": int(len(by_configuration)),
                    "minimum": float(by_configuration.min()),
                    "median": float(by_configuration.median()),
                    "maximum": float(by_configuration.max()),
                },
            }
        )
    return rows


def _fit_precomputed_umap(distance: np.ndarray, config: UMAPConfig) -> tuple[np.ndarray, float]:
    """Research-only precomputed-distance UMAP for the frozen mixed sensitivity."""

    from umap import UMAP

    started = time.perf_counter()
    model = UMAP(
        n_components=2,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric="precomputed",
        random_state=config.seed,
        n_jobs=1,
    )
    coordinates = np.asarray(model.fit_transform(distance), dtype=float)
    return coordinates, time.perf_counter() - started


def _within_workload_configuration_from_graph(
    graph: np.ndarray,
    workloads: pd.DataFrame,
    values: np.ndarray,
    *,
    value_kind: str,
    eligibility: dict[str, Any],
    permutations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the PR 1 conditioned null using a reusable frozen local graph.

    ``within_workload_configuration_analysis`` is deliberately generic and
    rebuilds its graph on each call.  Stage 5 needs all 16 configuration fields
    on the same graph, so this execution-only vectorization preserves its pair
    statistic and seeded within-cell shuffles while avoiding repeated graph
    construction and Python-level pair loops.
    """

    if value_kind not in {"categorical", "numeric"}:
        raise ValueError("value_kind must be categorical or numeric.")
    rows = np.repeat(np.arange(len(graph)), graph.shape[1])
    neighbors = graph.ravel()
    original = np.asarray(values, dtype=object)
    if len(original) != len(graph):
        raise ValueError("Configuration values must align with the workload graph.")
    valid_neighbor = neighbors >= 0
    if value_kind == "categorical":
        missing = pd.isna(original)
        valid_pair = valid_neighbor & ~missing[rows] & ~missing[neighbors.clip(min=0)]

        def pair_values(current: np.ndarray) -> np.ndarray:
            return (current[rows[valid_pair]] == current[neighbors[valid_pair]]).astype(float)

    else:
        numeric_original = pd.to_numeric(pd.Series(original), errors="coerce").to_numpy(dtype=float)
        valid_pair = (
            valid_neighbor
            & np.isfinite(numeric_original[rows])
            & np.isfinite(numeric_original[neighbors.clip(min=0)])
        )

        def pair_values(current: np.ndarray) -> np.ndarray:
            numeric = pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(dtype=float)
            return np.abs(numeric[rows[valid_pair]] - numeric[neighbors[valid_pair]])

    valid_rows = rows[valid_pair]

    def summary(current: np.ndarray) -> dict[str, Any]:
        compared = pair_values(current)
        totals = np.bincount(valid_rows, weights=compared, minlength=len(graph))
        counts = np.bincount(valid_rows, minlength=len(graph))
        per_row = np.full(len(graph), np.nan, dtype=float)
        usable = counts > 0
        per_row[usable] = totals[usable] / counts[usable]
        return {
            "kind": value_kind,
            "per_row": per_row,
            "eligible_rows": int(usable.sum()),
            "excluded_rows": int((~usable).sum()),
            "eligible_neighbor_pairs": int(len(compared)),
            "mean": float(np.mean(compared)) if len(compared) else None,
        }

    observed = summary(original)
    if observed["mean"] is None:
        return {
            "observed": observed,
            "workload_conditioned_null": {
                "null_mean": None,
                "null_standard_deviation": None,
                "p_value": None,
                "alternative": None,
                "permutations": permutations,
                "seed": seed,
            },
        }
    keys = workloads[["isl", "osl", "conc"]].astype(str).agg("|".join, axis=1)
    cells = [np.flatnonzero(keys.eq(key).to_numpy()) for key in sorted(keys.unique())]
    generator = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = original.copy()
        for positions in cells:
            shuffled[positions] = original[positions][generator.permutation(len(positions))]
        null_values[index] = summary(shuffled)["mean"]
    more_extreme = (
        null_values >= observed["mean"]
        if value_kind == "categorical"
        else null_values <= observed["mean"]
    )
    return {
        "observed": observed,
        "workload_conditioned_null": {
            "null_mean": float(null_values.mean()),
            "null_standard_deviation": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
            "p_value": float((more_extreme.sum() + 1) / (len(null_values) + 1)),
            "alternative": "greater homophily" if value_kind == "categorical" else "smaller absolute difference",
            "permutations": permutations,
            "seed": seed,
        },
    }


def _hierarchy(representations: dict[str, np.ndarray], data: Any) -> dict[str, Any]:
    """Evaluate workload and configuration structure for named geometries."""

    workloads = data.cohort[["isl", "osl", "conc"]]
    groups = data.cohort[GROUP_COLUMN]
    result_by_representation = {}
    for name, representation in representations.items():
        graphs = _nested_graphs(representation)
        purity = {}
        for k in K_VALUES:
            value = workload_neighborhood_purity(graphs[k], workloads)
            purity[str(k)] = {
                **{key: item for key, item in value.items() if key != "per_row"},
                "grouped_bootstrap": _boot(value["per_row"], groups),
            }
        configurations = {}
        within_workload = {
            k: within_workload_knn_indices(representation, workloads, k=k)
            for k in K_VALUES
        }
        for feature in (item for item in PCA_FEATURES if item not in {"isl", "osl", "conc"}):
            kind = "categorical" if feature in (*CATEGORICAL_FEATURES, *BOOLEAN_FEATURES) else "numeric"
            configurations[feature] = {}
            for k in K_VALUES:
                within_graph, eligibility = within_workload[k]
                analysis = _within_workload_configuration_from_graph(
                    within_graph,
                    workloads,
                    data.cohort[feature].to_numpy(),
                    value_kind=kind,
                    eligibility=eligibility,
                    permutations=200,
                    seed=42,
                )
                null = analysis["workload_conditioned_null"]
                configurations[feature][str(k)] = {
                    "kind": kind,
                    "eligibility": {
                        key: value
                        for key, value in eligibility.items()
                        if key != "eligible_mask"
                    },
                    "observed_mean": analysis["observed"]["mean"],
                    "observed_grouped_bootstrap": (
                        _boot(analysis["observed"]["per_row"], groups)
                        if analysis["observed"]["eligible_rows"]
                        else None
                    ),
                    "eligible_neighbor_pairs": analysis["observed"]["eligible_neighbor_pairs"],
                    "null_mean": null.get("null_mean"),
                    "null_standard_deviation": null.get("null_standard_deviation"),
                    "p_value": null.get("p_value"),
                    "alternative": null.get("alternative"),
                }
        result_by_representation[name] = {
            "workload_purity": purity,
            "configuration_within_workload": configurations,
        }
    return result_by_representation


def _ablation_matrix(data: Any, features: list[str]) -> tuple[np.ndarray, list[str]]:
    preprocessor = make_preprocessor_for_features(features)
    matrix = np.asarray(preprocessor.fit_transform(data.cohort[features]), dtype=np.float32)
    return matrix, list(preprocessor.get_feature_names_out())


def _load_frozen_pca_15(data: Any, artifact_path: Path) -> np.ndarray:
    """Project the canonical matrix with the existing Stage 2 PCA basis.

    The promoted PCA overlay artifact predates the representation-artifact
    schema used by AE/VAE, so its compatible identity lives under
    ``shared_basis`` rather than top-level cohort hashes.  Validate that live
    contract instead of inventing a second PCA artifact or retraining it.
    """

    pca = json.loads(artifact_path.read_text(encoding="utf-8"))
    basis = pca.get("shared_basis", {})
    if pca.get("schema_version") != "pca-target-overlays-v2":
        raise ValueError("Preserved PCA artifact schema is incompatible.")
    if pca.get("dump", {}).get("version") != "db-dump/2026-07-20":
        raise ValueError("Preserved PCA artifact source snapshot is incompatible.")
    if (
        basis.get("cohort_rows") != len(data.cohort)
        or basis.get("full_eligible_row_count") != len(data.cohort)
        or basis.get("feature_order") != list(PCA_FEATURES)
        or basis.get("target_metrics_in_inputs") != []
    ):
        raise ValueError("Preserved PCA artifact does not match the canonical Stage 5 contract.")
    preprocessing = basis.get("preprocessing", {})
    if preprocessing.get("feature_order") != list(PCA_FEATURES):
        raise ValueError("Preserved PCA artifact source feature order is incompatible.")
    if preprocessing["encoded_feature_names"] != data.encoded_feature_names:
        raise ValueError("Preserved PCA artifact encoded feature order is incompatible.")
    components = np.asarray(preprocessing["pca_components"], dtype=float)
    mean = np.asarray(preprocessing["pca_mean"], dtype=float)
    if components.shape[0] < 15 or components.shape[1] != len(data.encoded_feature_names) or mean.shape != (len(data.encoded_feature_names),):
        raise ValueError("Preserved PCA basis dimensionality is incompatible.")
    return (data.matrix - mean) @ components[:15].T


def _load_existing_representations(data: Any, artifacts: Path) -> dict[str, np.ndarray]:
    representations = {
        "PCA-15": _load_frozen_pca_15(data, artifacts / "pca-db-dump-2026-07-20.json")
    }
    for method, filename in (
        ("AE-15", "representation-ae-final-db-dump-2026-07-20.json"),
        ("VAE-15", "representation-vae-final-db-dump-2026-07-20.json"),
    ):
        artifact, companion_path = load_final_representation_artifact(
            artifacts / filename,
            # The current canonical identity hashes semantic row IDs; the
            # already-promoted final AE/VAE artifacts predate that hash
            # migration.  Their source/feature/outcome/companion contracts are
            # still validated here, then row IDs and values are proven equal by
            # ``align_companion_to_active_cohort`` below.
            expected_cohort_hash=None,
        )
        if artifact["encoded_feature_order"] != data.encoded_feature_names:
            raise ValueError(f"{method} artifact encoded feature order is incompatible.")
        companion = pd.read_parquet(companion_path)
        aligned_companion, _alignment = align_companion_to_active_cohort(
            companion,
            data,
            expected_pca_input_hash=artifact.get("semantic_identity", {}).get("pca_input_hash"),
        )
        aligned = aligned_companion.loc[aligned_companion["seed"].eq(42)].copy().set_index("row_id")
        if aligned.index.astype(str).tolist() != data.row_ids:
            raise ValueError(f"{method} companion row-ID order is incompatible.")
        representations[method] = aligned[[f"z{index}" for index in range(1, 16)]].to_numpy(dtype=float)
    return representations


def _consensus_summary(agreement: dict[str, Any]) -> dict[str, float]:
    """Summarize recurring neighbors against each row's candidate-set union.

    With three or more methods, the number of distinct neighbors recurring in
    at least two methods can legitimately exceed ``k``.  Dividing that count
    by ``k`` is therefore not a fraction.  The union of all per-method
    candidate neighbors is the stable denominator for the reported recurrence
    fraction.
    """

    recurring = agreement["consensus"]["recurring_neighbors"]
    graphs = list(agreement["neighbor_graphs"].values())
    union_counts = [
        len({int(neighbor) for graph in graphs for neighbor in graph[row]})
        for row in range(len(recurring))
    ]
    recurring_counts = [len(row) for row in recurring]
    if not union_counts or any(count == 0 for count in union_counts):
        raise ValueError("Consensus candidate neighborhoods must be non-empty.")
    return {
        "consensus_neighbor_count": float(np.mean(recurring_counts)),
        "candidate_neighbor_union_count": float(np.mean(union_counts)),
        "consensus_fraction": float(
            np.mean([count / union for count, union in zip(recurring_counts, union_counts, strict=True)])
        ),
    }


def _outcomes(data: Any, coordinates: np.ndarray) -> dict[str, Any]:
    """Join post-freeze outcomes by row ID and emit descriptive-only summaries."""

    coordinate_frame = pd.DataFrame(
        {"row_id": data.row_ids, "x": coordinates[:, 0], "y": coordinates[:, 1]}
    )
    outcome_frame = data.cohort[list(OUTCOME_TARGETS)].copy()
    outcome_frame.insert(0, "row_id", data.row_ids)
    joined = coordinate_frame.merge(outcome_frame, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(data.row_ids):
        raise RuntimeError("Post-freeze outcome join did not preserve canonical row identity.")
    graph = exact_knn_indices(coordinates, 30)
    workload_key = data.cohort[["isl", "osl", "conc"]].astype(str).agg("|".join, axis=1)
    rows = {}
    for target in OUTCOME_TARGETS:
        values = pd.to_numeric(joined[target], errors="coerce").to_numpy(dtype=float)
        present = np.isfinite(values)
        local = []
        for row, neighbors in enumerate(graph):
            if present[row]:
                observed_neighbors = neighbors[present[neighbors]]
                local.extend(abs(values[row] - values[observed_neighbors]).tolist())
        adjusted = pd.DataFrame(
            {"workload": workload_key, "x": coordinates[:, 0], "y": coordinates[:, 1], "target": values}
        )
        adjusted[["x", "y", "target"]] = adjusted[["x", "y", "target"]].sub(
            adjusted.groupby("workload")[["x", "y", "target"]].transform("mean")
        )
        valid_adjusted = adjusted["target"].notna()
        rows[target] = {
            "available_rows": int(present.sum()),
            "x_pearson": float(pd.Series(coordinates[present, 0]).corr(pd.Series(values[present]), method="pearson")),
            "y_pearson": float(pd.Series(coordinates[present, 1]).corr(pd.Series(values[present]), method="pearson")),
            "x_spearman": float(pd.Series(coordinates[present, 0]).corr(pd.Series(values[present]), method="spearman")),
            "y_spearman": float(pd.Series(coordinates[present, 1]).corr(pd.Series(values[present]), method="spearman")),
            "mean_absolute_difference_among_30_coordinate_neighbors": float(np.mean(local)) if local else None,
            "within_exact_workload": {
                "x_pearson": float(adjusted.loc[valid_adjusted, "x"].corr(adjusted.loc[valid_adjusted, "target"], method="pearson")),
                "y_pearson": float(adjusted.loc[valid_adjusted, "y"].corr(adjusted.loc[valid_adjusted, "target"], method="pearson")),
                "x_spearman": float(adjusted.loc[valid_adjusted, "x"].corr(adjusted.loc[valid_adjusted, "target"], method="spearman")),
                "y_spearman": float(adjusted.loc[valid_adjusted, "y"].corr(adjusted.loc[valid_adjusted, "target"], method="spearman")),
            },
        }
    return rows


def run_stage5_results(
    aggregate: pd.DataFrame,
    *,
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Execute the frozen Stage 5 protocol and write non-promoted research artifacts."""

    output = Path(artifact_dir)
    cache_dir = output / CACHE_DIRECTORY
    data = canonical_stage5_data(aggregate)
    row_ids = data.row_ids
    groups = data.cohort[GROUP_COLUMN]
    reference_graphs = _nested_graphs(data.matrix)
    reference_trust = _ReferenceTrustworthiness(
        data.matrix, temporary_directory=cache_dir
    )
    projections: list[pd.DataFrame] = []
    umap2: list[dict[str, Any]] = []
    umap2_coordinates: dict[str, np.ndarray] = {}
    for config in UMAP_2_GRID:
        parameter_id = _parameter_id("umap2", **asdict(config))
        identity = _cache_identity(
            data,
            method="umap-2",
            parameters=asdict(config),
            encoded_feature_names=data.encoded_feature_names,
        )

        def fit_current(config: UMAPConfig = config) -> tuple[np.ndarray, dict[str, Any]]:
            fitted = fit_stage5_umap(
                data.matrix, config, encoded_feature_names=data.encoded_feature_names
            )
            return fitted.coordinates, {
                "runtime_seconds": fitted.runtime_seconds,
                "software": fitted.software,
            }

        coordinates, metadata, execution, cache_key = _cached_coordinates(cache_dir, identity, fit_current)
        if coordinates.shape != (len(row_ids), 2):
            raise RuntimeError("Cached UMAP-2 coordinates have incompatible row alignment.")
        metric = _metrics(
            coordinates,
            reference_graphs,
            groups=groups,
            trustworthiness_reference=reference_trust,
        )
        umap2.append(
            {
                "parameter_id": parameter_id,
                "config": asdict(config),
                "runtime_seconds": metadata["runtime_seconds"],
                "execution": execution,
                "cache_key": cache_key,
                "metrics": _compact_metrics(metric),
            }
        )
        umap2_coordinates[parameter_id] = coordinates
        projections.append(
            _projection_rows(
                row_ids,
                coordinates,
                method="umap-2",
                parameter_id=parameter_id,
                seed=config.seed,
                n_neighbors=config.n_neighbors,
                min_dist=config.min_dist,
            )
        )
        del metric
        gc.collect()
    canonical_id = _parameter_id("umap2", n_components=2, metric="euclidean", **UMAP_CANONICAL_DISPLAY)
    canonical_coordinates = umap2_coordinates[canonical_id]
    seed_stability = []
    for neighbors in (15, 50, 100):
        for distance in (0.0, 0.1, 0.5):
            selected = {
                str(seed): _nested_graphs(
                    umap2_coordinates[
                        _parameter_id(
                            "umap2",
                            n_components=2,
                            n_neighbors=neighbors,
                            min_dist=distance,
                            seed=seed,
                            metric="euclidean",
                        )
                    ]
                )
                for seed in STAGE5_SEEDS
            }
            result = _graph_seed_stability(selected)
            seed_stability.append({"n_neighbors": neighbors, "min_dist": distance, "pairs": result["pairs"], "mean_jaccard": result["mean_jaccard"]})
            del selected
            gc.collect()
    hierarchy = _hierarchy(
        {"encoded_original": data.matrix, "canonical_umap_2": canonical_coordinates}, data
    )

    ablations = {}
    for name, features in (
        ("workload_only", ["isl", "osl", "conc"]),
        ("configuration_only", [item for item in PCA_FEATURES if item not in {"isl", "osl", "conc"}]),
    ):
        matrix, names = _ablation_matrix(data, features)
        graphs = _nested_graphs(matrix)
        ablation_trust = _ReferenceTrustworthiness(
            matrix, temporary_directory=cache_dir
        )
        runs, run_graphs = [], {}
        for seed in STAGE5_SEEDS:
            config = UMAPConfig(n_components=2, n_neighbors=15, min_dist=0.1, seed=seed)
            parameter_id = _parameter_id(name, seed=seed, n_neighbors=15, min_dist=0.1)
            identity = _cache_identity(
                data, method=name, parameters=asdict(config), encoded_feature_names=names
            )

            def fit_current(
                config: UMAPConfig = config,
                matrix: np.ndarray = matrix,
                names: list[str] = names,
            ) -> tuple[np.ndarray, dict[str, Any]]:
                fitted = fit_stage5_umap(matrix, config, encoded_feature_names=names)
                return fitted.coordinates, {
                    "runtime_seconds": fitted.runtime_seconds,
                    "software": fitted.software,
                }

            coordinates, metadata, execution, cache_key = _cached_coordinates(
                cache_dir, identity, fit_current
            )
            if coordinates.shape != (len(row_ids), 2):
                raise RuntimeError(f"Cached {name} coordinates have incompatible row alignment.")
            metric = _metrics(
                coordinates,
                graphs,
                groups=groups,
                trustworthiness_reference=ablation_trust,
            )
            runs.append(
                {
                    "parameter_id": parameter_id,
                    "seed": seed,
                    "runtime_seconds": metadata["runtime_seconds"],
                    "execution": execution,
                    "cache_key": cache_key,
                    "metrics": _compact_metrics(metric),
                }
            )
            run_graphs[str(seed)] = metric["candidate_graphs"]
            projections.append(
                _projection_rows(
                    row_ids,
                    coordinates,
                    method=name,
                    parameter_id=parameter_id,
                    seed=seed,
                    n_neighbors=15,
                    min_dist=0.1,
                )
            )
        ablations[name] = {
            "features": features,
            "runs": runs,
            "seed_stability": _graph_seed_stability(run_graphs),
        }
        ablation_trust.close()

    mixed_largest = source_mixed_knn_indices(data.cohort, max(ALL_K_VALUES))
    mixed_graphs = {k: mixed_largest[:, :k] for k in ALL_K_VALUES}
    mixed_overlap = _mixed_disagreement(reference_graphs, mixed_graphs, data)
    mixed_distance = materialize_source_mixed_distance_matrix(data.cohort, allow_large=True)
    mixed_trust = _ReferenceTrustworthiness(
        distances=mixed_distance, temporary_directory=cache_dir
    )
    mixed_runs = []
    mixed_run_graphs: dict[str, dict[int, np.ndarray]] = {}
    mixed_coordinates: dict[str, np.ndarray] = {}
    for seed in STAGE5_SEEDS:
        config = UMAPConfig(n_components=2, n_neighbors=15, min_dist=0.1, seed=seed)
        parameter_id = _parameter_id("mixed_umap2", seed=seed, n_neighbors=15, min_dist=0.1)
        identity = _cache_identity(
            data,
            method="mixed-umap-2",
            parameters=asdict(config),
            encoded_feature_names=list(PCA_FEATURES),
            extra={"mixed_distance": "source-balanced-v1"},
        )

        def fit_current(config: UMAPConfig = config) -> tuple[np.ndarray, dict[str, Any]]:
            coordinates, runtime = _fit_precomputed_umap(mixed_distance, config)
            return coordinates, {"runtime_seconds": runtime}

        coordinates, metadata, execution, cache_key = _cached_coordinates(
            cache_dir, identity, fit_current
        )
        if coordinates.shape != (len(row_ids), 2):
            raise RuntimeError("Cached mixed-distance UMAP coordinates have incompatible row alignment.")
        metric = _metrics(
            coordinates,
            mixed_graphs,
            groups=groups,
            trustworthiness_reference=mixed_trust,
        )
        mixed_runs.append(
            {
                "parameter_id": parameter_id,
                "seed": seed,
                "runtime_seconds": metadata["runtime_seconds"],
                "execution": execution,
                "cache_key": cache_key,
                "metrics": _compact_metrics(metric),
            }
        )
        mixed_run_graphs[str(seed)] = metric["candidate_graphs"]
        mixed_coordinates[str(seed)] = coordinates
        projections.append(
            _projection_rows(
                row_ids,
                coordinates,
                method="mixed-umap-2",
                parameter_id=parameter_id,
                seed=seed,
                n_neighbors=15,
                min_dist=0.1,
            )
        )
    mixed_hierarchy = _hierarchy({"canonical_mixed_umap_2": mixed_coordinates["42"]}, data)
    mixed_stability = _graph_seed_stability(mixed_run_graphs)
    mixed_trust.close()
    del mixed_distance

    tsne = []
    for config in TSNE_2_GRID:
        parameter_id = _parameter_id("tsne2", **asdict(config))
        identity = _cache_identity(
            data,
            method="tsne-2",
            parameters=asdict(config),
            encoded_feature_names=data.encoded_feature_names,
        )

        def fit_current(config: Any = config) -> tuple[np.ndarray, dict[str, Any]]:
            fitted = fit_stage5_tsne(
                data.matrix, config, encoded_feature_names=data.encoded_feature_names
            )
            return fitted.coordinates, {
                "runtime_seconds": fitted.runtime_seconds,
                "final_kl_divergence": fitted.final_kl_divergence,
                "iterations": fitted.iterations,
                "software": fitted.software,
            }

        coordinates, metadata, execution, cache_key = _cached_coordinates(
            cache_dir, identity, fit_current
        )
        if coordinates.shape != (len(row_ids), 2):
            raise RuntimeError("Cached t-SNE coordinates have incompatible row alignment.")
        metric = _metrics(
            coordinates,
            reference_graphs,
            groups=groups,
            trustworthiness_reference=reference_trust,
        )
        tsne.append(
            {
                "parameter_id": parameter_id,
                "config": asdict(config),
                "runtime_seconds": metadata["runtime_seconds"],
                "final_kl_divergence": metadata.get("final_kl_divergence"),
                "iterations": metadata.get("iterations"),
                "execution": execution,
                "cache_key": cache_key,
                "metrics": _compact_metrics(metric),
            }
        )
        projections.append(
            _projection_rows(
                row_ids,
                coordinates,
                method="tsne-2",
                parameter_id=parameter_id,
                seed=config.seed,
                perplexity=config.perplexity,
            )
        )
        del metric
        del coordinates
        gc.collect()

    projection_path = output / PROJECTION_ARTIFACT
    projection = pd.concat(projections, ignore_index=True)
    expected_projection_rows = len(row_ids) * (27 + 9 + 3 + 3 + 3)
    if not (
        len(umap2) == 27
        and len(tsne) == 9
        and len(ablations["workload_only"]["runs"]) == 3
        and len(ablations["configuration_only"]["runs"]) == 3
        and len(mixed_runs) == 3
    ):
        raise RuntimeError("A frozen Stage 5 structural grid is incomplete.")
    if len(projection) != expected_projection_rows or projection.duplicated(["row_id", "parameter_id"]).any():
        raise RuntimeError("Stage 5 projection artifact alignment/count validation failed.")
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    structural_path = output / STRUCTURAL_ARTIFACT
    existing_structural: dict[str, Any] | None = None
    if structural_path.exists():
        # A post-freeze retry must preserve the original structural boundary and
        # its coordinate companion byte-for-byte.  Only a future explicit
        # structural-serialization repair may replace this artifact.
        structural_sha = require_structural_freeze(structural_path)
        existing_structural = json.loads(structural_path.read_text(encoding="utf-8"))
        expected_identity = {
            "source_dump": "db-dump/2026-07-20",
            "cohort_hash": data.cohort_hash,
            "row_key_hash": data.row_key_hash,
            "encoded_feature_order": data.encoded_feature_names,
            "target_metrics_in_inputs": [],
        }
        for key, expected in expected_identity.items():
            if existing_structural.get(key) != expected:
                raise RuntimeError(f"Existing structural freeze has incompatible {key}.")
        if existing_structural.get("run_counts") != {
            "umap_2": 27,
            "tsne_2": 9,
            "workload_only_umap_2": 3,
            "configuration_only_umap_2": 3,
            "mixed_distance_umap_2": 3,
        }:
            raise RuntimeError("Existing structural freeze has incomplete frozen run counts.")
        expected_projection_hash = existing_structural.get("projection_artifact", {}).get("sha256")
        if not projection_path.exists() or hashlib.sha256(projection_path.read_bytes()).hexdigest() != expected_projection_hash:
            raise RuntimeError("Existing structural freeze's projection companion is incompatible.")
    else:
        projection.to_parquet(projection_path, index=False, compression="zstd")
    structural = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "phase": "structural_freeze",
        "source_schema_version": STAGE5_SCHEMA_VERSION,
        "source_dump": "db-dump/2026-07-20",
        "cohort_rows": len(data.cohort), "configurations": int(groups.nunique()),
        "cohort_hash": data.cohort_hash, "row_key_hash": data.row_key_hash,
        "semantic_identity": data.semantic_identity, "feature_order": list(PCA_FEATURES),
        "encoded_feature_order": data.encoded_feature_names, "target_metrics_in_inputs": [],
        "software_versions": software_versions({"umap-learn": importlib.metadata.version("umap-learn"), "numba": importlib.metadata.version("numba"), "pynndescent": importlib.metadata.version("pynndescent")}),
        "reference_neighbors": {str(k): {"rows": len(row_ids), "k": k} for k in ALL_K_VALUES},
        "run_counts": {
            "umap_2": len(umap2),
            "tsne_2": len(tsne),
            "workload_only_umap_2": len(ablations["workload_only"]["runs"]),
            "configuration_only_umap_2": len(ablations["configuration_only"]["runs"]),
            "mixed_distance_umap_2": len(mixed_runs),
        },
        "umap_2": {"runs": umap2, "seed_stability": seed_stability, "canonical_display": UMAP_CANONICAL_DISPLAY},
        "hierarchy": hierarchy, "ablations": ablations,
        "mixed_distance": {
            "reference_overlap": mixed_overlap,
            "runs": mixed_runs,
            "seed_stability": mixed_stability,
            "hierarchy": mixed_hierarchy,
            "dtype": "float32",
            "rows": len(row_ids),
            "expected_dense_bytes": mixed_distance_memory_bytes(len(row_ids)),
        },
        "tsne_2": {"runs": tsne, "canonical_display": TSNE_CANONICAL_DISPLAY},
        "projection_artifact": {"filename": projection_path.name, "rows": len(projection), "sha256": hashlib.sha256(projection_path.read_bytes()).hexdigest()},
        "cache": {"directory": str(cache_dir), "entries_are_temporary_non_promoted": True},
        "warnings": ["Structural artifact contains no outcome overlays.", "UMAP-2 and t-SNE coordinates are descriptive; no clustering was performed.", "Trustworthiness uses a reusable exact original-space rank matrix and is scikit-learn formula equivalent."],
    }
    if existing_structural is None:
        _write_json(structural_path, structural)
        structural_sha = require_structural_freeze(structural_path)

    outcomes = _outcomes(data, canonical_coordinates)
    umap15, umap15_coordinates = [], {}
    for config in UMAP_15_GRID:
        parameter_id = _parameter_id("umap15", **asdict(config))
        identity = _cache_identity(
            data,
            method="umap-15",
            parameters=asdict(config),
            encoded_feature_names=data.encoded_feature_names,
        )

        def fit_current(config: UMAPConfig = config) -> tuple[np.ndarray, dict[str, Any]]:
            fitted = fit_stage5_umap(
                data.matrix, config, encoded_feature_names=data.encoded_feature_names
            )
            return fitted.coordinates, {
                "runtime_seconds": fitted.runtime_seconds,
                "software": fitted.software,
            }

        coordinates, metadata, execution, cache_key = _cached_coordinates(
            cache_dir, identity, fit_current
        )
        if coordinates.shape != (len(row_ids), 15):
            raise RuntimeError("Cached UMAP-15 coordinates have incompatible row alignment.")
        metric = _metrics(
            coordinates,
            reference_graphs,
            groups=groups,
            trustworthiness_reference=reference_trust,
        )
        umap15.append(
            {
                "parameter_id": parameter_id,
                "config": asdict(config),
                "runtime_seconds": metadata["runtime_seconds"],
                "execution": execution,
                "cache_key": cache_key,
                "metrics": _compact_metrics(metric),
            }
        )
        umap15_coordinates[parameter_id] = coordinates
        del metric
        gc.collect()
    umap15_stability = []
    for neighbors in (15, 50, 100):
        for distance in (0.0, 0.1, 0.5):
            selected = {
                str(seed): _nested_graphs(
                    umap15_coordinates[
                        _parameter_id(
                            "umap15",
                            n_components=15,
                            n_neighbors=neighbors,
                            min_dist=distance,
                            seed=seed,
                            metric="euclidean",
                        )
                    ]
                )
                for seed in STAGE5_SEEDS
            }
            result = _graph_seed_stability(selected)
            umap15_stability.append({"n_neighbors": neighbors, "min_dist": distance, "mean_jaccard": result["mean_jaccard"], "pairs": result["pairs"]})
            del selected
            gc.collect()
    grouped = []
    for partition_seed in PARTITION_SEEDS:
        for split in grouped_partition_definitions(data, partition_seed=partition_seed):
            validation = np.asarray(split["validation_indices"], dtype=int)
            from modeling.representation_analysis import fit_fold_preprocessor
            fold = fit_fold_preprocessor(data, split["train_indices"], validation)
            validation_reference = _nested_graphs(fold.validation_matrix)
            validation_trust = _ReferenceTrustworthiness(
                fold.validation_matrix, temporary_directory=cache_dir
            )
            for seed in STAGE5_SEEDS:
                config = UMAPConfig(n_components=15, n_neighbors=15, min_dist=0.1, seed=seed)
                identity = _cache_identity(
                    data,
                    method="grouped-umap-15-transform",
                    parameters={**asdict(config), "partition_seed": partition_seed, "fold": split["fold"]},
                    encoded_feature_names=fold.encoded_feature_names,
                    extra={
                        "train_row_ids_sha256": hashlib.sha256("\n".join(split["train_row_ids"]).encode()).hexdigest(),
                        "validation_row_ids_sha256": hashlib.sha256("\n".join(split["validation_row_ids"]).encode()).hexdigest(),
                    },
                )

                def fit_current(
                    split: dict[str, Any] = split,
                    config: UMAPConfig = config,
                ) -> tuple[np.ndarray, dict[str, Any]]:
                    transformed = grouped_umap_transform(data, split, config)
                    return transformed["validation_coordinates"], {
                        "runtime_seconds": transformed["fitted"].runtime_seconds,
                        "train_configurations": transformed["train_configurations"],
                        "validation_configurations": transformed["validation_configurations"],
                        "group_overlap": transformed["group_overlap"],
                    }

                coordinates, metadata, execution, cache_key = _cached_coordinates(
                    cache_dir, identity, fit_current
                )
                if coordinates.shape != (len(validation), 15):
                    raise RuntimeError("Cached grouped UMAP-15 transform has incompatible validation rows.")
                metrics = _metrics(
                    coordinates,
                    validation_reference,
                    groups=data.cohort.iloc[validation][GROUP_COLUMN],
                    trustworthiness_reference=validation_trust,
                )
                grouped.append(
                    {
                        "partition_seed": partition_seed,
                        "fold": split["fold"],
                        "seed": seed,
                        "train_rows": len(split["train_indices"]),
                        "validation_rows": len(validation),
                        "train_configurations": metadata["train_configurations"],
                        "validation_configurations": metadata["validation_configurations"],
                        "group_overlap": metadata["group_overlap"],
                        "runtime_seconds": metadata["runtime_seconds"],
                        "execution": execution,
                        "cache_key": cache_key,
                        "metrics": [
                            row for row in _compact_metrics(metrics) if row["k"] in K_VALUES
                        ],
                    }
                )
            validation_trust.close()
    if len(umap15) != 27 or len(grouped) != 27:
        raise RuntimeError("A frozen Stage 5 UMAP-15 grid or grouped validation grid is incomplete.")
    canonical_umap15 = umap15_coordinates[_parameter_id("umap15", n_components=15, n_neighbors=15, min_dist=0.1, seed=42, metric="euclidean")]
    representations = _load_existing_representations(data, output)
    representations["UMAP-15"] = canonical_umap15
    cross_method = {}
    for k in K_VALUES:
        core = cross_method_neighbor_agreement({key: representations[key] for key in ("PCA-15", "AE-15", "VAE-15")}, row_ids={key: row_ids for key in ("PCA-15", "AE-15", "VAE-15")}, k=k)
        extended = cross_method_neighbor_agreement(representations, row_ids={key: row_ids for key in representations}, k=k)
        for result in (core, extended):
            for pair in result["pairs"]:
                rank = local_rank_agreement(
                    result["neighbor_graphs"][pair["left_method"]],
                    result["neighbor_graphs"][pair["right_method"]],
                )
                pair["local_rank_agreement"] = rank["mean"]
                pair["rank_eligible_rows"] = rank["eligible_rows"]
        core_summary = _consensus_summary(core)
        extended_summary = _consensus_summary(extended)
        cross_method[str(k)] = {
            "core_pairs": core["pairs"],
            "extended_pairs": extended["pairs"],
            "core_consensus_neighbor_count": core_summary["consensus_neighbor_count"],
            "core_candidate_neighbor_union_count": core_summary["candidate_neighbor_union_count"],
            "core_consensus_fraction": core_summary["consensus_fraction"],
            "extended_consensus_neighbor_count": extended_summary["consensus_neighbor_count"],
            "extended_candidate_neighbor_union_count": extended_summary["candidate_neighbor_union_count"],
            "extended_consensus_fraction": extended_summary["consensus_fraction"],
        }
    method_fidelity = {}
    for name, values in representations.items():
        method_fidelity[name] = _compact_metrics(
            _metrics(
                values,
                reference_graphs,
                groups=groups,
                trustworthiness_reference=reference_trust,
            )
        )
    reference_trust.close()
    final = {"schema_version": RESULT_SCHEMA_VERSION, "phase": "final", "source_dump": structural["source_dump"], "cohort_rows": len(data.cohort), "configurations": int(groups.nunique()), "cohort_hash": data.cohort_hash, "row_key_hash": data.row_key_hash, "feature_order": list(PCA_FEATURES), "target_metrics_in_inputs": [], "structural_artifact": {"filename": structural_path.name, "sha256": structural_sha}, "projection_artifact": structural["projection_artifact"], "outcome_overlays": {"canonical_umap2_parameter_id": canonical_id, "descriptive_only": True, "metrics": outcomes}, "umap_15": {"runs": umap15, "seed_stability": umap15_stability, "grouped_held_out": grouped, "run_counts": {"umap_15": len(umap15), "grouped_held_out": len(grouped)}, "promotion_status": "not_promoted_human_review_required"}, "cross_method": {"method_fidelity": method_fidelity, "consensus": cross_method, "tsne_included": False}, "warnings": ["Outcome overlays were calculated only after the structural artifact hash was frozen.", "Stage 5 is not added to the active artifact index.", "UMAP-15 promotion remains a human decision."], "software_versions": structural["software_versions"]}
    final_path = output / FINAL_ARTIFACT
    final_sha = _write_json(final_path, final)
    return {"structural_path": structural_path, "structural_sha256": structural_sha, "final_path": final_path, "final_sha256": final_sha, "projection_path": projection_path, "data": data, "structural": structural, "final": final}
