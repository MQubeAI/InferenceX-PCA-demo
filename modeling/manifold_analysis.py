"""Research-only Stage 5 UMAP/t-SNE fitting and protocol orchestration helpers.

Generic neighborhood statistics live in :mod:`modeling.neighborhood_analysis`.
This module uses the existing canonical representation cohort and deliberately
does not import UMAP at module import time, so dashboard startup remains light.
"""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.manifold import TSNE

from modeling.neighborhood_analysis import (
    DEFAULT_GROUPED_BOOTSTRAP_REPLICATES,
    DEFAULT_GROUPED_BOOTSTRAP_SEED,
    DEFAULT_NEIGHBORHOOD_K_VALUES,
    SENSITIVITY_NEIGHBORHOOD_K,
)
from modeling.pca_target_analysis import (
    OUTCOME_PREFIXES,
    OUTCOME_TERMS,
    PCA_FEATURES,
    source_feature_for_encoded,
    validate_pca_feature_schema,
)
from modeling.representation_analysis import (
    GROUP_COLUMN,
    RANDOM_SEEDS,
    SOURCE_DUMP_VERSION,
    CanonicalRepresentationData,
    canonical_representation_data,
    fit_fold_preprocessor,
    software_versions,
)


STAGE5_SCHEMA_VERSION = "manifold-analysis-stage5-v1"
STAGE5_NEIGHBORHOOD_K_VALUES = DEFAULT_NEIGHBORHOOD_K_VALUES
STAGE5_SENSITIVITY_NEIGHBORHOOD_K = SENSITIVITY_NEIGHBORHOOD_K
STAGE5_GROUPED_BOOTSTRAP_REPLICATES = DEFAULT_GROUPED_BOOTSTRAP_REPLICATES
STAGE5_GROUPED_BOOTSTRAP_SEED = DEFAULT_GROUPED_BOOTSTRAP_SEED
STAGE5_SEEDS = RANDOM_SEEDS
UMAP_N_NEIGHBORS = (15, 50, 100)
UMAP_MIN_DIST = (0.0, 0.1, 0.5)
UMAP_METRIC = "euclidean"
UMAP_CANONICAL_DISPLAY = {"n_neighbors": 15, "min_dist": 0.1, "seed": 42}
TSNE_PERPLEXITIES = (15, 30, 50)
TSNE_CANONICAL_DISPLAY = {"perplexity": 30, "seed": 42}
TSNE_N_COMPONENTS = 2
TSNE_INITIALIZATION = "pca"
TSNE_LEARNING_RATE: str | float = "auto"
TSNE_MAX_ITERATIONS = 1_000
TSNE_EARLY_EXAGGERATION = 12.0
TSNE_METRIC = "euclidean"
TSNE_METHOD = "barnes_hut"


@dataclass(frozen=True)
class UMAPConfig:
    n_components: int
    n_neighbors: int
    min_dist: float
    seed: int
    metric: str = UMAP_METRIC


@dataclass(frozen=True)
class TSNEConfig:
    perplexity: int
    seed: int
    n_components: int = TSNE_N_COMPONENTS
    init: str = TSNE_INITIALIZATION
    learning_rate: str | float = TSNE_LEARNING_RATE
    max_iter: int = TSNE_MAX_ITERATIONS
    early_exaggeration: float = TSNE_EARLY_EXAGGERATION
    metric: str = TSNE_METRIC
    method: str = TSNE_METHOD


def _umap_grid(n_components: int) -> tuple[UMAPConfig, ...]:
    return tuple(
        UMAPConfig(
            n_components=n_components,
            n_neighbors=neighbors,
            min_dist=min_dist,
            seed=seed,
        )
        for neighbors, min_dist, seed in product(UMAP_N_NEIGHBORS, UMAP_MIN_DIST, STAGE5_SEEDS)
    )


UMAP_2_GRID = _umap_grid(2)
UMAP_15_GRID = _umap_grid(15)
TSNE_2_GRID = tuple(
    TSNEConfig(perplexity=perplexity, seed=seed)
    for perplexity, seed in product(TSNE_PERPLEXITIES, STAGE5_SEEDS)
)


@dataclass
class FittedUMAP:
    """A fitted research UMAP model with alignment and runtime metadata."""

    model: Any
    coordinates: np.ndarray
    config: UMAPConfig
    runtime_seconds: float
    software: dict[str, str]
    encoded_feature_names: tuple[str, ...]

    def transform(
        self,
        matrix: Any,
        *,
        encoded_feature_names: Iterable[str] | None = None,
    ) -> np.ndarray:
        values, feature_names = _stage5_structural_matrix(
            matrix, encoded_feature_names=encoded_feature_names
        )
        if feature_names != self.encoded_feature_names:
            raise ValueError("UMAP transform feature order does not match the fitted structural matrix.")
        transformed = np.asarray(self.model.transform(values), dtype=float)
        if transformed.shape != (len(values), self.config.n_components):
            raise RuntimeError("UMAP transform returned an unexpected coordinate shape.")
        return transformed


@dataclass
class FittedTSNE:
    """A descriptive-only t-SNE fit; no transform interface is exposed."""

    coordinates: np.ndarray
    config: TSNEConfig
    runtime_seconds: float
    final_kl_divergence: float | None
    iterations: int | None
    software: dict[str, str]


def _structural_matrix(matrix: Any) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        leakage = [
            str(column)
            for column in matrix.columns
            if str(column).startswith(OUTCOME_PREFIXES)
            or any(term in str(column).lower() for term in OUTCOME_TERMS)
        ]
        if leakage:
            raise ValueError(
                "A Stage 5 structural matrix must not contain outcome columns: "
                + ", ".join(sorted(set(leakage)))
            )
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 2:
        raise ValueError("A Stage 5 structural matrix needs at least three rows and two columns.")
    if not np.isfinite(values).all():
        raise ValueError("A Stage 5 structural matrix cannot contain NaN or infinite values.")
    return values


def _validate_stage5_encoded_feature_names(
    encoded_feature_names: Iterable[str],
    *,
    width: int,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in encoded_feature_names)
    if len(names) != width:
        raise ValueError("Encoded feature names must be aligned one-for-one with matrix columns.")
    if len(names) != len(set(names)):
        raise ValueError("Encoded feature names must be unique.")
    unexpected = [name for name in names if source_feature_for_encoded(name) not in PCA_FEATURES]
    if unexpected:
        raise ValueError(
            "Encoded feature names contain non-structural or outcome fields: "
            + ", ".join(unexpected)
        )
    return names


def _stage5_structural_matrix(
    matrix: Any,
    *,
    encoded_feature_names: Iterable[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Require auditable source provenance for every Stage 5 fit/transform matrix."""

    names = encoded_feature_names
    if names is None and isinstance(matrix, pd.DataFrame):
        names = matrix.columns
    if names is None:
        raise ValueError(
            "Stage 5 matrices require audited encoded_feature_names; use canonical_stage5_data "
            "or fold-local encoded feature names."
        )
    values = _structural_matrix(matrix)
    return values, _validate_stage5_encoded_feature_names(names, width=values.shape[1])


def validate_stage5_structural_columns(columns: Iterable[str]) -> None:
    """Fail closed when anything other than the frozen 19 source fields is supplied."""

    validate_pca_feature_schema(tuple(columns))


def validate_stage5_data(data: CanonicalRepresentationData) -> None:
    """Verify canonical cohort/matrix/row identity alignment before a Stage 5 fit."""

    validate_stage5_structural_columns(PCA_FEATURES)
    matrix = _structural_matrix(data.matrix)
    if len(data.cohort) != len(matrix) or len(data.row_ids) != len(matrix):
        raise ValueError("Canonical cohort, matrix, and row IDs must have identical row counts.")
    if len(set(data.row_ids)) != len(data.row_ids):
        raise ValueError("Canonical Stage 5 row IDs must be unique.")
    if GROUP_COLUMN not in data.cohort:
        raise ValueError("Canonical Stage 5 data is missing config_id grouping.")
    if matrix.shape[1] != len(data.encoded_feature_names):
        raise ValueError("Canonical encoded feature names do not match the structural matrix.")
    _validate_stage5_encoded_feature_names(
        data.encoded_feature_names,
        width=matrix.shape[1],
    )


def canonical_stage5_data(
    aggregate: pd.DataFrame,
    *,
    enforce_snapshot_counts: bool = True,
) -> CanonicalRepresentationData:
    """Reuse the authoritative representation cohort and preprocessing unchanged."""

    data = canonical_representation_data(
        aggregate, enforce_snapshot_counts=enforce_snapshot_counts
    )
    validate_stage5_data(data)
    return data


def validate_projection_alignment(coordinates: Any, row_ids: Iterable[str]) -> None:
    """Ensure a stored/future projection has exactly one coordinate row per row ID."""

    array = np.asarray(coordinates)
    identifiers = list(row_ids)
    if array.ndim != 2:
        raise ValueError("Projection coordinates must be two-dimensional.")
    if len(array) != len(identifiers):
        raise ValueError("Projection row count does not match projection row IDs.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Projection row IDs must be unique.")


def _validate_umap_config(config: UMAPConfig, *, stage5_fixed: bool) -> None:
    if config.n_components < 2:
        raise ValueError("UMAP requires at least two output dimensions.")
    if config.n_neighbors < 2:
        raise ValueError("UMAP n_neighbors must be at least two.")
    if config.min_dist < 0:
        raise ValueError("UMAP min_dist must be non-negative.")
    if stage5_fixed and (
        config.n_components not in {2, 15}
        or config.n_neighbors not in UMAP_N_NEIGHBORS
        or config.min_dist not in UMAP_MIN_DIST
        or config.seed not in STAGE5_SEEDS
        or config.metric != UMAP_METRIC
    ):
        raise ValueError("Unsupported parameter value in the fixed Stage 5 UMAP path.")


def _import_umap() -> Any:
    """Import UMAP only when a research fit is explicitly requested."""

    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - environment-specific path
        raise RuntimeError(
            "UMAP research dependency is unavailable; install requirements-representation.txt."
        ) from exc
    return UMAP


def fit_umap(
    matrix: Any,
    config: UMAPConfig,
    *,
    encoded_feature_names: Iterable[str] | None = None,
    stage5_fixed: bool = False,
) -> FittedUMAP:
    """Fit a UMAP projection.  Set ``stage5_fixed`` for frozen-protocol checks."""

    values, feature_names = _stage5_structural_matrix(
        matrix, encoded_feature_names=encoded_feature_names
    )
    _validate_umap_config(config, stage5_fixed=stage5_fixed)
    if config.n_neighbors >= len(values):
        raise ValueError("UMAP n_neighbors must be smaller than the structural row count.")
    UMAP = _import_umap()
    started = time.perf_counter()
    model = UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.seed,
        n_jobs=1,
    )
    coordinates = np.asarray(model.fit_transform(values), dtype=float)
    if coordinates.shape != (len(values), config.n_components):
        raise RuntimeError("UMAP fit returned an unexpected coordinate shape.")
    return FittedUMAP(
        model=model,
        coordinates=coordinates,
        config=config,
        runtime_seconds=time.perf_counter() - started,
        software=software_versions({"umap-learn": importlib.metadata.version("umap-learn")}),
        encoded_feature_names=feature_names,
    )


def fit_stage5_umap(
    matrix: Any,
    config: UMAPConfig,
    *,
    encoded_feature_names: Iterable[str] | None = None,
) -> FittedUMAP:
    """Fit only a grid member permitted by the frozen Stage 5 protocol."""

    return fit_umap(
        matrix,
        config,
        encoded_feature_names=encoded_feature_names,
        stage5_fixed=True,
    )


def grouped_umap_transform(
    data: CanonicalRepresentationData,
    split: Mapping[str, Any],
    config: UMAPConfig,
) -> dict[str, Any]:
    """Fit UMAP on grouped train rows and transform its held-out configurations."""

    validate_stage5_data(data)
    required = {"train_indices", "validation_indices"}
    if missing := sorted(required - set(split)):
        raise ValueError("Grouped UMAP split is missing: " + ", ".join(missing))
    train = np.asarray(split["train_indices"], dtype=int)
    validation = np.asarray(split["validation_indices"], dtype=int)
    if not len(train) or not len(validation):
        raise ValueError("Grouped UMAP split must contain both train and validation rows.")
    if len(train) != len(set(train.tolist())) or len(validation) != len(set(validation.tolist())):
        raise ValueError("Grouped UMAP split cannot repeat row indices.")
    if train.min() < 0 or validation.min() < 0 or train.max() >= len(data.matrix) or validation.max() >= len(data.matrix):
        raise ValueError("Grouped UMAP split contains an out-of-range row index.")
    if set(train) & set(validation):
        raise ValueError("Grouped UMAP split reuses rows across train and validation.")
    train_groups = set(data.cohort.iloc[train][GROUP_COLUMN].astype(str))
    validation_groups = set(data.cohort.iloc[validation][GROUP_COLUMN].astype(str))
    if train_groups & validation_groups:
        raise ValueError("Grouped UMAP transform split leaks config_id values across train and validation.")
    train_row_ids = [data.row_ids[index] for index in train]
    validation_row_ids = [data.row_ids[index] for index in validation]
    for key, expected in (
        ("train_row_ids", train_row_ids),
        ("validation_row_ids", validation_row_ids),
    ):
        if key in split and list(split[key]) != expected:
            raise ValueError(f"Grouped UMAP split {key} does not match canonical row IDs.")
    fold_data = fit_fold_preprocessor(data, train, validation)
    fitted = fit_stage5_umap(
        fold_data.train_matrix,
        config,
        encoded_feature_names=fold_data.encoded_feature_names,
    )
    held_out = fitted.transform(
        fold_data.validation_matrix,
        encoded_feature_names=fold_data.encoded_feature_names,
    )
    validate_projection_alignment(held_out, validation_row_ids)
    return {
        "fitted": fitted,
        "train_row_ids": train_row_ids,
        "validation_row_ids": validation_row_ids,
        "validation_coordinates": held_out,
        "encoded_feature_names": fold_data.encoded_feature_names,
        "train_configurations": len(train_groups),
        "validation_configurations": len(validation_groups),
        "group_overlap": 0,
    }


def _validate_tsne_config(config: TSNEConfig, *, stage5_fixed: bool) -> None:
    if config.n_components != 2:
        raise ValueError("Stage 5 t-SNE supports exactly two output dimensions.")
    if config.perplexity <= 0:
        raise ValueError("t-SNE perplexity must be positive.")
    if stage5_fixed and (
        config.perplexity not in TSNE_PERPLEXITIES
        or config.seed not in STAGE5_SEEDS
        or config.init != TSNE_INITIALIZATION
        or config.learning_rate != TSNE_LEARNING_RATE
        or config.max_iter != TSNE_MAX_ITERATIONS
        or config.early_exaggeration != TSNE_EARLY_EXAGGERATION
        or config.metric != TSNE_METRIC
        or config.method != TSNE_METHOD
    ):
        raise ValueError("Unsupported parameter value in the fixed Stage 5 t-SNE path.")


def fit_tsne(
    matrix: Any,
    config: TSNEConfig,
    *,
    encoded_feature_names: Iterable[str] | None = None,
    stage5_fixed: bool = False,
) -> FittedTSNE:
    """Fit direct encoded-matrix t-SNE; deliberately do not expose transform."""

    values, _feature_names = _stage5_structural_matrix(
        matrix, encoded_feature_names=encoded_feature_names
    )
    _validate_tsne_config(config, stage5_fixed=stage5_fixed)
    if config.perplexity >= len(values):
        raise ValueError("t-SNE perplexity must be smaller than the structural row count.")
    model = TSNE(
        n_components=config.n_components,
        perplexity=config.perplexity,
        early_exaggeration=config.early_exaggeration,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        metric=config.metric,
        init=config.init,
        method=config.method,
        random_state=config.seed,
        n_jobs=1,
    )
    started = time.perf_counter()
    # Barnes-Hut's neighbor search delegates temporary-distance chunking to
    # scikit-learn's working-memory setting. Bound only that implementation
    # buffer for the 8,063-row cohort; frozen t-SNE parameters are unchanged.
    with config_context(working_memory=64):
        coordinates = np.asarray(model.fit_transform(values), dtype=float)
    if coordinates.shape != (len(values), config.n_components):
        raise RuntimeError("t-SNE fit returned an unexpected coordinate shape.")
    runtime = time.perf_counter() - started
    final_kl_divergence = (
        float(model.kl_divergence_) if hasattr(model, "kl_divergence_") else None
    )
    iterations = int(model.n_iter_) if hasattr(model, "n_iter_") else None
    # t-SNE has no transform API; its fitted object is not part of the result
    # contract, so free its substantial temporary state between frozen runs.
    del model
    return FittedTSNE(
        coordinates=coordinates,
        config=config,
        runtime_seconds=runtime,
        final_kl_divergence=final_kl_divergence,
        iterations=iterations,
        software=software_versions(),
    )


def fit_stage5_tsne(
    matrix: Any,
    config: TSNEConfig,
    *,
    encoded_feature_names: Iterable[str] | None = None,
) -> FittedTSNE:
    """Fit only one frozen t-SNE sensitivity-grid configuration."""

    return fit_tsne(
        matrix,
        config,
        encoded_feature_names=encoded_feature_names,
        stage5_fixed=True,
    )


def stage5_plan() -> dict[str, Any]:
    """Serializable, non-executing inventory of frozen Stage 5 work."""

    return {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "source_dump": SOURCE_DUMP_VERSION,
        "primary_input": "canonical encoded structural matrix",
        "feature_order": list(PCA_FEATURES),
        "target_metrics_in_inputs": [],
        "neighborhood_k_values": list(STAGE5_NEIGHBORHOOD_K_VALUES),
        "sensitivity_neighborhood_k": STAGE5_SENSITIVITY_NEIGHBORHOOD_K,
        "grouped_bootstrap": {
            "group_column": GROUP_COLUMN,
            "replicates": STAGE5_GROUPED_BOOTSTRAP_REPLICATES,
            "seed": STAGE5_GROUPED_BOOTSTRAP_SEED,
        },
        "umap_2_runs": len(UMAP_2_GRID),
        "umap_15_runs": len(UMAP_15_GRID),
        "tsne_2_runs": len(TSNE_2_GRID),
        "umap_canonical_display": UMAP_CANONICAL_DISPLAY,
        "tsne_canonical_display": TSNE_CANONICAL_DISPLAY,
        "umap_grid": [asdict(config) for config in UMAP_2_GRID],
        "umap_15_grid": [asdict(config) for config in UMAP_15_GRID],
        "tsne_grid": [asdict(config) for config in TSNE_2_GRID],
    }


def stage5_artifact_metadata(
    data: CanonicalRepresentationData,
    *,
    method: str,
    parameter_grid: Iterable[Mapping[str, Any]],
    runtime_seconds: float,
    projection_row_ids: Iterable[str],
) -> dict[str, Any]:
    """Prepare identity metadata for a future reviewed result artifact only."""

    validate_stage5_data(data)
    row_ids = list(projection_row_ids)
    if len(row_ids) != len(set(row_ids)) or not set(row_ids).issubset(set(data.row_ids)):
        raise ValueError("Projection row IDs must be a unique subset of canonical row IDs.")
    return {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "method": method,
        "source_dump": SOURCE_DUMP_VERSION,
        "cohort_hash": data.cohort_hash,
        "row_key_hash": data.row_key_hash,
        "semantic_identity": data.semantic_identity,
        "cohort_rows": len(data.cohort),
        "feature_order": list(PCA_FEATURES),
        "target_metrics_in_inputs": [],
        "software_versions": software_versions(),
        "parameter_grid": list(parameter_grid),
        "random_seeds": list(STAGE5_SEEDS),
        "neighborhood_k_values": list(STAGE5_NEIGHBORHOOD_K_VALUES),
        "sensitivity_neighborhood_k": STAGE5_SENSITIVITY_NEIGHBORHOOD_K,
        "runtime_seconds": float(runtime_seconds),
        "projection_row_ids": row_ids,
    }
