"""Shared, leakage-safe protocol for representation-learning experiments.

Outcome metrics are deliberately absent from preprocessing and representation
fitting.  They enter only the descriptive evaluation helpers near the end of
this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling.pca_target_analysis import (
    CATEGORICAL_FEATURES,
    ENERGY_TARGET,
    LATENCY_TARGET,
    NUMERIC_FEATURES,
    OUTPUT_TARGET,
    PCA_FEATURES,
    make_preprocessor,
    normalize_pca_inputs,
    shared_basis_cohort,
    source_feature_for_encoded,
    validate_pca_feature_schema,
)

REPRESENTATION_SCHEMA_VERSION = "representation-analysis-v1"
COMPARISON_SCHEMA_VERSION = "representation-comparison-v1"
FINAL_REPRESENTATION_SCHEMA_VERSION = "representation-analysis-final-v2"
FINAL_COMPARISON_SCHEMA_VERSION = "representation-comparison-final-v2"
SOURCE_DUMP_VERSION = "db-dump/2026-07-20"
EXPECTED_COHORT_ROWS = 8_063
EXPECTED_CONFIGURATIONS = 1_354
GROUP_FOLDS = 3
GROUP_COLUMN = "config_id"
ROW_KEY_COLUMNS = ("config_id", "benchmark_type", "isl", "osl", "conc")
OUTCOME_TARGETS = (LATENCY_TARGET, OUTPUT_TARGET, ENERGY_TARGET)
LATENT_DIMENSIONS = (2, 5, 10, 15)
RANDOM_SEEDS = (42, 123, 2026)
BOOLEAN_FEATURES = (
    "config_prefill_dp_attention",
    "config_decode_dp_attention",
    "config_disagg",
    "config_is_multinode",
)


@dataclass
class CanonicalRepresentationData:
    """The one canonical cohort and matrix used by PCA, AE, and VAE."""

    cohort: pd.DataFrame
    matrix: np.ndarray
    preprocessor: Any
    encoded_feature_names: list[str]
    row_ids: list[str]
    cohort_hash: str
    row_key_hash: str


@dataclass
class FoldPreprocessedData:
    """One train-fitted preprocessing state shared by every representation."""

    preprocessor: ColumnTransformer
    feature_order: list[str]
    encoded_feature_names: list[str]
    train_matrix: np.ndarray
    validation_matrix: np.ndarray
    all_matrix: np.ndarray


def _stable_scalar(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def representation_row_ids(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in ROW_KEY_COLUMNS if column not in frame]
    if missing:
        raise ValueError("Missing representation row-key fields: " + ", ".join(missing))
    keys = [
        "|".join(_stable_scalar(value) for value in row)
        for row in frame[list(ROW_KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Canonical representation row keys are not unique.")
    return [hashlib.sha256(key.encode("utf-8")).hexdigest() for key in keys]


def _sequence_hash(values: Iterable[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_representation_data(
    aggregate: pd.DataFrame,
    *,
    enforce_snapshot_counts: bool = True,
) -> CanonicalRepresentationData:
    """Reproduce the frozen PCA cohort and its established preprocessing matrix."""

    validate_pca_feature_schema(PCA_FEATURES)
    cohort = normalize_pca_inputs(shared_basis_cohort(aggregate))
    if GROUP_COLUMN not in cohort:
        raise ValueError("config_id is required for grouped representation validation.")
    row_ids = representation_row_ids(cohort)
    preprocessor = make_preprocessor()
    matrix = np.asarray(
        preprocessor.fit_transform(cohort[list(PCA_FEATURES)]),
        dtype=np.float32,
    )
    encoded_feature_names = list(preprocessor.get_feature_names_out())
    if enforce_snapshot_counts:
        if len(cohort) != EXPECTED_COHORT_ROWS:
            raise ValueError(
                f"Canonical cohort mismatch: expected {EXPECTED_COHORT_ROWS:,}, got {len(cohort):,}."
            )
        configurations = int(cohort[GROUP_COLUMN].nunique())
        if configurations != EXPECTED_CONFIGURATIONS:
            raise ValueError(
                "Canonical configuration count mismatch: "
                f"expected {EXPECTED_CONFIGURATIONS:,}, got {configurations:,}."
            )
    cohort_payload = {
        "row_ids": row_ids,
        "feature_order": list(PCA_FEATURES),
        "encoded_feature_names": encoded_feature_names,
    }
    cohort_hash = hashlib.sha256(
        json.dumps(cohort_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CanonicalRepresentationData(
        cohort=cohort,
        matrix=matrix,
        preprocessor=preprocessor,
        encoded_feature_names=encoded_feature_names,
        row_ids=row_ids,
        cohort_hash=cohort_hash,
        row_key_hash=_sequence_hash(row_ids),
    )


def validate_representation_feature_subset(features: Iterable[str]) -> list[str]:
    ordered = list(features)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Representation feature subsets cannot contain duplicates.")
    unexpected = [feature for feature in ordered if feature not in PCA_FEATURES]
    if unexpected:
        raise ValueError(
            "Representation feature subset contains fields outside the frozen schema: "
            + ", ".join(unexpected)
        )
    expected_order = [feature for feature in PCA_FEATURES if feature in ordered]
    if ordered != expected_order:
        raise ValueError("Representation feature subset violates canonical feature order.")
    return ordered


def make_preprocessor_for_features(features: Iterable[str]) -> ColumnTransformer:
    """Build the established preprocessing rules for a frozen feature subset."""

    ordered = validate_representation_feature_subset(features)
    numeric_features = [feature for feature in NUMERIC_FEATURES if feature in ordered]
    categorical_features = [
        feature for feature in CATEGORICAL_FEATURES if feature in ordered
    ]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                max_categories=30,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_fold_preprocessor(
    data: CanonicalRepresentationData,
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    *,
    feature_order: Iterable[str] = PCA_FEATURES,
) -> FoldPreprocessedData:
    """Fit imputation, scaling, and categories on training rows only."""

    features = validate_representation_feature_subset(feature_order)
    train = np.asarray(list(train_indices), dtype=int)
    validation = np.asarray(list(validation_indices), dtype=int)
    preprocessor = make_preprocessor_for_features(features)
    train_frame = data.cohort.iloc[train][features]
    preprocessor.fit(train_frame)
    return FoldPreprocessedData(
        preprocessor=preprocessor,
        feature_order=features,
        encoded_feature_names=list(preprocessor.get_feature_names_out()),
        train_matrix=np.asarray(
            preprocessor.transform(train_frame), dtype=np.float32
        ),
        validation_matrix=np.asarray(
            preprocessor.transform(data.cohort.iloc[validation][features]),
            dtype=np.float32,
        ),
        all_matrix=np.asarray(
            preprocessor.transform(data.cohort[features]), dtype=np.float32
        ),
    )


def source_feature_reconstruction_metrics(
    validation_frame: pd.DataFrame,
    actual_encoded: np.ndarray,
    reconstructed_encoded: np.ndarray,
    fold_data: FoldPreprocessedData,
) -> dict[str, Any]:
    """Score the 19 source features with equal feature-level contribution."""

    preprocessor = fold_data.preprocessor
    rows: list[dict[str, Any]] = []
    numeric_features = [
        feature for feature in NUMERIC_FEATURES if feature in fold_data.feature_order
    ]
    categorical_features = [
        feature for feature in CATEGORICAL_FEATURES if feature in fold_data.feature_order
    ]

    if numeric_features:
        numeric_slice = preprocessor.output_indices_["num"]
        numeric_reconstructed = reconstructed_encoded[:, numeric_slice]
        numeric_actual_encoded = actual_encoded[:, numeric_slice]
        numeric_pipeline = preprocessor.named_transformers_["num"]
        reconstructed_raw = numeric_pipeline.named_steps["scaler"].inverse_transform(
            numeric_reconstructed
        )
        for index, feature in enumerate(numeric_features):
            actual_raw = pd.to_numeric(
                validation_frame[feature], errors="coerce"
            ).to_numpy(dtype=float)
            observed = np.isfinite(actual_raw)
            predicted_raw = reconstructed_raw[:, index]
            raw_error = predicted_raw[observed] - actual_raw[observed]
            encoded_error = (
                numeric_reconstructed[:, index] - numeric_actual_encoded[:, index]
            )
            feature_type = "boolean" if feature in BOOLEAN_FEATURES else "numeric"
            row: dict[str, Any] = {
                "source_feature": feature,
                "feature_type": feature_type,
                "observed_rows": int(observed.sum()),
                "missing_rows": int((~observed).sum()),
                "mae": float(np.mean(np.abs(raw_error))) if observed.any() else None,
                "mse": float(np.mean(np.square(raw_error))) if observed.any() else None,
                "balanced_mae": float(np.mean(np.abs(encoded_error))),
                "balanced_mse": float(np.mean(np.square(encoded_error))),
                "exact_accuracy": None,
                "top2_accuracy": None,
                "unknown_validation_rows": 0,
                "confusion_matrix": [],
            }
            if feature_type == "boolean" and observed.any():
                actual_labels = (actual_raw[observed] >= 0.5).astype(int)
                predicted_labels = (predicted_raw[observed] >= 0.5).astype(int)
                row["exact_accuracy"] = float(
                    np.mean(actual_labels == predicted_labels)
                )
                row["confusion_matrix"] = [
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(
                            np.sum(
                                (actual_labels == actual)
                                & (predicted_labels == predicted)
                            )
                        ),
                    }
                    for actual in (0, 1)
                    for predicted in (0, 1)
                ]
            rows.append(row)

    if categorical_features:
        categorical_slice = preprocessor.output_indices_["cat"]
        categorical_reconstructed = reconstructed_encoded[:, categorical_slice]
        categorical_pipeline = preprocessor.named_transformers_["cat"]
        imputed_actual = categorical_pipeline.named_steps["imputer"].transform(
            validation_frame[categorical_features]
        )
        encoder = categorical_pipeline.named_steps["encoder"]
        offset = 0
        for feature_index, (feature, categories) in enumerate(
            zip(categorical_features, encoder.categories_, strict=True)
        ):
            width = len(categories)
            block = categorical_reconstructed[:, offset : offset + width]
            actual_values = np.asarray(imputed_actual[:, feature_index], dtype=str)
            category_values = np.asarray(categories, dtype=str)
            actual_positions = np.array(
                [
                    int(np.flatnonzero(category_values == value)[0])
                    if np.any(category_values == value)
                    else -1
                    for value in actual_values
                ],
                dtype=int,
            )
            predicted_positions = np.argmax(block, axis=1)
            top_count = min(2, width)
            top_positions = np.argpartition(
                block, kth=width - top_count, axis=1
            )[:, -top_count:]
            known = actual_positions >= 0
            exact = known & (predicted_positions == actual_positions)
            top2 = known & np.array(
                [
                    actual_position in predicted
                    for actual_position, predicted in zip(
                        actual_positions, top_positions, strict=True
                    )
                ]
            )
            predicted_values = category_values[predicted_positions]
            labels = sorted(
                set(actual_values.tolist()) | set(predicted_values.tolist())
            )
            confusion = []
            for actual in labels:
                for predicted in labels:
                    count = int(
                        np.sum(
                            (actual_values == actual)
                            & (predicted_values == predicted)
                        )
                    )
                    if count:
                        confusion.append(
                            {
                                "actual": actual,
                                "predicted": predicted,
                                "count": count,
                            }
                        )
            error_rate = 1.0 - float(np.mean(exact))
            rows.append(
                {
                    "source_feature": feature,
                    "feature_type": "categorical",
                    "observed_rows": len(actual_values),
                    "missing_rows": int(validation_frame[feature].isna().sum()),
                    "mae": error_rate,
                    "mse": error_rate,
                    "balanced_mae": error_rate,
                    "balanced_mse": error_rate,
                    "exact_accuracy": float(np.mean(exact)),
                    "top2_accuracy": float(np.mean(top2)),
                    "unknown_validation_rows": int((~known).sum()),
                    "confusion_matrix": confusion,
                }
            )
            offset += width

    ordered_rows = sorted(rows, key=lambda row: row["balanced_mse"], reverse=True)
    for rank, row in enumerate(ordered_rows, start=1):
        row["reconstruction_rank"] = rank
    by_type = []
    for feature_type in ("numeric", "boolean", "categorical"):
        selected = [row for row in rows if row["feature_type"] == feature_type]
        if selected:
            by_type.append(
                {
                    "feature_type": feature_type,
                    "features": len(selected),
                    "balanced_mae": float(
                        np.mean([row["balanced_mae"] for row in selected])
                    ),
                    "balanced_mse": float(
                        np.mean([row["balanced_mse"] for row in selected])
                    ),
                }
            )
    return {
        "features": ordered_rows,
        "balanced_source_mae": float(
            np.mean([row["balanced_mae"] for row in rows])
        ),
        "balanced_source_mse": float(
            np.mean([row["balanced_mse"] for row in rows])
        ),
        "by_feature_type": by_type,
        "definition": (
            "Each source feature receives one equal-weight loss: standardized error for "
            "numeric/boolean features and classification error for categorical features."
        ),
    }


def grouped_split_definitions(
    data: CanonicalRepresentationData,
    folds: int = GROUP_FOLDS,
) -> list[dict[str, Any]]:
    groups = data.cohort[GROUP_COLUMN].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    definitions: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(data.matrix, groups=groups)
    ):
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(f"Grouped split leakage in fold {fold}: {len(overlap)} groups.")
        definitions.append(
            {
                "fold": fold,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "train_row_ids": [data.row_ids[index] for index in train_indices],
                "validation_row_ids": [data.row_ids[index] for index in validation_indices],
                "train_configurations": len(train_groups),
                "validation_configurations": len(validation_groups),
                "group_overlap": 0,
            }
        )
    return definitions


def grouped_partition_definitions(
    data: CanonicalRepresentationData,
    *,
    partition_seed: int,
    folds: int = GROUP_FOLDS,
) -> list[dict[str, Any]]:
    """Create one independent shuffled grouped partition with zero leakage."""

    groups = data.cohort[GROUP_COLUMN].astype(str).to_numpy()
    splitter = GroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=partition_seed,
    )
    definitions = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(data.matrix, groups=groups)
    ):
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        if train_groups & validation_groups:
            raise RuntimeError(
                f"Grouped partition leakage in seed {partition_seed}, fold {fold}."
            )
        definitions.append(
            {
                "partition_seed": partition_seed,
                "fold": fold,
                "train_indices": train_indices.tolist(),
                "validation_indices": validation_indices.tolist(),
                "train_row_ids": [data.row_ids[index] for index in train_indices],
                "validation_row_ids": [
                    data.row_ids[index] for index in validation_indices
                ],
                "train_configurations": len(train_groups),
                "validation_configurations": len(validation_groups),
                "group_overlap": 0,
            }
        )
    return definitions


def split_metadata(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize split membership without duplicating positional indices."""

    return [
        {
            "fold": item["fold"],
            "train_row_ids": item["train_row_ids"],
            "validation_row_ids": item["validation_row_ids"],
            "train_configurations": item["train_configurations"],
            "validation_configurations": item["validation_configurations"],
            "group_overlap": item["group_overlap"],
        }
        for item in definitions
    ]


def reconstruction_metrics(
    actual: np.ndarray,
    reconstructed: np.ndarray,
    encoded_feature_names: list[str],
) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=float)
    reconstructed_array = np.asarray(reconstructed, dtype=float)
    if actual_array.shape != reconstructed_array.shape:
        raise ValueError("Actual and reconstructed matrices must have identical shapes.")
    error = reconstructed_array - actual_array
    encoded_rows = []
    for index, encoded_name in enumerate(encoded_feature_names):
        source = source_feature_for_encoded(encoded_name)
        encoded_rows.append(
            {
                "encoded_feature": encoded_name,
                "source_feature": source,
                "feature_type": "categorical" if source in CATEGORICAL_FEATURES else "numeric",
                "mse": float(np.mean(np.square(error[:, index]))),
                "mae": float(np.mean(np.abs(error[:, index]))),
            }
        )
    encoded = pd.DataFrame(encoded_rows)
    source = (
        encoded.groupby(["source_feature", "feature_type"], as_index=False)
        .agg(mse=("mse", "mean"), mae=("mae", "mean"), encoded_columns=("encoded_feature", "count"))
        .sort_values("mse", ascending=False)
    )
    by_type = (
        encoded.groupby("feature_type", as_index=False)
        .agg(mse=("mse", "mean"), mae=("mae", "mean"), encoded_columns=("encoded_feature", "count"))
    )
    return {
        "mse": float(mean_squared_error(actual_array, reconstructed_array)),
        "mae": float(mean_absolute_error(actual_array, reconstructed_array)),
        "by_encoded_feature": encoded.to_dict("records"),
        "by_source_feature": source.to_dict("records"),
        "by_feature_type": by_type.to_dict("records"),
    }


def matched_pca_reconstruction(
    data: CanonicalRepresentationData,
    dimensions: Iterable[int] = LATENT_DIMENSIONS,
) -> list[dict[str, Any]]:
    """Score the preserved full-cohort PCA transform at matched dimensions."""

    full = PCA(n_components=min(data.matrix.shape), random_state=42).fit(data.matrix)
    splits = grouped_split_definitions(data)
    rows = []
    for dimension in dimensions:
        coordinates = full.transform(data.matrix)[:, :dimension]
        reconstructed = coordinates @ full.components_[:dimension] + full.mean_
        metrics = reconstruction_metrics(data.matrix, reconstructed, data.encoded_feature_names)
        fold_metrics = []
        for split in splits:
            validation = np.asarray(split["validation_indices"])
            fold = reconstruction_metrics(
                data.matrix[validation],
                reconstructed[validation],
                data.encoded_feature_names,
            )
            fold_metrics.append(
                {"fold": split["fold"], "mse": fold["mse"], "mae": fold["mae"]}
            )
        rows.append(
            {
                "method": "PCA",
                "latent_dimension": int(dimension),
                "evaluation_note": (
                    "validation rows scored in grouped folds using the preserved full-cohort "
                    "PCA basis; PCA was not refit"
                ),
                "folds": fold_metrics,
                "validation_mse_mean": float(np.mean([row["mse"] for row in fold_metrics])),
                "validation_mae_mean": float(np.mean([row["mae"] for row in fold_metrics])),
                **metrics,
            }
        )
    return rows


def clustering_evaluation(
    embedding: np.ndarray,
    *,
    seed: int = 42,
    k_values: Iterable[int] = range(2, 11),
) -> dict[str, Any]:
    coordinates = np.asarray(embedding, dtype=float)
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(coordinates)
        counts = np.bincount(labels, minlength=k)
        labels_by_k[k] = labels
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(coordinates, labels, sample_size=min(4000, len(labels)), random_state=seed)),
                "davies_bouldin": float(davies_bouldin_score(coordinates, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(coordinates, labels)),
                "minimum_cluster_rows": int(counts.min()),
                "maximum_cluster_rows": int(counts.max()),
                "size_balance": float(counts.min() / counts.max()),
            }
        )
    best = max(rows, key=lambda row: row["silhouette"])
    return {
        "procedure": "k-means with n_init=20",
        "seed": seed,
        "scores": rows,
        "best_k": best["k"],
        "best_silhouette": best["silhouette"],
        "best_labels": labels_by_k[best["k"]].tolist(),
    }


def _neighbor_overlap(left: np.ndarray, right: np.ndarray, neighbors: int = 10) -> float:
    count = min(neighbors + 1, len(left))
    left_neighbors = NearestNeighbors(n_neighbors=count).fit(left).kneighbors(return_distance=False)
    right_neighbors = NearestNeighbors(n_neighbors=count).fit(right).kneighbors(return_distance=False)
    values = []
    for left_row, right_row in zip(left_neighbors, right_neighbors, strict=True):
        left_set = set(left_row[1:])
        right_set = set(right_row[1:])
        values.append(len(left_set & right_set) / max(1, len(left_set | right_set)))
    return float(np.mean(values))


def embedding_stability(embeddings: list[np.ndarray], sample_size: int = 2_000) -> dict[str, Any]:
    """Compare latent geometry; align coordinates before coordinate similarity."""

    if len(embeddings) < 2:
        return {"pairs": [], "mean_score": None}
    sample = np.linspace(0, len(embeddings[0]) - 1, min(sample_size, len(embeddings[0])), dtype=int)
    pairs = []
    for left_index in range(len(embeddings)):
        for right_index in range(left_index + 1, len(embeddings)):
            left = np.asarray(embeddings[left_index])[sample]
            right = np.asarray(embeddings[right_index])[sample]
            left_centered = left - left.mean(axis=0)
            right_centered = right - right.mean(axis=0)
            rotation, _ = orthogonal_procrustes(right_centered, left_centered)
            aligned = right_centered @ rotation
            coordinate_similarity = float(
                np.mean(
                    [
                        abs(pearsonr(left_centered[:, column], aligned[:, column]).statistic)
                        for column in range(left.shape[1])
                    ]
                )
            )
            distance_correlation = float(
                spearmanr(pdist(left_centered), pdist(right_centered)).statistic
            )
            neighbor_overlap = _neighbor_overlap(left_centered, aligned)
            score = float(np.mean([coordinate_similarity, distance_correlation, neighbor_overlap]))
            pairs.append(
                {
                    "left_run": left_index,
                    "right_run": right_index,
                    "procrustes_coordinate_similarity": coordinate_similarity,
                    "pairwise_distance_spearman": distance_correlation,
                    "nearest_neighbor_overlap": neighbor_overlap,
                    "stability_score": score,
                }
            )
    return {
        "alignment": "orthogonal Procrustes after centering",
        "pairs": pairs,
        "mean_score": float(np.mean([row["stability_score"] for row in pairs])),
    }


def outcome_overlay_evaluation(
    data: CanonicalRepresentationData,
    embedding: np.ndarray,
    cluster_labels: Iterable[int],
) -> dict[str, Any]:
    coordinates = np.asarray(embedding, dtype=float)
    labels = np.asarray(list(cluster_labels), dtype=int)
    result: dict[str, Any] = {}
    split_definitions = grouped_split_definitions(data)
    for target in OUTCOME_TARGETS:
        values = pd.to_numeric(data.cohort.get(target), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        associations = []
        quantiles = []
        for dimension in range(coordinates.shape[1]):
            associations.append(
                {
                    "dimension": dimension + 1,
                    "pearson": float(pearsonr(coordinates[valid, dimension], values[valid]).statistic),
                    "spearman": float(spearmanr(coordinates[valid, dimension], values[valid]).statistic),
                }
            )
            bins = pd.qcut(coordinates[valid, dimension], q=5, duplicates="drop")
            summary = (
                pd.DataFrame({"bin": bins.astype(str), "target": values[valid]})
                .groupby("bin", observed=True, sort=False)["target"]
                .agg(["count", "mean", "median"])
                .reset_index()
            )
            for row in summary.to_dict("records"):
                quantiles.append({"dimension": dimension + 1, **row})
        cluster_summary = (
            pd.DataFrame({"cluster": labels[valid], "target": values[valid]})
            .groupby("cluster")["target"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .to_dict("records")
        )
        probe_rows = []
        for split in split_definitions:
            train = np.array(split["train_indices"])
            validation = np.array(split["validation_indices"])
            train = train[np.isfinite(values[train])]
            validation = validation[np.isfinite(values[validation])]
            model = Ridge(alpha=1.0).fit(coordinates[train], values[train])
            prediction = model.predict(coordinates[validation])
            probe_rows.append(
                {
                    "fold": split["fold"],
                    "rows": len(validation),
                    "mae": float(mean_absolute_error(values[validation], prediction)),
                    "r2": float(r2_score(values[validation], prediction)),
                }
            )
        neighbors = NearestNeighbors(n_neighbors=min(11, valid.sum())).fit(coordinates[valid])
        neighbor_indices = neighbors.kneighbors(return_distance=False)[:, 1:]
        valid_targets = values[valid]
        neighbor_target = valid_targets[neighbor_indices].mean(axis=1)
        result[target] = {
            "usable_rows": int(valid.sum()),
            "associations": associations,
            "strongest_dimension": max(
                associations,
                key=lambda row: max(abs(row["pearson"]), abs(row["spearman"])),
            )["dimension"],
            "dimension_quantiles": quantiles,
            "cluster_summary": cluster_summary,
            "nearest_neighbor_target_correlation": float(
                spearmanr(valid_targets, neighbor_target).statistic
            ),
            "probe": {
                "model": "ridge regression (alpha=1.0), evaluation only",
                "grouping": GROUP_COLUMN,
                "folds": probe_rows,
                "mean_mae": float(np.mean([row["mae"] for row in probe_rows])),
                "mean_r2": float(np.mean([row["r2"] for row in probe_rows])),
            },
        }
    return result


def software_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    import platform
    from importlib.metadata import version

    names = ("numpy", "pandas", "scikit-learn", "scipy")
    values = {"python": platform.python_version()}
    values.update({name: version(name) for name in names})
    if extra:
        values.update(extra)
    return values


def artifact_common_metadata(
    data: CanonicalRepresentationData,
    splits: list[dict[str, Any]],
    *,
    method: str,
    source_dump: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": method,
        "source_dump": source_dump,
        "cohort_hash": data.cohort_hash,
        "row_key_hash": data.row_key_hash,
        "basis_row_identifiers": data.row_ids,
        "cohort_rows": len(data.cohort),
        "configurations": int(data.cohort[GROUP_COLUMN].nunique()),
        "feature_order": list(PCA_FEATURES),
        "encoded_feature_order": data.encoded_feature_names,
        "target_metrics_in_inputs": [],
        "preprocessing": {
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_missing_values": "median imputation",
            "numeric_scaling": "standard scaling",
            "categorical_missing_values": "most-frequent imputation",
            "categorical_encoding": "one-hot, handle_unknown=ignore, max_categories=30",
            "categorical_reconstruction": (
                "joint MSE on one-hot columns; outputs are continuous and are not guaranteed "
                "to be mutually exclusive categorical probabilities"
            ),
        },
        "split_definitions": split_metadata(splits),
        "grouped_validation": {"column": GROUP_COLUMN, "folds": len(splits), "group_overlap": 0},
        "random_seeds": [seed],
    }


def validate_representation_artifact(
    artifact: dict[str, Any],
    *,
    expected_method: str | None = None,
    expected_cohort_hash: str | None = None,
) -> None:
    required = {
        "schema_version",
        "method",
        "source_dump",
        "cohort_hash",
        "row_key_hash",
        "feature_order",
        "encoded_feature_order",
        "target_metrics_in_inputs",
        "split_definitions",
        "random_seeds",
        "software_versions",
    }
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError("Representation artifact is missing: " + ", ".join(missing))
    if artifact["schema_version"] != REPRESENTATION_SCHEMA_VERSION:
        raise ValueError("Representation artifact schema version is incompatible.")
    if artifact["source_dump"] != SOURCE_DUMP_VERSION:
        raise ValueError("Representation artifact source dump is incompatible.")
    if artifact["feature_order"] != list(PCA_FEATURES):
        raise ValueError("Representation artifact feature order is incompatible.")
    if artifact["target_metrics_in_inputs"]:
        raise ValueError("Representation artifact reports outcome leakage.")
    if expected_method and artifact["method"] != expected_method:
        raise ValueError(f"Expected {expected_method} artifact, found {artifact['method']}.")
    if expected_cohort_hash and artifact["cohort_hash"] != expected_cohort_hash:
        raise ValueError("Representation artifacts were built from different cohorts.")
    for split in artifact["split_definitions"]:
        if split.get("group_overlap") != 0:
            raise ValueError("Representation artifact contains grouped validation leakage.")


def load_representation_artifact(
    path: str | Path,
    *,
    expected_method: str | None = None,
    expected_cohort_hash: str | None = None,
) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_representation_artifact(
        artifact,
        expected_method=expected_method,
        expected_cohort_hash=expected_cohort_hash,
    )
    return artifact


def validate_final_representation_artifact(
    artifact: dict[str, Any],
    *,
    artifact_path: str | Path,
    expected_method: str | None = None,
    expected_cohort_hash: str | None = None,
    validate_companion: bool = True,
) -> Path:
    required = {
        "schema_version",
        "method",
        "source_dump",
        "cohort_hash",
        "row_key_hash",
        "cohort_rows",
        "feature_order",
        "encoded_feature_order",
        "target_metrics_in_inputs",
        "split_definitions",
        "random_seeds",
        "architecture",
        "hyperparameters",
        "runs",
        "summary",
        "embedding_companion",
        "software_versions",
    }
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError("Final representation artifact is missing: " + ", ".join(missing))
    if artifact["schema_version"] != FINAL_REPRESENTATION_SCHEMA_VERSION:
        raise ValueError("Final representation artifact schema version is incompatible.")
    if artifact["source_dump"] != SOURCE_DUMP_VERSION:
        raise ValueError("Final representation artifact source dump is incompatible.")
    if artifact["feature_order"] != list(PCA_FEATURES):
        raise ValueError("Final representation artifact feature order is incompatible.")
    if artifact["target_metrics_in_inputs"]:
        raise ValueError("Final representation artifact reports outcome leakage.")
    if artifact["random_seeds"] != list(RANDOM_SEEDS):
        raise ValueError("Final representation artifact does not contain the fixed seeds.")
    if expected_method and artifact["method"] != expected_method:
        raise ValueError(f"Expected {expected_method} artifact, found {artifact['method']}.")
    if expected_cohort_hash and artifact["cohort_hash"] != expected_cohort_hash:
        raise ValueError("Final representation artifacts were built from different cohorts.")
    for split in artifact["split_definitions"]:
        if split.get("group_overlap") != 0:
            raise ValueError("Final representation artifact contains grouped validation leakage.")
    companion = artifact["embedding_companion"]
    if companion.get("format") != "parquet":
        raise ValueError("Final embedding companion must use Parquet.")
    companion_path = Path(artifact_path).parent / companion["filename"]
    if validate_companion:
        if not companion_path.exists():
            raise FileNotFoundError(
                f"Embedding companion is missing: {companion_path.name}. "
                "Restore the matching Parquet artifact before loading this representation."
            )
        digest = hashlib.sha256(companion_path.read_bytes()).hexdigest()
        if digest != companion.get("sha256"):
            raise ValueError("Embedding companion checksum does not match its JSON metadata.")
        frame = pd.read_parquet(companion_path)
        if len(frame) != companion.get("rows"):
            raise ValueError("Embedding companion row count does not match its JSON metadata.")
        if sorted(frame["seed"].unique().tolist()) != list(RANDOM_SEEDS):
            raise ValueError("Embedding companion seeds do not match the fixed final seeds.")
        for seed in RANDOM_SEEDS:
            seed_rows = frame.loc[frame["seed"].eq(seed), "row_id"].astype(str).tolist()
            if len(seed_rows) != artifact["cohort_rows"] or _sequence_hash(seed_rows) != artifact["row_key_hash"]:
                raise ValueError("Embedding companion row alignment is incompatible.")
    return companion_path


def load_final_representation_artifact(
    path: str | Path,
    *,
    expected_method: str | None = None,
    expected_cohort_hash: str | None = None,
    validate_companion: bool = True,
) -> tuple[dict[str, Any], Path]:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    companion_path = validate_final_representation_artifact(
        artifact,
        artifact_path=artifact_path,
        expected_method=expected_method,
        expected_cohort_hash=expected_cohort_hash,
        validate_companion=validate_companion,
    )
    return artifact, companion_path


def write_json_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
