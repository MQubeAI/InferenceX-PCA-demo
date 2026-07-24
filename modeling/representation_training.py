"""Bounded Stage 2 training orchestration for AE and VAE artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from modeling.neural_representation import train_neural_fold
from modeling.representation_analysis import (
    LATENT_DIMENSIONS,
    OUTCOME_TARGETS,
    RANDOM_SEEDS,
    SOURCE_DUMP_VERSION,
    artifact_common_metadata,
    canonical_representation_data,
    clustering_evaluation,
    embedding_stability,
    grouped_split_definitions,
    matched_pca_reconstruction,
    outcome_overlay_evaluation,
    reconstruction_metrics,
    software_versions,
    write_json_artifact,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _representative_configurations(data: Any, embedding: np.ndarray) -> list[dict[str, Any]]:
    columns = ["config_id", "benchmark_type", "isl", "osl", "conc"]
    rows = []
    for dimension in range(embedding.shape[1]):
        order = np.argsort(embedding[:, dimension])
        for region, indices in (("low", order[:3]), ("high", order[-3:][::-1])):
            for rank, index in enumerate(indices, start=1):
                record = {
                    "dimension": dimension + 1,
                    "region": region,
                    "rank": rank,
                    "row_id": data.row_ids[index],
                    "latent_value": float(embedding[index, dimension]),
                }
                record.update(
                    {
                        column: _json_safe(data.cohort.iloc[index][column])
                        for column in columns
                    }
                )
                rows.append(record)
    return rows


def _embedding_records(data: Any, embedding: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for index, coordinate in enumerate(embedding):
        row = {
            "row_id": data.row_ids[index],
            "config_id": _json_safe(data.cohort.iloc[index]["config_id"]),
            "isl": _json_safe(data.cohort.iloc[index]["isl"]),
            "osl": _json_safe(data.cohort.iloc[index]["osl"]),
            "conc": _json_safe(data.cohort.iloc[index]["conc"]),
        }
        row.update({f"z{dimension + 1}": float(value) for dimension, value in enumerate(coordinate)})
        for target in OUTCOME_TARGETS:
            row[target] = _json_safe(data.cohort.iloc[index].get(target))
        records.append(row)
    return records


def run_screening_experiment(
    aggregate: Any,
    *,
    method: str,
    output_path: str | Path,
    weights_path: str | Path,
    seed: int = RANDOM_SEEDS[0],
    latent_dimensions: tuple[int, ...] = LATENT_DIMENSIONS,
    hidden_dimensions: tuple[int, int] = (64, 32),
    beta: float = 1.0,
    maximum_epochs: int = 150,
    patience: int = 12,
) -> dict[str, Any]:
    """Run one seed over the bounded dimension grid and serialize the artifact."""

    experiment_started = time.perf_counter()
    data = canonical_representation_data(aggregate)
    splits = grouped_split_definitions(data)
    results: dict[int, list[dict[str, Any]]] = {}
    screening_rows = []
    for dimension in latent_dimensions:
        fold_results = []
        for split in splits:
            trained = train_neural_fold(
                data.matrix,
                split["train_indices"],
                split["validation_indices"],
                method=method,
                latent_dimension=dimension,
                seed=seed,
                hidden_dimensions=hidden_dimensions,
                beta=beta,
                maximum_epochs=maximum_epochs,
                patience=patience,
            )
            validation = np.asarray(split["validation_indices"])
            metrics = reconstruction_metrics(
                data.matrix[validation],
                trained["reconstructed"][validation],
                data.encoded_feature_names,
            )
            trained["validation_metrics"] = metrics
            trained["fold"] = split["fold"]
            fold_results.append(trained)
        results[dimension] = fold_results
        stability = embedding_stability([result["embedding"] for result in fold_results])
        screening_rows.append(
            {
                "latent_dimension": dimension,
                "validation_mse_mean": float(
                    np.mean([result["validation_metrics"]["mse"] for result in fold_results])
                ),
                "validation_mse_std": float(
                    np.std([result["validation_metrics"]["mse"] for result in fold_results])
                ),
                "validation_mae_mean": float(
                    np.mean([result["validation_metrics"]["mae"] for result in fold_results])
                ),
                "validation_mae_std": float(
                    np.std([result["validation_metrics"]["mae"] for result in fold_results])
                ),
                "stability_score": stability["mean_score"],
                "parameter_count": fold_results[0]["parameter_count"],
                "runtime_seconds": float(sum(result["runtime_seconds"] for result in fold_results)),
                "fold_stability": stability,
            }
        )

    best_mse = min(row["validation_mse_mean"] for row in screening_rows)
    near_best = [
        row for row in screening_rows if row["validation_mse_mean"] <= best_mse * 1.01
    ]
    selected_row = min(
        near_best,
        key=lambda row: (row["latent_dimension"], -float(row["stability_score"] or 0)),
    )
    selected_dimension = int(selected_row["latent_dimension"])
    selected_folds = results[selected_dimension]
    representative = selected_folds[0]
    embedding = representative["embedding"]
    clustering = clustering_evaluation(embedding, seed=seed)
    original_clustering = clustering_evaluation(data.matrix, seed=seed)
    outcomes = outcome_overlay_evaluation(data, embedding, clustering["best_labels"])
    selected_validation = []
    for result in selected_folds:
        selected_validation.append(
            {
                "fold": result["fold"],
                "epochs_trained": result["epochs_trained"],
                "best_epoch": result["best_epoch"],
                "runtime_seconds": result["runtime_seconds"],
                "inference_seconds": result["inference_seconds"],
                "mse": result["validation_metrics"]["mse"],
                "mae": result["validation_metrics"]["mae"],
                "by_source_feature": result["validation_metrics"]["by_source_feature"],
                "by_feature_type": result["validation_metrics"]["by_feature_type"],
                "history": result["history"],
                "diagnostics": result["diagnostics"],
            }
        )

    import torch

    weights = Path(weights_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(representative["state_dict"], weights)
    weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    common = artifact_common_metadata(
        data,
        splits,
        method=method,
        source_dump=SOURCE_DUMP_VERSION,
        seed=seed,
    )
    artifact = {
        **common,
        "experiment_stage": "Stage 2 bounded screening",
        "status": "preliminary",
        "architecture": {
            "input_dimension": int(data.matrix.shape[1]),
            "hidden_dimensions": list(hidden_dimensions),
            "latent_dimension": selected_dimension,
            "decoder": list(reversed(hidden_dimensions)),
            "activation": "ReLU",
            "variational": method == "variational_autoencoder",
        },
        "hyperparameters": {
            "latent_dimensions_screened": list(latent_dimensions),
            "beta": beta if method == "variational_autoencoder" else None,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 256,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": patience,
            "early_stopping_minimum_delta": 1e-5,
            "selection_rule": (
                "smallest latent dimension within 1% of minimum mean grouped-validation "
                "reconstruction MSE; outcomes are not used"
            ),
        },
        "loss": (
            "mean squared reconstruction error"
            if method == "autoencoder"
            else f"mean squared reconstruction error + beta ({beta}) * mean KL divergence"
        ),
        "screening": screening_rows,
        "selected_result": {
            "latent_dimension": selected_dimension,
            "selection_basis": selected_row,
            "validation_folds": selected_validation,
            "validation_mse_mean": selected_row["validation_mse_mean"],
            "validation_mae_mean": selected_row["validation_mae_mean"],
            "stability": selected_row["fold_stability"],
            "parameter_count": representative["parameter_count"],
            "representative_fold": 0,
            "representative_embedding_scope": (
                "all cohort rows transformed by the fold-0 model; fold-0 validation "
                "configurations were not used for fitting"
            ),
            "diagnostics": representative["diagnostics"],
            "clustering": clustering,
            "original_feature_space_clustering": original_clustering,
            "outcome_overlays": outcomes,
            "representative_configurations": _representative_configurations(data, embedding),
            "embedding": _embedding_records(data, embedding),
        },
        "matched_dimension_pca": matched_pca_reconstruction(data),
        "weights": {
            "path": str(weights),
            "sha256": weights_hash,
            "bytes": weights.stat().st_size,
            "representative_fold": 0,
        },
        "software_versions": software_versions({"torch": representative["torch_version"]}),
        "computational_cost": {
            "screening_runtime_seconds": time.perf_counter() - experiment_started,
            "selected_parameter_count": representative["parameter_count"],
            "selected_transform_seconds": representative["inference_seconds"],
        },
    }
    safe_artifact = _json_safe(artifact)
    write_json_artifact(output_path, safe_artifact)
    safe_artifact["computational_cost"]["artifact_size_bytes"] = Path(output_path).stat().st_size
    write_json_artifact(output_path, safe_artifact)
    return safe_artifact

