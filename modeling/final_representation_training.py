"""Bounded Stage 3 multi-seed representation training and VAE beta diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from modeling.neural_representation import train_neural_fold
from modeling.pca_target_analysis import PCA_FEATURES
from modeling.pca_target_analysis import source_feature_for_encoded
from modeling.representation_analysis import (
    FINAL_REPRESENTATION_SCHEMA_VERSION,
    OUTCOME_TARGETS,
    RANDOM_SEEDS,
    SOURCE_DUMP_VERSION,
    artifact_common_metadata,
    canonical_representation_data,
    clustering_evaluation,
    embedding_stability,
    grouped_split_definitions,
    outcome_overlay_evaluation,
    reconstruction_metrics,
    software_versions,
    write_json_artifact,
)
from modeling.representation_training import (
    _json_safe,
    _representative_configurations,
)

FINAL_LATENT_DIMENSION = 15
FINAL_HIDDEN_DIMENSIONS = (64, 32)
DEFENSIBLE_ACTIVE_DIMENSIONS = 5


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _best_history_row(result: dict[str, Any]) -> dict[str, Any]:
    return result["history"][result["best_epoch"] - 1]


def _fold_record(
    result: dict[str, Any],
    *,
    seed: int,
    fold: int,
) -> dict[str, Any]:
    best = _best_history_row(result)
    return {
        "seed": seed,
        "fold": fold,
        "mse": result["validation_metrics"]["mse"],
        "mae": result["validation_metrics"]["mae"],
        "epochs_trained": result["epochs_trained"],
        "best_epoch": result["best_epoch"],
        "early_stopping_occurred": result["early_stopping_occurred"],
        "stopping_reason": result["stopping_reason"],
        "best_train_reconstruction": best["train_reconstruction"],
        "best_validation_reconstruction": best["validation_reconstruction"],
        "train_validation_gap": best["validation_reconstruction"] - best["train_reconstruction"],
        "best_train_kl": best["train_kl"],
        "best_validation_kl": best["validation_kl"],
        "best_validation_total": best["validation_total"],
        "runtime_seconds": result["runtime_seconds"],
        "transformation_seconds": result["inference_seconds"],
        "parameter_count": result["parameter_count"],
        "history": result["history"],
        "diagnostics": result["diagnostics"],
        "reconstruction_by_source_feature": result["validation_metrics"]["by_source_feature"],
        "reconstruction_by_feature_type": result["validation_metrics"]["by_feature_type"],
    }


def _aggregate_source_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for feature in record["reconstruction_by_source_feature"]:
            rows.append(
                {
                    "seed": record["seed"],
                    "fold": record["fold"],
                    **feature,
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["source_feature", "feature_type", "encoded_columns"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_standard_deviation=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_standard_deviation=("mae", "std"),
        )
        .sort_values("mse_mean", ascending=False)
    )
    return _json_safe(summary.fillna(0.0).to_dict("records"))


def _latent_traversal_summary(
    states: dict[str, Any],
    encoded_feature_names: list[str],
) -> list[dict[str, Any]]:
    """Summarize decoder change for a -1 to +1 traversal from latent zero."""

    rows = []
    for run_name, state in states.items():
        first_weight = state["decoder.0.weight"].detach().cpu().numpy()
        first_bias = state["decoder.0.bias"].detach().cpu().numpy()
        second_weight = state["decoder.2.weight"].detach().cpu().numpy()
        second_bias = state["decoder.2.bias"].detach().cpu().numpy()
        output_weight = state["decoder.4.weight"].detach().cpu().numpy()
        output_bias = state["decoder.4.bias"].detach().cpu().numpy()
        latent_dimension = first_weight.shape[1]

        def decode(latent: np.ndarray) -> np.ndarray:
            hidden = np.maximum(0.0, latent @ first_weight.T + first_bias)
            hidden = np.maximum(0.0, hidden @ second_weight.T + second_bias)
            return hidden @ output_weight.T + output_bias

        for dimension in range(latent_dimension):
            low = np.zeros((1, latent_dimension), dtype=np.float32)
            high = np.zeros((1, latent_dimension), dtype=np.float32)
            low[0, dimension] = -1.0
            high[0, dimension] = 1.0
            change = (decode(high) - decode(low))[0]
            for encoded_name, signed_change in zip(
                encoded_feature_names, change, strict=True
            ):
                rows.append(
                    {
                        "run": run_name,
                        "dimension": dimension + 1,
                        "source_feature": source_feature_for_encoded(encoded_name),
                        "signed_change": float(signed_change),
                        "absolute_change": abs(float(signed_change)),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["dimension", "source_feature"], as_index=False)
        .agg(
            signed_change_mean=("signed_change", "mean"),
            signed_change_standard_deviation=("signed_change", "std"),
            absolute_change_mean=("absolute_change", "mean"),
            absolute_change_standard_deviation=("absolute_change", "std"),
        )
        .sort_values(["dimension", "absolute_change_mean"], ascending=[True, False])
        .fillna(0.0)
    )
    return _json_safe(summary.to_dict("records"))


def _outcome_summary(seed_outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in OUTCOME_TARGETS:
        probe_rows = []
        neighbor_values = []
        strongest = []
        for seed, outcomes in seed_outcomes.items():
            target_result = outcomes[target]
            neighbor_values.append(target_result["nearest_neighbor_target_correlation"])
            strongest_row = max(
                target_result["associations"],
                key=lambda row: max(abs(row["pearson"]), abs(row["spearman"])),
            )
            strongest.append({"seed": seed, **strongest_row})
            for fold in target_result["probe"]["folds"]:
                probe_rows.append({"seed": seed, **fold})
        result[target] = {
            "probe_model": "ridge regression (alpha=1.0), post-hoc evaluation only",
            "fold_seed_results": probe_rows,
            "r2": _mean_std([row["r2"] for row in probe_rows]),
            "mae": _mean_std([row["mae"] for row in probe_rows]),
            "nearest_neighbor_target_correlation": _mean_std(neighbor_values),
            "strongest_dimensions_by_seed": strongest,
        }
    return result


def _cluster_summary(
    embeddings: dict[int, np.ndarray],
    *,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[int, list[int]]]:
    evaluations = {
        seed: clustering_evaluation(embeddings[seed], seed=seed) for seed in seeds
    }
    by_k = []
    for k in range(2, 11):
        rows = [
            next(row for row in evaluations[seed]["scores"] if row["k"] == k)
            for seed in seeds
        ]
        by_k.append(
            {
                "k": k,
                "silhouette": _mean_std([row["silhouette"] for row in rows]),
                "davies_bouldin": _mean_std([row["davies_bouldin"] for row in rows]),
                "calinski_harabasz": _mean_std([row["calinski_harabasz"] for row in rows]),
                "size_balance": _mean_std([row["size_balance"] for row in rows]),
            }
        )
    selected = max(by_k, key=lambda row: row["silhouette"]["mean"])
    selected_k = selected["k"]
    labels = {
        seed: KMeans(n_clusters=selected_k, random_state=seed, n_init=20)
        .fit_predict(embeddings[seed])
        .tolist()
        for seed in seeds
    }
    agreement = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            agreement.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(labels[left_seed], labels[right_seed])
                    ),
                }
            )
    return (
        {
            "procedure": "k-means, k=2 through 10, n_init=20",
            "selected_k": selected_k,
            "selection_rule": "highest mean silhouette across the three fixed seeds",
            "metrics_by_k": by_k,
            "selected_metrics": selected,
            "seed_best_k": {
                str(seed): evaluations[seed]["best_k"] for seed in seeds
            },
            "cluster_agreement": agreement,
            "cluster_stability_mean_ari": float(
                np.mean([row["adjusted_rand_index"] for row in agreement])
            ),
        },
        labels,
    )


def _write_embedding_companion(
    data: Any,
    embeddings: dict[int, np.ndarray],
    path: Path,
    cluster_labels: dict[int, list[int]],
) -> dict[str, Any]:
    frames = []
    base = pd.DataFrame(
        {
            "row_id": data.row_ids,
            "config_id": data.cohort["config_id"].astype(str).to_numpy(),
            "isl": pd.to_numeric(data.cohort["isl"], errors="coerce").to_numpy(),
            "osl": pd.to_numeric(data.cohort["osl"], errors="coerce").to_numpy(),
            "conc": pd.to_numeric(data.cohort["conc"], errors="coerce").to_numpy(),
        }
    )
    for target in OUTCOME_TARGETS:
        base[target] = pd.to_numeric(data.cohort[target], errors="coerce").to_numpy()
    for seed, embedding in embeddings.items():
        frame = base.copy()
        frame.insert(0, "fold_id", 0)
        frame.insert(0, "seed", seed)
        frame["cluster"] = np.asarray(cluster_labels[seed], dtype=np.int16)
        for dimension in range(embedding.shape[1]):
            frame[f"z{dimension + 1}"] = embedding[:, dimension].astype(np.float32)
        frames.append(frame)
    companion = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    companion.to_parquet(path, index=False, compression="zstd")
    return {
        "filename": path.name,
        "format": "parquet",
        "compression": "zstd",
        "rows": len(companion),
        "columns": companion.columns.tolist(),
        "seeds": list(embeddings),
        "representative_model_fold": 0,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def run_final_experiment(
    aggregate: Any,
    *,
    method: str,
    beta: float,
    output_path: str | Path,
    companion_path: str | Path,
    weights_path: str | Path,
    seeds: tuple[int, ...] = RANDOM_SEEDS,
    maximum_epochs: int,
    patience: int = 12,
) -> dict[str, Any]:
    """Train the fixed Stage 3 architecture across three seeds and three folds."""

    started = time.perf_counter()
    data = canonical_representation_data(aggregate)
    splits = grouped_split_definitions(data)
    all_records: list[dict[str, Any]] = []
    raw_results: dict[int, list[dict[str, Any]]] = {}
    representative_embeddings: dict[int, np.ndarray] = {}
    within_seed_stability: dict[str, Any] = {}
    weights_to_save: dict[str, Any] = {}
    torch_version = ""

    for seed in seeds:
        fold_results = []
        for split in splits:
            result = train_neural_fold(
                data.matrix,
                split["train_indices"],
                split["validation_indices"],
                method=method,
                latent_dimension=FINAL_LATENT_DIMENSION,
                seed=seed,
                hidden_dimensions=FINAL_HIDDEN_DIMENSIONS,
                beta=beta,
                maximum_epochs=maximum_epochs,
                patience=patience,
            )
            validation = np.asarray(split["validation_indices"])
            result["validation_metrics"] = reconstruction_metrics(
                data.matrix[validation],
                result["reconstructed"][validation],
                data.encoded_feature_names,
            )
            result["fold"] = split["fold"]
            fold_results.append(result)
            all_records.append(_fold_record(result, seed=seed, fold=split["fold"]))
            weights_to_save[f"seed_{seed}_fold_{split['fold']}"] = result["state_dict"]
            torch_version = result["torch_version"]
        raw_results[seed] = fold_results
        representative_embeddings[seed] = fold_results[0]["embedding"]
        within_seed_stability[str(seed)] = embedding_stability(
            [result["embedding"] for result in fold_results]
        )

    mse_values = [record["mse"] for record in all_records]
    mae_values = [record["mae"] for record in all_records]
    if not all(np.isfinite(mse_values + mae_values)):
        raise RuntimeError("Non-finite validation error detected; final training is unstable.")
    cross_seed_stability = embedding_stability(
        [representative_embeddings[seed] for seed in seeds]
    )
    clustering, cluster_labels = _cluster_summary(
        representative_embeddings,
        seeds=seeds,
    )
    seed_outcomes = {
        seed: outcome_overlay_evaluation(
            data, representative_embeddings[seed], cluster_labels[seed]
        )
        for seed in seeds
    }

    import torch

    weights = Path(weights_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights_to_save, weights)
    companion_metadata = _write_embedding_companion(
        data, representative_embeddings, Path(companion_path), cluster_labels
    )
    representative = raw_results[seeds[0]][0]
    common = artifact_common_metadata(
        data,
        splits,
        method=method,
        source_dump=SOURCE_DUMP_VERSION,
        seed=seeds[0],
    )
    common["schema_version"] = FINAL_REPRESENTATION_SCHEMA_VERSION
    common["random_seeds"] = list(seeds)
    artifact = {
        **common,
        "experiment_stage": "Stage 3 bounded final experiment",
        "status": "final",
        "architecture": {
            "input_dimension": int(data.matrix.shape[1]),
            "hidden_dimensions": list(FINAL_HIDDEN_DIMENSIONS),
            "latent_dimension": FINAL_LATENT_DIMENSION,
            "decoder": list(reversed(FINAL_HIDDEN_DIMENSIONS)),
            "activation": "ReLU",
            "variational": method == "variational_autoencoder",
        },
        "hyperparameters": {
            "beta": beta if method == "variational_autoencoder" else None,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 256,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": patience,
            "early_stopping_minimum_delta": 1e-5,
            "selection_role": "fixed before final multi-seed training; outcomes excluded",
        },
        "runs": all_records,
        "summary": {
            "validation_mse": _mean_std(mse_values),
            "validation_mae": _mean_std(mae_values),
            "epochs_trained": _mean_std(
                [float(record["epochs_trained"]) for record in all_records]
            ),
            "train_validation_gap": _mean_std(
                [record["train_validation_gap"] for record in all_records]
            ),
            "early_stopping_runs": int(
                sum(record["early_stopping_occurred"] for record in all_records)
            ),
            "maximum_epoch_runs": int(
                sum(not record["early_stopping_occurred"] for record in all_records)
            ),
            "parameter_count": representative["parameter_count"],
            "reconstruction_by_source_feature": _aggregate_source_errors(all_records),
            "within_seed_fold_stability": within_seed_stability,
            "cross_seed_stability": cross_seed_stability,
            "clustering": clustering,
            "outcome_overlays": _outcome_summary(seed_outcomes),
            "diagnostics_by_seed_fold": [
                {
                    "seed": record["seed"],
                    "fold": record["fold"],
                    **record["diagnostics"],
                }
                for record in all_records
            ],
            "decoder_sensitivity": {
                "mean_by_dimension": np.mean(
                    [
                        record["diagnostics"]["decoder_sensitivity_per_dimension"]
                        for record in all_records
                    ],
                    axis=0,
                ).tolist(),
                "standard_deviation_by_dimension": np.std(
                    [
                        record["diagnostics"]["decoder_sensitivity_per_dimension"]
                        for record in all_records
                    ],
                    axis=0,
                    ddof=1,
                ).tolist(),
            },
            "latent_traversals": _latent_traversal_summary(
                weights_to_save, data.encoded_feature_names
            ),
            "representative_configurations": _representative_configurations(
                data, representative_embeddings[seeds[0]]
            ),
        },
        "embedding_companion": companion_metadata,
        "weights": {
            "filename": weights.name,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "bytes": weights.stat().st_size,
            "included_runs": sorted(weights_to_save),
        },
        "software_versions": software_versions(
            {
                "torch": torch_version,
                "pyarrow": importlib.metadata.version("pyarrow"),
            }
        ),
        "computational_cost": {
            "total_runtime_seconds": time.perf_counter() - started,
            "training_runtime_seconds": float(
                sum(record["runtime_seconds"] for record in all_records)
            ),
            "transformation_seconds": float(
                sum(record["transformation_seconds"] for record in all_records)
            ),
            "parameter_count": representative["parameter_count"],
            "companion_size_bytes": companion_metadata["bytes"],
            "weights_size_bytes": weights.stat().st_size,
        },
    }
    safe = _json_safe(artifact)
    write_json_artifact(output_path, safe)
    safe["computational_cost"]["json_artifact_size_bytes"] = Path(output_path).stat().st_size
    write_json_artifact(output_path, safe)
    return safe


def _diagnostic_result_from_new_runs(
    data: Any,
    splits: list[dict[str, Any]],
    *,
    beta: float,
    seed: int,
    maximum_epochs: int,
    patience: int,
) -> dict[str, Any]:
    fold_results = []
    for split in splits:
        result = train_neural_fold(
            data.matrix,
            split["train_indices"],
            split["validation_indices"],
            method="variational_autoencoder",
            latent_dimension=FINAL_LATENT_DIMENSION,
            seed=seed,
            hidden_dimensions=FINAL_HIDDEN_DIMENSIONS,
            beta=beta,
            maximum_epochs=maximum_epochs,
            patience=patience,
        )
        validation = np.asarray(split["validation_indices"])
        result["validation_metrics"] = reconstruction_metrics(
            data.matrix[validation],
            result["reconstructed"][validation],
            data.encoded_feature_names,
        )
        result["fold"] = split["fold"]
        fold_results.append(result)
    representative = fold_results[0]
    clustering = clustering_evaluation(representative["embedding"], seed=seed)
    outcomes = outcome_overlay_evaluation(
        data, representative["embedding"], clustering["best_labels"]
    )
    folds = []
    for result in fold_results:
        best = _best_history_row(result)
        folds.append(
            {
                "fold": result["fold"],
                "mse": result["validation_metrics"]["mse"],
                "mae": result["validation_metrics"]["mae"],
                "kl_loss": best["validation_kl"],
                "total_objective": best["validation_total"],
                "epochs_trained": result["epochs_trained"],
                "active_latent_dimensions": result["diagnostics"]["active_latent_dimensions"],
                "average_kl_per_latent_dimension": result["diagnostics"]["average_kl_per_latent_dimension"],
                "latent_variance": result["diagnostics"]["latent_variance"],
                "posterior_collapse": result["diagnostics"]["posterior_collapse"],
                "decoder_ignores_latent_changes": result["diagnostics"]["decoder_ignores_latent_changes"],
            }
        )
    return {
        "beta": beta,
        "source": "new bounded diagnostic",
        "folds": folds,
        "validation_mse": _mean_std([row["mse"] for row in folds]),
        "validation_mae": _mean_std([row["mae"] for row in folds]),
        "validation_kl": _mean_std([row["kl_loss"] for row in folds]),
        "validation_total_objective": _mean_std([row["total_objective"] for row in folds]),
        "active_latent_dimensions": _mean_std(
            [float(row["active_latent_dimensions"]) for row in folds]
        ),
        "minimum_active_latent_dimensions": min(
            row["active_latent_dimensions"] for row in folds
        ),
        "posterior_collapse_any_fold": any(row["posterior_collapse"] for row in folds),
        "embedding_stability": embedding_stability(
            [result["embedding"] for result in fold_results]
        ),
        "clustering": {
            "best_k": clustering["best_k"],
            "best_silhouette": clustering["best_silhouette"],
            "scores": clustering["scores"],
        },
        "grouped_ridge_probes": {
            target: {
                "mean_r2": outcomes[target]["probe"]["mean_r2"],
                "mean_mae": outcomes[target]["probe"]["mean_mae"],
            }
            for target in OUTCOME_TARGETS
        },
        "runtime_seconds": float(sum(result["runtime_seconds"] for result in fold_results)),
    }


def _diagnostic_result_from_stage2(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    selected = artifact["selected_result"]
    folds = []
    for result in selected["validation_folds"]:
        best = result["history"][result["best_epoch"] - 1]
        diagnostics = result["diagnostics"]
        folds.append(
            {
                "fold": result["fold"],
                "mse": result["mse"],
                "mae": result["mae"],
                "kl_loss": best["validation_kl"],
                "total_objective": best["validation_total"],
                "epochs_trained": result["epochs_trained"],
                "active_latent_dimensions": diagnostics["active_latent_dimensions"],
                "average_kl_per_latent_dimension": diagnostics["average_kl_per_latent_dimension"],
                "latent_variance": diagnostics["latent_variance"],
                "posterior_collapse": diagnostics["posterior_collapse"],
                "decoder_ignores_latent_changes": diagnostics["decoder_ignores_latent_changes"],
            }
        )
    return {
        "beta": 1.0,
        "source": "reused fully comparable Stage 2 seed-42 result",
        "folds": folds,
        "validation_mse": _mean_std([row["mse"] for row in folds]),
        "validation_mae": _mean_std([row["mae"] for row in folds]),
        "validation_kl": _mean_std([row["kl_loss"] for row in folds]),
        "validation_total_objective": _mean_std([row["total_objective"] for row in folds]),
        "active_latent_dimensions": _mean_std(
            [float(row["active_latent_dimensions"]) for row in folds]
        ),
        "minimum_active_latent_dimensions": min(
            row["active_latent_dimensions"] for row in folds
        ),
        "posterior_collapse_any_fold": any(row["posterior_collapse"] for row in folds),
        "embedding_stability": selected["stability"],
        "clustering": {
            "best_k": selected["clustering"]["best_k"],
            "best_silhouette": selected["clustering"]["best_silhouette"],
            "scores": selected["clustering"]["scores"],
        },
        "grouped_ridge_probes": {
            target: {
                "mean_r2": selected["outcome_overlays"][target]["probe"]["mean_r2"],
                "mean_mae": selected["outcome_overlays"][target]["probe"]["mean_mae"],
            }
            for target in OUTCOME_TARGETS
        },
        "runtime_seconds": 0.0,
    }


def run_vae_beta_diagnostic(
    aggregate: Any,
    *,
    stage2_beta1_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    maximum_epochs: int = 150,
    patience: int = 12,
) -> dict[str, Any]:
    """Run only beta 0.1/0.5 and reuse the comparable beta-1 Stage 2 result."""

    started = time.perf_counter()
    data = canonical_representation_data(aggregate)
    splits = grouped_split_definitions(data)
    results = [
        _diagnostic_result_from_new_runs(
            data,
            splits,
            beta=beta,
            seed=seed,
            maximum_epochs=maximum_epochs,
            patience=patience,
        )
        for beta in (0.1, 0.5)
    ]
    results.append(_diagnostic_result_from_stage2(Path(stage2_beta1_path)))
    eligible = [
        row
        for row in results
        if not row["posterior_collapse_any_fold"]
        and row["minimum_active_latent_dimensions"] >= DEFENSIBLE_ACTIVE_DIMENSIONS
    ]
    selected = (
        sorted(
            eligible,
            key=lambda row: (
                -row["minimum_active_latent_dimensions"],
                row["validation_mse"]["mean"],
                -float(row["embedding_stability"]["mean_score"] or 0.0),
            ),
        )[0]
        if eligible
        else None
    )
    artifact = {
        "schema_version": "representation-vae-beta-diagnostic-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dump": SOURCE_DUMP_VERSION,
        "cohort_hash": data.cohort_hash,
        "row_key_hash": data.row_key_hash,
        "cohort_rows": len(data.cohort),
        "feature_order": list(PCA_FEATURES),
        "target_metrics_in_inputs": [],
        "split_definitions": artifact_common_metadata(
            data,
            splits,
            method="variational_autoencoder",
            source_dump=SOURCE_DUMP_VERSION,
            seed=seed,
        )["split_definitions"],
        "seed": seed,
        "latent_dimension": FINAL_LATENT_DIMENSION,
        "architecture": [51, 64, 32, 15, 32, 64, 51],
        "defensible_active_dimension_minimum": DEFENSIBLE_ACTIVE_DIMENSIONS,
        "selection_priority": [
            "avoid partial or complete posterior collapse",
            "maximize minimum active dimensions across folds",
            "minimize grouped validation reconstruction MSE",
            "prefer cross-fold embedding stability",
            "outcome probes excluded from selection",
        ],
        "results": results,
        "selected_beta": None if selected is None else selected["beta"],
        "selection_status": "no_defensible_beta" if selected is None else "selected",
        "selection_rationale": (
            "No beta met the predeclared active-dimension and collapse gate."
            if selected is None
            else (
                f"beta {selected['beta']} passed the >= {DEFENSIBLE_ACTIVE_DIMENSIONS} "
                "active dimensions in every fold gate and ranked first by the fixed priority."
            )
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    safe = _json_safe(artifact)
    write_json_artifact(output_path, safe)
    return safe
