"""Stage 4 methodological validation for the frozen representation experiment."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from modeling.final_representation_training import (
    _latent_traversal_summary,
    _mean_std,
)
from modeling.neural_representation import train_neural_fold
from modeling.pca_target_analysis import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PCA_FEATURES,
    source_feature_for_encoded,
)
from modeling.representation_analysis import (
    OUTCOME_TARGETS,
    RANDOM_SEEDS,
    SOURCE_DUMP_VERSION,
    canonical_representation_data,
    clustering_evaluation,
    embedding_stability,
    fit_fold_preprocessor,
    grouped_partition_definitions,
    grouped_split_definitions,
    outcome_overlay_evaluation,
    reconstruction_metrics,
    source_feature_reconstruction_metrics,
    software_versions,
    write_json_artifact,
)
from modeling.representation_training import _json_safe

PARTITION_SEEDS = (17, 29, 43)
COMPUTE_LIMIT_SECONDS = 190.0
STAGE4_SCHEMA_VERSION = "representation-validation-stage4-v1"
WORKLOAD_FEATURES = ("isl", "osl", "conc")
CONFIGURATION_FEATURES = tuple(
    feature for feature in PCA_FEATURES if feature not in WORKLOAD_FEATURES
)


def _sequence_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _split_metadata(data: Any, splits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in splits:
        validation = np.asarray(split["validation_indices"])
        config_ids = sorted(
            data.cohort.iloc[validation]["config_id"].astype(str).unique().tolist()
        )
        rows.append(
            {
                "partition_seed": split.get("partition_seed"),
                "fold": split["fold"],
                "train_rows": len(split["train_indices"]),
                "validation_rows": len(split["validation_indices"]),
                "train_configurations": split["train_configurations"],
                "validation_configurations": split["validation_configurations"],
                "validation_config_ids": config_ids,
                "validation_row_id_hash": _sequence_hash(
                    [data.row_ids[index] for index in validation]
                ),
                "group_overlap": split["group_overlap"],
            }
        )
    return rows


def _aggregate_source_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        for feature in record["source_metrics"]["features"]:
            rows.append(
                {
                    "partition_seed": record.get("partition_seed"),
                    "seed": record.get("seed"),
                    "fold": record["fold"],
                    **{key: value for key, value in feature.items() if key != "confusion_matrix"},
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for (feature, feature_type), group in frame.groupby(
        ["source_feature", "feature_type"], sort=False
    ):
        summaries.append(
            {
                "source_feature": feature,
                "feature_type": feature_type,
                "mae": _mean_std(group["mae"].dropna().astype(float).tolist()),
                "mse": _mean_std(group["mse"].dropna().astype(float).tolist()),
                "balanced_mae": _mean_std(group["balanced_mae"].astype(float).tolist()),
                "balanced_mse": _mean_std(group["balanced_mse"].astype(float).tolist()),
                "exact_accuracy": (
                    _mean_std(group["exact_accuracy"].dropna().astype(float).tolist())
                    if group["exact_accuracy"].notna().any()
                    else None
                ),
                "top2_accuracy": (
                    _mean_std(group["top2_accuracy"].dropna().astype(float).tolist())
                    if group["top2_accuracy"].notna().any()
                    else None
                ),
                "unknown_validation_rows": int(group["unknown_validation_rows"].sum()),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: row["balanced_mse"]["mean"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["reconstruction_rank"] = rank
    return {
        "features": ranked,
        "balanced_source_mae": _mean_std(
            [record["source_metrics"]["balanced_source_mae"] for record in records]
        ),
        "balanced_source_mse": _mean_std(
            [record["source_metrics"]["balanced_source_mse"] for record in records]
        ),
        "definition": records[0]["source_metrics"]["definition"],
    }


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "encoded_mse": _mean_std([record["encoded_mse"] for record in records]),
        "encoded_mae": _mean_std([record["encoded_mae"] for record in records]),
        "source_balanced": _aggregate_source_metrics(records),
        "runs": [
            {
                key: record.get(key)
                for key in (
                    "partition_seed",
                    "seed",
                    "fold",
                    "encoded_dimension",
                    "encoded_mse",
                    "encoded_mae",
                    "source_balanced_mse",
                    "source_balanced_mae",
                    "epochs_trained",
                    "best_epoch",
                    "early_stopping_occurred",
                    "runtime_seconds",
                    "active_latent_dimensions",
                    "posterior_collapse",
                )
            }
            for record in records
        ],
    }


def _run_pca_fold(
    data: Any,
    split: dict[str, Any],
    *,
    feature_order: tuple[str, ...] = tuple(PCA_FEATURES),
) -> tuple[dict[str, Any], dict[str, Any]]:
    fold_data = fit_fold_preprocessor(
        data,
        split["train_indices"],
        split["validation_indices"],
        feature_order=feature_order,
    )
    components = min(15, fold_data.train_matrix.shape[1])
    started = time.perf_counter()
    model = PCA(n_components=components, random_state=42).fit(fold_data.train_matrix)
    validation_embedding = model.transform(fold_data.validation_matrix)
    reconstructed = model.inverse_transform(validation_embedding)
    runtime = time.perf_counter() - started
    encoded = reconstruction_metrics(
        fold_data.validation_matrix,
        reconstructed,
        fold_data.encoded_feature_names,
    )
    validation = np.asarray(split["validation_indices"])
    source = source_feature_reconstruction_metrics(
        data.cohort.iloc[validation],
        fold_data.validation_matrix,
        reconstructed,
        fold_data,
    )
    record = {
        "method": "PCA",
        "partition_seed": split.get("partition_seed"),
        "seed": None,
        "fold": split["fold"],
        "encoded_dimension": fold_data.train_matrix.shape[1],
        "latent_dimension": components,
        "encoded_mse": encoded["mse"],
        "encoded_mae": encoded["mae"],
        "source_balanced_mse": source["balanced_source_mse"],
        "source_balanced_mae": source["balanced_source_mae"],
        "source_metrics": source,
        "runtime_seconds": runtime,
        "epochs_trained": None,
        "best_epoch": None,
        "early_stopping_occurred": None,
        "active_latent_dimensions": None,
        "posterior_collapse": None,
    }
    auxiliary = {
        "embedding": model.transform(fold_data.all_matrix),
        "model": model,
        "fold_data": fold_data,
        "split": split,
    }
    return record, auxiliary


def _run_neural_fold(
    data: Any,
    split: dict[str, Any],
    *,
    method: str,
    seed: int,
    feature_order: tuple[str, ...] = tuple(PCA_FEATURES),
    input_corruption: float = 0.0,
    kl_warmup_epochs: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fold_data = fit_fold_preprocessor(
        data,
        split["train_indices"],
        split["validation_indices"],
        feature_order=feature_order,
    )
    result = train_neural_fold(
        fold_data.all_matrix,
        split["train_indices"],
        split["validation_indices"],
        method=method,
        latent_dimension=15,
        seed=seed,
        hidden_dimensions=(64, 32),
        beta=0.1 if method == "variational_autoencoder" else 0.0,
        maximum_epochs=250 if method == "autoencoder" else 150,
        patience=12,
        input_corruption=input_corruption,
        kl_warmup_epochs=kl_warmup_epochs,
    )
    validation = np.asarray(split["validation_indices"])
    reconstructed = result["reconstructed"][validation]
    encoded = reconstruction_metrics(
        fold_data.validation_matrix,
        reconstructed,
        fold_data.encoded_feature_names,
    )
    source = source_feature_reconstruction_metrics(
        data.cohort.iloc[validation],
        fold_data.validation_matrix,
        reconstructed,
        fold_data,
    )
    diagnostics = result["diagnostics"]
    record = {
        "method": method,
        "partition_seed": split.get("partition_seed"),
        "seed": seed,
        "fold": split["fold"],
        "encoded_dimension": fold_data.train_matrix.shape[1],
        "latent_dimension": 15,
        "encoded_mse": encoded["mse"],
        "encoded_mae": encoded["mae"],
        "source_balanced_mse": source["balanced_source_mse"],
        "source_balanced_mae": source["balanced_source_mae"],
        "source_metrics": source,
        "runtime_seconds": result["runtime_seconds"],
        "epochs_trained": result["epochs_trained"],
        "best_epoch": result["best_epoch"],
        "early_stopping_occurred": result["early_stopping_occurred"],
        "active_latent_dimensions": diagnostics.get("active_latent_dimensions"),
        "posterior_collapse": diagnostics.get("posterior_collapse"),
        "average_kl_per_latent_dimension": diagnostics.get(
            "average_kl_per_latent_dimension"
        ),
        "latent_variance": diagnostics.get("latent_variance"),
        "decoder_sensitivity_per_dimension": diagnostics.get(
            "decoder_sensitivity_per_dimension"
        ),
        "decoder_ignores_latent_changes": diagnostics.get(
            "decoder_ignores_latent_changes"
        ),
        "validation_kl": (
            result["history"][result["best_epoch"] - 1]["validation_kl"]
            if method == "variational_autoencoder"
            else None
        ),
    }
    auxiliary = {
        "embedding": result["embedding"],
        "state_dict": result["state_dict"],
        "diagnostics": diagnostics,
        "fold_data": fold_data,
        "split": split,
    }
    return record, auxiliary


def _variation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    partition_means = (
        frame.groupby("partition_seed", dropna=False)["encoded_mse"].mean().tolist()
    )
    seed_frame = frame.dropna(subset=["seed"])
    seed_means = (
        seed_frame.groupby("seed")["encoded_mse"].mean().tolist()
        if not seed_frame.empty
        else []
    )
    return {
        "between_partition_mse": _mean_std(partition_means),
        "between_seed_mse": _mean_std(seed_means) if seed_means else None,
        "combined_run_mse": _mean_std(frame["encoded_mse"].tolist()),
        "partition_means": [
            {
                "partition_seed": int(partition),
                "mse": float(group["encoded_mse"].mean()),
                "mae": float(group["encoded_mae"].mean()),
            }
            for partition, group in frame.groupby("partition_seed")
        ],
        "seed_means": [
            {
                "seed": int(seed),
                "mse": float(group["encoded_mse"].mean()),
                "mae": float(group["encoded_mae"].mean()),
            }
            for seed, group in seed_frame.groupby("seed")
        ],
    }


def _geometry_robustness(
    embeddings: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    within_partition = {}
    for partition in PARTITION_SEEDS:
        within_partition[str(partition)] = embedding_stability(
            [embeddings[(partition, seed)] for seed in RANDOM_SEEDS]
        )
    within_seed = {}
    for seed in RANDOM_SEEDS:
        within_seed[str(seed)] = embedding_stability(
            [embeddings[(partition, seed)] for partition in PARTITION_SEEDS]
        )
    scores = [
        value["mean_score"]
        for value in [*within_partition.values(), *within_seed.values()]
        if value["mean_score"] is not None
    ]
    return {
        "within_partition_across_seed": within_partition,
        "within_seed_across_partition": within_seed,
        "combined_mean_stability": float(np.mean(scores)),
    }


def _feature_consistency(
    states: dict[str, Any],
    encoded_names: list[str],
) -> dict[str, Any]:
    importance_by_run = {}
    top_counts: Counter[str] = Counter()
    traversal_rows = []
    for run, state in states.items():
        traversals = _latent_traversal_summary({run: state}, encoded_names)
        traversal_rows.extend({"run": run, **row} for row in traversals)
        frame = pd.DataFrame(traversals)
        importance = (
            frame.groupby("source_feature")["absolute_change_mean"].mean().to_dict()
        )
        importance_by_run[run] = importance
        for feature in sorted(importance, key=importance.get, reverse=True)[:5]:
            top_counts[feature] += 1
    ordered_features = list(PCA_FEATURES)
    correlations = []
    run_names = sorted(importance_by_run)
    for left_index, left in enumerate(run_names):
        for right in run_names[left_index + 1 :]:
            left_values = [importance_by_run[left].get(feature, 0.0) for feature in ordered_features]
            right_values = [importance_by_run[right].get(feature, 0.0) for feature in ordered_features]
            correlations.append(
                {
                    "left_run": left,
                    "right_run": right,
                    "spearman": float(spearmanr(left_values, right_values).statistic),
                }
            )
    feature_rows = []
    for feature in ordered_features:
        values = [
            importance_by_run[run].get(feature, 0.0) for run in run_names
        ]
        feature_rows.append(
            {
                "source_feature": feature,
                "importance": _mean_std(values),
                "top5_run_frequency": top_counts[feature] / len(run_names),
            }
        )
    return {
        "features": sorted(
            feature_rows,
            key=lambda row: row["importance"]["mean"],
            reverse=True,
        ),
        "pairwise_rank_correlations": correlations,
        "mean_rank_correlation": float(
            np.nanmean([row["spearman"] for row in correlations])
        ),
        "latent_traversals": traversal_rows,
    }


def _pca_bootstrap_consistency(
    data: Any,
    partition_splits: dict[int, list[dict[str, Any]]],
    *,
    bootstrap_runs: int = 20,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    rows = []
    top_counts: Counter[str] = Counter()
    for partition, splits in partition_splits.items():
        split = splits[0]
        fold_data = fit_fold_preprocessor(
            data, split["train_indices"], split["validation_indices"]
        )
        base = PCA(n_components=15, random_state=42).fit(fold_data.train_matrix)
        train_indices = np.asarray(split["train_indices"])
        train_groups = data.cohort.iloc[train_indices]["config_id"].astype(str).to_numpy()
        unique_groups = np.unique(train_groups)
        source_names = [
            source_feature_for_encoded(name)
            for name in fold_data.encoded_feature_names
        ]
        bootstrap_values: dict[tuple[int, str], list[float]] = {}
        for _run in range(bootstrap_runs):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            positions = np.concatenate(
                [np.flatnonzero(train_groups == group) for group in sampled_groups]
            )
            fitted = PCA(n_components=15, random_state=42).fit(
                fold_data.train_matrix[positions]
            )
            aligned = fitted.components_.copy()
            for component in range(5):
                if np.dot(aligned[component], base.components_[component]) < 0:
                    aligned[component] *= -1
                for feature in PCA_FEATURES:
                    feature_positions = [
                        index
                        for index, source in enumerate(source_names)
                        if source == feature
                    ]
                    magnitude = float(
                        np.linalg.norm(aligned[component, feature_positions])
                    )
                    bootstrap_values.setdefault((component + 1, feature), []).append(
                        magnitude
                    )
            global_importance = {
                feature: float(
                    np.mean(
                        [
                            np.linalg.norm(
                                aligned[
                                    :5,
                                    [
                                        index
                                        for index, source in enumerate(source_names)
                                        if source == feature
                                    ],
                                ]
                            )
                        ]
                    )
                )
                for feature in PCA_FEATURES
            }
            for feature in sorted(
                global_importance, key=global_importance.get, reverse=True
            )[:5]:
                top_counts[feature] += 1
        for (component, feature), values in bootstrap_values.items():
            rows.append(
                {
                    "partition_seed": partition,
                    "component": component,
                    "source_feature": feature,
                    "mean_loading_magnitude": float(np.mean(values)),
                    "confidence_2_5": float(np.quantile(values, 0.025)),
                    "confidence_97_5": float(np.quantile(values, 0.975)),
                }
            )
    denominator = len(PARTITION_SEEDS) * bootstrap_runs
    return {
        "bootstrap_runs_per_partition": bootstrap_runs,
        "loading_confidence": rows,
        "top5_frequency": [
            {
                "source_feature": feature,
                "frequency": top_counts[feature] / denominator,
            }
            for feature in sorted(top_counts, key=top_counts.get, reverse=True)
        ],
    }


def _best_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.unique(left)
    right_values = np.unique(right)
    scores = np.zeros((len(left_values), len(right_values)))
    for left_index, left_value in enumerate(left_values):
        left_set = set(np.flatnonzero(left == left_value))
        for right_index, right_value in enumerate(right_values):
            right_set = set(np.flatnonzero(right == right_value))
            scores[left_index, right_index] = len(left_set & right_set) / max(
                1, len(left_set | right_set)
            )
    row_indices, column_indices = linear_sum_assignment(-scores)
    return float(np.mean(scores[row_indices, column_indices]))


def _config_labels(config_ids: pd.Series, labels: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"config_id": config_ids.astype(str), "label": labels})
    return (
        frame.groupby("config_id", sort=True)["label"]
        .agg(lambda values: values.value_counts().index[0])
        .to_numpy()
    )


def _cross_method_cluster_agreement(
    data: Any,
    pca_embeddings: dict[int, np.ndarray],
    ae_embeddings: dict[tuple[int, int], np.ndarray],
    vae_embeddings: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    rows = []
    for partition in PARTITION_SEEDS:
        for seed in RANDOM_SEEDS:
            labels = {
                "PCA": KMeans(n_clusters=3, random_state=seed, n_init=20).fit_predict(
                    pca_embeddings[partition]
                ),
                "AE": KMeans(n_clusters=2, random_state=seed, n_init=20).fit_predict(
                    ae_embeddings[(partition, seed)]
                ),
                "VAE": KMeans(n_clusters=2, random_state=seed, n_init=20).fit_predict(
                    vae_embeddings[(partition, seed)]
                ),
            }
            for left, right in (("PCA", "AE"), ("PCA", "VAE"), ("AE", "VAE")):
                left_config = _config_labels(data.cohort["config_id"], labels[left])
                right_config = _config_labels(data.cohort["config_id"], labels[right])
                rows.append(
                    {
                        "partition_seed": partition,
                        "seed": seed,
                        "left": left,
                        "right": right,
                        "adjusted_rand_index": float(
                            adjusted_rand_score(labels[left], labels[right])
                        ),
                        "normalized_mutual_information": float(
                            normalized_mutual_info_score(labels[left], labels[right])
                        ),
                        "row_cluster_overlap": _best_overlap(
                            labels[left], labels[right]
                        ),
                        "configuration_adjusted_rand_index": float(
                            adjusted_rand_score(left_config, right_config)
                        ),
                        "configuration_overlap": _best_overlap(
                            left_config, right_config
                        ),
                    }
                )
    summary = []
    frame = pd.DataFrame(rows)
    for (left, right), group in frame.groupby(["left", "right"], sort=False):
        summary.append(
            {
                "left": left,
                "right": right,
                "adjusted_rand_index": _mean_std(
                    group["adjusted_rand_index"].tolist()
                ),
                "normalized_mutual_information": _mean_std(
                    group["normalized_mutual_information"].tolist()
                ),
                "row_cluster_overlap": _mean_std(
                    group["row_cluster_overlap"].tolist()
                ),
                "configuration_adjusted_rand_index": _mean_std(
                    group["configuration_adjusted_rand_index"].tolist()
                ),
                "configuration_overlap": _mean_std(
                    group["configuration_overlap"].tolist()
                ),
            }
        )
    return {"runs": rows, "summary": summary}


def _ablation_evaluation(
    data: Any,
    current_splits: list[dict[str, Any]],
    *,
    feature_order: tuple[str, ...],
    label: str,
    compute_ledger: dict[str, float],
) -> dict[str, Any]:
    method_results = {}
    for method in ("PCA", "autoencoder", "variational_autoencoder"):
        records = []
        representative = None
        for split in current_splits:
            if method != "PCA" and split["fold"] != 0:
                continue
            if method == "PCA":
                record, auxiliary = _run_pca_fold(
                    data, split, feature_order=feature_order
                )
            else:
                record, auxiliary = _run_neural_fold(
                    data,
                    split,
                    method=method,
                    seed=42,
                    feature_order=feature_order,
                )
                compute_ledger["neural_training_seconds"] += record["runtime_seconds"]
            records.append(record)
            if split["fold"] == 0:
                representative = auxiliary["embedding"]
        clustering = clustering_evaluation(representative, seed=42)
        outcomes = outcome_overlay_evaluation(
            data, representative, clustering["best_labels"]
        )
        method_results[method] = {
            **_record_summary(records),
            "clustering": {
                "best_k": clustering["best_k"],
                "best_silhouette": clustering["best_silhouette"],
            },
            "grouped_ridge_probes": {
                target: {
                    "r2": outcomes[target]["probe"]["mean_r2"],
                    "mae": outcomes[target]["probe"]["mean_mae"],
                }
                for target in OUTCOME_TARGETS
            },
        }
    return {
        "label": label,
        "feature_order": list(feature_order),
        "source_feature_count": len(feature_order),
        "evaluation_scope": {
            "PCA": "all three grouped folds",
            "autoencoder": "seed 42, fixed fold 0 only (bounded exploratory ablation)",
            "variational_autoencoder": "seed 42, fixed fold 0 only (bounded exploratory ablation)",
        },
        "methods": method_results,
    }


def run_stage4_validation(
    aggregate: pd.DataFrame,
    *,
    stage3_ae_path: str | Path,
    stage3_vae_path: str | Path,
    stage3_comparison_path: str | Path,
    pca_artifact_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    data = canonical_representation_data(aggregate)
    pca_hash_before = hashlib.sha256(Path(pca_artifact_path).read_bytes()).hexdigest()
    stage3_ae = json.loads(Path(stage3_ae_path).read_text(encoding="utf-8"))
    stage3_vae = json.loads(Path(stage3_vae_path).read_text(encoding="utf-8"))
    stage3_comparison = json.loads(
        Path(stage3_comparison_path).read_text(encoding="utf-8")
    )
    for artifact in (stage3_ae, stage3_vae):
        if artifact["cohort_hash"] != data.cohort_hash:
            raise RuntimeError("Stage 3 cohort no longer matches the canonical cohort.")
        if artifact["feature_order"] != list(PCA_FEATURES):
            raise RuntimeError("Stage 3 feature order no longer matches the frozen schema.")
        if artifact["target_metrics_in_inputs"]:
            raise RuntimeError("Stage 3 artifact reports outcome leakage.")

    compute_ledger = {
        "limit_seconds": COMPUTE_LIMIT_SECONDS,
        "neural_training_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    current_splits = grouped_split_definitions(data)
    partition_splits = {
        seed: grouped_partition_definitions(data, partition_seed=seed)
        for seed in PARTITION_SEEDS
    }
    split_metadata = {
        "current": _split_metadata(data, current_splits),
        "independent_partitions": {
            str(seed): _split_metadata(data, splits)
            for seed, splits in partition_splits.items()
        },
    }
    if any(
        split["group_overlap"]
        for rows in [
            split_metadata["current"],
            *split_metadata["independent_partitions"].values(),
        ]
        for split in rows
    ):
        raise RuntimeError("Grouped partition leakage detected.")

    print("Stage 4: strict preprocessing on the current folds", flush=True)
    current_records = {"PCA": [], "AE": [], "VAE": []}
    current_auxiliary: dict[str, dict[int, Any]] = {"PCA": {}, "AE": {}, "VAE": {}}
    for split in current_splits:
        record, auxiliary = _run_pca_fold(data, split)
        current_records["PCA"].append(record)
        current_auxiliary["PCA"][split["fold"]] = auxiliary
        for method, label in (
            ("autoencoder", "AE"),
            ("variational_autoencoder", "VAE"),
        ):
            record, auxiliary = _run_neural_fold(
                data, split, method=method, seed=42
            )
            compute_ledger["neural_training_seconds"] += record["runtime_seconds"]
            current_records[label].append(record)
            current_auxiliary[label][split["fold"]] = auxiliary

    leakage_rows = []
    stage3_methods = {
        "PCA": stage3_comparison["methods"]["PCA-15"],
        "AE": stage3_comparison["methods"]["Autoencoder-15"],
        "VAE": stage3_comparison["methods"]["VAE-15"],
    }
    for method, records in current_records.items():
        strict_mse = float(np.mean([record["encoded_mse"] for record in records]))
        if method == "PCA":
            current_mse = stage3_methods[method]["validation_mse"]["mean"]
        else:
            seed42 = [
                row
                for row in (stage3_ae if method == "AE" else stage3_vae)["runs"]
                if row["seed"] == 42
            ]
            current_mse = float(np.mean([row["mse"] for row in seed42]))
        leakage_rows.append(
            {
                "method": method,
                "stage3_full_cohort_preprocessing_mse": current_mse,
                "strict_train_only_preprocessing_mse": strict_mse,
                "absolute_change": strict_mse - current_mse,
                "relative_change": (strict_mse - current_mse) / current_mse,
            }
        )

    print("Stage 4: three independent grouped partitionings", flush=True)
    partition_records = {"PCA": [], "AE": [], "VAE": []}
    pca_embeddings: dict[int, np.ndarray] = {}
    ae_embeddings: dict[tuple[int, int], np.ndarray] = {}
    vae_embeddings: dict[tuple[int, int], np.ndarray] = {}
    ae_states: dict[str, Any] = {}
    vae_states: dict[str, Any] = {}
    representative_encoded_names = None
    for partition, splits in partition_splits.items():
        for split in splits:
            pca_record, pca_aux = _run_pca_fold(data, split)
            partition_records["PCA"].append(pca_record)
            if split["fold"] == 0:
                pca_embeddings[partition] = pca_aux["embedding"]
            for seed in RANDOM_SEEDS:
                for method, label, embeddings, states in (
                    ("autoencoder", "AE", ae_embeddings, ae_states),
                    (
                        "variational_autoencoder",
                        "VAE",
                        vae_embeddings,
                        vae_states,
                    ),
                ):
                    record, auxiliary = _run_neural_fold(
                        data, split, method=method, seed=seed
                    )
                    compute_ledger["neural_training_seconds"] += record[
                        "runtime_seconds"
                    ]
                    if compute_ledger["neural_training_seconds"] > COMPUTE_LIMIT_SECONDS:
                        raise RuntimeError("Stage 4 compute limit exceeded.")
                    partition_records[label].append(record)
                    if split["fold"] == 0:
                        embeddings[(partition, seed)] = auxiliary["embedding"]
                        states[f"partition_{partition}_seed_{seed}"] = auxiliary[
                            "state_dict"
                        ]
                        representative_encoded_names = auxiliary[
                            "fold_data"
                        ].encoded_feature_names
        print(
            f"  partition {partition} complete; neural seconds="
            f"{compute_ledger['neural_training_seconds']:.1f}",
            flush=True,
        )

    partition_robustness = {
        method: {
            **_record_summary(records),
            "variation": _variation_summary(records),
        }
        for method, records in partition_records.items()
    }
    partition_robustness["AE"]["geometry"] = _geometry_robustness(ae_embeddings)
    partition_robustness["VAE"]["geometry"] = _geometry_robustness(vae_embeddings)

    print("Stage 4: bounded AE denoising and VAE KL warm-up", flush=True)
    ae_denoising_records = []
    ae_denoising_embeddings = []
    vae_warmup_records = []
    vae_warmup_embeddings = []
    for split in current_splits:
        ae_record, ae_aux = _run_neural_fold(
            data,
            split,
            method="autoencoder",
            seed=42,
            input_corruption=0.05,
        )
        vae_record, vae_aux = _run_neural_fold(
            data,
            split,
            method="variational_autoencoder",
            seed=42,
            kl_warmup_epochs=50,
        )
        compute_ledger["neural_training_seconds"] += (
            ae_record["runtime_seconds"] + vae_record["runtime_seconds"]
        )
        ae_denoising_records.append(ae_record)
        ae_denoising_embeddings.append(ae_aux["embedding"])
        vae_warmup_records.append(vae_record)
        vae_warmup_embeddings.append(vae_aux["embedding"])
        if compute_ledger["neural_training_seconds"] > COMPUTE_LIMIT_SECONDS:
            raise RuntimeError(
                "Stage 4 compute limit exceeded during bounded robustness experiments."
            )

    robustness_experiments = {
        "autoencoder": {
            "baseline": _record_summary(current_records["AE"]),
            "denoising_5_percent": _record_summary(ae_denoising_records),
            "baseline_stability": embedding_stability(
                [
                    current_auxiliary["AE"][fold]["embedding"]
                    for fold in range(3)
                ]
            ),
            "denoising_stability": embedding_stability(ae_denoising_embeddings),
            "decision_rule": (
                "Prefer denoising only if grouped reconstruction does not materially worsen "
                "and aligned stability improves."
            ),
        },
        "variational_autoencoder": {
            "baseline": _record_summary(current_records["VAE"]),
            "kl_warmup_50_epochs": _record_summary(vae_warmup_records),
            "baseline_stability": embedding_stability(
                [
                    current_auxiliary["VAE"][fold]["embedding"]
                    for fold in range(3)
                ]
            ),
            "warmup_stability": embedding_stability(vae_warmup_embeddings),
            "active_dimensions_baseline": [
                record["active_latent_dimensions"] for record in current_records["VAE"]
            ],
            "active_dimensions_warmup": [
                record["active_latent_dimensions"] for record in vae_warmup_records
            ],
            "decision_rule": (
                "Retain warm-up only if activity improves without unstable reconstruction."
            ),
        },
    }

    pca_consistency = _pca_bootstrap_consistency(data, partition_splits)
    ae_consistency = _feature_consistency(ae_states, representative_encoded_names)
    vae_consistency = _feature_consistency(vae_states, representative_encoded_names)
    cluster_agreement = _cross_method_cluster_agreement(
        data, pca_embeddings, ae_embeddings, vae_embeddings
    )

    print("Stage 4: two bounded feature-family ablations", flush=True)
    ablations = []
    for label, features in (
        ("configuration_only", CONFIGURATION_FEATURES),
        ("workload_only", WORKLOAD_FEATURES),
    ):
        ablations.append(
            _ablation_evaluation(
                data,
                current_splits,
                feature_order=features,
                label=label,
                compute_ledger=compute_ledger,
            )
        )
        if compute_ledger["neural_training_seconds"] > COMPUTE_LIMIT_SECONDS:
            raise RuntimeError("Stage 4 compute limit exceeded during ablation.")

    combined_ablation_reference = {}
    for method, label in (("PCA", "PCA"), ("AE", "AE"), ("VAE", "VAE")):
        representative = current_auxiliary[label][0]["embedding"]
        clustering = clustering_evaluation(representative, seed=42)
        outcomes = outcome_overlay_evaluation(
            data, representative, clustering["best_labels"]
        )
        combined_ablation_reference[method] = {
            **_record_summary(current_records[label]),
            "clustering": {
                "best_k": clustering["best_k"],
                "best_silhouette": clustering["best_silhouette"],
            },
            "grouped_ridge_probes": {
                target: {
                    "r2": outcomes[target]["probe"]["mean_r2"],
                    "mae": outcomes[target]["probe"]["mean_mae"],
                }
                for target in OUTCOME_TARGETS
            },
        }

    compute_ledger["wall_seconds"] = time.perf_counter() - wall_started
    pca_hash_after = hashlib.sha256(Path(pca_artifact_path).read_bytes()).hexdigest()
    if pca_hash_after != pca_hash_before:
        raise RuntimeError("PCA artifact changed during Stage 4.")
    artifact = {
        "schema_version": STAGE4_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dump": SOURCE_DUMP_VERSION,
        "cohort_hash": data.cohort_hash,
        "row_key_hash": data.row_key_hash,
        "cohort_rows": len(data.cohort),
        "configurations": int(data.cohort["config_id"].nunique()),
        "feature_order": list(PCA_FEATURES),
        "target_metrics_in_inputs": [],
        "pca_artifact_sha256_before": pca_hash_before,
        "pca_artifact_sha256_after": pca_hash_after,
        "stage3_artifacts_preserved": {
            "autoencoder": str(stage3_ae_path),
            "variational_autoencoder": str(stage3_vae_path),
            "comparison": str(stage3_comparison_path),
        },
        "strict_preprocessing": {
            "fit_scope": "training rows only within every grouped fold",
            "shared_across_methods": True,
            "unknown_categories": "ignored by the training-fitted one-hot encoder",
            "current_fold_comparison": leakage_rows,
            "strict_current_results": {
                method: _record_summary(records)
                for method, records in current_records.items()
            },
        },
        "source_feature_reconstruction": {
            "encoded_metrics_retained": True,
            "source_feature_count": 19,
            "strict_current": {
                method: _aggregate_source_metrics(records)
                for method, records in current_records.items()
            },
        },
        "split_definitions": split_metadata,
        "partition_robustness": partition_robustness,
        "robustness_experiments": robustness_experiments,
        "feature_consistency": {
            "PCA": pca_consistency,
            "AE": ae_consistency,
            "VAE": vae_consistency,
        },
        "cross_method_cluster_agreement": cluster_agreement,
        "ablations": {
            "combined_reference": combined_ablation_reference,
            "bounded_feature_families": ablations,
        },
        "external_validation_preparation": {
            "available_earlier_snapshot": "inferencex-pca-data (June rollback export)",
            "earlier_eligible_groups": 7_462,
            "current_snapshot": "db-dump/2026-07-20",
            "current_eligible_groups": 8_063,
            "new_groups_in_current_snapshot": 601,
            "later_snapshot_available_locally": False,
            "proposal": (
                "Use the June snapshot as development and freeze preprocessing/model choices "
                "before scoring the 601 new July groups as retrospective temporal validation. "
                "Because July informed the present methodology, reserve the next untouched "
                "official snapshot for prospective external validation."
            ),
            "execution_status": "planned only; not executed",
        },
        "compute_ledger": compute_ledger,
        "software_versions": software_versions(),
    }
    safe = _json_safe(artifact)
    write_json_artifact(output_path, safe)
    return safe


def augment_stage4_warmup_diagnostics(
    aggregate: pd.DataFrame,
    *,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Recreate only the fixed warm-up runs to retain fold-level collapse diagnostics."""

    data = canonical_representation_data(aggregate)
    path = Path(artifact_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != STAGE4_SCHEMA_VERSION:
        raise RuntimeError("Stage 4 artifact schema is incompatible.")
    if artifact.get("cohort_hash") != data.cohort_hash:
        raise RuntimeError("Stage 4 artifact cohort is incompatible.")
    records = []
    elapsed = 0.0
    for split in grouped_split_definitions(data):
        record, _auxiliary = _run_neural_fold(
            data,
            split,
            method="variational_autoencoder",
            seed=42,
            kl_warmup_epochs=50,
        )
        records.append(record)
        elapsed += record["runtime_seconds"]
    section = artifact["robustness_experiments"]["variational_autoencoder"]
    section["kl_warmup_50_epochs"] = _record_summary(records)
    section["kl_warmup_diagnostics"] = {
        "validation_kl": _mean_std([record["validation_kl"] for record in records]),
        "active_latent_dimensions": [
            record["active_latent_dimensions"] for record in records
        ],
        "average_kl_per_latent_dimension_by_fold": [
            record["average_kl_per_latent_dimension"] for record in records
        ],
        "latent_variance_by_fold": [
            record["latent_variance"] for record in records
        ],
        "decoder_sensitivity_per_dimension_by_fold": [
            record["decoder_sensitivity_per_dimension"] for record in records
        ],
        "decoder_ignores_latent_changes": [
            record["decoder_ignores_latent_changes"] for record in records
        ],
        "posterior_collapse": [
            record["posterior_collapse"] for record in records
        ],
    }
    artifact["compute_ledger"]["diagnostic_rerun_seconds"] = elapsed
    artifact["compute_ledger"]["total_neural_training_seconds"] = (
        artifact["compute_ledger"]["neural_training_seconds"] + elapsed
    )
    safe = _json_safe(artifact)
    write_json_artifact(path, safe)
    return safe
