#!/usr/bin/env python3
"""Build the final matched Stage 3 representation comparison artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apps import inferencex_pca_demo as app
from modeling.final_representation_training import _cluster_summary, _mean_std
from modeling.pca_target_analysis import PCA_FEATURES
from modeling.representation_analysis import (
    FINAL_COMPARISON_SCHEMA_VERSION,
    RANDOM_SEEDS,
    SOURCE_DUMP_VERSION,
    canonical_representation_data,
    grouped_split_definitions,
    load_final_representation_artifact,
    outcome_overlay_evaluation,
    reconstruction_metrics,
    write_json_artifact,
)
from scripts.build_july_pca_artifact import load_aggregate


def _pca_reconstruction(
    data: Any,
    embedding: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> dict[str, Any]:
    reconstructed = embedding @ components + mean
    folds = []
    source_rows = []
    for split in grouped_split_definitions(data):
        validation = np.asarray(split["validation_indices"])
        metrics = reconstruction_metrics(
            data.matrix[validation],
            reconstructed[validation],
            data.encoded_feature_names,
        )
        folds.append({"fold": split["fold"], "mse": metrics["mse"], "mae": metrics["mae"]})
        for row in metrics["by_source_feature"]:
            source_rows.append({"fold": split["fold"], **row})
    source_summary = []
    for feature in PCA_FEATURES:
        rows = [row for row in source_rows if row["source_feature"] == feature]
        source_summary.append(
            {
                "source_feature": feature,
                "feature_type": rows[0]["feature_type"],
                "encoded_columns": rows[0]["encoded_columns"],
                "mse_mean": float(np.mean([row["mse"] for row in rows])),
                "mse_standard_deviation": float(np.std([row["mse"] for row in rows], ddof=1)),
                "mae_mean": float(np.mean([row["mae"] for row in rows])),
                "mae_standard_deviation": float(np.std([row["mae"] for row in rows], ddof=1)),
            }
        )
    return {
        "fold_results": folds,
        "validation_mse": _mean_std([row["mse"] for row in folds]),
        "validation_mae": _mean_std([row["mae"] for row in folds]),
        "reconstruction_by_source_feature": sorted(
            source_summary, key=lambda row: row["mse_mean"], reverse=True
        ),
    }


def _pca_outcomes(data: Any, embedding: np.ndarray, labels: list[int]) -> dict[str, Any]:
    raw = outcome_overlay_evaluation(data, embedding, labels)
    return {
        target: {
            "probe_model": "ridge regression (alpha=1.0), post-hoc evaluation only",
            "fold_seed_results": [
                {"seed": None, **row} for row in values["probe"]["folds"]
            ],
            "r2": _mean_std([row["r2"] for row in values["probe"]["folds"]]),
            "mae": _mean_std([row["mae"] for row in values["probe"]["folds"]]),
            "nearest_neighbor_target_correlation": {
                "mean": values["nearest_neighbor_target_correlation"],
                "standard_deviation": 0.0,
                "minimum": values["nearest_neighbor_target_correlation"],
                "maximum": values["nearest_neighbor_target_correlation"],
            },
            "strongest_dimensions_by_seed": [
                {
                    "seed": None,
                    **max(
                        values["associations"],
                        key=lambda row: max(abs(row["pearson"]), abs(row["spearman"])),
                    ),
                }
            ],
        }
        for target, values in raw.items()
    }


def build_final_comparison(
    *,
    data_dir: str,
    pca_path: Path,
    ae_path: Path,
    vae_path: Path,
    beta_diagnostic_path: Path,
) -> dict[str, Any]:
    pca = json.loads(pca_path.read_text(encoding="utf-8"))
    ae, _ae_companion = load_final_representation_artifact(
        ae_path, expected_method="autoencoder"
    )
    vae, _vae_companion = load_final_representation_artifact(
        vae_path,
        expected_method="variational_autoencoder",
        expected_cohort_hash=ae["cohort_hash"],
    )
    diagnostic = json.loads(beta_diagnostic_path.read_text(encoding="utf-8"))
    if ae["split_definitions"] != vae["split_definitions"]:
        raise ValueError("Final AE and VAE fold definitions differ.")
    if pca["shared_basis"]["feature_order"] != ae["feature_order"]:
        raise ValueError("PCA and final neural feature orders differ.")
    if pca["shared_basis"]["full_eligible_row_count"] != ae["cohort_rows"]:
        raise ValueError("PCA and final neural cohort sizes differ.")
    _raw, aggregate, _metadata = load_aggregate(data_dir)
    data = canonical_representation_data(aggregate)
    if data.cohort_hash != ae["cohort_hash"]:
        raise ValueError("Active cohort differs from the final neural artifacts.")

    state = pca["shared_basis"]["preprocessing"]
    components = np.asarray(state["pca_components"], dtype=float)
    pca_mean = np.asarray(state["pca_mean"], dtype=float)
    centered = data.matrix - pca_mean
    pca15_embedding = centered @ components[:15].T
    pca5_embedding = pca15_embedding[:, :5]
    pca15_reconstruction = _pca_reconstruction(
        data, pca15_embedding, components[:15], pca_mean
    )
    pca15_clustering, pca15_labels = _cluster_summary(
        {seed: pca15_embedding for seed in RANDOM_SEEDS},
        seeds=RANDOM_SEEDS,
    )
    pca5_clustering, _pca5_labels = _cluster_summary(
        {seed: pca5_embedding for seed in RANDOM_SEEDS},
        seeds=RANDOM_SEEDS,
    )
    pca_outcomes = _pca_outcomes(data, pca15_embedding, pca15_labels[42])
    pca_loading_stability = float(
        np.mean(
            [
                row["cosine_similarity"]
                for row in pca["shared_basis"]["basis_comparison"]["components"]
            ]
        )
    )

    method_results = {
        "PCA-15": {
            "method": "PCA",
            "latent_dimension": 15,
            "linear_or_nonlinear": "linear",
            "validation_mse": pca15_reconstruction["validation_mse"],
            "validation_mae": pca15_reconstruction["validation_mae"],
            "reconstruction_by_source_feature": pca15_reconstruction[
                "reconstruction_by_source_feature"
            ],
            "fold_seed_results": pca15_reconstruction["fold_results"],
            "stability": {
                "score": pca_loading_stability,
                "protocol": "mean sign-aligned June-July loading cosine for PC1-PC5",
                "pairwise_distance_spearman": None,
                "nearest_neighbor_overlap": None,
                "procrustes_coordinate_similarity": None,
                "cluster_stability_mean_ari": pca15_clustering[
                    "cluster_stability_mean_ari"
                ],
            },
            "clustering": pca15_clustering,
            "outcome_overlays": pca_outcomes,
            "parameter_count": 0,
            "runtime_seconds": None,
            "transformation_seconds": None,
            "artifact_size_bytes": pca_path.stat().st_size,
            "interpretability": "high: signed source and encoded-feature loadings",
        },
        "Autoencoder-15": {
            "method": "Autoencoder",
            "latent_dimension": 15,
            "linear_or_nonlinear": "nonlinear deterministic",
            "validation_mse": ae["summary"]["validation_mse"],
            "validation_mae": ae["summary"]["validation_mae"],
            "reconstruction_by_source_feature": ae["summary"][
                "reconstruction_by_source_feature"
            ],
            "fold_seed_results": [
                {
                    key: row[key]
                    for key in (
                        "seed",
                        "fold",
                        "mse",
                        "mae",
                        "epochs_trained",
                        "early_stopping_occurred",
                        "train_validation_gap",
                    )
                }
                for row in ae["runs"]
            ],
            "stability": {
                "score": ae["summary"]["cross_seed_stability"]["mean_score"],
                "protocol": "cross-seed representative-fold geometry",
                "pairs": ae["summary"]["cross_seed_stability"]["pairs"],
                "cluster_stability_mean_ari": ae["summary"]["clustering"][
                    "cluster_stability_mean_ari"
                ],
            },
            "clustering": ae["summary"]["clustering"],
            "outcome_overlays": ae["summary"]["outcome_overlays"],
            "parameter_count": ae["summary"]["parameter_count"],
            "runtime_seconds": ae["computational_cost"]["total_runtime_seconds"],
            "transformation_seconds": ae["computational_cost"]["transformation_seconds"],
            "artifact_size_bytes": (
                ae_path.stat().st_size
                + ae["embedding_companion"]["bytes"]
                + ae["weights"]["bytes"]
            ),
            "interpretability": "medium-low: decoder sensitivity and latent traversals",
        },
        "VAE-15": {
            "method": "Variational Autoencoder",
            "latent_dimension": 15,
            "linear_or_nonlinear": "nonlinear variational",
            "selected_beta": vae["hyperparameters"]["beta"],
            "validation_mse": vae["summary"]["validation_mse"],
            "validation_mae": vae["summary"]["validation_mae"],
            "reconstruction_by_source_feature": vae["summary"][
                "reconstruction_by_source_feature"
            ],
            "fold_seed_results": [
                {
                    key: row[key]
                    for key in (
                        "seed",
                        "fold",
                        "mse",
                        "mae",
                        "epochs_trained",
                        "early_stopping_occurred",
                        "train_validation_gap",
                    )
                }
                for row in vae["runs"]
            ],
            "stability": {
                "score": vae["summary"]["cross_seed_stability"]["mean_score"],
                "protocol": "cross-seed representative-fold geometry",
                "pairs": vae["summary"]["cross_seed_stability"]["pairs"],
                "cluster_stability_mean_ari": vae["summary"]["clustering"][
                    "cluster_stability_mean_ari"
                ],
            },
            "clustering": vae["summary"]["clustering"],
            "outcome_overlays": vae["summary"]["outcome_overlays"],
            "diagnostics_by_seed_fold": vae["summary"]["diagnostics_by_seed_fold"],
            "partial_collapse": True,
            "partial_collapse_note": (
                "All runs pass the >=5 active-dimension gate, but only 5-7 of 15 "
                "dimensions are active."
            ),
            "parameter_count": vae["summary"]["parameter_count"],
            "runtime_seconds": vae["computational_cost"]["total_runtime_seconds"],
            "transformation_seconds": vae["computational_cost"]["transformation_seconds"],
            "artifact_size_bytes": (
                vae_path.stat().st_size
                + vae["embedding_companion"]["bytes"]
                + vae["weights"]["bytes"]
            ),
            "interpretability": (
                "medium-low: decoder sensitivity and traversals, constrained by partial collapse"
            ),
        },
    }
    conclusion_matrix = [
        {
            "criterion": "Reconstruction",
            "strongest evidence": "Autoencoder-15",
            "evidence": (
                f"MSE {ae['summary']['validation_mse']['mean']:.6f} +/- "
                f"{ae['summary']['validation_mse']['standard_deviation']:.6f}"
            ),
            "qualification": "Grouped folds and three seeds; no outcome-based selection.",
        },
        {
            "criterion": "Representation stability",
            "strongest evidence": "PCA",
            "evidence": f"June-July loading cosine {pca_loading_stability:.3f}",
            "qualification": "Neural stability is cross-seed latent-geometry stability.",
        },
        {
            "criterion": "Clustering",
            "strongest evidence": "Evidence-specific; no universal winner",
            "evidence": (
                f"PCA-15 silhouette {pca15_clustering['selected_metrics']['silhouette']['mean']:.3f}; "
                f"AE {ae['summary']['clustering']['selected_metrics']['silhouette']['mean']:.3f}; "
                f"VAE {vae['summary']['clustering']['selected_metrics']['silhouette']['mean']:.3f}"
            ),
            "qualification": "VAE clustering is shown adjacent to partial-collapse diagnostics.",
        },
        {
            "criterion": "Outcome preservation",
            "strongest evidence": "Depends on outcome",
            "evidence": "Grouped ridge probes are reported separately for all three outcomes.",
            "qualification": "Post-hoc only; outcomes never altered representation fitting.",
        },
        {
            "criterion": "Interpretability",
            "strongest evidence": "PCA",
            "evidence": "Direct signed loadings and stable component directions.",
            "qualification": "Neural decoder sensitivities are descriptive, not causal.",
        },
        {
            "criterion": "Compute cost",
            "strongest evidence": "PCA",
            "evidence": "No trainable parameters in the preserved linear basis.",
            "qualification": "Historical PCA training time was not recorded.",
        },
    ]
    return {
        "schema_version": FINAL_COMPARISON_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dump": SOURCE_DUMP_VERSION,
        "status": "final",
        "cohort_hash": ae["cohort_hash"],
        "row_key_hash": ae["row_key_hash"],
        "compatible_cohort": True,
        "cohort_rows": ae["cohort_rows"],
        "feature_order": list(PCA_FEATURES),
        "random_seeds": list(RANDOM_SEEDS),
        "latent_dimension": 15,
        "methods": method_results,
        "pca5_compact_clustering_reference": pca5_clustering,
        "vae_beta_diagnostic": diagnostic,
        "conclusion_matrix": conclusion_matrix,
        "publication_interpretation": {
            "evidence": (
                "AE-15 has the strongest matched reconstruction. PCA has the strongest "
                "established temporal and loading interpretability evidence. VAE beta 0.1 "
                "improves substantially over beta 1.0 but retains partial collapse."
            ),
            "interpretation": (
                "Nonlinear deterministic structure is supported for reconstruction, while "
                "the variational objective is not competitive on reconstruction and does not "
                "activate the full latent capacity."
            ),
            "unresolved_questions": (
                "External-snapshot stability, alternative categorical decoder heads, and "
                "cluster replication beyond the fixed cohort remain unresolved."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--pca", type=Path, default=app.PCA_TARGET_ARTIFACT_PATH)
    parser.add_argument(
        "--ae",
        type=Path,
        default=Path("artifacts/representation-ae-final-db-dump-2026-07-20.json"),
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=Path("artifacts/representation-vae-final-db-dump-2026-07-20.json"),
    )
    parser.add_argument(
        "--beta-diagnostic",
        type=Path,
        default=Path(
            "artifacts/representation-vae-beta-diagnostic-db-dump-2026-07-20.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/representation-comparison-final-db-dump-2026-07-20.json"),
    )
    args = parser.parse_args()
    artifact = build_final_comparison(
        data_dir=args.data_dir,
        pca_path=args.pca,
        ae_path=args.ae,
        vae_path=args.vae,
        beta_diagnostic_path=args.beta_diagnostic,
    )
    write_json_artifact(args.output, artifact)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

