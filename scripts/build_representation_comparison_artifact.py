#!/usr/bin/env python3
"""Build the Stage 2 comparison artifact from completed representation artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apps import inferencex_pca_demo as app
from modeling.pca_target_analysis import PCA_FEATURES
from modeling.representation_analysis import (
    COMPARISON_SCHEMA_VERSION,
    SOURCE_DUMP_VERSION,
    canonical_representation_data,
    clustering_evaluation,
    load_representation_artifact,
    outcome_overlay_evaluation,
    write_json_artifact,
)
from scripts.build_july_pca_artifact import load_aggregate


def _embedding(artifact: dict[str, Any]) -> np.ndarray:
    rows = artifact["selected_result"]["embedding"]
    columns = sorted(
        (column for column in rows[0] if column.startswith("z")),
        key=lambda value: int(value[1:]),
    )
    return np.asarray([[row[column] for column in columns] for row in rows], dtype=float)


def _neural_matched_rows(artifact: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        {
            "method": label,
            "latent dimension": row["latent_dimension"],
            "validation reconstruction MAE": row["validation_mae_mean"],
            "validation reconstruction MSE": row["validation_mse_mean"],
            "evaluation": "three grouped config_id folds; one screening seed",
        }
        for row in artifact["screening"]
    ]


def build_comparison(
    data_dir: str,
    pca_path: Path,
    ae_path: Path,
    vae_path: Path,
) -> dict[str, Any]:
    pca_artifact = json.loads(pca_path.read_text(encoding="utf-8"))
    ae = load_representation_artifact(ae_path, expected_method="autoencoder")
    vae = load_representation_artifact(
        vae_path,
        expected_method="variational_autoencoder",
        expected_cohort_hash=ae["cohort_hash"],
    )
    if pca_artifact["shared_basis"]["feature_order"] != ae["feature_order"]:
        raise ValueError("PCA and neural artifacts use different feature orders.")
    if pca_artifact["shared_basis"]["full_eligible_row_count"] != ae["cohort_rows"]:
        raise ValueError("PCA and neural artifacts use different cohort sizes.")
    _raw, aggregate, _metadata = load_aggregate(data_dir)
    data = canonical_representation_data(aggregate)
    if data.cohort_hash != ae["cohort_hash"]:
        raise ValueError("Active cohort does not match the neural artifacts.")

    pca_state = pca_artifact["shared_basis"]["preprocessing"]
    components = np.asarray(pca_state["pca_components"], dtype=float)
    pca_mean = np.asarray(pca_state["pca_mean"], dtype=float)
    pca_embedding = (data.matrix - pca_mean) @ components[:5].T
    pca_clustering = clustering_evaluation(pca_embedding)
    pca_outcomes = outcome_overlay_evaluation(
        data, pca_embedding, pca_clustering["best_labels"]
    )
    neural = (("Autoencoder", ae), ("Variational Autoencoder", vae))
    matched = [
        {
            "method": "PCA",
            "latent dimension": row["latent_dimension"],
            "validation reconstruction MAE": row["validation_mae_mean"],
            "validation reconstruction MSE": row["validation_mse_mean"],
            "evaluation": row["evaluation_note"],
        }
        for row in ae["matched_dimension_pca"]
    ]
    for label, artifact in neural:
        matched.extend(_neural_matched_rows(artifact, label))

    pca_stability = float(
        np.mean(
            [
                row["cosine_similarity"]
                for row in pca_artifact["shared_basis"]["basis_comparison"]["components"]
            ]
        )
    )
    methods = []
    clustering_rows = [
        {
            "method": "PCA",
            "latent dimension": 5,
            "best silhouette score": pca_clustering["best_silhouette"],
            "best k": pca_clustering["best_k"],
            "reference": "preserved PCA basis",
        }
    ]
    stability_rows = [
        {
            "method": "PCA",
            "stability score": pca_stability,
            "protocol": "mean sign-aligned June–July loading cosine",
        }
    ]
    pca_matched_5 = next(
        row for row in matched if row["method"] == "PCA" and row["latent dimension"] == 5
    )
    methods.append(
        {
            "method": "PCA",
            "linear or nonlinear": "linear",
            "latent dimension": 5,
            "trainable parameters": 0,
            "training time seconds": None,
            "validation reconstruction MAE": pca_matched_5["validation reconstruction MAE"],
            "validation reconstruction MSE": pca_matched_5["validation reconstruction MSE"],
            "stability score": pca_stability,
            "best silhouette score": pca_clustering["best_silhouette"],
            "best k": pca_clustering["best_k"],
            "interpretability level": "high: direct signed loadings",
            "artifact size bytes": pca_path.stat().st_size,
        }
    )
    for label, artifact in neural:
        selected = artifact["selected_result"]
        methods.append(
            {
                "method": label,
                "linear or nonlinear": "nonlinear",
                "latent dimension": selected["latent_dimension"],
                "trainable parameters": selected["parameter_count"],
                "training time seconds": artifact["computational_cost"]["screening_runtime_seconds"],
                "validation reconstruction MAE": selected["validation_mae_mean"],
                "validation reconstruction MSE": selected["validation_mse_mean"],
                "stability score": selected["stability"]["mean_score"],
                "best silhouette score": selected["clustering"]["best_silhouette"],
                "best k": selected["clustering"]["best_k"],
                "interpretability level": "medium-low: decoder sensitivity and traversals",
                "artifact size bytes": Path(
                    ae_path if label == "Autoencoder" else vae_path
                ).stat().st_size,
            }
        )
        clustering_rows.append(
            {
                "method": label,
                "latent dimension": selected["latent_dimension"],
                "best silhouette score": selected["clustering"]["best_silhouette"],
                "best k": selected["clustering"]["best_k"],
                "reference": "representative grouped fold",
            }
        )
        stability_rows.append(
            {
                "method": label,
                "stability score": selected["stability"]["mean_score"],
                "protocol": "fold-pair Procrustes, distance, and neighbor agreement",
            }
        )

    outcome_rows: dict[str, list[dict[str, Any]]] = {}
    all_outcomes = {
        "PCA": pca_outcomes,
        "Autoencoder": ae["selected_result"]["outcome_overlays"],
        "Variational Autoencoder": vae["selected_result"]["outcome_overlays"],
    }
    for target in all_outcomes["PCA"]:
        outcome_rows[target] = []
        for method, values in all_outcomes.items():
            result = values[target]
            strongest = max(
                result["associations"],
                key=lambda row: max(abs(row["pearson"]), abs(row["spearman"])),
            )
            outcome_rows[target].append(
                {
                    "method": method,
                    "strongest dimension": strongest.get("dimension"),
                    "Pearson": strongest["pearson"],
                    "Spearman": strongest["spearman"],
                    "neighbor target consistency": result["nearest_neighbor_target_correlation"],
                    "grouped ridge probe R2": result["probe"]["mean_r2"],
                    "grouped ridge probe MAE": result["probe"]["mean_mae"],
                }
            )

    interpretation = [
        {
            "method": "PCA",
            "evidence": "signed encoded and source-feature loadings; component quantiles",
            "guard": "descriptive, not causal",
        },
        {
            "method": "Autoencoder",
            "evidence": "decoder sensitivity, latent extremes, source-feature reconstruction error",
            "guard": "axes require alignment and have no direct causal meaning",
        },
        {
            "method": "Variational Autoencoder",
            "evidence": "decoder sensitivity, KL/active dimensions, latent extremes",
            "guard": "collapse diagnostics must pass before interpretation",
        },
    ]
    cost = [
        {
            "method": row["method"],
            "training time seconds": row["training time seconds"],
            "parameters": row["trainable parameters"],
            "artifact size bytes": row["artifact size bytes"],
        }
        for row in methods
    ]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dump": SOURCE_DUMP_VERSION,
        "status": "preliminary",
        "cohort_hash": ae["cohort_hash"],
        "row_key_hash": ae["row_key_hash"],
        "compatible_cohort": True,
        "cohort_rows": ae["cohort_rows"],
        "feature_order": list(PCA_FEATURES),
        "method_summary": methods,
        "matched_dimension_reconstruction": sorted(
            matched, key=lambda row: (row["latent dimension"], row["method"])
        ),
        "clustering": clustering_rows,
        "stability": stability_rows,
        "outcome_overlays": outcome_rows,
        "interpretability": interpretation,
        "computational_cost": cost,
        "conclusions": {
            "where_pca_performs_best": (
                "PCA currently provides the clearest direct feature attribution and the "
                "strongest established June–July basis-stability evidence."
            ),
            "where_autoencoder_adds_value": (
                "The autoencoder may add value where its matched-dimension reconstruction or "
                "neighborhood metrics improve; Stage 2 provides screening evidence only."
            ),
            "where_vae_adds_value": (
                "The VAE offers a regularized latent distribution and traversal diagnostics "
                "only if active-dimension and decoder-sensitivity checks rule out collapse."
            ),
            "whether_nonlinear_complexity_is_justified": (
                "Unresolved. One screening seed is insufficient to justify nonlinear complexity."
            ),
            "unresolved_questions": (
                "Multi-seed stability, final architecture choice, and resampled cluster stability "
                "remain for Stage 3."
            ),
            "limitations": (
                "PCA preserves the established full-cohort basis, while neural validation uses "
                "held-out configuration groups. Categorical one-hot outputs are reconstructed "
                "with joint MSE and are not guaranteed to be valid exclusive categories."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--pca", type=Path, default=app.PCA_TARGET_ARTIFACT_PATH)
    parser.add_argument("--ae", type=Path, default=app.AE_REPRESENTATION_ARTIFACT_PATH)
    parser.add_argument("--vae", type=Path, default=app.VAE_REPRESENTATION_ARTIFACT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=app.REPRESENTATION_COMPARISON_ARTIFACT_PATH,
    )
    args = parser.parse_args()
    artifact = build_comparison(args.data_dir, args.pca, args.ae, args.vae)
    write_json_artifact(args.output, artifact)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
