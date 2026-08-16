from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modeling.manifold_results import (
    ALL_K_VALUES,
    CACHE_DIRECTORY,
    _ReferenceTrustworthiness,
    _cache_identity,
    _cached_coordinates,
    _nested_graphs,
    _consensus_summary,
    _load_existing_representations,
    _load_frozen_pca_15,
    _outcomes,
    _parameter_id,
    _projection_rows,
    _within_workload_configuration_from_graph,
    require_structural_freeze,
    stable_artifact_hash,
)
from modeling.neighborhood_analysis import (
    trustworthiness_scores,
    within_workload_configuration_analysis,
    within_workload_knn_indices,
)
from modeling.representation_analysis import OUTCOME_TARGETS


class Stage5ResultsPipelineTests(unittest.TestCase):
    def test_consensus_fraction_uses_candidate_union_not_k(self) -> None:
        agreement = {
            "neighbor_graphs": {
                "pca": np.array([[1, 2]]),
                "ae": np.array([[1, 3]]),
                "vae": np.array([[2, 3]]),
            },
            "consensus": {"recurring_neighbors": [{"one": 2, "two": 2, "three": 2}]},
        }
        summary = _consensus_summary(agreement)
        # The former report divided three recurring neighbors by k=2.  A
        # recurring-neighbor count is not bounded by k once three methods are
        # involved, so that quantity was 1.5 and could not be a fraction.
        self.assertGreater(3.0 / 2.0, 1.0)
        self.assertEqual(summary["consensus_neighbor_count"], 3.0)
        self.assertEqual(summary["candidate_neighbor_union_count"], 3.0)
        self.assertEqual(summary["consensus_fraction"], 1.0)

    def test_current_legacy_pca_artifact_projects_without_retraining(self) -> None:
        artifact = json.loads(Path("artifacts/pca-db-dump-2026-07-20.json").read_text())
        encoded = artifact["shared_basis"]["preprocessing"]["encoded_feature_names"]
        data = SimpleNamespace(
            cohort=pd.DataFrame(index=range(8063)),
            encoded_feature_names=encoded,
            matrix=np.zeros((2, len(encoded)), dtype=float),
        )
        projection = _load_frozen_pca_15(
            data, Path("artifacts/pca-db-dump-2026-07-20.json")
        )
        self.assertEqual(projection.shape, (2, 15))

    def test_current_final_embeddings_align_semantically_despite_legacy_cohort_hash(self) -> None:
        from scripts.build_july_pca_artifact import load_aggregate
        from modeling.manifold_analysis import canonical_stage5_data

        _raw, aggregate, _metadata = load_aggregate(".data/inferencex-db-dump-2026-07-20")
        data = canonical_stage5_data(aggregate)
        ae = json.loads(Path("artifacts/representation-ae-final-db-dump-2026-07-20.json").read_text())
        self.assertNotEqual(ae["cohort_hash"], data.cohort_hash)
        representations = _load_existing_representations(data, Path("artifacts"))
        self.assertEqual(set(representations), {"PCA-15", "AE-15", "VAE-15"})
        self.assertTrue(all(values.shape == (8063, 15) for values in representations.values()))

    def test_tsne_parameter_id_retains_the_frozen_barnes_hut_method(self) -> None:
        parameter_id = _parameter_id(
            "tsne2", perplexity=30, seed=42, method="barnes_hut"
        )
        self.assertEqual(
            parameter_id,
            "tsne2|method=barnes_hut|perplexity=30|seed=42",
        )

    def test_post_freeze_outcome_summary_handles_missing_neighbor_values(self) -> None:
        rows = 32
        outcomes = {
            target: np.arange(rows, dtype=float) + index
            for index, target in enumerate(OUTCOME_TARGETS)
        }
        outcomes[OUTCOME_TARGETS[0]][1] = np.nan
        data = SimpleNamespace(
            row_ids=[f"row-{index}" for index in range(rows)],
            cohort=pd.DataFrame(
                {
                    "isl": [1] * 16 + [2] * 16,
                    "osl": [16] * 16 + [32] * 16,
                    "conc": [1] * rows,
                    **outcomes,
                }
            ),
        )
        result = _outcomes(
            data,
            np.column_stack([np.arange(rows, dtype=float), np.arange(rows, dtype=float) % 3]),
        )
        self.assertEqual(result[OUTCOME_TARGETS[0]]["available_rows"], 31)
        self.assertIsNotNone(
            result[OUTCOME_TARGETS[0]]["mean_absolute_difference_among_30_coordinate_neighbors"]
        )

    def test_cached_reference_trustworthiness_matches_sklearn_for_all_frozen_k(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(240, 7))
        candidate = reference[:, :3] + rng.normal(scale=0.1, size=(240, 3))
        trustworthiness = _ReferenceTrustworthiness(reference)
        cached = trustworthiness.scores(_nested_graphs(candidate))
        expected = trustworthiness_scores(reference, candidate, ALL_K_VALUES)
        for k in ALL_K_VALUES:
            self.assertAlmostEqual(cached[k], expected[k], places=12)
        trustworthiness.close()

    def test_cache_key_contains_scientific_identity_and_stale_entries_fail_closed(self) -> None:
        data = SimpleNamespace(cohort_hash="cohort-a", row_key_hash="rows-a")
        first = _cache_identity(
            data,
            method="umap-2",
            parameters={"seed": 42},
            encoded_feature_names=["num__isl"],
        )
        changed = _cache_identity(
            data,
            method="umap-2",
            parameters={"seed": 123},
            encoded_feature_names=["num__isl"],
        )
        self.assertNotEqual(stable_artifact_hash(first), stable_artifact_hash(changed))
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / CACHE_DIRECTORY
            calls = 0

            def fit() -> tuple[np.ndarray, dict[str, float]]:
                nonlocal calls
                calls += 1
                return np.zeros((3, 2)), {"runtime_seconds": 1.0}

            first_result = _cached_coordinates(cache_dir, first, fit)
            second_result = _cached_coordinates(cache_dir, first, fit)
            self.assertEqual(calls, 1)
            self.assertEqual(first_result[2], "fresh_fit")
            self.assertEqual(second_result[2], "validated_cache")

            cache_key = hashlib.sha256(
                json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            path = cache_dir / f"{cache_key}.npz"
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    identity=np.asarray("different-identity"),
                    coordinates=np.zeros((3, 2)),
                    metadata=np.asarray("{}"),
                )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _cached_coordinates(cache_dir, first, fit)

    def test_structural_freeze_is_outcome_free_and_hashes_exact_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structural.json"
            artifact = {
                "phase": "structural_freeze",
                "target_metrics_in_inputs": [],
                "umap_2": {"runs": []},
            }
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(require_structural_freeze(path), hashlib.sha256(path.read_bytes()).hexdigest())
            artifact["outcome_overlays"] = {"metrics": {"metrics_tput_per_gpu": 1.0}}
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain outcome overlays"):
                require_structural_freeze(path)

    def test_artifact_hash_and_projection_rows_are_stable_and_row_aligned(self) -> None:
        self.assertEqual(
            stable_artifact_hash({"b": [2], "a": 1}),
            stable_artifact_hash({"a": 1, "b": [2]}),
        )
        frame = _projection_rows(
            ["row-b", "row-a"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            method="umap-2",
            parameter_id="frozen-run",
            seed=42,
            n_neighbors=15,
            min_dist=0.1,
        )
        self.assertEqual(frame["row_id"].tolist(), ["row-b", "row-a"])
        self.assertFalse(frame.duplicated(["row_id", "parameter_id"]).any())
        self.assertEqual(set(frame.columns), {
            "row_id", "method", "parameter_id", "seed", "n_neighbors", "min_dist", "perplexity", "x", "y"
        })

    def test_vectorized_conditioned_null_preserves_pr1_statistic_for_both_value_kinds(self) -> None:
        workloads = pd.DataFrame(
            {
                "isl": [1] * 6 + [2] * 6,
                "osl": [16] * 6 + [32] * 6,
                "conc": [1] * 12,
            }
        )
        geometry = np.array([[index] for index in range(12)], dtype=float)
        graph, eligibility = within_workload_knn_indices(geometry, workloads, k=2)
        cases = {
            "categorical": np.array(["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 3),
            "numeric": np.array([1, 1, 2, 3, 3, 4, 2, 2, 3, 4, 4, 5], dtype=float),
        }
        for value_kind, values in cases.items():
            with self.subTest(value_kind=value_kind):
                optimized = _within_workload_configuration_from_graph(
                    graph,
                    workloads,
                    values,
                    value_kind=value_kind,
                    eligibility=eligibility,
                    permutations=31,
                    seed=42,
                )
                generic = within_workload_configuration_analysis(
                    geometry,
                    workloads,
                    values,
                    k=2,
                    value_kind=value_kind,
                    permutations=31,
                    seed=42,
                )
                for field in (
                    "mean",
                    "eligible_rows",
                    "excluded_rows",
                    "eligible_neighbor_pairs",
                ):
                    self.assertEqual(optimized["observed"][field], generic["observed"][field])
                for field in (
                    "null_mean",
                    "null_standard_deviation",
                    "p_value",
                    "alternative",
                ):
                    self.assertEqual(
                        optimized["workload_conditioned_null"][field],
                        generic["workload_conditioned_null"][field],
                    )


if __name__ == "__main__":
    unittest.main()
