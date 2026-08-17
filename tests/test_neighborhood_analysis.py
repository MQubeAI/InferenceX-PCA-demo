from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modeling.neighborhood_analysis import (
    categorical_neighbor_homophily,
    consensus_neighbors,
    cross_method_neighbor_agreement,
    cross_seed_neighborhood_stability,
    exact_knn_indices,
    fit_source_mixed_distance_spec,
    grouped_bootstrap,
    local_rank_agreement,
    neighbor_jaccard,
    neighbor_recall,
    source_mixed_distance_block,
    source_mixed_knn_indices,
    trustworthiness_scores,
    within_workload_knn_indices,
    within_workload_configuration_analysis,
    within_workload_permutation_null,
    workload_neighborhood_purity,
)
from modeling.pca_target_analysis import PCA_FEATURES


def source_frame(rows: int = 3) -> pd.DataFrame:
    """Small valid frozen-source frame with no outcome-dependent inputs."""

    values = {}
    for feature in PCA_FEATURES:
        if feature == "isl":
            values[feature] = [0.0, 1.0, 0.0][:rows]
        elif feature == "config_hardware":
            values[feature] = ["a", "a", "b"][:rows]
        elif feature.startswith("config_") and feature in {
            "config_framework",
            "config_model",
            "config_precision",
            "config_spec_method",
        }:
            values[feature] = ["same"] * rows
        else:
            values[feature] = [0.0] * rows
    return pd.DataFrame(values)


class NeighborhoodAgreementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.geometry = self.rng.normal(size=(80, 5))
        self.neighbors = exact_knn_indices(self.geometry, 10)

    def test_identical_representation_has_perfect_recall_and_jaccard(self) -> None:
        self.assertEqual(neighbor_recall(self.neighbors, self.neighbors)["mean"], 1.0)
        self.assertEqual(neighbor_jaccard(self.neighbors, self.neighbors)["mean"], 1.0)

    def test_identical_representation_has_trustworthiness_one(self) -> None:
        score = trustworthiness_scores(self.geometry, self.geometry.copy(), [10])[10]
        self.assertAlmostEqual(score, 1.0, places=12)

    def test_self_neighbors_are_excluded(self) -> None:
        for row, row_neighbors in enumerate(self.neighbors):
            self.assertNotIn(row, row_neighbors.tolist())
        explicit_query = exact_knn_indices(self.geometry, 10, query=self.geometry.copy())
        for row, row_neighbors in enumerate(explicit_query):
            self.assertNotIn(row, row_neighbors.tolist())
        tied = exact_knn_indices(np.zeros((8, 2)), 3)
        for row, row_neighbors in enumerate(tied):
            self.assertNotIn(row, row_neighbors.tolist())

    def test_independent_geometry_lowers_agreement(self) -> None:
        unrelated = exact_knn_indices(self.rng.normal(size=(80, 5)), 10)
        self.assertLess(neighbor_jaccard(self.neighbors, unrelated)["mean"], 0.5)

    def test_invalid_k_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "k=80 is invalid"):
            exact_knn_indices(self.geometry, 80)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            trustworthiness_scores(self.geometry, self.geometry, [10.5])

    def test_rank_agreement_reports_insufficient_shared_neighbors(self) -> None:
        reference = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        candidate = np.array([[3, 2, 1], [3, 0, 2], [3, 0, 1], [2, 0, 1]])
        result = local_rank_agreement(reference, candidate, minimum_shared_neighbors=4)
        self.assertEqual(result["eligible_rows"], 0)
        self.assertEqual(result["insufficient_shared_rows"], 4)
        self.assertIsNone(result["mean"])
        self.assertTrue(np.isnan(result["per_row"]).all())

    def test_rank_agreement_uses_only_shared_neighbor_ranks(self) -> None:
        reference = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        reversed_ranks = np.array([[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]])
        result = local_rank_agreement(reference, reversed_ranks)
        np.testing.assert_allclose(result["per_row"], -1.0)
        self.assertEqual(result["eligible_rows"], 4)

    def test_cross_seed_stability_ignores_rotation_and_reflection(self) -> None:
        rotation_reflection = np.diag([-1.0, 1.0, -1.0, 1.0, 1.0])
        row_ids = [f"row-{index}" for index in range(len(self.geometry))]
        result = cross_seed_neighborhood_stability(
            {"seed_42": self.geometry, "seed_123": self.geometry @ rotation_reflection},
            row_ids={"seed_42": row_ids, "seed_123": row_ids},
            k_values=[10],
        )
        self.assertEqual(result["mean_jaccard"], 1.0)

    def test_cross_method_comparison_aligns_row_ids_and_consensus_is_explicit(self) -> None:
        row_ids = [f"row-{index}" for index in range(len(self.geometry))]
        result = cross_method_neighbor_agreement(
            {"pca-15": self.geometry, "ae-15": self.geometry[::-1]},
            row_ids={"pca-15": row_ids, "ae-15": row_ids[::-1]},
            k=10,
        )
        self.assertEqual(result["pairs"][0]["neighbor_jaccard"], 1.0)
        self.assertEqual(result["consensus"]["minimum_methods"], 2)
        self.assertTrue(all(result["consensus"]["recurring_neighbors"]))
        graphs = {"pca-15": self.neighbors, "ae-15": self.neighbors}
        consensus = consensus_neighbors(graphs, row_ids={name: row_ids for name in graphs})
        self.assertTrue(
            all(
                set(row) == {row_ids[neighbor] for neighbor in self.neighbors[index]}
                for index, row in enumerate(consensus["recurring_neighbors"])
            )
        )
        with self.assertRaisesRegex(ValueError, "row_ids are required"):
            cross_method_neighbor_agreement({"pca-15": self.geometry, "ae-15": self.geometry}, k=10)

    def test_labelled_outcomes_cannot_enter_neighborhood_geometry(self) -> None:
        labelled = pd.DataFrame(
            {
                "isl": [1.0, 2.0, 3.0],
                "metrics_tput_per_gpu": [10.0, 20.0, 30.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "outcome columns"):
            exact_knn_indices(labelled, 1)


class GroupAndHierarchyTests(unittest.TestCase):
    def test_grouped_bootstrap_keeps_complete_configuration_groups_and_is_reproducible(self) -> None:
        values = np.array([1.0, 2.0, 10.0, 20.0, 30.0])
        groups = np.array(["a", "a", "b", "b", "b"], dtype=object)
        first = grouped_bootstrap(values, groups, replicates=12, seed=42)
        second = grouped_bootstrap(values, groups, replicates=12, seed=42)
        self.assertEqual(first["sampled_groups"], second["sampled_groups"])
        np.testing.assert_array_equal(first["replicate_values"], second["replicate_values"])
        group_rows = {"a": 2, "b": 3}
        for selected, row_count in zip(
            first["sampled_groups"], first["sampled_row_counts"], strict=True
        ):
            self.assertEqual(row_count, sum(group_rows[group] for group in selected))
        group_values = {"a": values[:2], "b": values[2:]}
        for selected, estimate in zip(first["sampled_groups"], first["replicate_values"], strict=True):
            expected = np.mean(np.concatenate([group_values[group] for group in selected]))
            self.assertEqual(estimate, expected)
        compact = grouped_bootstrap(values, groups, replicates=2, seed=42, return_samples=False)
        self.assertNotIn("sampled_groups", compact)
        self.assertNotIn("sampled_row_counts", compact)

    def test_workload_purity_on_hand_constructed_hierarchy(self) -> None:
        neighbors = np.array([[1, 2], [0, 3], [3, 0], [2, 1]])
        workloads = [(1, 1, 1), (1, 1, 1), (2, 2, 1), (2, 2, 1)]
        purity = workload_neighborhood_purity(neighbors, workloads)
        np.testing.assert_allclose(purity["per_row"], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(purity["mean"], 0.5)
        with self.assertRaisesRegex(ValueError, "non-missing"):
            workload_neighborhood_purity(neighbors, [(1, 1, None)] * 4)

    def test_within_workload_null_distinguishes_positive_and_nonpositive_cases(self) -> None:
        # Two workload cells, each split into two locally coherent configuration values.
        workloads = [(1, 1, 1)] * 6 + [(2, 2, 2)] * 6
        geometry = np.array([[index] for index in range(12)], dtype=float)
        graph = np.array(
            [
                [1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4],
                [7, 8], [6, 8], [6, 7], [10, 11], [9, 11], [9, 10],
            ]
        )
        # The helper also reports enough rows/cells for the requested local graph.
        _local, eligibility = within_workload_knn_indices(geometry, workloads, k=2)
        self.assertEqual(eligibility["eligible_cells"], 2)
        self.assertEqual(eligibility["eligible_rows"], 12)
        positive = within_workload_permutation_null(
            graph,
            workloads,
            ["a"] * 3 + ["b"] * 3 + ["a"] * 3 + ["b"] * 3,
            value_kind="categorical",
            permutations=199,
            seed=42,
        )
        self.assertEqual(positive["mean"], 1.0)
        self.assertLess(positive["p_value"], 0.05)
        nonpositive = within_workload_permutation_null(
            graph,
            workloads,
            ["a", "b", "b", "a", "a", "b"] * 2,
            value_kind="categorical",
            permutations=199,
            seed=42,
        )
        self.assertGreater(nonpositive["p_value"], 0.2)
        self.assertLessEqual(nonpositive["mean"], positive["mean"])
        fixed_within_cells = within_workload_permutation_null(
            graph,
            workloads,
            ["a"] * 6 + ["b"] * 6,
            value_kind="categorical",
            permutations=11,
            seed=42,
        )
        np.testing.assert_allclose(fixed_within_cells["null_values"], 1.0)
        self.assertIsNotNone(categorical_neighbor_homophily(graph, ["a"] * 12)["mean"])
        combined = within_workload_configuration_analysis(
            geometry,
            workloads,
            [0, 0, 0, 1, 1, 1] * 2,
            k=2,
            value_kind="numeric",
            permutations=5,
            seed=42,
        )
        self.assertEqual(combined["eligibility"]["eligible_rows"], 12)
        self.assertIsNotNone(combined["observed"]["mean"])


class MixedDistanceTests(unittest.TestCase):
    def test_mixed_type_distance_has_hand_computable_values_and_missing_policy(self) -> None:
        frame = source_frame()
        frame.loc[2, "config_framework"] = None
        distance = source_mixed_distance_block(frame.iloc[:1], frame)
        # Row 0 to row 1 differs only in normalized isl: one equal-weight source feature.
        self.assertAlmostEqual(float(distance[0, 1]), 1 / len(PCA_FEATURES))
        # Row 0 to row 2 differs in hardware plus a one-sided missing framework value.
        self.assertAlmostEqual(float(distance[0, 2]), 2 / len(PCA_FEATURES))

    def test_mixed_distance_weights_original_source_features_equally(self) -> None:
        frame = source_frame()
        distance = source_mixed_distance_block(frame, frame)
        self.assertAlmostEqual(float(distance[0, 1]), float(distance[0, 2]))
        self.assertAlmostEqual(float(distance[0, 1]), 1 / len(PCA_FEATURES))
        graph = source_mixed_knn_indices(frame, 1, chunk_size=2)
        for row, neighbors in enumerate(graph):
            self.assertNotIn(row, neighbors.tolist())

    def test_mixed_distance_zero_range_and_both_missing_are_explicit(self) -> None:
        reference = source_frame(rows=1)
        query = reference.copy()
        query.loc[0, "osl"] = 999.0
        query.loc[0, "config_framework"] = None
        reference.loc[0, "config_framework"] = None
        spec = fit_source_mixed_distance_spec(reference)
        distance = source_mixed_distance_block(query, reference, spec=spec)
        self.assertEqual(float(distance[0, 0]), 0.0)

    def test_mixed_distance_clips_out_of_range_numeric_differences_per_feature(self) -> None:
        reference = source_frame(rows=2)
        query = reference.iloc[:1].copy()
        query.loc[0, "isl"] = 10.0
        distance = source_mixed_distance_block(query, reference)
        self.assertAlmostEqual(float(distance[0, 0]), 1 / len(PCA_FEATURES))

    def test_outcomes_cannot_enter_mixed_distance_feature_path(self) -> None:
        frame = source_frame()
        with self.assertRaisesRegex(ValueError, "frozen configuration/workload schema"):
            fit_source_mixed_distance_spec(
                frame.assign(metrics_tput_per_gpu=1.0),
                feature_order=(*PCA_FEATURES, "metrics_tput_per_gpu"),
            )
        baseline = source_mixed_distance_block(frame.iloc[:1], frame)
        with_outcome = source_mixed_distance_block(
            frame.assign(metrics_tput_per_gpu=[1.0, 1000.0, -1000.0]).iloc[:1],
            frame.assign(metrics_tput_per_gpu=[1.0, 1000.0, -1000.0]),
        )
        np.testing.assert_allclose(with_outcome, baseline)


if __name__ == "__main__":
    unittest.main()
