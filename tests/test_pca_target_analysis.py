from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from apps import inferencex_pca_demo as app
from modeling.pca_target_analysis import (
    ENERGY_TARGET,
    OUTPUT_TARGET,
    PCA_FEATURES,
    compare_bases,
    explained_variance_table,
    fit_shared_pca,
    target_overlay,
    validate_pca_feature_schema,
)


ARTIFACT = Path("artifacts/pca-db-dump-2026-07-20.json")


class PcaTargetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_active_dump_and_exact_july_counts(self) -> None:
        self.assertEqual(app.ACTIVE_DUMP_VERSION, "db-dump/2026-07-20")
        self.assertTrue(app.DEFAULT_DATA_DIR.endswith("db-dump-2026-07-20"))
        self.assertEqual(
            self.artifact["counts"],
            {
                "raw_rows": 81_851,
                "aggregate_rows": 8_239,
                "configurations": 1_368,
                "energy_raw_rows": 5_175,
                "energy_aggregate_groups": 2_766,
                "energy_configurations": 305,
            },
        )

    def test_pca_feature_schema_has_no_outcome_leakage_and_preserves_order(self) -> None:
        validate_pca_feature_schema(PCA_FEATURES)
        self.assertEqual(self.artifact["shared_basis"]["feature_order"], list(PCA_FEATURES))
        self.assertEqual(self.artifact["shared_basis"]["target_metrics_in_inputs"], [])
        for feature in PCA_FEATURES:
            lowered = feature.lower()
            self.assertFalse(feature.startswith("metrics_"))
            self.assertFalse(any(term in lowered for term in ("latency", "throughput", "power", "energy", "tput", "tpot", "ttft", "itl", "e2el")))
        with self.assertRaises(ValueError):
            validate_pca_feature_schema([*PCA_FEATURES[:-1], OUTPUT_TARGET])

    def test_artifact_has_dump_provenance_and_valid_explained_variance(self) -> None:
        self.assertEqual(self.artifact["dump"]["version"], app.ACTIVE_DUMP_VERSION)
        self.assertIn("official_release_url", self.artifact["dump"])
        explained = self.artifact["shared_basis"]["explained_variance"]
        ratios = np.array([row["explained_variance_ratio"] for row in explained])
        cumulative = np.array([row["cumulative_explained_variance"] for row in explained])
        self.assertAlmostEqual(float(ratios.sum()), 1.0, places=10)
        np.testing.assert_allclose(cumulative, np.cumsum(ratios))
        encoded = self.artifact["shared_basis"]["preprocessing"]["encoded_feature_names"]
        components = self.artifact["shared_basis"]["preprocessing"]["pca_components"]
        self.assertTrue(all(len(component) == len(encoded) for component in components))

    def test_exact_selected_target_is_raw_identity_throughput(self) -> None:
        target = self.artifact["targets"][OUTPUT_TARGET]
        self.assertEqual(target["raw_target"], "metrics_tput_per_gpu")
        self.assertEqual(target["transformation"], "identity")
        self.assertEqual(target["inverse_transformation"], "identity")
        self.assertEqual(target["unit"], "tokens/second/GPU")
        self.assertEqual(target["direction"], "higher is better")
        validation = target["historical_validation_context"]
        self.assertEqual(validation["rows"], 4096)
        self.assertEqual(validation["folds"], 3)
        self.assertEqual(validation["grouping"], "config_id")
        self.assertAlmostEqual(validation["r2_mean"], 0.961979)
        self.assertAlmostEqual(validation["mae"], 338.540384)

    def test_energy_is_observed_only_and_not_imputed(self) -> None:
        target = self.artifact["targets"][ENERGY_TARGET]
        self.assertEqual(target["raw_measured_rows"], 5_175)
        self.assertEqual(target["usable_rows"], 2_766)
        self.assertEqual(target["unique_configurations"], 305)
        self.assertEqual(target["workload_support"]["benchmark_type"], ["single_turn"])
        self.assertEqual(target["workload_support"]["isl"], [1024.0, 8192.0])
        self.assertEqual(target["workload_support"]["osl"], [1024.0])
        self.assertEqual(target["direction"], "lower is better")


class PcaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, source = app.data_source_status(app.DEFAULT_DATA_DIR)
        manifest = app.build_dataset_manifest(source)
        benchmarks, _configs, joined, _info = app.load_joined_data(
            app.DEFAULT_DATA_DIR, manifest["fingerprint"]
        )
        aggregate, _metadata = app.build_analysis_frame(
            joined, "Median aggregate per config/workload/concurrency"
        )
        cls.benchmarks = benchmarks
        cls.joined = joined
        cls.aggregate = aggregate
        cls.result = fit_shared_pca(aggregate)

    def test_raw_export_reproduces_canonical_counts(self) -> None:
        self.assertEqual(len(self.benchmarks), 81_851)
        self.assertEqual(len(self.aggregate), 8_239)
        self.assertEqual(self.joined["config_id"].nunique(), 1_368)
        self.assertEqual(self.joined[ENERGY_TARGET].notna().sum(), 5_175)
        self.assertEqual(self.aggregate[ENERGY_TARGET].notna().sum(), 2_766)

    def test_shared_basis_projection_uses_identical_feature_order(self) -> None:
        self.assertEqual(self.result.source_features, list(PCA_FEATURES))
        self.assertEqual(len(self.result.cohort), 8_063)
        self.assertEqual(len(self.result.scores), len(self.result.cohort))
        self.assertFalse(self.result.cohort["benchmark_type"].ne("single_turn").any())

    def test_missing_energy_is_excluded_without_imputation(self) -> None:
        overlay = target_overlay(self.result, ENERGY_TARGET)
        self.assertEqual(overlay["usable_rows"], 2_766)
        self.assertFalse(overlay["frame"][ENERGY_TARGET].isna().any())

    def test_sign_aligned_basis_comparison_handles_sign_flip(self) -> None:
        flipped = copy.deepcopy(self.result)
        flipped.pca.components_[0] *= -1
        comparison = compare_bases(self.result, flipped)
        pc1 = comparison["components"].iloc[0]
        self.assertEqual(pc1["sign_alignment"], -1)
        self.assertAlmostEqual(pc1["cosine_similarity"], 1.0, places=10)
        self.assertAlmostEqual(pc1["loading_correlation"], 1.0, places=10)

    def test_explained_variance_dimensions_are_consistent(self) -> None:
        explained = explained_variance_table(self.result)
        self.assertEqual(len(explained), len(self.result.pca.components_))
        self.assertEqual(len(self.result.pca.components_[0]), len(self.result.encoded_feature_names))
        self.assertAlmostEqual(explained["explained_variance_ratio"].sum(), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
