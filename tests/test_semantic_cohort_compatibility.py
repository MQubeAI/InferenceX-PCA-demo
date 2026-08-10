from __future__ import annotations

import unittest

import pandas as pd

from apps import inferencex_pca_demo as app
from modeling.dataset_checkpoint import VERIFIED_CHECKPOINT
from modeling.pca_target_analysis import (
    CATEGORICAL_FEATURES,
    ENERGY_TARGET,
    LATENCY_TARGET,
    NUMERIC_FEATURES,
    OUTPUT_TARGET,
    PCA_FEATURES,
)
from modeling.representation_analysis import (
    OUTCOME_TARGETS,
    ROW_KEY_COLUMNS,
    align_companion_to_active_cohort,
    canonical_representation_data,
)


def synthetic_raw_rows() -> pd.DataFrame:
    """Small repeated raw fixture with all frozen PCA inputs and targets."""

    records: list[dict[str, object]] = []
    configurations = (
        (10, 1024, 1024, 8),
        (11, 2048, 1024, 16),
        (12, 4096, 2048, 32),
        (13, 8192, 1024, 64),
    )
    for index, (config_id, isl, osl, conc) in enumerate(configurations):
        for repetition in (0, 1):
            row: dict[str, object] = {
                "config_id": config_id,
                "benchmark_type": "single_turn",
                "isl": isl,
                "osl": osl,
                "conc": conc,
                LATENCY_TARGET: 0.01 * (index + 1) + repetition * 0.001,
                OUTPUT_TARGET: 100.0 * (index + 1) + repetition,
                ENERGY_TARGET: None if index % 2 else 0.1 * (index + 1) + repetition * 0.01,
            }
            for feature in NUMERIC_FEATURES:
                if feature not in row:
                    row[feature] = index + 1
            for feature in CATEGORICAL_FEATURES:
                row[feature] = f"{feature}_{index % 2}"
            records.append(row)
    return pd.DataFrame(records)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    return app.aggregate_analysis_frame(raw, list(ROW_KEY_COLUMNS))


def companion_for(data) -> pd.DataFrame:
    base = data.cohort[["config_id", "isl", "osl", "conc", *OUTCOME_TARGETS]].copy()
    base.insert(0, "row_id", data.row_ids)
    base.insert(0, "seed", 42)
    base["z1"] = range(len(base))
    return base


class SemanticCohortCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = synthetic_raw_rows()
        self.aggregate = aggregate(self.raw)
        self.data = canonical_representation_data(
            self.aggregate, enforce_snapshot_counts=False
        )
        self.companion = companion_for(self.data)

    def test_raw_shuffle_changes_legacy_order_but_not_semantic_identity(self) -> None:
        shuffled = self.raw.sample(frac=1, random_state=2026).reset_index(drop=True)
        shuffled_data = canonical_representation_data(
            aggregate(shuffled), enforce_snapshot_counts=False
        )
        self.assertNotEqual(
            self.data.legacy_ordered_row_key_hash,
            shuffled_data.legacy_ordered_row_key_hash,
        )
        self.assertNotEqual(
            self.data.legacy_ordered_cohort_hash,
            shuffled_data.legacy_ordered_cohort_hash,
        )
        self.assertEqual(self.data.semantic_identity, shuffled_data.semantic_identity)
        _aligned, report = align_companion_to_active_cohort(self.companion, shuffled_data)
        self.assertEqual(report["row_key_set"]["missing_row_ids"], 0)
        self.assertEqual(report["row_key_set"]["extra_row_ids"], 0)

    def test_aggregate_shuffle_has_identical_deterministic_order_and_identity(self) -> None:
        shuffled_aggregate = self.aggregate.sample(frac=1, random_state=99).reset_index(drop=True)
        shuffled_data = canonical_representation_data(
            shuffled_aggregate, enforce_snapshot_counts=False
        )
        self.assertEqual(self.data.row_ids, shuffled_data.row_ids)
        self.assertEqual(self.data.cohort[list(ROW_KEY_COLUMNS)].to_dict("records"),
                         shuffled_data.cohort[list(ROW_KEY_COLUMNS)].to_dict("records"))
        self.assertEqual(self.data.semantic_identity, shuffled_data.semantic_identity)

    def test_historical_companion_sequence_is_accepted_and_aligned_by_row_id(self) -> None:
        shuffled_companion = self.companion.sample(frac=1, random_state=7).reset_index(drop=True)
        aligned, report = align_companion_to_active_cohort(shuffled_companion, self.data)
        self.assertEqual(aligned["row_id"].tolist(), self.data.row_ids)
        self.assertEqual(aligned["z1"].tolist(), self.companion.set_index("row_id").loc[self.data.row_ids, "z1"].tolist())
        self.assertTrue(all(target["exact_match"] for target in report["targets"].values()))

    def test_missing_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "row-ID set mismatch"):
            align_companion_to_active_cohort(self.companion.iloc[1:], self.data)

    def test_extra_row_is_rejected(self) -> None:
        extra = self.companion.iloc[[0]].copy()
        extra["row_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "row-ID set mismatch"):
            align_companion_to_active_cohort(pd.concat([self.companion, extra]), self.data)

    def test_duplicate_companion_row_is_rejected(self) -> None:
        duplicate = pd.concat([self.companion, self.companion.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate row IDs"):
            align_companion_to_active_cohort(duplicate, self.data)

    def test_changed_workload_key_is_rejected(self) -> None:
        changed = self.aggregate.copy()
        changed.loc[changed.index[0], "conc"] = 999
        changed_data = canonical_representation_data(changed, enforce_snapshot_counts=False)
        with self.assertRaisesRegex(ValueError, "row-ID set mismatch"):
            align_companion_to_active_cohort(self.companion, changed_data)

    def test_changed_config_derived_pca_feature_is_detected(self) -> None:
        changed = self.aggregate.copy()
        changed.loc[changed.index[0], "config_prefill_tp"] = 999
        changed_data = canonical_representation_data(changed, enforce_snapshot_counts=False)
        self.assertNotEqual(
            self.data.semantic_identity["pca_input_hash"],
            changed_data.semantic_identity["pca_input_hash"],
        )
        with self.assertRaisesRegex(ValueError, "PCA source features differ"):
            align_companion_to_active_cohort(
                self.companion,
                changed_data,
                expected_pca_input_hash=self.data.semantic_identity["pca_input_hash"],
            )

    def test_changed_target_is_detected(self) -> None:
        changed = self.aggregate.copy()
        changed.loc[changed.index[0], LATENCY_TARGET] = 99.0
        changed_data = canonical_representation_data(changed, enforce_snapshot_counts=False)
        with self.assertRaisesRegex(ValueError, f"target mismatch for {LATENCY_TARGET}"):
            align_companion_to_active_cohort(self.companion, changed_data)

    def test_target_mismatch_in_any_saved_seed_is_detected(self) -> None:
        second_seed = self.companion.copy()
        second_seed["seed"] = 123
        second_seed.loc[second_seed.index[0], OUTPUT_TARGET] = -1.0
        with self.assertRaisesRegex(ValueError, f"target mismatch for {OUTPUT_TARGET}, seed 123"):
            align_companion_to_active_cohort(
                pd.concat([self.companion, second_seed], ignore_index=True), self.data
            )

    def test_all_pca_features_are_present_in_the_semantic_identity(self) -> None:
        self.assertEqual(len(PCA_FEATURES), 19)
        self.assertIn("pca_input_hash", self.data.semantic_identity)


class JulyArtifactSemanticCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _status, source = app.data_source_status(app.DEFAULT_DATA_DIR)
        if source["checkpoint"]["status"] != VERIFIED_CHECKPOINT:
            raise unittest.SkipTest("requires the verified recovered July checkpoint")
        manifest = app.build_dataset_manifest(source)
        _benchmarks, _configs, joined, _source = app.load_joined_data(
            app.DEFAULT_DATA_DIR, manifest["fingerprint"]
        )
        aggregate_frame, _metadata = app.build_analysis_frame(
            joined, "Median aggregate per config/workload/concurrency"
        )
        cls.data = canonical_representation_data(aggregate_frame)

    def test_committed_july_ae_and_vae_companions_accept_recovered_rows_by_identity(self) -> None:
        for method, path in (
            ("autoencoder", app.AE_REPRESENTATION_ARTIFACT_PATH),
            ("variational_autoencoder", app.VAE_REPRESENTATION_ARTIFACT_PATH),
        ):
            _artifact, companion = app.load_neural_representation_artifact(str(path), method)
            aligned, report = align_companion_to_active_cohort(companion, self.data)
            self.assertEqual(len(aligned), 3 * len(self.data.cohort))
            self.assertEqual(report["row_key_set"]["missing_row_ids"], 0)
            self.assertEqual(report["row_key_set"]["extra_row_ids"], 0)
            self.assertTrue(
                all(target["exact_match"] for target in report["targets"].values())
            )


if __name__ == "__main__":
    unittest.main()
