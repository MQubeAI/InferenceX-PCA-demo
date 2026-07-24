from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path

from apps import inferencex_pca_demo as app
from modeling.pca_target_analysis import OUTCOME_PREFIXES, PCA_FEATURES
from modeling.representation_analysis import (
    EXPECTED_COHORT_ROWS,
    canonical_representation_data,
    grouped_split_definitions,
    load_final_representation_artifact,
    validate_final_representation_artifact,
    validate_representation_artifact,
)

PCA_ARTIFACT = Path("artifacts/pca-db-dump-2026-07-20.json")
AE_ARTIFACT = Path("artifacts/representation-ae-db-dump-2026-07-20.json")
VAE_ARTIFACT = Path("artifacts/representation-vae-db-dump-2026-07-20.json")
COMPARISON_ARTIFACT = Path(
    "artifacts/representation-comparison-db-dump-2026-07-20.json"
)
FINAL_AE_ARTIFACT = Path(
    "artifacts/representation-ae-final-db-dump-2026-07-20.json"
)
FINAL_VAE_ARTIFACT = Path(
    "artifacts/representation-vae-final-db-dump-2026-07-20.json"
)
FINAL_COMPARISON_ARTIFACT = Path(
    "artifacts/representation-comparison-final-db-dump-2026-07-20.json"
)
VAE_BETA_DIAGNOSTIC = Path(
    "artifacts/representation-vae-beta-diagnostic-db-dump-2026-07-20.json"
)
STAGE4_VALIDATION_ARTIFACT = Path(
    "artifacts/representation-validation-stage4-db-dump-2026-07-20.json"
)


class SharedRepresentationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, probe = app.data_source_status(app.DEFAULT_DATA_DIR)
        manifest = app.build_dataset_manifest(probe)
        _benchmarks, _configs, joined, _source = app.load_joined_data(
            app.DEFAULT_DATA_DIR, manifest["fingerprint"]
        )
        aggregate, _metadata = app.build_analysis_frame(
            joined, "Median aggregate per config/workload/concurrency"
        )
        cls.data = canonical_representation_data(aggregate)

    def test_shared_cohort_is_exact_and_row_aligned(self) -> None:
        self.assertEqual(len(self.data.cohort), EXPECTED_COHORT_ROWS)
        self.assertEqual(len(self.data.matrix), EXPECTED_COHORT_ROWS)
        self.assertEqual(len(self.data.row_ids), EXPECTED_COHORT_ROWS)
        self.assertEqual(len(set(self.data.row_ids)), EXPECTED_COHORT_ROWS)
        self.assertEqual(self.data.cohort["config_id"].nunique(), 1_354)

    def test_feature_order_is_frozen_and_outcomes_are_excluded(self) -> None:
        self.assertEqual(
            list(PCA_FEATURES),
            [
                "isl",
                "osl",
                "conc",
                "config_prefill_tp",
                "config_prefill_ep",
                "config_prefill_dp_attention",
                "config_prefill_num_workers",
                "config_decode_tp",
                "config_decode_ep",
                "config_decode_dp_attention",
                "config_decode_num_workers",
                "config_num_prefill_gpu",
                "config_hardware",
                "config_framework",
                "config_model",
                "config_precision",
                "config_spec_method",
                "config_disagg",
                "config_is_multinode",
            ],
        )
        self.assertFalse(any(feature.startswith(OUTCOME_PREFIXES) for feature in PCA_FEATURES))

    def test_grouped_splits_have_no_configuration_leakage(self) -> None:
        groups = self.data.cohort["config_id"].astype(str)
        for split in grouped_split_definitions(self.data):
            train = set(groups.iloc[split["train_indices"]])
            validation = set(groups.iloc[split["validation_indices"]])
            self.assertFalse(train & validation)
            self.assertEqual(split["group_overlap"], 0)

    def test_pca_artifact_remains_byte_unchanged(self) -> None:
        self.assertEqual(
            hashlib.sha256(PCA_ARTIFACT.read_bytes()).hexdigest(),
            "7857c1e0e3d29ee7dc1c7027a6fa2872d276dda2ad56a265cc1b9f0dd28951c8",
        )


class RepresentationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ae = json.loads(AE_ARTIFACT.read_text(encoding="utf-8"))
        cls.vae = json.loads(VAE_ARTIFACT.read_text(encoding="utf-8"))
        cls.comparison = json.loads(COMPARISON_ARTIFACT.read_text(encoding="utf-8"))

    def test_ae_and_vae_metadata_are_valid_and_compatible(self) -> None:
        validate_representation_artifact(self.ae, expected_method="autoencoder")
        validate_representation_artifact(
            self.vae,
            expected_method="variational_autoencoder",
            expected_cohort_hash=self.ae["cohort_hash"],
        )
        self.assertEqual(self.ae["basis_row_identifiers"], self.vae["basis_row_identifiers"])
        self.assertEqual(self.ae["feature_order"], self.vae["feature_order"])
        self.assertEqual(self.ae["target_metrics_in_inputs"], [])
        self.assertEqual(self.vae["target_metrics_in_inputs"], [])

    def test_comparison_checks_exact_cohort_and_feature_compatibility(self) -> None:
        self.assertTrue(self.comparison["compatible_cohort"])
        self.assertEqual(self.comparison["cohort_hash"], self.ae["cohort_hash"])
        self.assertEqual(self.comparison["row_key_hash"], self.ae["row_key_hash"])
        self.assertEqual(self.comparison["feature_order"], list(PCA_FEATURES))
        self.assertEqual(self.comparison["status"], "preliminary")

    def test_vae_reports_collapse_diagnostics(self) -> None:
        diagnostics = self.vae["selected_result"]["diagnostics"]
        self.assertIn("average_kl_per_latent_dimension", diagnostics)
        self.assertIn("active_latent_dimensions", diagnostics)
        self.assertIn("latent_variance", diagnostics)
        self.assertIn("decoder_ignores_latent_changes", diagnostics)
        self.assertIn("posterior_collapse", diagnostics)

    def test_streamlit_import_does_not_import_neural_framework(self) -> None:
        completed = subprocess.run(
            [
                str(Path(".venv-streamlit/bin/python")),
                "-c",
                "import sys; import apps.inferencex_pca_demo; "
                "raise SystemExit(1 if 'torch' in sys.modules else 0)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class FinalRepresentationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ae, cls.ae_companion = load_final_representation_artifact(
            FINAL_AE_ARTIFACT,
            expected_method="autoencoder",
        )
        cls.vae, cls.vae_companion = load_final_representation_artifact(
            FINAL_VAE_ARTIFACT,
            expected_method="variational_autoencoder",
            expected_cohort_hash=cls.ae["cohort_hash"],
        )
        cls.comparison = json.loads(
            FINAL_COMPARISON_ARTIFACT.read_text(encoding="utf-8")
        )
        cls.beta = json.loads(VAE_BETA_DIAGNOSTIC.read_text(encoding="utf-8"))

    def test_final_seeds_folds_cohort_and_features_are_identical(self) -> None:
        expected_seeds = [42, 123, 2026]
        self.assertEqual(self.ae["random_seeds"], expected_seeds)
        self.assertEqual(self.vae["random_seeds"], expected_seeds)
        self.assertEqual(self.ae["split_definitions"], self.vae["split_definitions"])
        self.assertEqual(self.ae["cohort_hash"], self.vae["cohort_hash"])
        self.assertEqual(self.ae["row_key_hash"], self.vae["row_key_hash"])
        self.assertEqual(self.ae["feature_order"], self.vae["feature_order"])
        self.assertEqual(self.ae["target_metrics_in_inputs"], [])
        self.assertEqual(self.vae["target_metrics_in_inputs"], [])
        for split in self.ae["split_definitions"]:
            self.assertEqual(split["group_overlap"], 0)
            self.assertFalse(
                set(split["train_row_ids"]) & set(split["validation_row_ids"])
            )

    def test_ae_records_early_stopping_and_extended_cap(self) -> None:
        self.assertEqual(self.ae["hyperparameters"]["maximum_epochs"], 250)
        self.assertEqual(self.ae["hyperparameters"]["early_stopping_patience"], 12)
        self.assertEqual(len(self.ae["runs"]), 9)
        for run in self.ae["runs"]:
            self.assertIn("early_stopping_occurred", run)
            self.assertIn(
                run["stopping_reason"],
                {"early_stopping_patience", "maximum_epoch_cap"},
            )
            self.assertLessEqual(run["epochs_trained"], 250)

    def test_vae_beta_and_active_dimension_metadata(self) -> None:
        self.assertEqual(self.beta["selected_beta"], 0.1)
        self.assertEqual(
            [row["beta"] for row in self.beta["results"]],
            [0.1, 0.5, 1.0],
        )
        self.assertEqual(self.vae["hyperparameters"]["beta"], 0.1)
        active = [
            row["diagnostics"]["active_latent_dimensions"] for row in self.vae["runs"]
        ]
        self.assertGreaterEqual(min(active), 5)
        self.assertLess(max(active), 15)
        self.assertFalse(
            any(row["diagnostics"]["posterior_collapse"] for row in self.vae["runs"])
        )

    def test_multi_seed_summaries_recompute_exactly(self) -> None:
        for artifact in (self.ae, self.vae):
            mse = [row["mse"] for row in artifact["runs"]]
            mae = [row["mae"] for row in artifact["runs"]]
            self.assertAlmostEqual(
                artifact["summary"]["validation_mse"]["mean"],
                statistics.mean(mse),
            )
            self.assertAlmostEqual(
                artifact["summary"]["validation_mse"]["standard_deviation"],
                statistics.stdev(mse),
            )
            self.assertAlmostEqual(
                artifact["summary"]["validation_mae"]["mean"],
                statistics.mean(mae),
            )
            self.assertAlmostEqual(
                artifact["summary"]["validation_mae"]["standard_deviation"],
                statistics.stdev(mae),
            )

    def test_parquet_companions_are_compact_aligned_and_versioned(self) -> None:
        import pandas as pd

        for artifact, path in (
            (self.ae, self.ae_companion),
            (self.vae, self.vae_companion),
        ):
            companion = artifact["embedding_companion"]
            self.assertEqual(companion["format"], "parquet")
            self.assertEqual(companion["compression"], "zstd")
            self.assertLess(companion["bytes"], 3_000_000)
            frame = pd.read_parquet(path)
            self.assertEqual(len(frame), 8_063 * 3)
            self.assertEqual(sorted(frame["seed"].unique().tolist()), [42, 123, 2026])
            self.assertIn("cluster", frame)
            self.assertIn("fold_id", frame)
            self.assertEqual(sorted(frame["fold_id"].unique().tolist()), [0])

    def test_missing_companion_has_a_useful_error(self) -> None:
        changed = json.loads(json.dumps(self.ae))
        changed["embedding_companion"]["filename"] = "missing-final-embeddings.parquet"
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "artifact.json"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Embedding companion is missing.*Restore the matching Parquet",
            ):
                validate_final_representation_artifact(
                    changed,
                    artifact_path=artifact_path,
                    expected_method="autoencoder",
                )

    def test_final_comparison_is_matched_and_dashboard_uses_uncertainty(self) -> None:
        self.assertEqual(self.comparison["status"], "final")
        self.assertEqual(self.comparison["latent_dimension"], 15)
        self.assertEqual(self.comparison["random_seeds"], [42, 123, 2026])
        self.assertTrue(self.comparison["compatible_cohort"])
        self.assertEqual(
            set(self.comparison["methods"]),
            {"PCA-15", "Autoencoder-15", "VAE-15"},
        )
        source = inspect.getsource(app.render_representation_comparison_dashboard)
        self.assertIn('"MSE std"', source)
        self.assertIn('"R² std"', source)
        self.assertIn("PCA-5 compact reference", source)
        self.assertIn("No universal winner", source)


class Stage4ValidationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = json.loads(
            STAGE4_VALIDATION_ARTIFACT.read_text(encoding="utf-8")
        )

    def test_frozen_cohort_features_and_pca_are_preserved(self) -> None:
        artifact = self.artifact
        self.assertEqual(artifact["schema_version"], "representation-validation-stage4-v1")
        self.assertEqual(artifact["cohort_rows"], 8_063)
        self.assertEqual(artifact["configurations"], 1_354)
        self.assertEqual(artifact["feature_order"], list(PCA_FEATURES))
        self.assertEqual(artifact["target_metrics_in_inputs"], [])
        self.assertEqual(
            artifact["pca_artifact_sha256_before"],
            artifact["pca_artifact_sha256_after"],
        )
        self.assertEqual(
            artifact["pca_artifact_sha256_after"],
            hashlib.sha256(PCA_ARTIFACT.read_bytes()).hexdigest(),
        )

    def test_every_partition_is_grouped_and_disjoint(self) -> None:
        definitions = self.artifact["split_definitions"]
        self.assertEqual(set(definitions["independent_partitions"]), {"17", "29", "43"})
        for partition in [
            definitions["current"],
            *definitions["independent_partitions"].values(),
        ]:
            self.assertEqual(len(partition), 3)
            validation_groups = []
            for fold in partition:
                self.assertEqual(fold["group_overlap"], 0)
                validation_groups.append(set(fold["validation_config_ids"]))
            self.assertFalse(validation_groups[0] & validation_groups[1])
            self.assertFalse(validation_groups[0] & validation_groups[2])
            self.assertFalse(validation_groups[1] & validation_groups[2])
            self.assertEqual(
                sum(len(groups) for groups in validation_groups),
                self.artifact["configurations"],
            )

    def test_partition_results_have_fixed_neural_seeds(self) -> None:
        for method in ("AE", "VAE"):
            runs = self.artifact["partition_robustness"][method]["runs"]
            self.assertEqual(len(runs), 27)
            self.assertEqual(sorted({run["seed"] for run in runs}), [42, 123, 2026])
            self.assertEqual(sorted({run["partition_seed"] for run in runs}), [17, 29, 43])

    def test_source_feature_metrics_are_balanced_and_complete(self) -> None:
        source = self.artifact["source_feature_reconstruction"]["strict_current"]
        for method in ("PCA", "AE", "VAE"):
            features = source[method]["features"]
            self.assertEqual(len(features), 19)
            self.assertEqual(
                {row["source_feature"] for row in features},
                set(PCA_FEATURES),
            )
            categorical = [row for row in features if row["feature_type"] == "categorical"]
            self.assertTrue(categorical)
            self.assertTrue(
                all(row["exact_accuracy"] is not None for row in categorical)
            )
            self.assertTrue(all(row["top2_accuracy"] is not None for row in categorical))

    def test_bounded_neural_interventions_retain_diagnostics(self) -> None:
        experiments = self.artifact["robustness_experiments"]
        self.assertIn("denoising_5_percent", experiments["autoencoder"])
        diagnostics = experiments["variational_autoencoder"][
            "kl_warmup_diagnostics"
        ]
        self.assertEqual(diagnostics["active_latent_dimensions"], [7, 7, 7])
        self.assertEqual(len(diagnostics["average_kl_per_latent_dimension_by_fold"]), 3)
        self.assertTrue(
            all(
                len(values) == 15
                for values in diagnostics["average_kl_per_latent_dimension_by_fold"]
            )
        )
        self.assertFalse(any(diagnostics["posterior_collapse"]))
        self.assertFalse(any(diagnostics["decoder_ignores_latent_changes"]))

    def test_dashboard_loader_rejects_missing_or_incompatible_validation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                app.load_representation_validation_artifact(str(missing))
            changed = json.loads(json.dumps(self.artifact))
            changed["cohort_rows"] = 8_062
            incompatible = Path(directory) / "incompatible.json"
            incompatible.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cohort is incompatible"):
                app.load_representation_validation_artifact(str(incompatible))


if __name__ == "__main__":
    unittest.main()
