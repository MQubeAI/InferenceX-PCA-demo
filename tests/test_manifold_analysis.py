from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling.manifold_analysis import (
    STAGE5_GROUPED_BOOTSTRAP_REPLICATES,
    STAGE5_GROUPED_BOOTSTRAP_SEED,
    STAGE5_NEIGHBORHOOD_K_VALUES,
    STAGE5_SCHEMA_VERSION,
    TSNE_2_GRID,
    TSNE_CANONICAL_DISPLAY,
    TSNEConfig,
    UMAP_15_GRID,
    UMAP_2_GRID,
    UMAP_CANONICAL_DISPLAY,
    UMAPConfig,
    canonical_stage5_data,
    fit_stage5_tsne,
    fit_stage5_umap,
    grouped_umap_transform,
    stage5_plan,
    validate_projection_alignment,
    validate_stage5_data,
    validate_stage5_structural_columns,
)
from modeling.pca_target_analysis import PCA_FEATURES
from modeling.representation_analysis import fit_fold_preprocessor
from scripts import run_manifold_analysis as runner


HAS_UMAP = importlib.util.find_spec("umap") is not None


class ManifoldProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.random.default_rng(42).normal(size=(80, 7)).astype(np.float32)
        self.encoded_feature_names = tuple(
            f"num__{feature}" for feature in PCA_FEATURES[: self.matrix.shape[1]]
        )

    @staticmethod
    def canonical_fixture() -> object:
        rows = []
        for config in range(20):
            for workload in range(4):
                row = {
                    "config_id": f"config-{config}",
                    "benchmark_type": "single_turn",
                    "metrics_tput_per_gpu": float(config),
                    "metrics_joules_per_output_token": float(config + 1),
                    "metrics_median_tpot": float(config + 2),
                }
                for feature in PCA_FEATURES:
                    if feature in {
                        "config_hardware",
                        "config_framework",
                        "config_model",
                        "config_precision",
                        "config_spec_method",
                    }:
                        row[feature] = "validation-only" if config >= 15 else "training"
                    elif feature.startswith("config_"):
                        row[feature] = float(config % 2)
                    elif feature == "isl":
                        row[feature] = float(1 + workload)
                    elif feature == "osl":
                        row[feature] = float(16 + workload)
                    elif feature == "conc":
                        row[feature] = float(1 + workload)
                    else:
                        row[feature] = float(workload)
                rows.append(row)
        return canonical_stage5_data(pd.DataFrame(rows), enforce_snapshot_counts=False)

    def test_frozen_grid_cardinality_and_protocol_constants(self) -> None:
        self.assertEqual(len(UMAP_2_GRID), 27)
        self.assertEqual(len(UMAP_15_GRID), 27)
        self.assertEqual(len(TSNE_2_GRID), 9)
        self.assertEqual(STAGE5_NEIGHBORHOOD_K_VALUES, (10, 30, 50))
        self.assertEqual(STAGE5_GROUPED_BOOTSTRAP_REPLICATES, 200)
        self.assertEqual(STAGE5_GROUPED_BOOTSTRAP_SEED, 42)
        self.assertEqual(UMAP_CANONICAL_DISPLAY, {"n_neighbors": 15, "min_dist": 0.1, "seed": 42})
        self.assertEqual(TSNE_CANONICAL_DISPLAY, {"perplexity": 30, "seed": 42})
        self.assertEqual(stage5_plan()["schema_version"], STAGE5_SCHEMA_VERSION)

    def test_tsne_output_is_two_dimensional_and_reproducible(self) -> None:
        config = TSNEConfig(perplexity=15, seed=42)
        first = fit_stage5_tsne(
            self.matrix, config, encoded_feature_names=self.encoded_feature_names
        )
        second = fit_stage5_tsne(
            self.matrix, config, encoded_feature_names=self.encoded_feature_names
        )
        self.assertEqual(first.coordinates.shape, (80, 2))
        np.testing.assert_allclose(first.coordinates, second.coordinates, rtol=1e-6, atol=1e-6)
        self.assertIsNotNone(first.final_kl_divergence)
        self.assertFalse(hasattr(first, "transform"))

    def test_stage5_tsne_rejects_unsupported_dimensions_before_fitting(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            fit_stage5_tsne(
                self.matrix,
                TSNEConfig(perplexity=15, seed=42, n_components=3),
                encoded_feature_names=self.encoded_feature_names,
            )

    def test_stage5_umap_rejects_unsupported_grid_member_before_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported parameter"):
            fit_stage5_umap(
                self.matrix,
                UMAPConfig(n_components=2, n_neighbors=16, min_dist=0.1, seed=42),
                encoded_feature_names=self.encoded_feature_names,
            )

    @unittest.skipUnless(HAS_UMAP, "umap-learn is a research dependency not installed in this environment")
    def test_umap_shape_transform_and_seed_reproducibility(self) -> None:
        config = UMAPConfig(n_components=2, n_neighbors=15, min_dist=0.1, seed=42)
        first = fit_stage5_umap(
            self.matrix, config, encoded_feature_names=self.encoded_feature_names
        )
        second = fit_stage5_umap(
            self.matrix, config, encoded_feature_names=self.encoded_feature_names
        )
        self.assertEqual(first.coordinates.shape, (80, 2))
        np.testing.assert_allclose(first.coordinates, second.coordinates, rtol=1e-6, atol=1e-6)
        self.assertEqual(
            first.transform(
                self.matrix[:5], encoded_feature_names=self.encoded_feature_names
            ).shape,
            (5, 2),
        )

    def test_grouped_umap_transform_uses_train_only_preprocessing(self) -> None:
        data = self.canonical_fixture()
        groups = data.cohort["config_id"].astype(str)
        train = np.flatnonzero(groups.str.replace("config-", "").astype(int) < 15)
        validation = np.flatnonzero(groups.str.replace("config-", "").astype(int) >= 15)
        split = {
            "train_indices": train.tolist(),
            "validation_indices": validation.tolist(),
            "train_row_ids": [data.row_ids[index] for index in train],
            "validation_row_ids": [data.row_ids[index] for index in validation],
        }
        captured = []

        class FakeUMAP:
            def transform(self, values: object, **_kwargs: object) -> np.ndarray:
                return np.zeros((len(values), 2))

        def fake_fit(values: object, _config: object, **_kwargs: object) -> FakeUMAP:
            captured.append(np.asarray(values))
            return FakeUMAP()

        with patch("modeling.manifold_analysis.fit_stage5_umap", side_effect=fake_fit):
            result = grouped_umap_transform(
                data,
                split,
                UMAPConfig(n_components=2, n_neighbors=15, min_dist=0.1, seed=42),
            )
        expected = fit_fold_preprocessor(data, train, validation)
        np.testing.assert_allclose(captured[0], expected.train_matrix)
        self.assertNotIn("cat__config_hardware_validation-only", result["encoded_feature_names"])
        self.assertEqual(result["validation_coordinates"].shape, (len(validation), 2))
        self.assertEqual(result["group_overlap"], 0)

    def test_row_alignment_and_outcome_leakage_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_projection_alignment(np.zeros((3, 2)), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "frozen configuration/workload schema"):
            validate_stage5_structural_columns((*PCA_FEATURES, "metrics_tput_per_gpu"))
        with self.assertRaisesRegex(ValueError, "outcome columns"):
            fit_stage5_tsne(
                pd.DataFrame(
                    {
                        "isl": [1.0, 2.0, 3.0],
                        "metrics_tput_per_gpu": [1.0, 2.0, 3.0],
                    }
                ),
                TSNEConfig(perplexity=15, seed=42),
            )
        with self.assertRaisesRegex(ValueError, "require audited encoded_feature_names"):
            fit_stage5_tsne(self.matrix, TSNEConfig(perplexity=15, seed=42))
        invalid = SimpleNamespace(
            matrix=np.zeros((4, 2)),
            cohort=pd.DataFrame({"config_id": ["a", "b", "c"]}),
            row_ids=["a", "b", "c", "d"],
            encoded_feature_names=["x", "y"],
        )
        with self.assertRaisesRegex(ValueError, "identical row counts"):
            validate_stage5_data(invalid)
        encoded_outcome = SimpleNamespace(
            matrix=np.zeros((3, 2)),
            cohort=pd.DataFrame({"config_id": ["a", "b", "c"]}),
            row_ids=["a", "b", "c"],
            encoded_feature_names=["num__isl", "num__metrics_tput_per_gpu"],
        )
        with self.assertRaisesRegex(ValueError, "non-structural or outcome"):
            validate_stage5_data(encoded_outcome)

    def test_runner_without_execute_never_calls_real_fits_or_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unavailable = f"{directory}/not-a-checkpoint"
            output = io.StringIO()
            with patch.object(runner, "_execute_frozen_projection_fits") as execute:
                with contextlib.redirect_stdout(output):
                    result = runner.main(["--data-dir", unavailable])
            self.assertEqual(result, 0)
            execute.assert_not_called()
            self.assertEqual(list(__import__("pathlib").Path(directory).iterdir()), [])
            self.assertIn("Validation-only", output.getvalue())

    def test_dashboard_import_stays_free_of_torch_and_umap(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import apps.inferencex_pca_demo; "
                "raise SystemExit(1 if {'torch', 'umap', 'numba', 'pynndescent'} & set(sys.modules) else 0)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
