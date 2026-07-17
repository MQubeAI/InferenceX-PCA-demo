from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from apps import inferencex_pca_demo as app


def fixture_frame() -> pd.DataFrame:
    rows = []
    for config_id in range(1, 6):
        for repeat in range(8):
            rows.append(
                {
                    "config_id": config_id,
                    "isl": 128 + 16 * repeat,
                    "conc": 1 + repeat % 4,
                    "config_prefill_tp": 1 + config_id % 3,
                    "config_hardware": f"gpu-{config_id % 2}",
                    "config_framework": f"framework-{config_id % 3}",
                    "metrics_p99_itl": config_id * 0.2 + repeat * 0.03,
                }
            )
    return pd.DataFrame(rows)


class AnalysisWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = fixture_frame()
        self.features = ["isl", "conc", "config_prefill_tp", "config_hardware", "config_framework"]

    def test_signature_changes_with_controls(self) -> None:
        metadata = {"dataset_fingerprint": "dataset", "analysis_unit": "unit", "analysis_row_count": 40}
        first = app.analysis_signature(metadata, "pca", {"seed": 42})
        second = app.analysis_signature(metadata, "pca", {"seed": 43})
        self.assertNotEqual(first, second)

    def test_grouped_evaluation_has_no_group_overlap(self) -> None:
        result, error = app.grouped_rf_evaluation(
            self.frame,
            self.features,
            "metrics_p99_itl",
            max_rows=100,
            seed=42,
            n_estimators=10,
            n_splits=5,
            permutation_repeats=2,
        )
        self.assertEqual(error, "")
        assert result is not None
        self.assertEqual(result["split_mode"], "Grouped cross-validation by config_id")
        self.assertEqual(int(result["fold_metrics"]["group_overlap"].max()), 0)
        self.assertEqual(result["fold_count"], 5)

    def test_pca_stability_is_deterministic(self) -> None:
        first, error = app.pca_stability_summary(self.frame, self.features, 100, 11, 3)
        second, second_error = app.pca_stability_summary(self.frame, self.features, 100, 11, 3)
        self.assertEqual(error, "")
        self.assertEqual(second_error, "")
        self.assertEqual(first["runs"], 3)
        np.testing.assert_allclose(
            first["component_similarity"]["mean_sign_aligned_loading_similarity"],
            second["component_similarity"]["mean_sign_aligned_loading_similarity"],
        )


if __name__ == "__main__":
    unittest.main()
