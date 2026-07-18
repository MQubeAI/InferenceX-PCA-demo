from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from apps import inferencex_pca_demo as app


class DashboardUiTests(unittest.TestCase):
    def test_navigation_has_exactly_four_presentable_tabs(self) -> None:
        self.assertEqual(
            app.MAIN_TAB_LABELS,
            ("Overview", "Data Understanding", "PCA", "Model Results"),
        )
        main_source = inspect.getsource(app.main)
        self.assertIn("st.tabs(MAIN_TAB_LABELS)", main_source)
        self.assertNotIn("st.radio(", main_source)
        for label in app.REMOVED_TOP_LEVEL_SECTION_LABELS:
            self.assertNotIn(label, main_source)

    def test_normal_startup_does_not_fit_or_run_models(self) -> None:
        main_source = inspect.getsource(app.main)
        for operation in (
            "fit_pca_analysis",
            "pca_stability_summary",
            "grouped_rf_evaluation",
            "evaluate_models",
            "run_tabfm_comparison_subprocess",
        ):
            self.assertNotIn(operation, main_source)

    def test_csv_first_and_json_fallback_data_loading_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"config_id": [1], "metrics_tput_per_gpu": [10.0]}).to_csv(
                path / "benchmark_results.csv", index=False
            )
            pd.DataFrame({"id": [1], "hardware": ["gpu"]}).to_csv(
                path / "configs.csv", index=False
            )
            benchmarks, configs, joined, source = app.load_joined_data(str(path), "csv-test")
            self.assertEqual(source["active_mode"], "CSV")
            self.assertEqual(len(benchmarks), 1)
            self.assertIn("config_hardware", configs)
            self.assertIn("config_hardware", joined)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "benchmark_results.json").write_text('[{"config_id": 1}]', encoding="utf-8")
            (path / "configs.json").write_text('[{"id": 1, "hardware": "gpu"}]', encoding="utf-8")
            _benchmarks, _configs, joined, source = app.load_joined_data(str(path), "json-test")
            self.assertEqual(source["active_mode"], "JSON fallback")
            self.assertIn("config_hardware", joined)


if __name__ == "__main__":
    unittest.main()
