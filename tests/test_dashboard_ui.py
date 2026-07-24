from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from apps import inferencex_pca_demo as app


class DashboardUiTests(unittest.TestCase):
    def test_navigation_has_exactly_four_presentable_tabs(self) -> None:
        self.assertEqual(
            app.MAIN_TAB_LABELS,
            (
                "Overview",
                "Data Understanding",
                "Representation Analysis",
                "Model Results",
            ),
        )
        main_source = inspect.getsource(app.main)
        self.assertIn("st.tabs(MAIN_TAB_LABELS)", main_source)
        self.assertNotIn("st.radio(", main_source)
        for label in app.REMOVED_TOP_LEVEL_SECTION_LABELS:
            self.assertNotIn(label, main_source)

    def test_overview_formatting_is_compact_without_losing_precision_helpers(self) -> None:
        self.assertEqual(app.format_compact_count(79_830), "79.8K")
        self.assertEqual(app.format_compact_count(7_462), "7,462")
        self.assertEqual(app.format_compact_count(1_197), "1,197")
        self.assertEqual(app.format_overview_r2(0.961979), "0.962")
        self.assertEqual(app.format_overview_mae(338.540384), "338.5")
        self.assertEqual(app.format_overview_percentage(0.9534), "95.3%")

    def test_page_shell_uses_revised_title_collapsed_sidebar_and_research_paused_wording(self) -> None:
        app_source = inspect.getsource(app)
        main_source = inspect.getsource(app.main)
        overview_source = inspect.getsource(app.render_overview)
        self.assertIn('page_title="InferenceX Benchmark Research"', app_source)
        self.assertIn('initial_sidebar_state="collapsed"', app_source)
        self.assertIn("Research dashboard", main_source)
        self.assertIn("Research paused", overview_source)
        self.assertNotIn("Do not continue", overview_source)
        self.assertIn("Exact full-context R²", overview_source)

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

    def test_energy_measurements_preserve_tabs_and_are_observed_only(self) -> None:
        main_source = inspect.getsource(app.main)
        energy_source = inspect.getsource(app.render_energy_measurements_dashboard)
        self.assertEqual(len(app.MAIN_TAB_LABELS), 4)
        self.assertIn("render_energy_measurements_dashboard(joined)", main_source)
        self.assertIn("Find observed measurement", energy_source)
        self.assertIn("Observed benchmark measurements only", energy_source)
        self.assertIn("not predictions", energy_source)
        for forbidden in (".fit(", "model.predict(", "provider.predict(", "CatBoost", "RandomForest", "TabFM"):
            self.assertNotIn(forbidden, energy_source)

    def test_july_pca_sections_preserve_four_tab_shell(self) -> None:
        source = inspect.getsource(app.render_pca_dashboard)
        self.assertEqual(len(app.MAIN_TAB_LABELS), 4)
        self.assertIn("Median TPOT", source)
        self.assertIn("Throughput per GPU", source)
        self.assertIn("Joules per output token", source)
        self.assertIn("Latency-focused descriptive overlay on the shared configuration PCA", source)
        self.assertIn("Final supervised target shown as a descriptive overlay", source)
        self.assertIn("Observed-energy descriptive overlay on the measured subset", source)
        self.assertIn("target is a color/association overlay, not a PCA input", source)
        self.assertIn("Build interactive target projections", source)
        self.assertIn("full eligible", source)
        self.assertIn("dataset in the cumulative July 20 snapshot", source)
        self.assertIn("Use optional log1p color scale (display only)", source)
        self.assertNotIn("grouped_rf_evaluation", source)

    def test_representation_analysis_has_five_subpages_and_pca_is_not_duplicated(self) -> None:
        self.assertEqual(
            app.REPRESENTATION_SUBPAGE_LABELS,
            (
                "Principal Component Analysis",
                "Autoencoder",
                "Variational Autoencoder",
                "Results and Comparison",
                "Research Validation",
            ),
        )
        source = inspect.getsource(app.render_representation_analysis_dashboard)
        main_source = inspect.getsource(app.main)
        self.assertIn("st.tabs(REPRESENTATION_SUBPAGE_LABELS)", source)
        self.assertIn("render_pca_dashboard", source)
        self.assertIn("render_representation_analysis_dashboard", main_source)
        self.assertIn("render_research_validation_dashboard", source)
        self.assertNotIn("render_pca_dashboard(", main_source)

    def test_research_validation_is_artifact_only_and_reports_method_limits(self) -> None:
        source = inspect.getsource(app.render_research_validation_dashboard)
        self.assertIn("Train-only preprocessing sensitivity", source)
        self.assertIn("Equal-weight reconstruction over 19 source features", source)
        self.assertIn("Three independent grouped partition assignments", source)
        self.assertIn("cross-method agreement", source)
        self.assertIn("Ablations are exploratory", source)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("import torch", inspect.getsource(app))

    def test_neural_pages_use_artifacts_and_expose_useful_empty_and_error_states(self) -> None:
        source = inspect.getsource(app.render_neural_representation_dashboard)
        self.assertIn("Training is intentionally", source)
        self.assertIn("artifact is incompatible or unreadable", source)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("import torch", inspect.getsource(app))

    def test_streamlit_apptest_renders_representation_pages_without_errors(self) -> None:
        tested = AppTest.from_file(
            "apps/inferencex_pca_demo.py",
            default_timeout=90,
        ).run()
        self.assertEqual(len(tested.exception), 0)
        self.assertEqual(len(tested.error), 0)
        labels = [tab.label for tab in tested.tabs]
        for label in (*app.MAIN_TAB_LABELS, *app.REPRESENTATION_SUBPAGE_LABELS):
            self.assertIn(label, labels)

    def test_model_results_are_marked_historical_on_july_data(self) -> None:
        source = inspect.getsource(app.render_model_results_dashboard)
        self.assertIn("historical experiments on the June snapshot", source)
        self.assertIn("not applied to cumulative-snapshot rows", source)

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
