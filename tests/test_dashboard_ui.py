from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_representation_analysis_has_exact_stage5_subpages_and_pca_is_not_duplicated(self) -> None:
        self.assertEqual(
            app.REPRESENTATION_SUBPAGE_LABELS,
            (
                "Principal Component Analysis",
                "Manifold Analysis",
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
        self.assertIn("render_manifold_analysis_dashboard", source)
        self.assertIn("render_representation_analysis_dashboard", main_source)
        self.assertIn("render_research_validation_dashboard", source)
        self.assertNotIn("render_pca_dashboard(", main_source)

    def test_manifold_page_is_artifact_only_with_fixed_canonical_defaults(self) -> None:
        source = inspect.getsource(app.render_manifold_analysis_dashboard)
        app_source = inspect.getsource(app)
        self.assertIn("Nonlinear Neighborhood Geometry", source)
        self.assertIn("PARTIALLY SUPPORTED", source)
        self.assertIn("not promoted", source)
        self.assertIn("n_neighbors=15, min_dist=0.1", source)
        self.assertIn("seed=42", source)
        self.assertIn("perplexity=30", source)
        self.assertIn("Post-hoc performance and energy overlays", source)
        self.assertIn("canonical_frame", source)
        for figure_title in (
            "All 27 frozen UMAP-2 settings: local-neighborhood sensitivity",
            "Canonical UMAP-2: preservation of original-space neighborhoods",
            "Exact workload purity: original structural space versus canonical UMAP-2",
            "Ablation: local-neighborhood recovery at k=10",
            "Source-balanced mixed-distance sensitivity",
            "Local-neighborhood preservation comparison — not a global geometry leaderboard",
            "Core neighborhood consensus with UMAP-15 as sensitivity only",
            "UMAP-15: in-cohort versus grouped held-out recovery",
        ):
            self.assertIn(figure_title, source)
        self.assertIn("Log display", source)
        self.assertIn("display only", source)
        self.assertNotIn("_manifold_fidelity_table(sensitivity_run)", source)
        self.assertNotIn('method="UMAP-15"', source)
        self.assertNotIn("UMAP-15 sensitivity view", source)
        self.assertIn("aligned_projection_frame", app_source)
        for forbidden in (
            "modeling.manifold_results",
            "fit_stage5_umap",
            "fit_stage5_tsne",
            "run_stage5_results",
            "KMeans",
            "HDBSCAN",
        ):
            self.assertNotIn(forbidden, app_source)

    def test_manifold_primary_labels_are_human_readable(self) -> None:
        self.assertEqual(app._manifold_label("config_prefill_tp"), "Prefill tensor parallelism")
        self.assertEqual(app._manifold_label("config_prefill_ep"), "Prefill expert parallelism")
        self.assertEqual(app._manifold_label("config_prefill_num_workers"), "Prefill worker count")
        self.assertEqual(app._manifold_label("config_decode_tp"), "Decode tensor parallelism")
        self.assertEqual(app._manifold_label("metrics_median_tpot"), "Median TPOT")

    def test_manifold_page_handles_absent_optional_capability_without_affecting_core_dashboard(self) -> None:
        active_dataset = {"verification": {"artifact_status": "valid"}}
        with (
            patch.object(app, "STAGE5_ARTIFACT_PATHS", None),
            patch.object(app, "STAGE5_ARTIFACT_CAPABILITY_ERROR", None),
            patch.object(app, "render_section_intro"),
            patch.object(app.st, "info") as info,
        ):
            app.render_manifold_analysis_dashboard(
                active_dataset=active_dataset,
                active_representation_data=None,  # The unavailable state returns before cohort access.
            )
        info.assert_called_once()
        self.assertIn("not available for this snapshot", info.call_args.args[0])
        self.assertEqual(active_dataset["verification"]["artifact_status"], "valid")

    def test_manifold_page_isolates_a_corrupt_optional_capability(self) -> None:
        active_dataset = {"verification": {"artifact_status": "valid"}}
        with (
            patch.object(app, "STAGE5_ARTIFACT_PATHS", None),
            patch.object(app, "STAGE5_ARTIFACT_CAPABILITY_ERROR", "projection SHA mismatch"),
            patch.object(app, "render_section_intro"),
            patch.object(app.st, "error") as error,
        ):
            app.render_manifold_analysis_dashboard(
                active_dataset=active_dataset,
                active_representation_data=None,
            )
        error.assert_called_once()
        self.assertIn("failed closed", error.call_args.args[0])
        self.assertEqual(active_dataset["verification"]["artifact_status"], "valid")

    def test_dashboard_import_remains_research_dependency_isolated(self) -> None:
        # Test the dashboard module's dependency boundary, not global interpreter
        # state populated when unittest discovery imports the separate Stage 5
        # research test module.
        app_source = inspect.getsource(app)
        imports = set()
        for node in ast.walk(ast.parse(app_source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        for module_name in ("umap", "numba", "pynndescent", "torch", "modeling.manifold_results"):
            self.assertNotIn(module_name, imports)

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

    def test_streamlit_apptest_handles_checkpoint_states_without_errors(self) -> None:
        tested = AppTest.from_file(
            "apps/inferencex_pca_demo.py",
            default_timeout=90,
        ).run()
        self.assertEqual(len(tested.exception), 0)
        self.assertEqual(len(tested.error), 0)
        labels = [tab.label for tab in tested.tabs]
        for label in app.MAIN_TAB_LABELS:
            self.assertIn(label, labels)
        self.assertIn("Manifold Analysis", labels)
        self.assertTrue(any("Nonlinear Neighborhood Geometry" in item.value for item in tested.markdown))
        if not Path(app.DEFAULT_DATA_DIR).exists():
            self.assertTrue(any("not installed" in info.value for info in tested.info))

    def test_model_results_are_marked_historical_on_july_data(self) -> None:
        source = inspect.getsource(app.render_model_results_dashboard)
        self.assertIn("historical experiments on the June snapshot", source)
        self.assertIn("not applied to cumulative-snapshot rows", source)

    def test_raw_csv_is_preferred_and_legacy_csv_json_loading_is_preserved(self) -> None:
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
            pd.DataFrame({"config_id": [1], "metrics": ['{"tput_per_gpu": 10.0}']}).to_csv(
                path / "benchmark_results_raw.csv", index=False
            )
            pd.DataFrame({"config_id": [1], "metrics_tput_per_gpu": [5.0]}).to_csv(
                path / "benchmark_results.csv", index=False
            )
            pd.DataFrame({"id": [1], "hardware": ["gpu"]}).to_csv(
                path / "configs.csv", index=False
            )
            benchmarks, _configs, joined, source = app.load_joined_data(str(path), "raw-test")
            self.assertEqual(source["active_mode"], "Raw CSV")
            self.assertEqual(float(benchmarks["metrics_tput_per_gpu"].iloc[0]), 10.0)
            self.assertIn("metrics_tput_per_gpu", joined)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "benchmark_results.json").write_text('[{"config_id": 1}]', encoding="utf-8")
            (path / "configs.json").write_text('[{"id": 1, "hardware": "gpu"}]', encoding="utf-8")
            _benchmarks, _configs, joined, source = app.load_joined_data(str(path), "json-test")
            self.assertEqual(source["active_mode"], "JSON fallback")
            self.assertIn("config_hardware", joined)


if __name__ == "__main__":
    unittest.main()
