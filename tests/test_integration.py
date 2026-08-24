from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from apps import inferencex_pca_demo as app
from modeling import integration


def capability_frame() -> pd.DataFrame:
    values = {
        "canonical_hardware_id": "epoch_hardware:nvidia-b200",
        "epoch_hardware_name": "NVIDIA B200",
        "vendor": "NVIDIA",
        "epoch_hardware_type": "GPU",
        "epoch_release_date": "2024-11-15",
        "epoch_last_modified": "2026-08-10T00:00:00Z",
        "source": "Epoch AI ml_hardware.csv",
        "source_snapshot": "2026-08-20",
        "provenance": "https://example.invalid/b200",
        "source_notes": "per GPU",
    }
    for index, column in enumerate(integration.CAPABILITY_FEATURE_COLUMNS, 1):
        values[column] = float(index)
    values["epoch_fp4_peak_flops"] = np.nan
    return pd.DataFrame([values])


def mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    hardware = integration.hardware_mapping(capability_frame(), ["b200", "h100"])
    models = integration.model_mapping(pd.DataFrame({"Model": ["GLM-5"]}), ["glm5"])
    return hardware, models


def throughput_fixture() -> pd.DataFrame:
    rows = []
    for index, hardware in enumerate(("b200", "h100"), 1):
        rows.append(
            {
                "config_id": f"config-{index}",
                "benchmark_type": "single_turn",
                "isl": 1024,
                "osl": 1024,
                "conc": 32,
                "date": "2026-07-01",
                "config_hardware": hardware,
                "config_model": "glm5",
                "config_framework": "vllm",
                "config_precision": "fp8",
                "metrics_tput_per_gpu": float(index * 100),
            }
        )
    return pd.DataFrame(rows)


class IntegrationInvariantTests(unittest.TestCase):
    def test_raw_epoch_hashes_are_unchanged_by_read_only_audit(self) -> None:
        if not integration.EPOCH_HASH_MANIFEST.exists():
            self.skipTest("Epoch raw snapshot is not installed")
        before = integration.verify_epoch_raw_hashes()
        integration.audit_epoch_sources()
        after = integration.verify_epoch_raw_hashes()
        self.assertEqual(before, after)

    def test_pipeline_preserves_raw_epoch_hashes(self) -> None:
        if not integration.EPOCH_HASH_MANIFEST.exists() or not Path(app.DEFAULT_DATA_DIR).exists():
            self.skipTest("Epoch or frozen InferenceX snapshot is not installed")
        before = integration.verify_epoch_raw_hashes()
        with tempfile.TemporaryDirectory() as directory:
            result = integration.run_integration_pipeline(
                app.DEFAULT_DATA_DIR,
                Path(directory) / "derived",
                run_experiments=False,
            )
        after = integration.verify_epoch_raw_hashes()
        self.assertTrue(result["raw_hashes_verified"])
        self.assertEqual(before, after)

    def test_hardware_enrichment_preserves_rows_configs_and_target_values(self) -> None:
        source = throughput_fixture()
        hardware, models = mappings()
        view, validation = integration.build_throughput_integration_view(
            source, hardware, capability_frame(), models
        )
        self.assertEqual(len(view), len(source))
        self.assertEqual(view.config_id.nunique(), source.config_id.nunique())
        np.testing.assert_allclose(view.metrics_tput_per_gpu, source.metrics_tput_per_gpu)
        self.assertTrue(validation["invariant_results"]["row_count_preserved"]["pass"])
        self.assertTrue(validation["invariant_results"]["target_values_unchanged"]["pass"])
        self.assertFalse(any("cluster" in name.lower() for name in view.columns))
        self.assertFalse(any("data_center" in name.lower() for name in view.columns))

    def test_one_hardware_label_has_at_most_one_accepted_entity(self) -> None:
        hardware, _models = mappings()
        accepted = hardware.loc[hardware.mapping_status.eq("accepted")]
        self.assertTrue(accepted.groupby("inferencex_hardware").canonical_hardware_id.nunique().le(1).all())
        duplicate = pd.concat([hardware, hardware.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            integration.validate_hardware_mapping(duplicate)

    def test_ambiguous_hardware_mapping_remains_unjoined(self) -> None:
        hardware, models = mappings()
        h100 = hardware.loc[hardware.inferencex_hardware.eq("h100")].iloc[0]
        self.assertTrue(h100.ambiguity)
        self.assertEqual(h100.mapping_status, "unresolved")
        self.assertTrue(pd.isna(h100.canonical_hardware_id))
        view, _validation = integration.build_throughput_integration_view(
            throughput_fixture(), hardware, capability_frame(), models
        )
        h100_view = view.loc[view.config_hardware.eq("h100")].iloc[0]
        self.assertTrue(pd.isna(h100_view.epoch_memory_capacity_bytes))

    def test_missing_capability_is_never_converted_to_zero(self) -> None:
        hardware, models = mappings()
        view, _validation = integration.build_throughput_integration_view(
            throughput_fixture(), hardware, capability_frame(), models
        )
        b200 = view.loc[view.config_hardware.eq("b200")].iloc[0]
        self.assertTrue(pd.isna(b200.epoch_fp4_peak_flops))
        self.assertNotEqual(b200.epoch_fp4_peak_flops, 0)

    def test_artificial_analysis_absence_is_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status = integration.artificial_analysis_status(Path(directory) / "missing")
        self.assertEqual(status["status"], "not_loaded")
        self.assertEqual(status["message"], "Artificial Analysis data not loaded.")

    def test_observed_and_unsupported_counterfactuals_are_distinct(self) -> None:
        source = throughput_fixture()
        counterpart = source.iloc[[0]].copy()
        counterpart["config_hardware"] = "b300"
        counterpart["config_id"] = "config-3"
        counterpart["metrics_tput_per_gpu"] = 300.0
        observed_view = pd.concat([source, counterpart], ignore_index=True)
        observed = integration.observed_counterfactual(observed_view, observed_view.iloc[0], "b300")
        unsupported = integration.observed_counterfactual(source, source.iloc[0], "mi355x")
        self.assertEqual(observed["status"], "observed")
        self.assertTrue(observed["observed"])
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertFalse(unsupported["observed"])

    def test_prototype_renderer_has_no_prediction_runtime(self) -> None:
        source = inspect.getsource(app.render_hardware_integration_prototype)
        self.assertIn("Observed throughput", source)
        self.assertIn("Prediction: unsupported", source)
        self.assertIn("Artificial Analysis data not loaded", source)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("model.predict(", source)


if __name__ == "__main__":
    unittest.main()
