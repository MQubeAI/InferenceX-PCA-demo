from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from modeling.energy_measurements import (
    CONFIG_FIELDS,
    ENERGY_TARGET,
    MATCH_FIELDS,
    exact_observed_lookup,
    mark_dominated_comparisons,
    measured_energy_rows,
    nearest_measured_configurations,
    observed_energy_conversions,
    unsupported_selection_fields,
)


def configuration(**overrides):
    row = {
        "config_id": "config-a",
        "benchmark_type": "single_turn",
        "isl": 1024,
        "osl": 1024,
        "conc": 8,
        "config_model": "model-a",
        "config_hardware": "gpu-a",
        "config_framework": "framework-a",
        "config_precision": "fp8",
        "config_spec_method": "none",
        "config_disagg": False,
        "config_is_multinode": False,
        "config_prefill_tp": 1,
        "config_prefill_ep": 1,
        "config_prefill_dp_attention": False,
        "config_prefill_num_workers": 1,
        "config_decode_tp": 1,
        "config_decode_ep": 1,
        "config_decode_dp_attention": False,
        "config_decode_num_workers": 1,
        "config_num_prefill_gpu": 1,
        "config_num_decode_gpu": 1,
        ENERGY_TARGET: 2.0,
        "metrics_tput_per_gpu": 10.0,
        "metrics_avg_power_w": 500.0,
        "date": "2026-06-01",
    }
    row.update(overrides)
    return row


class EnergyMeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                configuration(),
                configuration(**{ENERGY_TARGET: 4.0, "date": "2026-06-03"}),
                configuration(
                    config_id="config-b", config_hardware="gpu-b", config_model="model-b",
                    **{ENERGY_TARGET: 1.5, "metrics_tput_per_gpu": 20.0},
                ),
                configuration(
                    config_id="config-c", config_hardware="gpu-b", config_model="model-a",
                    isl=8192, conc=16, **{ENERGY_TARGET: 5.0, "metrics_tput_per_gpu": 8.0},
                ),
                configuration(config_id="config-unmeasured", config_precision="fp4", **{ENERGY_TARGET: None}),
            ]
        )
        self.selection = {field: self.frame.iloc[0][field] for field in MATCH_FIELDS}

    def test_exact_measurement_returns_observed_median_and_metadata(self) -> None:
        result = exact_observed_lookup(self.frame, self.selection, price_per_kwh=0.20)
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["label"], "Observed energy measurement")
        self.assertFalse(result["is_prediction"])
        self.assertEqual(result["match_count"], 2)
        self.assertEqual(result["joules_per_output_token"], 3.0)
        self.assertEqual(result["minimum"], 2.0)
        self.assertEqual(result["maximum"], 4.0)
        self.assertEqual(result["config_ids"], ["config-a"])
        self.assertEqual(result["date_start"], "2026-06-01")
        self.assertEqual(result["date_end"], "2026-06-03")

    def test_supported_but_absent_combination_never_returns_prediction(self) -> None:
        selection = dict(self.selection)
        selection["config_model"] = "model-b"
        result = exact_observed_lookup(self.frame, selection)
        self.assertEqual(result["status"], "no_exact_match")
        self.assertEqual(result["label"], "No exact measured energy result")
        self.assertFalse(result["is_prediction"])
        self.assertNotIn("joules_per_output_token", result)

    def test_unsupported_workload_is_blocked(self) -> None:
        selection = dict(self.selection)
        selection["isl"] = 4096
        self.assertEqual(unsupported_selection_fields(self.frame, selection), ["isl"])
        self.assertEqual(exact_observed_lookup(self.frame, selection)["status"], "unsupported")

    def test_nearest_rows_are_comparisons_and_prefer_same_workload(self) -> None:
        selection = dict(self.selection)
        selection["config_model"] = "model-b"
        nearest = nearest_measured_configurations(self.frame, selection, limit=3)
        self.assertTrue(nearest["comparison_only"].all())
        self.assertTrue(nearest["same_workload"].all())
        self.assertFalse((nearest["differing_fields"] == "none").any())
        self.assertTrue((nearest["isl"] == 1024).all())
        self.assertTrue((nearest["osl"] == 1024).all())

    def test_missing_targets_are_excluded(self) -> None:
        measured = measured_energy_rows(self.frame)
        self.assertNotIn("config-unmeasured", measured["config_id"].tolist())
        self.assertNotIn("fp4", measured["config_precision"].tolist())

    def test_observed_energy_conversions_and_cost_are_exact(self) -> None:
        result = observed_energy_conversions(3.6, 0.15)
        self.assertEqual(result["tokens_per_kwh"], 1_000_000)
        self.assertEqual(result["kwh_per_million_output_tokens"], 1.0)
        self.assertEqual(result["electricity_cost_per_million_output_tokens"], 0.15)

    def test_dominance_is_limited_to_supplied_comparison_rows(self) -> None:
        comparisons = pd.DataFrame(
            [
                {"config_id": "a", ENERGY_TARGET: 2.0, "metrics_tput_per_gpu": 10.0, "same_workload": True},
                {"config_id": "b", ENERGY_TARGET: 1.0, "metrics_tput_per_gpu": 20.0, "same_workload": True},
            ]
        )
        marked = mark_dominated_comparisons(comparisons)
        self.assertTrue(marked.loc[marked.config_id == "a", "dominated_in_comparison"].item())
        self.assertFalse(marked.loc[marked.config_id == "b", "dominated_in_comparison"].item())

    def test_match_schema_contains_configuration_and_workload_only(self) -> None:
        self.assertTrue(set(CONFIG_FIELDS).issubset(MATCH_FIELDS))
        self.assertNotIn(ENERGY_TARGET, MATCH_FIELDS)
        self.assertFalse(any(field.startswith("metrics_") for field in MATCH_FIELDS))

    def test_repository_energy_counts_reconcile_when_local_data_is_present(self) -> None:
        benchmark_path = Path("inferencex-pca-data/benchmark_results.csv")
        if not benchmark_path.exists():
            self.skipTest("Local benchmark CSV is unavailable")
        benchmark = pd.read_csv(benchmark_path, low_memory=False)
        measured = benchmark[benchmark[ENERGY_TARGET].notna()]
        grouped = measured.groupby(
            ["config_id", "benchmark_type", "isl", "osl", "conc"], dropna=False
        )[ENERGY_TARGET].median()
        self.assertEqual(len(measured), 3_794)
        self.assertEqual(len(grouped), 2_135)


if __name__ == "__main__":
    unittest.main()
