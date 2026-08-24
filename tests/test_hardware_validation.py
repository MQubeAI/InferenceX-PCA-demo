from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from modeling import hardware_validation as validation


def fixture() -> pd.DataFrame:
    rows = []
    capabilities = list(validation.CAPABILITY_FEATURE_COLUMNS)
    for hardware_index, hardware in enumerate(("b200", "b300", "mi300x"), 1):
        for config_index in range(1, 5):
            row = {
                "config_id": f"{hardware}-{config_index}",
                "benchmark_type": "single_turn",
                "isl": 1024.0,
                "osl": 1024.0,
                "conc": float(config_index * 8),
                "config_hardware": hardware,
                "config_framework": "vllm",
                "config_model": "model-a",
                "config_precision": "fp8" if config_index % 2 else "bf16",
                "config_spec_method": "mtp",
                "config_disagg": bool(config_index % 2),
                "config_is_multinode": False,
                "config_prefill_tp": 1.0,
                "config_prefill_ep": 1.0,
                "config_prefill_dp_attention": False,
                "config_prefill_num_workers": 1.0,
                "config_decode_tp": 1.0,
                "config_decode_ep": 1.0,
                "config_decode_dp_attention": False,
                "config_decode_num_workers": 1.0,
                "config_num_prefill_gpu": 1.0,
                "metrics_tput_per_gpu": float(hardware_index * 100 + config_index * 5),
                "vendor": "AMD" if hardware == "mi300x" else "NVIDIA",
                "epoch_hardware_name": hardware,
            }
            for capability_index, capability in enumerate(capabilities, 1):
                row[capability] = float(hardware_index * 10 + capability_index)
            if hardware == "mi300x":
                row["epoch_intranode_bandwidth_bytes_per_second"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class HardwareValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = fixture()
        self.base = [
            "isl", "osl", "conc", "config_prefill_tp", "config_prefill_ep",
            "config_prefill_dp_attention", "config_prefill_num_workers", "config_decode_tp",
            "config_decode_ep", "config_decode_dp_attention", "config_decode_num_workers",
            "config_num_prefill_gpu", "config_hardware", "config_framework", "config_model",
            "config_precision", "config_spec_method", "config_disagg", "config_is_multinode",
        ]

    def test_matched_precision_is_explicit_and_unknown_remains_missing(self) -> None:
        frame = self.frame.copy()
        frame.loc[frame.index[0], "config_precision"] = "new_precision"
        enriched, details = validation.add_matched_precision_peak(frame)
        self.assertTrue(pd.isna(enriched.loc[frame.index[0], validation.MATCHED_PRECISION_FEATURE]))
        self.assertIn("new_precision", details["unmapped_precision_labels"])
        fp8 = enriched.loc[enriched.config_precision.eq("fp8")].iloc[0]
        self.assertEqual(
            fp8[validation.MATCHED_PRECISION_FEATURE], fp8["epoch_fp8_peak_flops"]
        )

    def test_metrics_include_relative_and_log_ratio_diagnostics(self) -> None:
        metrics = validation.prediction_metrics([1.0, 10.0, 100.0], [1.0, 8.0, 130.0])
        self.assertEqual(metrics["rows"], 3)
        self.assertIn("p90_absolute_percentage_error", metrics)
        self.assertIn("median_absolute_log_ratio", metrics)
        self.assertGreater(metrics["within_30_percent"], 0.0)

    def test_fixed_folds_are_group_isolated_for_every_representation(self) -> None:
        representations = {
            "A_identity_only": self.base,
            "C_capabilities_only": [column for column in self.base if column != "config_hardware"]
            + list(validation.CAPABILITY_FEATURE_COLUMNS),
        }
        folds = validation.deterministic_grouped_folds(self.frame, 3)
        results, predictions = validation.evaluate_fixed_folds(
            self.frame, representations, ["global_median", "ridge"], folds
        )
        self.assertEqual(set(predictions), {
            ("A_identity_only", "global_median"), ("A_identity_only", "ridge"),
            ("C_capabilities_only", "global_median"), ("C_capabilities_only", "ridge"),
        })
        for representation in results.values():
            for model in representation["models"].values():
                self.assertTrue(all(row["group_overlap"] == 0 for row in model["folds"]))

    def test_hardware_profile_distance_never_uses_held_out_measurements(self) -> None:
        features = list(validation.CAPABILITY_FEATURE_COLUMNS)
        profiles = validation._hardware_profiles(self.frame, features)
        first = validation._nearest_hardware_profile(profiles, "b200", features)
        changed = self.frame.copy()
        changed.loc[changed.config_hardware.eq("b200"), "metrics_tput_per_gpu"] = 99_999.0
        second = validation._nearest_hardware_profile(
            validation._hardware_profiles(changed, features), "b200", features
        )
        self.assertEqual(first["nearest"], second["nearest"])
        self.assertFalse(first["vendor_novelty"])

    def test_holdout_excludes_all_held_out_configs_from_training(self) -> None:
        metadata = {
            "capability_feature_columns": list(validation.CAPABILITY_FEATURE_COLUMNS),
            "representations": {
                "C_capabilities_only": [column for column in self.base if column != "config_hardware"]
                + list(validation.CAPABILITY_FEATURE_COLUMNS)
            },
        }
        result = validation.held_out_hardware_transfer(self.frame, metadata)
        self.assertEqual(len(result["records"]), 3)
        self.assertTrue(all(row["group_overlap"] == 0 for row in result["records"]))
        self.assertTrue(all("capability_random_forest" in row["metrics"] for row in result["records"]))


if __name__ == "__main__":
    unittest.main()
