from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling import comparison, throughput_uncertainty as uncertainty


def fixture_frame() -> pd.DataFrame:
    rows = []
    for config_id in range(12):
        for repeat in range(4):
            rows.append({
                "config_id": config_id,
                "isl": 128 + 16 * repeat,
                "osl": 32 + 8 * repeat,
                "conc": 1 + repeat,
                "benchmark_type": "offline" if repeat % 2 else "online",
                "config_hardware": f"gpu-{config_id % 2}",
                "config_model": f"model-{config_id % 3}",
                "config_framework": f"framework-{config_id % 2}",
                "config_precision": "bf16" if config_id % 2 else "fp16",
                "metrics_tput_per_gpu": 100.0 + config_id * 5 + repeat * 2,
            })
    return pd.DataFrame(rows)


class ThroughputUncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = fixture_frame()
        self.features = [
            "isl", "osl", "conc", "benchmark_type", "config_hardware",
            "config_model", "config_framework", "config_precision",
        ]

    def test_grouped_role_split_is_deterministic_and_disjoint(self) -> None:
        first = uncertainty.split_outer_training_groups(self.frame, 42)
        second = uncertainty.split_outer_training_groups(self.frame, 42)
        self.assertEqual({name: list(part.index) for name, part in first.items()}, {name: list(part.index) for name, part in second.items()})
        groups = [set(part.config_id) for part in first.values()]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(set().union(*groups), set(self.frame.config_id))

    def test_conformal_interval_construction_uses_calibration_only(self) -> None:
        prediction = np.array([10.0, 20.0])
        global_intervals, global_quantiles = uncertainty.global_split_conformal_intervals(prediction, np.array([1.0, -2.0, 3.0]), (0.5,))
        self.assertEqual(global_quantiles[0.5], 2.0)
        np.testing.assert_allclose(global_intervals[0.5][0], [8.0, 18.0])
        scale_intervals, scale_quantiles = uncertainty.conformalized_scale_intervals(prediction, np.array([2.0, 4.0]), np.array([2.0, 2.0]), np.array([3.0, 1.0]), (0.5,))
        self.assertEqual(scale_quantiles[0.5], 2.0)
        np.testing.assert_allclose(scale_intervals[0.5][1], [16.0, 22.0])
        quantile_intervals, quantile_quantiles = uncertainty.conformalized_quantile_intervals(
            prediction, np.array([-3.0, 4.0]), {0.5: np.array([-1.0, -1.0])}, {0.5: np.array([1.0, 1.0])}, {0.5: np.array([-2.0, -2.0])}, {0.5: np.array([2.0, 2.0])}, (0.5,)
        )
        self.assertEqual(quantile_quantiles[0.5], 3.0)
        np.testing.assert_allclose(quantile_intervals[0.5][0], [5.0, 15.0])

    def test_coverage_summary_and_subgroups(self) -> None:
        frame = pd.DataFrame({"config_hardware": ["a", "b"], "conc": [1, 1]})
        summary = uncertainty.summarize_interval(
            np.array([0.0, 10.0]), np.array([-1.0, 1.0]), np.array([1.0, 5.0]),
            0.8, frame, np.array(["q1", "q2"]), subgroup_minimum_support=1,
        )
        self.assertEqual(summary["empirical_coverage"], 0.5)
        self.assertEqual(summary["average_interval_width"], 3.0)
        self.assertEqual(summary["coverage_and_width_by_throughput_bin"][0]["rows"], 1)
        self.assertEqual(summary["worst_subgroup_undercoverage"]["value"], "b")

    def test_default_subgroup_support_excludes_one_row_groups_but_retains_them(self) -> None:
        frame = pd.DataFrame({"config_hardware": ["covered"] * 20 + ["one-row"] + ["large-miss"] * 20})
        observed = np.zeros(len(frame))
        lower = np.full(len(frame), -1.0)
        upper = np.r_[np.ones(20), np.array([-1.0]), -np.ones(20)]
        summary = uncertainty.summarize_interval(observed, lower, upper, 0.8, frame, np.array(["q1"] * len(frame)))
        subgroup_rows = summary["coverage_by_feature"]["config_hardware"]
        self.assertIn("one-row", [row["value"] for row in subgroup_rows])
        self.assertEqual(summary["subgroup_minimum_support"], 20)
        self.assertEqual(summary["subgroup_groups_excluded_from_worst_ranking"], 1)
        self.assertEqual(summary["worst_subgroup_undercoverage"]["value"], "large-miss")

    def test_subgroup_support_is_configurable(self) -> None:
        frame = pd.DataFrame({"config_hardware": ["a", "b"]})
        summary = uncertainty.summarize_interval(
            np.array([0.0, 2.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
            0.8, frame, np.array(["q1", "q2"]), subgroup_minimum_support=2,
        )
        self.assertEqual(summary["subgroup_groups_excluded_from_worst_ranking"], 2)
        self.assertIsNone(summary["worst_subgroup_undercoverage"])

    def test_outer_evaluation_has_one_tabfm_call_per_fold_and_no_validation_target_leakage(self) -> None:
        calls: list[tuple[str, np.ndarray]] = []

        def fake_tabfm(context, query, y_context, *_args, **_kwargs):
            calls.append(("tabfm", np.asarray(y_context, dtype=float)))
            return query["isl"].to_numpy(dtype=float) * 0.1, [], {"mock": True}

        def fake_catboost(train, query, target, loss, _seed):
            calls.append((loss, np.asarray(target, dtype=float)))
            if loss == "RMSE":
                return np.full(len(query), np.log1p(3.0))
            alpha = float(loss.split("=")[1])
            return np.full(len(query), -4.0 if alpha < 0.5 else 4.0)

        with patch.object(uncertainty, "_tabfm_fit_predict", side_effect=fake_tabfm), patch.object(uncertainty, "_catboost_predict", side_effect=fake_catboost):
            result = uncertainty.evaluate_throughput_uncertainty(self.frame, self.features, max_rows=100, seed=42, n_splits=3)

        self.assertEqual(result["tabfm"]["invocation_count"], 3)
        self.assertEqual(sum(name == "tabfm" for name, _ in calls), 3)
        self.assertTrue(all(row["outer_group_overlap"] == 0 for row in result["folds"]))
        self.assertTrue(all(row["tabfm_invocations"] == 1 for row in result["folds"]))
        tabfm_lengths = [len(target) for name, target in calls if name == "tabfm"]
        residual_model_lengths = [len(target) for name, target in calls if name != "tabfm"]
        self.assertEqual(tabfm_lengths, [row["partition_rows"]["context"] for row in result["folds"]])
        self.assertEqual(residual_model_lengths, [row["partition_rows"]["uncertainty_train"] for row in result["folds"] for _ in range(7)])
        self.assertTrue(all(np.max(target) < 1_000 for _name, target in calls))
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn('"predictions"', serialized)
        self.assertNotIn('"residuals"', serialized)

    def test_validation_target_change_does_not_change_calibration(self) -> None:
        work, _ = comparison.prepare_model_frame(self.frame, self.features, "metrics_tput_per_gpu", 100, 42)
        _outer_train, validation_i = comparison.deterministic_grouped_folds(work, 1)[0]
        changed = self.frame.copy()
        changed.loc[work.iloc[validation_i].index, "metrics_tput_per_gpu"] = 1_000_000.0

        def fake_tabfm(_context, query, _y_context, *_args, **_kwargs):
            return query["isl"].to_numpy(dtype=float) * 0.1, [], {"mock": True}

        def fake_catboost(_train, query, _target, loss, _seed):
            if loss == "RMSE":
                return np.full(len(query), np.log1p(2.0))
            return np.full(len(query), -2.0 if "alpha=0.0" in loss or "alpha=0.1" in loss or "alpha=0.25" in loss else 2.0)

        with patch.object(uncertainty, "_tabfm_fit_predict", side_effect=fake_tabfm), patch.object(uncertainty, "_catboost_predict", side_effect=fake_catboost):
            first = uncertainty.evaluate_throughput_uncertainty(self.frame, self.features, max_rows=100, seed=42, n_splits=1)
            second = uncertainty.evaluate_throughput_uncertainty(changed, self.features, max_rows=100, seed=42, n_splits=1)
        self.assertEqual(first["folds"][0]["calibration_quantiles"], second["folds"][0]["calibration_quantiles"])

    def test_mac_launcher_accepts_script_and_argument_separator(self) -> None:
        launcher = (Path(__file__).parents[1] / "scripts" / "run_tabfm_mac.sh").read_text()
        self.assertIn("throughput_uncertainty_diagnostics.py", launcher)
        self.assertIn('--) shift; forwarded+=("$@"); break ;;', launcher)


if __name__ == "__main__":
    unittest.main()
