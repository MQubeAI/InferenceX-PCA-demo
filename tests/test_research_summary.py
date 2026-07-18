from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from modeling.research_summary import (
    apply_subgroup_support_to_uncertainty_artifact,
    build_research_summary,
    read_aggregate_artifact,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class ResearchSummaryTests(unittest.TestCase):
    def make_artifacts(self, directory: Path) -> None:
        write_json(directory / "model-diagnostics-4096.json", {
            "aggregate_only": True,
            "experiments": {"throughput": {"target": "metrics_tput_per_gpu", "models": {"tabfm": {"metrics": {"r2": {"mean": .961979, "std": .008605}, "mae": {"mean": 338.540384}}}}}},
        })
        write_json(directory / "median-tpot-tail-model-4096-seed-42.json", {
            "aggregate_only": True, "two_stage": {"summary": {"global_tabfm": {"mae_mean": .0059}}},
        })
        write_json(directory / "throughput-residual-diagnostics.json", {
            "aggregate_only": True, "result": {"residual_diagnostics": {"heteroskedasticity": {"conditional_on_target": True}}},
        })
        interval = {
            "nominal_coverage": .95, "empirical_coverage": .9534,
            "average_interval_width": 2485.442, "interval_score": 4739.169,
            "coverage_by_feature": {"conc": [
                {"value": "one", "rows": 1, "coverage": 0.0, "average_width": 1.0},
                {"value": "large", "rows": 20, "coverage": .8, "average_width": 2.0},
            ]},
        }
        write_json(directory / "throughput-uncertainty-4096-seed-42.json", {
            "aggregate_only": True,
            "result": {"point_model": {"r2": .913897, "mae": 449.8}, "intervals": {"conditional_scale": {"0.95": interval, "0.8": {**interval, "nominal_coverage": .8}, "0.5": {**interval, "nominal_coverage": .5}}}},
        })

    def test_selection_is_deterministic_and_keeps_throughput_r2_contexts_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.make_artifacts(directory)
            first = build_research_summary(directory)
            second = build_research_summary(directory)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["selected_throughput_point_model"]["metrics"]["r2"], .961979)
        self.assertEqual(first["selected_uncertainty_method"]["uncertainty_evaluation_point_model"]["r2"], .913897)
        interval = first["selected_uncertainty_method"]["intervals"]["0.95"]
        self.assertEqual(interval["subgroup_minimum_support"], 20)
        self.assertEqual(interval["worst_subgroup_undercoverage"]["value"], "large")

    def test_missing_artifacts_are_reported_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary = build_research_summary(temp)
        self.assertTrue(all(value == "missing" for value in summary["artifact_status"].values()))
        self.assertIsNone(summary["selected_throughput_point_model"]["metrics"])

    def test_aggregate_only_enforcement_rejects_row_level_prediction_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            write_json(path, {"aggregate_only": True, "predictions": [1.0]})
            with self.assertRaises(ValueError):
                read_aggregate_artifact(path)

    def test_legacy_uncertainty_artifact_can_be_upgraded_without_model_output(self) -> None:
        artifact = {
            "aggregate_only": True,
            "result": {
                "intervals": {
                    "conditional_scale": {
                        "0.95": {
                            "nominal_coverage": .95,
                            "coverage_by_feature": {
                                "conc": [{"value": "one", "rows": 1, "coverage": 0.0, "average_width": 1.0}],
                            },
                        },
                    },
                },
            },
        }
        upgraded = apply_subgroup_support_to_uncertainty_artifact(artifact)
        report = upgraded["result"]["intervals"]["conditional_scale"]["0.95"]
        self.assertEqual(upgraded["controls"]["subgroup_minimum_support"], 20)
        self.assertEqual(report["subgroup_groups_excluded_from_worst_ranking"], 1)
        self.assertIsNone(report["worst_subgroup_undercoverage"])

    def test_normal_app_import_does_not_import_tabfm(self) -> None:
        command = [sys.executable, "-c", "import apps.inferencex_pca_demo; import sys; assert not any(name.lower().startswith('tabfm') for name in sys.modules)"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
