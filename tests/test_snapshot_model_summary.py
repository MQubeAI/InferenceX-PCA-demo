from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_snapshot_model_summary import SummaryError, build_snapshot_model_summary


def manifest() -> dict:
    return {"dataset_id": "inferencex-db-dump-2026-07-27", "source": {"release": "db-dump/2026-07-27"}}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class SnapshotModelSummaryTests(unittest.TestCase):
    def test_projects_only_finite_aggregate_tabfm_and_uncertainty_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tabfm, uncertainty = root / "tabfm.json", root / "uncertainty.json"
            write_json(tabfm, {"aggregate_only": True, "result": {"metrics": {"r2_mean": 0.91, "r2_std": 0.02, "mae_mean": 12.5}}})
            write_json(
                uncertainty,
                {
                    "aggregate_only": True,
                    "result": {
                        "point_model": {"r2": 0.8, "mae": 20.0},
                        "intervals": {
                            "conditional_scale": {
                                "0.8": {
                                    "nominal_coverage": 0.8,
                                    "empirical_coverage": 0.81,
                                    "average_interval_width": 3.0,
                                    "median_interval_width": 2.0,
                                    "interval_score": 4.0,
                                    "coverage_by_feature": {"hardware": [{"value": "h200", "rows": 30, "coverage": 0.81}]},
                                }
                            }
                        },
                    },
                },
            )
            result = build_snapshot_model_summary(snapshot_manifest=manifest(), tabfm_path=tabfm, uncertainty_path=uncertainty)
            self.assertTrue(result["aggregate_only"])
            self.assertEqual(result["selected_throughput_point_model"]["metrics"]["r2"], 0.91)
            self.assertEqual(result["selected_uncertainty_method"]["intervals"]["0.8"]["subgroup_minimum_support"], 20)

    def test_rejects_non_finite_or_non_aggregate_runner_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tabfm, uncertainty = root / "tabfm.json", root / "uncertainty.json"
            write_json(tabfm, {"aggregate_only": True, "result": {"metrics": {"r2_mean": float("nan"), "r2_std": 0.02, "mae_mean": 12.5}}})
            write_json(uncertainty, {"aggregate_only": False})
            with self.assertRaises((SummaryError, ValueError)):
                build_snapshot_model_summary(snapshot_manifest=manifest(), tabfm_path=tabfm, uncertainty_path=uncertainty)


if __name__ == "__main__":
    unittest.main()
