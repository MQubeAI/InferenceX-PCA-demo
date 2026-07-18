from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from apps import inferencex_pca_demo as app
from modeling import comparison
from modeling import diagnostics
from scripts import model_diagnostics


def fixture_frame() -> pd.DataFrame:
    rows = []
    for config_id in range(1, 7):
        for repeat in range(5):
            rows.append({
                "config_id": config_id,
                "isl": float(128 + repeat * 16) if repeat else np.nan,
                "conc": 1 + repeat,
                "config_hardware": None if repeat == 1 else f"gpu-{config_id % 2}",
                "config_framework": f"framework-{config_id % 3}",
                "metrics_p99_itl": np.nan if (config_id == 1 and repeat == 0) else config_id * 0.2 + repeat * 0.05,
            })
    return pd.DataFrame(rows)


class ModelComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = fixture_frame()
        self.features = ["isl", "conc", "config_hardware", "config_framework"]

    def test_grouped_folds_are_deterministic_and_isolated(self) -> None:
        work, _ = comparison.prepare_model_frame(self.frame, self.features, "metrics_p99_itl", 100, 42)
        first = comparison.deterministic_grouped_folds(work, 5)
        second = comparison.deterministic_grouped_folds(work, 5)
        self.assertEqual(len(first), len(second))
        for (train_a, valid_a), (train_b, valid_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(valid_a, valid_b)
            self.assertFalse(set(work.iloc[train_a].config_id) & set(work.iloc[valid_a].config_id))

    def test_models_share_fold_records_and_exclude_missing_targets(self) -> None:
        result = comparison.evaluate_models(self.frame, self.features, "metrics_p99_itl", ["random_forest", "catboost"], 100, 42, 3)
        self.assertEqual(result["preparation"]["missing_target_rows_excluded"], 1)
        rf = result["models"]["random_forest"]["folds"]
        cat = result["models"]["catboost"]["folds"]
        self.assertEqual([(row["train_rows"], row["validation_rows"]) for row in rf], [(row["train_rows"], row["validation_rows"]) for row in cat])
        self.assertTrue(all(row["group_overlap"] == 0 for row in rf + cat))

    def test_fold_preprocessing_is_local_and_includes_missing_indicators(self) -> None:
        train = self.frame.iloc[:10].copy()
        valid = self.frame.iloc[10:15].copy()
        train.loc[train.index[0], "isl"] = np.nan
        valid.loc[valid.index[0], "isl"] = 9_999.0
        processor = comparison.FoldPreprocessor.fit(train, self.features)
        transformed = processor.transform(train)
        self.assertIn("isl__missing", transformed)
        self.assertEqual(transformed.loc[train.index[0], "isl"], train["isl"].median())
        self.assertEqual(transformed.loc[train.index[1], "config_hardware"], comparison.MISSING_CATEGORY)

    def test_missingness_report_flags_target_eligibility_and_structure(self) -> None:
        self.frame["mostly_empty"] = np.nan
        report = comparison.missingness_report(self.frame, self.features + ["mostly_empty"], ["metrics_p99_itl"])
        self.assertEqual(report["usable_rows_per_target"]["metrics_p99_itl"], len(self.frame) - 1)
        record = next(item for item in report["columns"] if item["column"] == "mostly_empty")
        self.assertTrue(record["likely_structural_or_not_applicable"])

    def test_adapter_failure_is_captured_per_fold(self) -> None:
        with patch.object(comparison, "_catboost_fit_predict", side_effect=RuntimeError("intentional")):
            result = comparison.evaluate_models(self.frame, self.features, "metrics_p99_itl", ["catboost"], 100, 42, 2)
        self.assertTrue(all("RuntimeError: intentional" in row["failure"] for row in result["models"]["catboost"]["folds"]))

    def test_tabfm_unavailable_and_mocked_adapter_behavior(self) -> None:
        with patch.object(comparison, "model_available", return_value=(False, "unavailable")):
            unavailable = comparison.evaluate_models(self.frame, self.features, "metrics_p99_itl", ["tabfm"], 100, 42, 1)
        self.assertTrue(all(row["failure"] == "unavailable" for row in unavailable["models"]["tabfm"]["folds"]))
        def fake_tabfm(_train, valid, _target, _seed, _cap, _metadata=None):
            return np.array([1.0] * len(valid)), [], {"available_context_rows": 20, "used_context_rows": 8, "device": "cpu"}
        with patch.object(comparison, "model_available", return_value=(True, "mock")), patch.object(comparison, "_tabfm_fit_predict", side_effect=fake_tabfm):
            mocked = comparison.evaluate_models(self.frame, self.features, "metrics_p99_itl", ["tabfm"], 100, 42, 2, 8)
        self.assertTrue(all(row["used_context_rows"] == 8 for row in mocked["models"]["tabfm"]["folds"]))

    def test_subprocess_boundary_and_stale_signature(self) -> None:
        def fake_run(command, **_kwargs):
            output = command[command.index("--output") + 1]
            with open(output, "w", encoding="utf-8") as handle:
                json.dump({"comparison": {"models": {"tabfm": {"available": True, "folds": []}}}}, handle)
            return subprocess.CompletedProcess(command, 0, "ok", "")

        with patch.object(app.subprocess, "run", side_effect=fake_run) as run:
            result, error = app.run_tabfm_comparison_subprocess("inferencex-pca-data", "metrics_p99_itl", 1, 100, 42, 16, "Raw benchmark rows")
        self.assertEqual(error, "")
        self.assertTrue(result["available"])
        self.assertIn(".venv-tabfm/bin/python", run.call_args.args[0][0])
        metadata = {"dataset_fingerprint": "a", "analysis_unit": "unit", "analysis_row_count": 3}
        self.assertNotEqual(app.analysis_signature(metadata, "model-comparison", {"folds": 1}), app.analysis_signature(metadata, "model-comparison", {"folds": 2}))

    def test_comparison_payload_is_aggregate_only(self) -> None:
        result = comparison.evaluate_models(self.frame, self.features, "metrics_p99_itl", ["random_forest"], 100, 42, 2)
        serialized = json.dumps(result, default=str)
        self.assertNotIn("predictions", serialized)
        self.assertNotIn("metrics_p99_itl\": [", serialized)

    def test_context_selection_is_deterministic_bounded_and_target_free(self) -> None:
        train, valid = self.frame.iloc[:20].copy(), self.frame.iloc[20:].copy()
        train["metrics_p99_itl"] = np.arange(len(train))
        first, meta = diagnostics.select_context(train, valid, self.features, 7, "coverage", 42)
        second, _ = diagnostics.select_context(train.assign(metrics_p99_itl=-99), valid, self.features, 7, "coverage", 42)
        self.assertEqual(len(first), 7); self.assertTrue(set(first.index).issubset(train.index))
        self.assertListEqual(list(first.index), list(second.index)); self.assertEqual(meta["context_rows_used"], 7)

    def test_context_selection_does_not_use_validation_targets(self) -> None:
        train, valid = self.frame.iloc[:20].copy(), self.frame.iloc[20:].copy()
        valid["metrics_p99_itl"] = np.arange(len(valid), dtype=float)
        first, _ = diagnostics.select_context(train, valid, self.features, 7, "nearest", 42)
        second, _ = diagnostics.select_context(train, valid.assign(metrics_p99_itl=-999.0), self.features, 7, "nearest", 42)
        self.assertListEqual(list(first.index), list(second.index))

    def test_fold_diagnostics_are_deterministic_and_aggregate_only(self) -> None:
        train, valid = self.frame.iloc[:20].copy(), self.frame.iloc[20:].copy()
        prediction = np.linspace(0.1, 1.0, len(valid))
        first = diagnostics.fold_difficulty(train, valid, "metrics_p99_itl", prediction)
        second = diagnostics.fold_difficulty(train, valid, "metrics_p99_itl", prediction)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        hardware = first["category_coverage"]["config_hardware"]
        self.assertIn("training_distribution", hardware)
        self.assertIn("validation_distribution", hardware)
        serialized = json.dumps(first)
        self.assertNotIn('"actual"', serialized)
        self.assertNotIn('"residual"', serialized)

    def test_subgroup_totals_match_fold_totals(self) -> None:
        valid = self.frame.iloc[20:].copy()
        prediction = np.linspace(0.1, 1.0, len(valid))
        attribution = diagnostics.error_attribution(valid, "metrics_p99_itl", prediction, ["config_hardware", "conc"])
        totals = attribution["fold_totals"]
        for rows in attribution["by_feature"].values():
            self.assertEqual(sum(row["rows"] for row in rows), totals["rows"])
            self.assertAlmostEqual(sum(row["total_absolute_error"] for row in rows), totals["total_absolute_error"])
            self.assertEqual(sum(row["large_residual_count"] for row in rows), totals["large_residual_count"])
            self.assertEqual(sum(row["underprediction_count"] for row in rows), totals["underprediction_count"])
            self.assertEqual(sum(row["overprediction_count"] for row in rows), totals["overprediction_count"])

    def test_seed_specific_artifact_resume_isolated(self) -> None:
        self.assertTrue(model_diagnostics.artifact_seed_is_compatible({"controls": {"seed": 123}}, 123))
        self.assertFalse(model_diagnostics.artifact_seed_is_compatible({"controls": {"seed": 42}}, 123))

    def test_stratified_and_nearest_context(self) -> None:
        train, valid = self.frame.iloc[:20], self.frame.iloc[20:]
        random, _ = diagnostics.select_context(train, valid, self.features, 6, "random", 42)
        stratified, _ = diagnostics.select_context(train, valid, self.features, 6, "stratified", 42)
        nearest, _ = diagnostics.select_context(train, valid, self.features, 6, "nearest", 42)
        self.assertGreaterEqual(stratified["config_framework"].nunique(), random["config_framework"].nunique())
        self.assertEqual(len(nearest), 6)

    def test_transform_inverse_and_known_config_rows(self) -> None:
        values = pd.Series([0.0, 1.0, 9.0]); transformed = diagnostics.target_transform(values, "log1p")
        np.testing.assert_allclose(diagnostics.inverse_target(transformed.to_numpy(), "log1p"), values)
        work, _ = comparison.prepare_model_frame(self.frame, self.features, "metrics_p99_itl", 100, 42)
        for train, valid in diagnostics.known_config_folds(work, 3, 42):
            self.assertFalse(set(train) & set(valid))
            self.assertTrue(set(work.iloc[valid].config_id).issubset(set(work.iloc[train].config_id)))


if __name__ == "__main__":
    unittest.main()
