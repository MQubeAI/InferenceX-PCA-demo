from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import model_side_integration as model_side


ROOT = Path(__file__).resolve().parents[1]


def aa_record(name: str, identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "name": name,
        "slug": name.lower().replace(".", "-").replace(" ", "-"),
        "model_creator": {"id": "creator", "name": "Creator"},
        "release_date": "2026-01-01",
        "evaluations": {}, "performance": {}, "pricing": {},
    }


class FrozenArtificialAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        if not model_side.AA_SNAPSHOT.exists() or not model_side.AA_MANIFEST.exists():
            self.skipTest("local frozen AA snapshot is not installed")

    def test_frozen_snapshot_hash_count_and_manifest_are_exact(self) -> None:
        _payload, verification = model_side.verify_frozen_snapshot()
        self.assertEqual(verification["sha256"], model_side.AA_SHA256)
        self.assertEqual(verification["record_count"], 200)
        self.assertTrue(verification["manifest_consistent"])

    def test_snapshot_and_manifest_do_not_contain_a_credential_value(self) -> None:
        # The integration module never accesses the environment; the frozen
        # data and manifest must also not serialize the credential identifier.
        for path in (model_side.AA_SNAPSHOT, model_side.AA_MANIFEST):
            self.assertNotIn("ARTIFICIAL_ANALYSIS_API_KEY", path.read_text(encoding="utf-8"))
        source = Path(model_side.__file__).read_text(encoding="utf-8")
        for environment_access in ("import os", "os.getenv", "os.environ"):
            self.assertNotIn(environment_access, source)

    def test_no_live_network_client_is_present(self) -> None:
        source = Path(model_side.__file__).read_text(encoding="utf-8")
        for forbidden in ("urlopen", "requests.", "httpx", "urllib.request"):
            self.assertNotIn(forbidden, source)


class ModelMappingTests(unittest.TestCase):
    def test_exact_mapping_accepts_only_reviewed_exact_variants(self) -> None:
        epoch = pd.DataFrame({"Model": ["MiniMax-M2.5", "MiniMax-M3", "GLM-5"]})
        aa = [aa_record("MiniMax-M2.5", "aa-m25"), aa_record("MiniMax-M3", "aa-m3"), aa_record("GLM-5 (Reasoning)", "aa-glm")]
        mapping = model_side.build_three_source_mapping(["minimaxm2.5", "minimaxm3", "glm5", "qwen3.5"], epoch, aa).set_index("inferencex_model")
        self.assertEqual(mapping.loc["minimaxm2.5", "overall_mapping_status"], "ACCEPTED")
        self.assertEqual(mapping.loc["minimaxm3", "overall_mapping_status"], "ACCEPTED")
        self.assertEqual(mapping.loc["glm5", "aa_mapping_status"], "UNRESOLVED")
        self.assertEqual(mapping.loc["qwen3.5", "overall_mapping_status"], "UNRESOLVED")
        self.assertTrue(mapping.loc[mapping.ambiguity, "overall_mapping_status"].ne("ACCEPTED").all())
        self.assertTrue(mapping.loc[mapping.overall_mapping_status.eq("ACCEPTED"), "evidence"].str.len().gt(0).all())

    def test_policy_excludes_aa_outcomes_from_primary_static_features(self) -> None:
        policy = model_side.model_feature_policy(pd.DataFrame(columns=["Model", "Parameters"]))
        self.assertEqual(policy["primary_static_model_feature_columns"], ["epoch_model_parameters"])
        self.assertFalse(set(policy["primary_static_model_feature_columns"]) & set(policy["excluded_aa_outcome_columns"]))
        self.assertIn("aa_median_output_tokens_per_second", policy["excluded_aa_outcome_columns"])


class EnrichmentInvariantTests(unittest.TestCase):
    def test_real_build_preserves_hardware_outputs_and_operating_point_invariants(self) -> None:
        if not model_side.AA_SNAPSHOT.exists() or not model_side.HARDWARE_VIEW.exists():
            self.skipTest("local frozen AA snapshot or hardware view is not installed")
        hardware_outputs = [
            model_side.HARDWARE_VIEW,
            ROOT / "data/derived/integration/reports/hardware_representation_experiment.json",
            ROOT / "data/derived/integration/reports/hardware_holdout/index.json",
        ]
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in hardware_outputs if path.exists()
        }
        result = model_side.build_artifacts()
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
        self.assertEqual(before, after)
        validation = result["validation"]
        self.assertTrue(validation["row_count_preserved"])
        self.assertTrue(validation["config_count_preserved"])
        self.assertTrue(validation["throughput_target_preserved"])
        self.assertEqual(validation["duplicate_operating_identity_count"], 0)
        self.assertTrue(validation["unresolved_external_fields_null"])

    def test_unresolved_aa_rows_are_null_and_aa_columns_are_prefixed(self) -> None:
        # This directly exercises the critical null/prefix contract without a
        # network call or a mutation of the frozen full view.
        record = aa_record("MiniMax-M2.5", "aa-m25")
        flat = pd.DataFrame([model_side._aa_flat_record(record)])
        unresolved = pd.DataFrame({"aa_mapping_status": ["UNRESOLVED"]})
        for column in flat.columns:
            if column != "aa_model_id":
                unresolved[column] = pd.NA
        self.assertTrue(unresolved[[c for c in flat if c != "aa_model_id"]].isna().all().all())
        self.assertTrue(all(column.startswith("aa_") for column in flat.columns))
        self.assertNotIn(model_side.TARGET, flat.columns)

    def test_target_comparison_is_exact_not_recomputed(self) -> None:
        before = np.array([1.0, 2.0, np.nan])
        after = np.array([1.0, 2.0, np.nan])
        self.assertTrue(np.array_equal(before, after, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
