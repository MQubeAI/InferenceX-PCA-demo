from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modeling.dcbench import build_evaluation_inputs, build_questions, score
from modeling.dcbench_source import FROZEN_SOURCE_MANIFEST, FROZEN_SOURCE_SLICE, sha256_file
from modeling.dcbench_v0_2_candidates import build_candidates


def _contains_gold(value):
    if isinstance(value, dict):
        return any("gold" in key or _contains_gold(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_gold(item) for item in value)
    return False

class DcBenchTests(unittest.TestCase):
    def test_bank_is_unique_complete_and_provenanced(self):
        questions = build_questions()
        self.assertEqual(len(questions), 12)
        self.assertEqual(len({question["question_id"] for question in questions}), len(questions))
        for question in questions:
            self.assertTrue(question["gold_answer"])
            self.assertTrue(question["evidence"])
            self.assertTrue(question["source_snapshot"])
            for evidence in question["evidence"]:
                self.assertTrue((ROOT / evidence["source_path"]).exists())

    def test_frozen_slice_hash_and_contract_are_valid(self):
        manifest = json.loads(FROZEN_SOURCE_MANIFEST.read_text())
        self.assertTrue(FROZEN_SOURCE_SLICE.is_file())
        self.assertEqual(manifest["slice_sha256"], sha256_file(FROZEN_SOURCE_SLICE))
        self.assertEqual(manifest["slice_rows"], 23)
        self.assertEqual(manifest["slice_columns"], 26)

    def test_frozen_builds_do_not_need_full_enriched_view(self):
        import modeling.dcbench_source as source
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            source, "FULL_ENRICHED_VIEW", Path(directory) / "missing_full_view.csv"
        ):
            self.assertEqual(len(build_questions()), 12)
            self.assertEqual(len(build_candidates()), 28)

    def test_rebuild_is_deterministic(self):
        self.assertEqual(build_questions(), build_questions())

    def test_committed_bank_matches_frozen_rebuild(self):
        persisted = [
            json.loads(line)
            for line in (ROOT / "data/derived/dcbench/questions/inference_v0_1.jsonl").read_text().splitlines()
        ]
        self.assertEqual(persisted, build_questions())

    def test_numeric_and_categorical_scoring(self):
        questions = build_questions()
        self.assertEqual(score(questions[0], str(questions[0]["gold_numeric_value"])), 1)
        self.assertEqual(score(questions[0], "999999"), 0)
        self.assertEqual(score(questions[2], questions[2]["gold_answer"]), 1)

    def test_no_unsupported_or_unresolved_epoch_spec_question(self):
        for question in build_questions():
            self.assertNotEqual(question["answerability_mode"], "unsupported_current_system")
            self.assertNotIn("H100", question["question"])
            self.assertNotIn("H200", question["question"])

    def test_model_facing_inputs_exclude_gold_fields_and_resolve_project_evidence(self):
        closed, evidence = build_evaluation_inputs(build_questions())
        self.assertEqual(len(closed), 12)
        self.assertEqual(len(evidence), 12)
        for item in [*closed, *evidence]:
            self.assertFalse(_contains_gold(item))
        by_id = {item["question_id"]: item for item in evidence}
        self.assertIn("random_forest_results", by_id["inference_v0_1_007"]["evidence"][0])
        self.assertIn("capability_random_forest_by_held_out_sku", by_id["inference_v0_1_009"]["evidence"][0])
