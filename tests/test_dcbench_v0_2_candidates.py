from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.dcbench_v0_2_candidates import FIXED_CONFIGURATION, build_candidates


class DcBenchV02CandidateTests(unittest.TestCase):
    def test_candidate_pool_is_deterministic_and_complete(self) -> None:
        candidates = build_candidates()
        self.assertEqual(len(candidates), 28)
        self.assertEqual(len({candidate["question_id"] for candidate in candidates}), 28)
        self.assertEqual(candidates, build_candidates())
        for candidate in candidates:
            self.assertTrue(candidate["gold_answer"])
            self.assertTrue(candidate["evidence"])
            self.assertNotIn("H100", candidate["question"])
            self.assertNotIn("H200", candidate["question"])

    def test_committed_candidate_pool_matches_frozen_rebuild(self) -> None:
        persisted = [
            json.loads(line)
            for line in (ROOT / "data/derived/dcbench/questions/inference_v0_2_candidates.jsonl").read_text().splitlines()
        ]
        self.assertEqual(persisted, build_candidates())

    def test_controlled_comparisons_retain_every_fixed_field(self) -> None:
        for candidate in build_candidates():
            for evidence in candidate["evidence"]:
                rows = evidence.get("candidate_rows", [])
                fixed = evidence.get("fixed_configuration")
                if not fixed or len(rows) < 2:
                    continue
                self.assertEqual(set(fixed), set(FIXED_CONFIGURATION))
                for row in rows:
                    for field, expected in fixed.items():
                        self.assertEqual(row[field], expected)
