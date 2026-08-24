from __future__ import annotations
import json, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from modeling.dcbench import build_questions, score

class DcBenchTests(unittest.TestCase):
 def test_bank_is_unique_complete_and_provenanced(self):
  qs=build_questions(); self.assertEqual(len(qs),12); self.assertEqual(len({q['question_id'] for q in qs}),len(qs))
  for q in qs:
   self.assertTrue(q['gold_answer']); self.assertTrue(q['evidence']); self.assertTrue(q['source_snapshot'])
   for e in q['evidence']:
    self.assertTrue((ROOT/e['source_path']).exists())
 def test_rebuild_is_deterministic(self): self.assertEqual(build_questions(),build_questions())
 def test_numeric_and_categorical_scoring(self):
  qs=build_questions(); self.assertEqual(score(qs[0],str(qs[0]['gold_numeric_value'])),1); self.assertEqual(score(qs[0],'999999'),0); self.assertEqual(score(qs[2],qs[2]['gold_answer']),1)
 def test_no_unsupported_or_unresolved_epoch_spec_question(self):
  for q in build_questions():
   self.assertNotEqual(q['answerability_mode'],'unsupported_current_system'); self.assertNotIn('H100',q['question']); self.assertNotIn('H200',q['question'])
