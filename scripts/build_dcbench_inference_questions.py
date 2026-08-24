from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.dcbench import ROOT, build_questions
out=ROOT/"data/derived/dcbench/questions/inference_v0_1.jsonl"; out.parent.mkdir(parents=True,exist_ok=True)
questions=build_questions(); out.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in questions),encoding="utf-8")
fixtures=ROOT/"data/derived/dcbench/fixtures"; fixtures.mkdir(parents=True,exist_ok=True)
(fixtures/"perfect_responses.jsonl").write_text("".join(json.dumps({"question_id":q["question_id"],"response":q["gold_answer"],"fixture":True})+"\n" for q in questions),encoding="utf-8")
(fixtures/"wrong_responses.jsonl").write_text("".join(json.dumps({"question_id":q["question_id"],"response":"__deliberately_wrong__","fixture":True})+"\n" for q in questions),encoding="utf-8")
inputs=ROOT/"data/derived/dcbench/evaluation_inputs"; inputs.mkdir(parents=True,exist_ok=True)
closed=[{"question_id":q["question_id"],"question":q["question"],"context_mode":"closed_book","expected_answer_format":q["answer_type"]} for q in questions]
evidence=[{**x,"context_mode":"evidence_supplied","evidence":q["evidence"],"derivation_contract":q["derivation"]} for x,q in zip(closed,questions)]
for name,items in (("closed_book.jsonl",closed),("evidence_supplied.jsonl",evidence)):
 (inputs/name).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in items),encoding="utf-8")
templates=ROOT/"data/derived/dcbench/evaluation_responses/templates"; templates.mkdir(parents=True,exist_ok=True)
for model in ("gpt","claude","qwen","kimi"):
 (templates/f"{model}_responses.jsonl").write_text("".join(json.dumps({"model":model,"run_id":"fill_in","question_id":q["question_id"],"response":"","context_mode":"fill_in","timestamp":"fill_in","notes":"paste complete raw response"})+"\n" for q in questions),encoding="utf-8")
card=ROOT/"docs/dcbench/inference-question-bank-v0.1.md"
sections=["# Inference question bank v0.1\n\nGenerated from the frozen derived view and validation artifacts. Objective operations are not hidden chain-of-thought requirements.\n"]
for q in questions:
    sections.append(f"## {q['question_id']} — {q['category']} ({q['difficulty']})\n\n**Question:** {q['question']}\n\n**Gold answer:** {q['gold_answer']}\n\n**Why/evidence:** {q['required_sources']} — {' → '.join(q['derivation'])}.\n\n**Failure indicates:** {', '.join(q['required_reasoning'])} failure.\n")
card.write_text("\n".join(sections),encoding="utf-8")
print(out)
