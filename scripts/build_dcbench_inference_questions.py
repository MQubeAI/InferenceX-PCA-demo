"""Build the frozen, clean-CI-safe v0.1 DC Bench seed artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.dcbench import build_evaluation_inputs, build_questions  # noqa: E402


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("frozen", "full"),
        default="frozen",
        help="Use the committed source slice (default) or explicitly request the local full enriched view.",
    )
    args = parser.parse_args()
    questions = build_questions(source_mode=args.source)
    _write_jsonl(ROOT / "data/derived/dcbench/questions/inference_v0_1.jsonl", questions)
    fixtures = ROOT / "data/derived/dcbench/fixtures"
    _write_jsonl(
        fixtures / "perfect_responses.jsonl",
        [{"question_id": question["question_id"], "response": question["gold_answer"], "fixture": True} for question in questions],
    )
    _write_jsonl(
        fixtures / "wrong_responses.jsonl",
        [{"question_id": question["question_id"], "response": "__deliberately_wrong__", "fixture": True} for question in questions],
    )
    closed, evidence = build_evaluation_inputs(questions, source_mode=args.source)
    _write_jsonl(ROOT / "data/derived/dcbench/evaluation_inputs/closed_book.jsonl", closed)
    _write_jsonl(ROOT / "data/derived/dcbench/evaluation_inputs/evidence_supplied.jsonl", evidence)
    card = ROOT / "docs/dcbench/inference-question-bank-v0.1.md"
    sections = [
        "# Inference question bank v0.1\n",
        "Generated from the committed immutable source slice and compact validation artifacts. "
        "Objective operations are not hidden chain-of-thought requirements.\n",
    ]
    for question in questions:
        sections.append(
            f"## {question['question_id']} — {question['category']} ({question['difficulty']})\n\n"
            f"**Question:** {question['question']}\n\n"
            f"**Gold answer:** {question['gold_answer']}\n\n"
            f"**Why/evidence:** {question['required_sources']} — {' → '.join(question['derivation'])}.\n\n"
            f"**Failure indicates:** {', '.join(question['required_reasoning'])} failure.\n"
        )
    card.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
