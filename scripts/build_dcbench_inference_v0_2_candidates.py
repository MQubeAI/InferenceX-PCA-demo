"""Build the separate, review-only v0.2 DC Bench candidate pool."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.dcbench_v0_2_candidates import VERSION, build_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("frozen", "full"),
        default="frozen",
        help="Use the committed source slice (default) or explicitly request the local full enriched view.",
    )
    args = parser.parse_args()
    candidates = build_candidates(source_mode=args.source)
    output = ROOT / "data/derived/dcbench/questions/inference_v0_2_candidates.jsonl"
    output.write_text("".join(json.dumps(question, sort_keys=True) + "\n" for question in candidates), encoding="utf-8")
    report = ROOT / "docs/dcbench/inference-question-bank-v0.2-candidates.md"
    sections = [
        "# Inference question bank v0.2 candidates\n",
        "These 28 mechanically verified items are a separate candidate pool created because no target frontier-model API or approved interface was available for v0.1 baseline collection. v0.1 remains frozen. Candidate selection still requires review after real baseline results exist.\n",
    ]
    for question in candidates:
        sections.append(
            f"## {question['question_id']} — {question['category']} ({question['difficulty']})\n\n"
            f"**Question:** {question['question']}\n\n"
            f"**Gold answer:** {question['gold_answer']}\n\n"
            f"**Evidence:** {question['required_sources']}\n\n"
            f"**Operations:** {' → '.join(question['derivation'])}\n"
        )
    report.write_text("\n".join(sections), encoding="utf-8")
    summary = {
        "version": VERSION,
        "questions": len(candidates),
        "categories": {category: sum(question["category"] == category for question in candidates) for category in sorted({question["category"] for question in candidates})},
        "difficulty": {difficulty: sum(question["difficulty"] == difficulty for question in candidates) for difficulty in sorted({question["difficulty"] for question in candidates})},
    }
    summary_path = ROOT / "data/derived/dcbench/reports/inference_v0_2_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
