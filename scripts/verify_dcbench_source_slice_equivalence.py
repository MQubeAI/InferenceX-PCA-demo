"""Prove that the compact DC Bench source slice preserves frozen benchmark content."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.dcbench import build_evaluation_inputs, build_questions  # noqa: E402
from modeling.dcbench_v0_2_candidates import build_candidates  # noqa: E402

REPORT_JSON = ROOT / "data/derived/dcbench/reports/source_slice_equivalence.json"
REPORT_MD = ROOT / "data/derived/dcbench/reports/source_slice_equivalence.md"


def _normalize(value: Any) -> Any:
    """Remove source-location provenance while retaining all benchmark content."""
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in value.items()
            if key not in {"operating_point_source", "source_path"}
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize(item) for item in value)
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_equivalence() -> dict[str, Any]:
    full_v01 = build_questions(source_mode="full")
    frozen_v01 = build_questions(source_mode="frozen")
    full_inputs = build_evaluation_inputs(full_v01, source_mode="full")
    frozen_inputs = build_evaluation_inputs(frozen_v01, source_mode="frozen")
    full_v02 = build_candidates(source_mode="full")
    frozen_v02 = build_candidates(source_mode="frozen")
    comparisons = {
        "v0_1_questions": (full_v01, frozen_v01, 12),
        "v0_1_evaluation_inputs": (full_inputs, frozen_inputs, 2),
        "v0_2_candidate_questions": (full_v02, frozen_v02, 28),
    }
    records: dict[str, Any] = {}
    for name, (full, frozen, expected_count) in comparisons.items():
        equivalent = _normalize(full) == _normalize(frozen)
        if not equivalent:
            raise AssertionError(f"Frozen source slice changed scientific content for {name}.")
        records[name] = {
            "equivalent_except_source_provenance": True,
            "full_normalized_sha256": _digest(full),
            "frozen_normalized_sha256": _digest(frozen),
            "expected_count": expected_count,
        }
    persisted_v01 = [json.loads(line) for line in (ROOT / "data/derived/dcbench/questions/inference_v0_1.jsonl").read_text().splitlines()]
    persisted_v02 = [json.loads(line) for line in (ROOT / "data/derived/dcbench/questions/inference_v0_2_candidates.jsonl").read_text().splitlines()]
    if _normalize(persisted_v01) != _normalize(frozen_v01):
        raise AssertionError("Persisted v0.1 question bank differs from frozen rebuild.")
    if _normalize(persisted_v02) != _normalize(frozen_v02):
        raise AssertionError("Persisted v0.2 candidate pool differs from frozen rebuild.")
    records["persisted_v0_1"] = {"equivalent_to_frozen_rebuild": True, "questions": len(persisted_v01)}
    records["persisted_v0_2"] = {"equivalent_to_frozen_rebuild": True, "questions": len(persisted_v02)}
    return {"status": "PASS", "allowed_difference": "operating-point source provenance only", "comparisons": records}


def main() -> None:
    report = verify_equivalence()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "# DC Bench source-slice equivalence\n\n"
        "**PASS.** The committed frozen source slice reproduces v0.1, its two model-facing input bundles, "
        "and the v0.2 candidate pool exactly after excluding only operating-point source-location provenance. "
        "The full enriched view remains a local extraction/reconstruction input, not a CI dependency.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
