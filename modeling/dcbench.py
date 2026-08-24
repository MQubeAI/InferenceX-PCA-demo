"""Deterministic seed builder and scorer for the DC Bench inference track."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.dcbench_source import (
    ROOT,
    V0_1_CONTROLLED_ANCHORS,
    V0_1_DIRECT_ANCHORS,
    anchored_rows,
    load_operating_point_source,
)

VALIDATION = ROOT / "data/derived/integration/reports/validation/controlled_hardware_validation.json"
TRANSFER = ROOT / "data/derived/integration/reports/validation/held_out_hardware_transfer.json"
TARGET = "metrics_tput_per_gpu"
VERSION = "inference_v0_1"

FIXED = [
    "benchmark_type",
    "isl",
    "osl",
    "conc",
    "config_model",
    "config_framework",
    "config_precision",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_decode_tp",
    "config_decode_ep",
    "config_disagg",
    "config_is_multinode",
]


def _safe(value: Any) -> Any:
    value = value.item() if hasattr(value, "item") else value
    return None if isinstance(value, float) and math.isnan(value) else value


def _row_evidence(row: pd.Series, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": source["path"],
        "row": {column: _safe(row[column]) for column in ["config_id", "config_hardware", *FIXED, TARGET]},
    }


def _question(
    qid: str,
    category: str,
    difficulty: str,
    question: str,
    answer: str,
    answer_type: str,
    mode: str,
    evidence: list[dict[str, Any]],
    derivation: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "track": "inference_performance",
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": str(answer),
        "answer_type": answer_type,
        "gold_numeric_value": extra.pop("numeric", None),
        "gold_numeric_tolerance": extra.pop("tolerance", None),
        "choices": extra.pop("choices", None),
        "required_reasoning": extra.pop("reasoning", ["retrieve"]),
        "required_sources": extra.pop("sources", ["InferenceX July 20 aggregate"]),
        "answerability_mode": mode,
        "evidence": evidence,
        "derivation": derivation,
        "source_snapshot": "InferenceX db-dump/2026-07-20",
        "artifact_version": VERSION,
        "notes": extra.pop("notes", ""),
        **extra,
    }


def _controlled_pair(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve the historical v0.1 pair by immutable identities, not outcome order."""
    pair = anchored_rows(frame, V0_1_CONTROLLED_ANCHORS)
    if not pair["hardware_mapping_status"].eq("accepted").all() or pair[TARGET].isna().any():
        raise ValueError("Frozen v0.1 controlled-comparison anchors are not eligible.")
    for column in FIXED:
        if pair[column].nunique(dropna=False) != 1:
            raise ValueError(f"Frozen v0.1 controlled pair differs in fixed field: {column}")
    return pair


def build_questions(source_mode: str = "frozen") -> list[dict[str, Any]]:
    """Build v0.1 from an explicit operating-point source boundary.

    ``frozen`` is the clean-CI default. ``full`` is only for local provenance
    reconstruction against the excluded complete enriched view.
    """
    view, source = load_operating_point_source(source_mode)
    view = view.loc[view[TARGET].notna()].copy()
    direct = anchored_rows(view, V0_1_DIRECT_ANCHORS)
    a, b = _controlled_pair(view).iloc
    base = ", ".join(f"{column.replace('config_', '')}={a[column]}" for column in FIXED[:6])
    av, bv = float(a[TARGET]), float(b[TARGET])
    high, low = (a, b) if av >= bv else (b, a)
    pct = (float(high[TARGET]) / float(low[TARGET]) - 1) * 100

    questions: list[dict[str, Any]] = []
    for number, (_, row) in enumerate(direct.iterrows(), 1):
        questions.append(
            _question(
                f"{VERSION}_{number:03d}",
                "direct_retrieval",
                "easy",
                f"For the measured operating point with config_id {row.config_id}, hardware {row.config_hardware}, ISL {row.isl}, OSL {row.osl}, and concurrency {row.conc}, what throughput per GPU was reported?",
                f"{float(row[TARGET]):.6f} tokens/s/GPU",
                "numeric",
                "table_only",
                [_row_evidence(row, source)],
                ["retrieve the exact aggregate row"],
                numeric=float(row[TARGET]),
                tolerance=1e-3,
                reasoning=["retrieve", "read metric"],
            )
        )
    questions += [
        _question(
            f"{VERSION}_003",
            "controlled_comparison",
            "medium",
            f"Holding {base} fixed, which of {a.config_hardware} and {b.config_hardware} has higher measured throughput per GPU?",
            high.config_hardware,
            "categorical",
            "multirow_reasoning",
            [_row_evidence(a, source), _row_evidence(b, source)],
            ["verify fixed variables", "compare throughput"],
            reasoning=["filter", "compare"],
        ),
        _question(
            f"{VERSION}_004",
            "quantitative_multirow_reasoning",
            "medium",
            f"For the same controlled pair ({a.config_hardware} versus {b.config_hardware}; {base}), by what percent is the higher measured throughput per GPU greater than the lower?",
            f"{pct:.3f}%",
            "numeric",
            "multirow_reasoning",
            [_row_evidence(a, source), _row_evidence(b, source)],
            ["(higher/lower - 1) * 100"],
            numeric=pct,
            tolerance=0.05,
            reasoning=["filter", "calculate", "compare"],
        ),
        _question(
            f"{VERSION}_005",
            "configuration_tradeoff_reasoning",
            "medium",
            f"Among these two measured candidates with {base}, select the deployment maximizing throughput per GPU: {a.config_hardware} ({av:.3f}) or {b.config_hardware} ({bv:.3f}) tokens/s/GPU.",
            high.config_hardware,
            "categorical",
            "multirow_reasoning",
            [_row_evidence(a, source), _row_evidence(b, source)],
            ["compare supplied measured candidates"],
            reasoning=["retrieve", "optimize"],
        ),
        _question(
            f"{VERSION}_006",
            "planning_constrained_decision",
            "hard",
            f"A demand of 10,000 output tokens/s must use one of two measured candidates with {base}: {a.config_hardware} ({av:.3f} tokens/s/GPU) or {b.config_hardware} ({bv:.3f} tokens/s/GPU). Assuming throughput scales linearly only for this bookkeeping calculation, which requires fewer GPUs: ceil(demand / measured throughput per GPU)?",
            high.config_hardware,
            "categorical",
            "multirow_reasoning",
            [_row_evidence(a, source), _row_evidence(b, source)],
            ["ceil(10000/a)", "ceil(10000/b)", "choose smaller"],
            reasoning=["calculate", "constrained_select"],
            notes="This is a measured-candidate sizing calculation, not a scaling-law claim.",
        ),
        _question(
            f"{VERSION}_007",
            "research_result_interpretation",
            "hard",
            "In the controlled July Epoch-resolved cohort, did adding Epoch capabilities to hardware identity materially improve ordinary interpolation?",
            "No; the difference was small and not material.",
            "short_text",
            "project_analysis",
            [{"source_path": str(VALIDATION.relative_to(ROOT)), "json_path": "results.A_identity_only.random_forest and B_identity_plus_capabilities.random_forest"}],
            ["compare fixed-fold A and B", "apply recorded interpretation"],
            sources=["controlled hardware validation"],
            reasoning=["interpret_experiment"],
        ),
        _question(
            f"{VERSION}_008",
            "research_result_interpretation",
            "hard",
            "What does near-equal capability-only and identity-only Random Forest accuracy support?",
            "Physical capability descriptors preserve much of the predictive information; it does not prove reconstruction of GPU identity.",
            "short_text",
            "project_analysis",
            [{"source_path": str(VALIDATION.relative_to(ROOT)), "json_path": "results.C_capabilities_only.random_forest"}],
            ["compare A/C", "respect claim boundary"],
            sources=["controlled hardware validation"],
            reasoning=["interpret_experiment", "avoid_overclaim"],
        ),
    ]
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    records = transfer["records"]
    best = min(records, key=lambda record: record["metrics"]["capability_random_forest"]["mae"])
    weak = max(
        records,
        key=lambda record: record["metrics"]["capability_random_forest"]["median_absolute_percentage_error"],
    )
    for suffix, record, prompt in (
        ("009", best, "Which held-out SKU had the lowest capability-RF MAE?"),
        ("010", weak, "Which held-out SKU had the highest capability-RF MdAPE?"),
    ):
        questions.append(
            _question(
                f"{VERSION}_{suffix}",
                "generalization_transfer_reasoning",
                "hard",
                prompt,
                record["held_out_sku"],
                "categorical",
                "project_analysis",
                [{"source_path": str(TRANSFER.relative_to(ROOT)), "held_out_hardware": record["held_out_sku"]}],
                ["compare held-out metrics"],
                sources=["held-out hardware transfer"],
                reasoning=["retrieve", "rank"],
            )
        )
    questions += [
        _question(
            f"{VERSION}_011",
            "support_uncertainty_reasoning",
            "hard",
            "Does the seven-SKU capability-distance versus MdAPE association establish a calibrated support threshold for future hardware?",
            "No; it is descriptive across only seven SKUs and does not define a threshold.",
            "short_text",
            "project_analysis",
            [{"source_path": str(TRANSFER.relative_to(ROOT)), "field": "support_relationship"}],
            ["interpret descriptive Spearman evidence"],
            sources=["held-out hardware transfer"],
            reasoning=["interpret_statistics", "avoid_overclaim"],
        ),
        _question(
            f"{VERSION}_012",
            "planning_constrained_decision",
            "hard",
            "A planner wants a hardware-transfer case with an interpolation-like descriptor profile and the lower of the two such MdAPEs. Which SKU should it choose: GB200 or MI325X?",
            "MI325X",
            "categorical",
            "project_analysis",
            [
                {"source_path": str(TRANSFER.relative_to(ROOT)), "held_out_sku": "GB200"},
                {"source_path": str(TRANSFER.relative_to(ROOT)), "held_out_sku": "MI325X"},
            ],
            ["identify interpolation-like records", "compare MdAPE"],
            sources=["held-out hardware transfer"],
            reasoning=["filter", "compare", "constrained_select"],
        ),
    ]
    for question in questions:
        question["operating_point_source"] = source
    return questions


def _random_forest_summary(validation: dict[str, Any], representation: str) -> dict[str, Any]:
    pooled = validation["results"][representation]["models"]["random_forest"]["summary"]["pooled_oof"]
    return {
        "representation": representation,
        "r2": pooled["r2"],
        "mae_tokens_per_second_per_gpu": pooled["mae"],
        "median_absolute_percentage_error": pooled["median_absolute_percentage_error"],
        "within_20_percent": pooled["within_20_percent"],
        "rows": pooled["rows"],
    }


def build_evaluation_inputs(
    questions: list[dict[str, Any]] | None = None, source_mode: str = "frozen"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build model-facing inputs without exposing gold-answer fields."""
    questions = build_questions(source_mode) if questions is None else questions
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    transfer_by_sku = {record["held_out_sku"].casefold(): record for record in transfer["records"]}
    closed = [
        {
            "question_id": question["question_id"],
            "question": question["question"],
            "context_mode": "closed_book",
            "expected_answer_format": question["answer_type"],
        }
        for question in questions
    ]
    evidence = []
    for prompt, question in zip(closed, questions, strict=True):
        suffix = question["question_id"].rsplit("_", 1)[-1]
        if suffix in {"007", "008"}:
            representations = ["A_identity_only", "B_identity_plus_capabilities"] if suffix == "007" else ["A_identity_only", "C_capabilities_only"]
            resolved = [{
                "source_path": str(VALIDATION.relative_to(ROOT)),
                "protocol": {
                    "cohort_rows": validation["primary_cohort"]["rows"],
                    "cohort_configs": validation["primary_cohort"]["configs"],
                    "grouped_validation": validation["primary_cohort"]["split"],
                },
                "random_forest_results": [_random_forest_summary(validation, representation) for representation in representations],
            }]
        elif suffix in {"009", "010"}:
            metric = "mae" if suffix == "009" else "median_absolute_percentage_error"
            resolved = [{
                "source_path": str(TRANSFER.relative_to(ROOT)),
                "metric": metric,
                "capability_random_forest_by_held_out_sku": [
                    {"held_out_sku": record["held_out_sku"], metric: record["metrics"]["capability_random_forest"][metric]}
                    for record in transfer["records"]
                ],
            }]
        elif suffix == "011":
            resolved = [{
                "source_path": str(TRANSFER.relative_to(ROOT)),
                "support_error_relationship": transfer["support_error_relationship"],
                "claim_boundary": transfer["claim_boundary"],
            }]
        elif suffix == "012":
            resolved = [{
                "source_path": str(TRANSFER.relative_to(ROOT)),
                "held_out_skus": [
                    {
                        "held_out_sku": sku,
                        "median_absolute_percentage_error": transfer_by_sku[sku.casefold()]["metrics"]["capability_random_forest"]["median_absolute_percentage_error"],
                        "normalized_distance": transfer_by_sku[sku.casefold()]["support"]["nearest"]["normalized_distance"],
                        "outside_dimension_count": transfer_by_sku[sku.casefold()]["support"]["outside_dimension_count"],
                    }
                    for sku in ("GB200", "MI325X")
                ],
            }]
        else:
            resolved = question["evidence"]
        evidence.append({**prompt, "context_mode": "evidence_supplied", "evidence": resolved, "derivation_contract": question["derivation"]})
    return closed, evidence


def extract_contract_answer(raw_response: str) -> str:
    """Extract an ``Answer:`` line from the documented manual prompt contract."""
    match = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", str(raw_response))
    if not match:
        raise ValueError("No non-empty 'Answer:' line found in the raw response")
    return match.group(1).strip()


def score(question: dict[str, Any], response: str) -> float:
    """Score a response using the deterministic v0.1 answer contract."""
    expected = str(question["gold_answer"]).strip().casefold()
    received = str(response).strip().casefold()
    if question["answer_type"] == "numeric":
        numbers = re.findall(r"[-+]?\d*\.?\d+", received)
        if not numbers:
            return 0.0
        return float(abs(float(numbers[0]) - float(question["gold_numeric_value"])) <= float(question["gold_numeric_tolerance"]))
    return float(received == expected)
