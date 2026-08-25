"""Deterministic, separate v0.2 candidate pool for DC Bench inference tasks.

The frozen v0.1 seed is intentionally not imported or modified here.  These
items are candidates for review after baseline collection becomes available.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.dcbench import ROOT, TARGET, TRANSFER, VALIDATION
from modeling.dcbench_source import load_operating_point_source

VERSION = "inference_v0_2_candidate"
FIXED_CONFIGURATION = [
    "benchmark_type",
    "isl",
    "osl",
    "conc",
    "config_model",
    "config_framework",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_dp_attention",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_dp_attention",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_num_decode_gpu",
    "workers",
    "offload_mode",
]
NO_CONCURRENCY = [column for column in FIXED_CONFIGURATION if column != "conc"]


def _safe(value: Any) -> Any:
    value = value.item() if hasattr(value, "item") else value
    return None if pd.isna(value) else value


def _same(left: Any, right: Any) -> bool:
    return (pd.isna(left) and pd.isna(right)) or left == right


def _row(row: pd.Series) -> dict[str, Any]:
    columns = ["config_id", "config_hardware", *FIXED_CONFIGURATION, TARGET]
    return {column: _safe(row[column]) for column in columns}


def _fixed(row: pd.Series, columns: list[str] = FIXED_CONFIGURATION) -> dict[str, Any]:
    return {column: _safe(row[column]) for column in columns}


def _question(
    number: int,
    category: str,
    difficulty: str,
    prompt: str,
    answer: str,
    answer_type: str,
    mode: str,
    evidence: list[dict[str, Any]],
    derivation: list[str],
    *,
    numeric: float | None = None,
    tolerance: float | None = None,
    reasoning: list[str] | None = None,
    sources: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "question_id": f"{VERSION}_{number:03d}",
        "track": "inference_performance",
        "category": category,
        "difficulty": difficulty,
        "question": prompt,
        "gold_answer": answer,
        "answer_type": answer_type,
        "gold_numeric_value": numeric,
        "gold_numeric_tolerance": tolerance,
        "choices": None,
        "required_reasoning": reasoning or ["retrieve"],
        "required_sources": sources or ["InferenceX July 20 aggregate"],
        "answerability_mode": mode,
        "evidence": evidence,
        "derivation": derivation,
        "source_snapshot": "InferenceX db-dump/2026-07-20",
        "artifact_version": VERSION,
        "notes": notes,
    }


def _accepted_view(source_mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    view, source = load_operating_point_source(source_mode)
    return (
        view.loc[
            view["hardware_mapping_status"].eq("accepted")
            & view[TARGET].notna()
            & view["benchmark_type"].eq("single_turn")
        ].copy(),
        source,
    )


def _hardware_groups(view: pd.DataFrame) -> list[pd.DataFrame]:
    groups = []
    for _, group in view.groupby(FIXED_CONFIGURATION, dropna=False, sort=True):
        deduplicated = group.sort_values(["config_hardware", "config_id"]).drop_duplicates("config_hardware")
        if deduplicated["config_hardware"].nunique() >= 2:
            groups.append(deduplicated.reset_index(drop=True))
    return groups


def _select_hardware_group(
    groups: list[pd.DataFrame], model: str, precision: str, required_hardware: tuple[str, ...]
) -> pd.DataFrame:
    required = set(required_hardware)
    for group in groups:
        if group.iloc[0]["config_model"] != model or group.iloc[0]["config_precision"] != precision:
            continue
        if required.issubset(set(group["config_hardware"])):
            return group.loc[group["config_hardware"].isin(required_hardware)].copy()
    raise ValueError(f"No controlled group for {model}, {precision}, {required_hardware}")


def _comparison_evidence(rows: pd.DataFrame, source: dict[str, Any]) -> list[dict[str, Any]]:
    reference = rows.iloc[0]
    for _, row in rows.iloc[1:].iterrows():
        assert all(_same(reference[column], row[column]) for column in FIXED_CONFIGURATION)
    return [
        {
            "source_path": source["path"],
            "fixed_configuration": _fixed(reference),
            "candidate_rows": [_row(row) for _, row in rows.iterrows()],
        }
    ]


def _concurrency_rows(view: pd.DataFrame) -> pd.DataFrame:
    for _, group in view.groupby(["config_hardware", *NO_CONCURRENCY], dropna=False, sort=True):
        selected = group.sort_values(["conc", "config_id"]).drop_duplicates("conc")
        if (
            selected.iloc[0]["config_hardware"] == "b200"
            and selected.iloc[0]["config_model"] == "dsr1"
            and selected.iloc[0]["config_framework"] == "dynamo-sglang"
            and selected.iloc[0]["config_precision"] == "fp4"
            and len(selected) >= 4
        ):
            return selected.iloc[[0, 1, 2, 3]].copy()
    raise ValueError("No deterministic concurrency comparison found")


def _rf_summary(validation: dict[str, Any], representation: str) -> dict[str, Any]:
    pooled = validation["results"][representation]["models"]["random_forest"]["summary"]["pooled_oof"]
    return {
        "representation": representation,
        "r2": pooled["r2"],
        "mae_tokens_per_second_per_gpu": pooled["mae"],
        "median_absolute_percentage_error": pooled["median_absolute_percentage_error"],
        "within_20_percent": pooled["within_20_percent"],
    }


def build_candidates(source_mode: str = "frozen") -> list[dict[str, Any]]:
    """Build the review-only v0.2 pool from an explicit source boundary."""
    view, source = _accepted_view(source_mode)
    groups = _hardware_groups(view)
    pair_a = _select_hardware_group(groups, "dsr1", "fp4", ("b200", "b300"))
    pair_b = _select_hardware_group(groups, "dsv4", "fp4", ("b300", "mi355x"))
    pair_c = _select_hardware_group(groups, "minimaxm3", "fp8", ("b300", "gb300"))
    pair_d = _select_hardware_group(groups, "qwen3.5", "fp8", ("gb200", "gb300"))
    triple = _select_hardware_group(groups, "dsv4", "fp4", ("b200", "b300", "mi355x"))
    concurrency = _concurrency_rows(view)
    validation = json.loads(VALIDATION.read_text())
    transfer = json.loads(TRANSFER.read_text())
    questions: list[dict[str, Any]] = []

    for number, hardware in enumerate(("b200", "b300", "gb200", "mi355x"), start=1):
        row = view.loc[view["config_hardware"].eq(hardware)].sort_values(["config_id", "isl", "osl", "conc"]).iloc[0]
        value = float(row[TARGET])
        questions.append(
            _question(
                number,
                "direct_retrieval",
                "easy",
                f"For the measured operating point with config_id {row.config_id}, hardware {row.config_hardware}, ISL {row.isl}, OSL {row.osl}, and concurrency {row.conc}, what throughput per GPU was reported?",
                f"{value:.6f} tokens/s/GPU",
                "numeric",
                "table_only",
                [{"source_path": source["path"], "row": _row(row)}],
                ["retrieve the measured aggregate row"],
                numeric=value,
                tolerance=1e-3,
                reasoning=["retrieve", "read metric"],
            )
        )

    comparisons = (pair_a, pair_b, pair_c, pair_d)
    for number, rows in enumerate(comparisons, start=5):
        high = rows.loc[rows[TARGET].idxmax()]
        names = " and ".join(rows["config_hardware"].tolist())
        questions.append(
            _question(
                number,
                "controlled_comparison",
                "medium",
                f"The evidence gives two measured candidates with all workload and serving-configuration fields fixed. Which of {names} has higher throughput per GPU?",
                str(high["config_hardware"]),
                "categorical",
                "multirow_reasoning",
                _comparison_evidence(rows, source),
                ["verify every listed fixed field", "compare measured throughput per GPU"],
                reasoning=["filter", "compare"],
            )
        )

    quantitative_pairs = (pair_a, pair_b, pair_c)
    for number, rows in enumerate(quantitative_pairs, start=9):
        high = float(rows[TARGET].max())
        low = float(rows[TARGET].min())
        if number == 9:
            value = (high / low - 1) * 100
            wording = "By what percent is the higher measured throughput per GPU greater than the lower?"
            derivation = ["(higher / lower - 1) * 100"]
            answer = f"{value:.3f}%"
        elif number == 10:
            value = high - low
            wording = "What is the absolute throughput-per-GPU difference between the two measured candidates?"
            derivation = ["higher throughput minus lower throughput"]
            answer = f"{value:.3f} tokens/s/GPU"
        else:
            value = high / low
            wording = "What is the ratio of higher to lower measured throughput per GPU?"
            derivation = ["higher throughput divided by lower throughput"]
            answer = f"{value:.4f}"
        questions.append(
            _question(
                number,
                "quantitative_multirow_reasoning",
                "medium",
                f"For the two controlled measured candidates in the evidence, {wording}",
                answer,
                "numeric",
                "multirow_reasoning",
                _comparison_evidence(rows, source),
                derivation,
                numeric=value,
                tolerance=0.05 if number == 9 else 1e-3,
                reasoning=["verify fixed variables", "calculate", "compare"],
            )
        )

    rank_order = triple.sort_values(TARGET, ascending=False)["config_hardware"].tolist()
    questions.append(
        _question(
            12,
            "quantitative_multirow_reasoning",
            "medium",
            "Rank the three controlled measured candidates in the evidence from highest to lowest throughput per GPU, using ` > ` between hardware labels.",
            " > ".join(rank_order),
            "ranking",
            "multirow_reasoning",
            _comparison_evidence(triple, source),
            ["verify fixed variables", "sort measured throughput per GPU descending"],
            reasoning=["filter", "rank"],
        )
    )

    concurrency_evidence = [
        {
            "source_path": source["path"],
            "fixed_configuration_except_concurrency": _fixed(concurrency.iloc[0], NO_CONCURRENCY),
            "candidate_rows": [_row(row) for _, row in concurrency.iterrows()],
        }
    ]
    low_conc, mid_conc, high_conc, max_conc = (concurrency.iloc[index] for index in range(4))
    growth = (float(high_conc[TARGET]) / float(mid_conc[TARGET]) - 1) * 100
    questions.extend(
        [
            _question(
                13,
                "configuration_tradeoff_reasoning",
                "medium",
                "With every listed field except concurrency fixed, which listed concurrency has the highest measured throughput per GPU?",
                str(int(max_conc["conc"])),
                "categorical",
                "multirow_reasoning",
                concurrency_evidence,
                ["hold all non-concurrency fields fixed", "select the largest measured throughput"],
                reasoning=["filter", "optimize"],
            ),
            _question(
                14,
                "quantitative_multirow_reasoning",
                "hard",
                f"With every listed field except concurrency fixed, by what percent does measured throughput per GPU change from concurrency {int(mid_conc['conc'])} to {int(high_conc['conc'])}?",
                f"{growth:.3f}%",
                "numeric",
                "multirow_reasoning",
                concurrency_evidence,
                ["(throughput at higher concurrency / throughput at lower concurrency - 1) * 100"],
                numeric=growth,
                tolerance=0.05,
                reasoning=["filter", "calculate"],
            ),
        ]
    )

    for number, rows, demand in ((15, pair_a, 1000), (16, pair_c, 1000), (17, triple, 500)):
        gpu_counts = {row["config_hardware"]: math.ceil(demand / float(row[TARGET])) for _, row in rows.iterrows()}
        best = min(gpu_counts, key=gpu_counts.get)
        questions.append(
            _question(
                number,
                "planning_constrained_decision",
                "hard",
                f"A demand of {demand:,} output tokens/s must use one listed measured candidate. For bookkeeping only, assume linear scaling and choose the hardware requiring the fewest GPUs: ceil(demand / measured throughput per GPU).",
                best,
                "categorical",
                "multirow_reasoning",
                _comparison_evidence(rows, source),
                ["calculate ceil(demand / measured throughput per GPU) for each candidate", "choose the smallest count"],
                reasoning=["calculate", "constrained_select"],
                notes="Measured-candidate sizing calculation only; this does not claim a scaling law.",
            )
        )

    representations = ["A_identity_only", "B_identity_plus_capabilities", "C_capabilities_only", "D_capabilities_plus_matched_precision"]
    rf_evidence = [
        {
            "source_path": str(VALIDATION.relative_to(ROOT)),
            "cohort": {"rows": validation["primary_cohort"]["rows"], "configs": validation["primary_cohort"]["configs"], "split": validation["primary_cohort"]["split"]},
            "random_forest_results": [_rf_summary(validation, representation) for representation in representations],
        }
    ]
    summary = {entry["representation"]: entry for entry in rf_evidence[0]["random_forest_results"]}
    best_r2 = max(summary, key=lambda name: summary[name]["r2"])
    best_mae = min(summary, key=lambda name: summary[name]["mae_tokens_per_second_per_gpu"])
    questions.extend(
        [
            _question(18, "research_result_interpretation", "hard", "Which controlled Random Forest representation has the highest pooled out-of-fold R²?", best_r2, "categorical", "project_analysis", rf_evidence, ["compare pooled out-of-fold R² across A/B/C/D"], reasoning=["retrieve", "rank"], sources=["controlled hardware validation"]),
            _question(19, "research_result_interpretation", "hard", "Which controlled Random Forest representation has the lowest pooled out-of-fold MAE?", best_mae, "categorical", "project_analysis", rf_evidence, ["compare pooled out-of-fold MAE across A/B/C/D"], reasoning=["retrieve", "rank"], sources=["controlled hardware validation"]),
            _question(20, "research_result_interpretation", "hard", "Did adding matched precision peak compute to the capability-only representation materially improve ordinary interpolation?", "No; D was slightly worse than C on both R² and MAE.", "short_text", "project_analysis", rf_evidence, ["compare C and D under the fixed protocol"], reasoning=["interpret_experiment", "avoid_overclaim"], sources=["controlled hardware validation"]),
            _question(21, "research_result_interpretation", "hard", "Does the A/B/C/D result support claiming that Epoch capabilities materially improve ordinary interpolation?", "No; the ordinary-interpolation differences are small and not material.", "short_text", "project_analysis", rf_evidence, ["compare A/B/C/D", "respect recorded claim boundary"], reasoning=["interpret_experiment", "avoid_overclaim"], sources=["controlled hardware validation"]),
        ]
    )

    transfer_rows = []
    for record in transfer["records"]:
        metrics = record["metrics"]
        capability = metrics["capability_random_forest"]
        transfer_rows.append(
            {
                "held_out_sku": record["held_out_sku"],
                "test_rows": record["test_rows"],
                "test_configs": record["test_configs"],
                "capability_rf_mae_tokens_per_second_per_gpu": capability["mae"],
                "capability_rf_mdape": capability["median_absolute_percentage_error"],
                "capability_rf_within_20_percent": capability["within_20_percent"],
                "capability_rf_r2": capability["r2"],
                "global_median_mae_tokens_per_second_per_gpu": metrics["global_training_median"]["mae"],
                "nearest_analogue_mae_tokens_per_second_per_gpu": metrics["nearest_hardware_capability_analogue"]["mae"],
                "nearest_training_hardware": record["support"]["nearest"]["training_hardware"],
                "normalized_distance": record["support"]["nearest"]["normalized_distance"],
                "outside_dimension_count": record["support"]["outside_dimension_count"],
            }
        )
    transfer_evidence = [{"source_path": str(TRANSFER.relative_to(ROOT)), "held_out_sku_records": transfer_rows, "claim_boundary": transfer["claim_boundary"]}]
    best_transfer_mae = min(transfer_rows, key=lambda row: row["capability_rf_mae_tokens_per_second_per_gpu"])["held_out_sku"]
    worst_transfer_mdape = max(transfer_rows, key=lambda row: row["capability_rf_mdape"])["held_out_sku"]
    questions.extend(
        [
            _question(22, "generalization_transfer_reasoning", "hard", "Which held-out measured SKU has the lowest capability-Random-Forest MAE?", best_transfer_mae, "categorical", "project_analysis", transfer_evidence, ["rank held-out capability-RF MAE"], reasoning=["retrieve", "rank"], sources=["held-out hardware transfer"]),
            _question(23, "generalization_transfer_reasoning", "hard", "Which held-out measured SKU has the highest capability-Random-Forest MdAPE?", worst_transfer_mdape, "categorical", "project_analysis", transfer_evidence, ["rank held-out capability-RF MdAPE"], reasoning=["retrieve", "rank"], sources=["held-out hardware transfer"]),
            _question(24, "generalization_transfer_reasoning", "hard", "For held-out GB300, which has lower MAE: capability Random Forest or the nearest hardware-capability analogue baseline?", "capability Random Forest", "categorical", "project_analysis", transfer_evidence, ["compare the two GB300 MAEs"], reasoning=["filter", "compare"], sources=["held-out hardware transfer"]),
            _question(25, "generalization_transfer_reasoning", "hard", "Does the held-out-SKU study support calling this unreleased-hardware prediction?", "No; it is observed held-out-SKU transfer, not unreleased-hardware prediction.", "short_text", "project_analysis", transfer_evidence, ["read the claim boundary", "avoid unsupported extrapolation"], reasoning=["interpret_experiment", "avoid_overclaim"], sources=["held-out hardware transfer"]),
        ]
    )

    farthest = max(transfer_rows, key=lambda row: row["normalized_distance"])["held_out_sku"]
    questions.extend(
        [
            _question(26, "support_uncertainty_reasoning", "hard", "Which held-out SKU has the largest nearest-training-hardware normalized capability distance?", farthest, "categorical", "project_analysis", transfer_evidence, ["rank normalized nearest-hardware distances"], reasoning=["retrieve", "rank"], sources=["held-out hardware transfer"]),
            _question(27, "support_uncertainty_reasoning", "hard", "Between MI300X and MI325X, which has the higher capability-Random-Forest MdAPE despite a larger nearest-hardware distance?", "MI300X", "categorical", "project_analysis", transfer_evidence, ["filter MI300X and MI325X", "compare distance and MdAPE"], reasoning=["filter", "compare"], sources=["held-out hardware transfer"]),
            _question(28, "support_uncertainty_reasoning", "hard", "Does the seven-SKU distance-versus-MdAPE association define a calibrated support threshold for arbitrary future hardware?", "No; it is descriptive across seven SKUs and does not define a calibrated threshold.", "short_text", "project_analysis", [{"source_path": str(TRANSFER.relative_to(ROOT)), "support_error_relationship": transfer["support_error_relationship"], "claim_boundary": transfer["claim_boundary"]}], ["interpret seven-SKU descriptive association", "avoid threshold overclaim"], reasoning=["interpret_statistics", "avoid_overclaim"], sources=["held-out hardware transfer"]),
        ]
    )
    assert len(questions) == 28
    for question in questions:
        question["operating_point_source"] = source
    return questions
