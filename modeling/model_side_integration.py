"""Reproducible, source-specific Artificial Analysis model-side integration.

This module reads the already frozen Artificial Analysis snapshot only.  It has
no HTTP client and never reads environment variables.  Artificial Analysis
serving observations remain source-prefixed contextual fields; they are never
throughput predictors or replacements for InferenceX observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.integration import TARGET, epoch_snapshot_date, load_existing_throughput_cohort
from modeling.dataset_checkpoint import default_data_dir, load_data_manifest

ROOT = Path(__file__).resolve().parents[1]
AA_SNAPSHOT = ROOT / "data/external/artificial_analysis/raw/artificial_analysis_language_models_free_20260825T030353569731Z.json"
AA_MANIFEST = ROOT / "data/external/artificial_analysis/manifest.json"
EPOCH_MODELS = ROOT / "data/external/epoch/raw/all_ai_models.csv"
INTEGRATION = ROOT / "data/derived/integration"
HARDWARE_VIEW = INTEGRATION / "views/inferencex_throughput_epoch_hardware_v1.csv"
AA_SHA256 = "6f588e1f8bd7dad6074544e5652e821bc5944170e647f6f92c914fc03cdc0f49"
AA_SNAPSHOT_ID = "artificial_analysis/free-models/2026-08-25"
IX_SNAPSHOT_ID = "db-dump/2026-07-20"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")


def verify_frozen_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the approved payload and manifest agree exactly."""
    if not AA_SNAPSHOT.exists():
        raise RuntimeError(f"Frozen AA snapshot missing: {AA_SNAPSHOT}")
    raw = AA_SNAPSHOT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != AA_SHA256:
        raise RuntimeError(f"Frozen AA snapshot SHA-256 mismatch: {digest}")
    payload = json.loads(raw)
    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 200:
        raise RuntimeError("Frozen AA snapshot record count is not exactly 200.")
    if not AA_MANIFEST.exists():
        raise RuntimeError("Artificial Analysis manifest is missing.")
    manifest = json.loads(AA_MANIFEST.read_text(encoding="utf-8"))
    relative = str(AA_SNAPSHOT.relative_to(ROOT))
    matches = [
        item for item in manifest.get("snapshots", [])
        if item.get("file") == relative
        and item.get("sha256") == AA_SHA256
        and item.get("raw_record_count") == 200
        and item.get("response_status") == 200
    ]
    if len(matches) != 1:
        raise RuntimeError("AA manifest does not contain one exact approved snapshot entry.")
    return payload, {
        "snapshot": relative,
        "sha256": digest,
        "record_count": len(records),
        "manifest_exact_entry_count": len(matches),
        "manifest_consistent": True,
    }


def _leaf_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        # Include containers as fields as well as their leaves.  This preserves
        # actual dict-versus-null counts for optional nested AA objects.
        result: list[tuple[str, Any]] = [(prefix, value)] if prefix else []
        for key, nested in value.items():
            result.extend(_leaf_paths(nested, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return [(prefix, value)]


def _aa_classification(path: str) -> tuple[str, str, str, str]:
    """Return category, entity level, eligibility, and conservative meaning."""
    if path in {"id", "name", "slug", "model_creator", "model_creator.id", "model_creator.name"}:
        return "A. identity/provenance", "model record", "contextual/display only", "source model identity or creator"
    if path == "release_date":
        return "A. identity/provenance", "model record", "contextual/display only", "model release-date metadata"
    if path == "evaluations" or path.startswith("evaluations."):
        return "C. benchmark/capability outcome", "AA model benchmark aggregate", "leakage-prone; excluded", "Artificial Analysis capability index"
    if path == "performance" or path.startswith("performance."):
        return "D. API/provider serving-performance observation", "AA model API observation; provider aggregation not declared", "semantically incompatible; excluded", "current API-serving observation"
    if path == "pricing" or path.startswith("pricing.") or path.startswith("artificial_analysis_intelligence_index_cost"):
        return "E. pricing/business metadata", "AA model pricing aggregate", "leakage-prone; excluded", "current pricing or cost metadata"
    return "F. other", "response metadata", "contextual/display only", "response-tier, version, or pagination metadata"


def source_audit(payload: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    records = payload["data"]
    leaves: dict[str, list[Any]] = {}
    for record in records:
        for path, value in _leaf_paths(record):
            leaves.setdefault(path, []).append(value)
    fields = []
    for path in sorted(leaves):
        values = leaves[path]
        present = len(values)
        nulls = sum(value is None for value in values)
        category, level, eligibility, meaning = _aa_classification(path)
        unit = "not stated by payload"
        if path == "performance.median_output_tokens_per_second":
            unit = "tokens/second (provider/API aggregate; not per GPU)"
        elif path.startswith("performance."):
            unit = "seconds"
        elif path.startswith("pricing.price_1m_"):
            unit = "price per 1M tokens; currency not supplied in payload"
        elif path.startswith("evaluations."):
            unit = "index points"
        elif path.endswith("cost_per_task.total_cost") or path.endswith(".total_cost"):
            unit = "cost unit not declared in payload"
        fields.append({
            "path": f"data[].{path}", "data_type": sorted({type(v).__name__ for v in values if v is not None}),
            "present_count": present, "null_count_when_present": nulls,
            "missing_count": len(records) - present + nulls,
            "missing_percentage": round((len(records) - present + nulls) * 100 / len(records), 3),
            "classification": category, "entity_level": level, "prediction_eligibility": eligibility,
            "unit": unit, "likely_meaning": meaning,
        })
    response_fields = []
    for path, value in _leaf_paths({key: value for key, value in payload.items() if key != "data"}):
        category, level, eligibility, meaning = _aa_classification(path)
        response_fields.append({"path": path, "data_type": type(value).__name__, "classification": category,
                                "entity_level": level, "prediction_eligibility": eligibility,
                                "unit": "not applicable", "likely_meaning": meaning})
    return {
        "source": "Artificial Analysis Data API Free endpoint", "endpoint": "https://artificialanalysis.ai/api/v2/language/models/free",
        "snapshot": verification, "top_level_structure": {key: type(value).__name__ for key, value in payload.items()},
        "entity_level": "one current Artificial Analysis model record; performance fields are API/model observations with no provider identity or aggregation context declared in this Free payload, not an InferenceX deployment operating point",
        "response_fields": response_fields, "record_fields": fields,
        "interpretation": "The Free response contains identity, release, AA benchmark indices, provider/API serving aggregates, and pricing. It contains no hardware, deployment configuration, total parameter, active-parameter, MoE, layer, hidden-size, or attention-configuration field.",
    }


# Each decision was manually reviewed against exact version labels; it is not a
# fuzzy matcher. AA parenthetical operating/reasoning variants are deliberately
# not silently equated to an InferenceX model label.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "dsr1": {"epoch": ["DeepSeek-R1", "DeepSeek-R1 (May 2025)"], "epoch_status": "UNRESOLVED", "aa": ["DeepSeek R1 (Jan '25)", "DeepSeek R1 Distill Llama 8B", "DeepSeek R1 Distill Qwen 1.5B", "DeepSeek R1 Distill Llama 70B"], "aa_status": "UNRESOLVED", "reason": "InferenceX shorthand lacks release/version; AA additionally contains base and distill variants."},
    "dsv4": {"epoch": ["DeepSeek-V4-Pro", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro-0813", "DeepSeek V4 Flash 0731"], "epoch_status": "UNRESOLVED", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "InferenceX V4 shorthand cannot select an Epoch Pro/Flash/dated variant; no AA V4 record."},
    "glm5": {"epoch": ["GLM-5"], "epoch_status": "ACCEPTED", "aa": ["GLM-5 (Reasoning)", "GLM-5-Turbo"], "aa_status": "UNRESOLVED", "reason": "Epoch exact match. AA distinguishes a reasoning offering and Turbo offering, while InferenceX does not."},
    "glm5.1": {"epoch": ["GLM-5.1"], "epoch_status": "ACCEPTED", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "Exact Epoch match; no GLM-5.1 AA record in the frozen Free response."},
    "glm5.2": {"epoch": ["GLM-5.2"], "epoch_status": "ACCEPTED", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "Exact Epoch match; no GLM-5.2 AA record in the frozen Free response."},
    "gptoss120b": {"epoch": ["gpt-oss-120b"], "epoch_status": "ACCEPTED", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "Exact Epoch match; no exact gpt-oss-120b AA record in the frozen Free response."},
    "kimik2.5": {"epoch": ["Kimi K2.5"], "epoch_status": "ACCEPTED", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "Exact Epoch match; AA has K2.6 offerings, not K2.5, so there is no exact AA candidate."},
    "llama70b": {"epoch": ["Llama 2-70B", "Llama 3-70B", "Llama 3.1-70B", "Llama 3.3 70B"], "epoch_status": "UNRESOLVED", "aa": ["DeepSeek R1 Distill Llama 70B", "Hermes 3 - Llama-3.1 70B", "Hermes 4 - Llama-3.1 70B (Non-reasoning)"], "aa_status": "UNRESOLVED", "reason": "Family/size shorthand does not identify Llama release or fine-tune; AA candidates are derivative offerings."},
    "minimaxm2.5": {"epoch": ["MiniMax-M2.5"], "epoch_status": "ACCEPTED", "aa": ["MiniMax-M2.5"], "aa_status": "ACCEPTED", "reason": "Exact version label is unique in both Epoch and AA."},
    "minimaxm3": {"epoch": ["MiniMax-M3"], "epoch_status": "ACCEPTED", "aa": ["MiniMax-M3"], "aa_status": "ACCEPTED", "reason": "Exact version label is unique in both Epoch and AA."},
    "qwen3.5": {"epoch": ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B", "Qwen3.5 397B-A17B", "Qwen3.5-Omni-Flash", "Qwen3.5-Omni-Plus"], "epoch_status": "UNRESOLVED", "aa": ["Qwen3.5 0.8B (Non-reasoning)", "Qwen3.5 2B (Non-reasoning)", "Qwen3.5 4B (Non-reasoning)", "Qwen3.5 9B (Reasoning)", "Qwen3.5 9B (Non-reasoning)", "Qwen3.5 27B (Reasoning)", "Qwen3.5 27B (Non-reasoning)", "Qwen3.5 35B A3B (Reasoning)", "Qwen3.5 122B A10B (Reasoning)", "Qwen3.5 122B A10B (Non-reasoning)", "Qwen3.5 397B A17B (Reasoning)", "Qwen3.5 397B A17B (Non-reasoning)", "Qwen3.5 Omni Plus"], "aa_status": "UNRESOLVED", "reason": "InferenceX family label omits size, dense/MoE choice, mode, and Omni variant."},
}


def build_three_source_mapping(inferencex_models: list[str], epoch: pd.DataFrame, aa_records: list[dict[str, Any]]) -> pd.DataFrame:
    epoch_names = set(epoch["Model"].dropna().astype(str))
    aa_by_name = {record["name"]: record for record in aa_records}
    rows = []
    for label in sorted(set(inferencex_models)):
        spec = MODEL_SPECS.get(label)
        if spec is None:
            spec = {"epoch": [], "epoch_status": "NO_CANDIDATE", "aa": [], "aa_status": "NO_CANDIDATE", "reason": "No manually reviewed mapping decision exists."}
        epoch_candidates = [name for name in spec["epoch"] if name in epoch_names]
        aa_candidates = [name for name in spec["aa"] if name in aa_by_name]
        epoch_status = spec["epoch_status"]
        aa_status = spec["aa_status"]
        if epoch_status == "ACCEPTED" and len(epoch_candidates) != 1:
            raise ValueError(f"Accepted Epoch mapping lacks one exact source record: {label}")
        if aa_status == "ACCEPTED" and len(aa_candidates) != 1:
            raise ValueError(f"Accepted AA mapping lacks one exact source record: {label}")
        canonical = f"canonical_model:{_slug(epoch_candidates[0])}" if epoch_status == "ACCEPTED" else pd.NA
        if epoch_status == aa_status == "ACCEPTED":
            overall, confidence, ambiguity = "ACCEPTED", "HIGH", False
        elif "UNRESOLVED" in {epoch_status, aa_status}:
            overall, confidence, ambiguity = "UNRESOLVED", "NONE", True
        else:
            overall, confidence, ambiguity = "NO_CANDIDATE", "NONE", False
        aa_record = aa_by_name[aa_candidates[0]] if aa_status == "ACCEPTED" else {}
        rows.append({
            "inferencex_model": label, "canonical_model_id": canonical,
            "epoch_model_id": f"epoch_model:{_slug(epoch_candidates[0])}" if epoch_status == "ACCEPTED" else pd.NA,
            "epoch_model_name": epoch_candidates[0] if epoch_status == "ACCEPTED" else pd.NA,
            "epoch_mapping_status": epoch_status,
            "aa_model_id": aa_record.get("id") if aa_status == "ACCEPTED" else pd.NA,
            "aa_model_name": aa_record.get("name") if aa_status == "ACCEPTED" else pd.NA,
            "aa_mapping_status": aa_status, "overall_mapping_status": overall,
            "confidence": confidence, "ambiguity": ambiguity,
            "candidate_count": len(epoch_candidates) + len(aa_candidates),
            "epoch_candidate_count": len(epoch_candidates), "aa_candidate_count": len(aa_candidates),
            "evidence": "Manual exact-version review. " + spec["reason"], "notes": spec["reason"],
            "inferencex_snapshot": IX_SNAPSHOT_ID, "epoch_snapshot": epoch_snapshot_date(), "aa_snapshot": AA_SNAPSHOT_ID,
        })
    mapping = pd.DataFrame(rows)
    if mapping.inferencex_model.duplicated().any():
        raise ValueError("Every InferenceX model must have one mapping decision.")
    if mapping.loc[mapping.ambiguity, "overall_mapping_status"].eq("ACCEPTED").any():
        raise ValueError("Ambiguous model mappings cannot be accepted.")
    return mapping


def _aa_flat_record(record: dict[str, Any]) -> dict[str, Any]:
    perf = record.get("performance", {})
    evaluation = record.get("evaluations", {})
    pricing = record.get("pricing", {})
    creator = record.get("model_creator", {})
    cost = record.get("artificial_analysis_intelligence_index_cost") or {}
    return {
        "aa_model_id": record.get("id"), "aa_model_name": record.get("name"), "aa_slug": record.get("slug"),
        "aa_model_creator_id": creator.get("id"), "aa_model_creator_name": creator.get("name"), "aa_release_date": record.get("release_date"),
        "aa_artificial_analysis_intelligence_index": evaluation.get("artificial_analysis_intelligence_index"),
        "aa_artificial_analysis_coding_index": evaluation.get("artificial_analysis_coding_index"),
        "aa_artificial_analysis_agentic_index": evaluation.get("artificial_analysis_agentic_index"),
        "aa_median_output_tokens_per_second": perf.get("median_output_tokens_per_second"),
        "aa_median_time_to_first_token_seconds": perf.get("median_time_to_first_token_seconds"),
        "aa_median_time_to_first_answer_token_seconds": perf.get("median_time_to_first_answer_token_seconds"),
        "aa_median_end_to_end_response_time_seconds": perf.get("median_end_to_end_response_time_seconds"),
        "aa_price_1m_cache_hit_tokens": pricing.get("price_1m_cache_hit_tokens"),
        "aa_price_1m_cache_write_tokens": pricing.get("price_1m_cache_write_tokens"),
        "aa_price_1m_input_tokens": pricing.get("price_1m_input_tokens"), "aa_price_1m_output_tokens": pricing.get("price_1m_output_tokens"),
        "aa_intelligence_index_cost_total_cost": cost.get("total_cost"),
        "aa_intelligence_index_cost_per_task_total_cost": (cost.get("cost_per_task") or {}).get("total_cost"),
        "aa_source": "Artificial Analysis Data API Free endpoint", "aa_source_snapshot": AA_SNAPSHOT_ID,
    }


STATIC_PRIMARY_FEATURES = ["epoch_model_parameters"]
AA_OUTCOME_COLUMNS = [
    "aa_artificial_analysis_intelligence_index", "aa_artificial_analysis_coding_index", "aa_artificial_analysis_agentic_index",
    "aa_median_output_tokens_per_second", "aa_median_time_to_first_token_seconds", "aa_median_time_to_first_answer_token_seconds",
    "aa_median_end_to_end_response_time_seconds", "aa_price_1m_cache_hit_tokens", "aa_price_1m_cache_write_tokens",
    "aa_price_1m_input_tokens", "aa_price_1m_output_tokens", "aa_intelligence_index_cost_total_cost",
    "aa_intelligence_index_cost_per_task_total_cost",
]


def model_feature_policy(epoch: pd.DataFrame) -> dict[str, Any]:
    records = []
    static = {"Parameters", "Parameters notes", "Numerical format"}
    identity = {"Model", "Organization", "Publication date", "Reference", "Link", "Authors", "Base model", "Hugging Face developer id"}
    for field in epoch.columns:
        if field in static:
            category = "A. STATIC / PRE-DEPLOYMENT MODEL DESCRIPTOR"
            eligible = "conditional: only normalized, exact-version static values; primary currently only Parameters"
        elif field in identity:
            category = "E. IDENTITY / PROVENANCE"
            eligible = "contextual/display only"
        elif any(token in field.lower() for token in ("training", "finetune", "compute", "hardware", "power", "dataset", "epoch", "batch", "utilization", "chip-hours")):
            category = "E. IDENTITY / PROVENANCE"
            eligible = "excluded: training/development metadata is not a serving-throughput descriptor"
        else:
            category = "B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME" if field in {"WikiText and Penn Treebank data"} else "E. IDENTITY / PROVENANCE"
            eligible = "contextual/display only"
        records.append({"source": "Epoch all_ai_models.csv", "field": field, "category": category, "primary_feature_eligibility": eligible})
    for column in _aa_flat_record({"id": None}).keys():
        if column in AA_OUTCOME_COLUMNS:
            category = "C. API / PROVIDER PERFORMANCE OUTCOME" if "median_" in column else ("D. PRICING / BUSINESS METADATA" if "price" in column or "cost" in column else "B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME")
            eligible = "excluded from primary physical-throughput representation"
        else:
            category, eligible = "E. IDENTITY / PROVENANCE", "contextual/display only"
        records.append({"source": "Artificial Analysis Free endpoint", "field": column, "category": category, "primary_feature_eligibility": eligible})
    return {"policy_version": "v1", "primary_static_model_feature_columns": STATIC_PRIMARY_FEATURES,
            "excluded_aa_outcome_columns": AA_OUTCOME_COLUMNS, "records": records,
            "rule": "Only model-intrinsic, pre-deployment static descriptors can enter a future primary unseen-model physical-throughput representation. No AA outcome, performance, latency, capability-index, or pricing field may enter it."}


DESCRIPTORS = [
    ("total_parameters", "Epoch Parameters", "epoch_model_parameters", "Direct numeric field; six accepted Epoch mappings are non-null."),
    ("active_parameters_per_token", "Epoch Parameters notes", None, "Only unstructured notes for GLM-5.2 and MiniMax-M2.5; not normalized."),
    ("dense_vs_moe", None, None, "No structured source field."), ("number_of_experts", None, None, "No structured source field."),
    ("active_experts_per_token", None, None, "No structured source field."), ("architecture_family", None, None, "No structured source field."),
    ("layer_count", None, None, "No structured source field."), ("hidden_size", None, None, "No structured source field."),
    ("attention_head_count", None, None, "No structured source field."), ("kv_head_count", None, None, "No structured source field."),
    ("context_length", None, None, "No structured source field; MiniMax-M3 abstract mentions 1M context but is unstructured."),
    ("vocabulary_size", None, None, "No structured source field."), ("attention_architecture", None, None, "No structured source field; MiniMax-M3 abstract mentions MSA but is unstructured."),
    ("parameter_precision", "Epoch Numerical format", None, "Column exists but has no accepted-model value."),
    ("kv_cache_relevant_characteristics", None, None, "No structured source field."),
]


def descriptor_coverage(mapping: pd.DataFrame, epoch: pd.DataFrame) -> pd.DataFrame:
    accepted = mapping.loc[mapping.epoch_mapping_status.eq("ACCEPTED"), ["inferencex_model", "epoch_model_name"]]
    meta = accepted.merge(epoch, left_on="epoch_model_name", right_on="Model", how="left", validate="one_to_one")
    rows = []
    all_models = mapping.inferencex_model.tolist()
    for descriptor, epoch_field, primary, note in DESCRIPTORS:
        available = pd.Series(False, index=meta.index)
        if descriptor == "total_parameters":
            available = pd.to_numeric(meta["Parameters"], errors="coerce").notna()
        missing = sorted(set(all_models) - set(meta.loc[available, "inferencex_model"]))
        rows.append({"descriptor": descriptor, "inferencex_availability": "no structured descriptor field", "epoch_source_field": epoch_field or "none",
                     "epoch_available_accepted_models": int(available.sum()), "aa_availability": "no field in Free response",
                     "combined_coverage_models": int(available.sum()), "combined_coverage_percent": round(100 * available.sum() / len(all_models), 2),
                     "models_missing": "; ".join(missing), "semantic_confidence": "HIGH" if descriptor == "total_parameters" else "INSUFFICIENT",
                     "primary_physical_throughput_eligibility": "conditional" if descriptor == "total_parameters" else "not eligible", "notes": note})
    return pd.DataFrame(rows)


def _epoch_metadata(mapping: pd.DataFrame, epoch: pd.DataFrame) -> pd.DataFrame:
    cols = ["Model", "Organization", "Publication date", "Parameters", "Parameters notes", "Numerical format", "Domain", "Task", "Base model", "Reference", "Link"]
    accepted = mapping.loc[mapping.epoch_mapping_status.eq("ACCEPTED"), ["inferencex_model", "epoch_model_id", "epoch_model_name"]]
    result = accepted.merge(epoch[cols], left_on="epoch_model_name", right_on="Model", how="left", validate="one_to_one")
    return result.rename(columns={"Organization": "epoch_model_organization", "Publication date": "epoch_model_publication_date", "Parameters": "epoch_model_parameters", "Parameters notes": "epoch_model_parameters_notes", "Numerical format": "epoch_model_numerical_format", "Domain": "epoch_model_domain", "Task": "epoch_model_task", "Base model": "epoch_model_base_model", "Reference": "epoch_model_reference", "Link": "epoch_model_link"}).drop(columns="Model")


def build_model_enrichment(mapping: pd.DataFrame, epoch: pd.DataFrame, aa_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not HARDWARE_VIEW.exists():
        raise FileNotFoundError(f"Existing validated hardware view is required: {HARDWARE_VIEW}")
    before = pd.read_csv(HARDWARE_VIEW, low_memory=False)
    base = before.copy()
    base["inferencex_model"] = base["config_model"].astype(str).str.strip().str.lower()
    before_target = pd.to_numeric(base[TARGET], errors="coerce").to_numpy()
    map_columns = ["inferencex_model", "canonical_model_id", "epoch_model_id", "epoch_model_name", "epoch_mapping_status", "aa_model_id", "aa_model_name", "aa_mapping_status", "overall_mapping_status"]
    view = base.merge(mapping[map_columns], on="inferencex_model", how="left", validate="many_to_one", suffixes=("", "_model_side"))
    metadata = _epoch_metadata(mapping, epoch)
    view = view.merge(metadata, on=["inferencex_model", "epoch_model_id", "epoch_model_name"], how="left", validate="many_to_one")
    aa = pd.DataFrame([_aa_flat_record(record) for record in aa_records])
    view = view.merge(aa, on="aa_model_id", how="left", validate="many_to_one", suffixes=("", "_observation"))
    # A failed/ambiguous link is null by construction, including all external observations.
    unresolved = view.aa_mapping_status.ne("ACCEPTED")
    for column in aa.columns:
        if column != "aa_model_id":
            view.loc[unresolved, column] = pd.NA
    after_target = pd.to_numeric(view[TARGET], errors="coerce").to_numpy()
    key = ["config_id", "benchmark_type", "isl", "osl", "conc"]
    target_same = len(before_target) == len(after_target) and np.array_equal(before_target, after_target, equal_nan=True)
    validation = {
        "rows_before": len(before), "rows_after": len(view), "configs_before": int(before.config_id.nunique(dropna=False)), "configs_after": int(view.config_id.nunique(dropna=False)),
        "row_count_preserved": len(before) == len(view), "config_count_preserved": int(before.config_id.nunique(dropna=False)) == int(view.config_id.nunique(dropna=False)),
        "throughput_target_preserved": target_same, "duplicate_operating_identity_count": int(view.duplicated(key).sum()),
        "unresolved_external_fields_null": bool(view.loc[unresolved, [c for c in aa.columns if c != "aa_model_id"]].isna().all().all()),
        "aa_source_prefixed": bool(all(column.startswith("aa_") for column in aa.columns)),
        "aa_metrics_do_not_alias_throughput": TARGET not in AA_OUTCOME_COLUMNS and all(column != TARGET for column in AA_OUTCOME_COLUMNS),
    }
    if not all([validation["row_count_preserved"], validation["config_count_preserved"], validation["throughput_target_preserved"], validation["duplicate_operating_identity_count"] == 0, validation["unresolved_external_fields_null"], validation["aa_source_prefixed"]]):
        raise AssertionError(f"Model enrichment invariant failure: {validation}")
    return view, validation


def _metric_contract() -> list[dict[str, str]]:
    return [
        {"aa_source_field": "performance.median_output_tokens_per_second", "aa_definition": "current AA API/model output-speed observation", "aa_entity_level": "AA model record; provider aggregation not declared", "aa_unit": "tokens/s, not per GPU", "related_inferencex_field": "metrics_tput_per_gpu", "inferencex_definition": "measured InferenceX deployment throughput per GPU", "inferencex_unit": "tokens/s/GPU", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "Hardware, workload, concurrency, aggregation, and denominator differ; provider context is absent. Never overwrite or alias."},
        {"aa_source_field": "performance.median_time_to_first_token_seconds", "aa_definition": "current AA API/model median time to first token", "aa_entity_level": "AA model record; provider aggregation not declared", "aa_unit": "seconds", "related_inferencex_field": "metrics_median_ttft", "inferencex_definition": "measured deployment median TTFT at an operating point", "inferencex_unit": "not declared in cohort schema", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "No matched request, server, provider, or measurement protocol."},
        {"aa_source_field": "performance.median_time_to_first_answer_token_seconds", "aa_definition": "current AA API/model answer-start latency", "aa_entity_level": "AA model record; provider aggregation not declared", "aa_unit": "seconds", "related_inferencex_field": "metrics_median_ttft", "inferencex_definition": "deployment median TTFT", "inferencex_unit": "not declared in cohort schema", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "AA answer-token definition and provider context are source-specific."},
        {"aa_source_field": "performance.median_end_to_end_response_time_seconds", "aa_definition": "current AA API/model end-to-end response latency", "aa_entity_level": "AA model record; provider aggregation not declared", "aa_unit": "seconds", "related_inferencex_field": "metrics_median_e2el", "inferencex_definition": "measured deployment median end-to-end latency", "inferencex_unit": "not declared in cohort schema", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "AA aggregate and absent provider context must remain source-specific."},
        {"aa_source_field": "evaluations.artificial_analysis_*_index", "aa_definition": "AA benchmark/capability index", "aa_entity_level": "AA model benchmark aggregate", "aa_unit": "index points", "related_inferencex_field": "none", "inferencex_definition": "none", "inferencex_unit": "n/a", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "Capability outcome, not physical serving descriptor."},
        {"aa_source_field": "pricing.price_1m_*_tokens and intelligence_index_cost", "aa_definition": "current provider price/cost metadata", "aa_entity_level": "AA model offering", "aa_unit": "price per 1M tokens; currency not supplied", "related_inferencex_field": "none", "inferencex_definition": "none", "inferencex_unit": "n/a", "comparability": "NOT_DIRECTLY_COMPARABLE", "caveat": "Business metadata can change and is not a measured InferenceX metric."},
    ]


def _model_vocabulary(cohort: pd.DataFrame) -> list[dict[str, Any]]:
    fields = ["config_hardware", "benchmark_type", "config_precision", "isl", "osl"]
    rows = []
    for model, group in cohort.groupby("config_model", dropna=False):
        entry = {"model_label": str(model), "row_count": int(len(group)), "unique_config_count": int(group.config_id.nunique(dropna=False))}
        for field in fields:
            values = group[field].dropna().unique().tolist()
            entry[field.replace("config_", "") + "_coverage"] = sorted([_json(value) for value in values], key=str)
        rows.append(entry)
    return rows


def _overlap(mapping: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Any]:
    indexed = mapping.set_index("inferencex_model")
    labels = cohort.config_model.astype(str).str.lower()
    aa = labels.map(indexed.aa_mapping_status).eq("ACCEPTED")
    both = labels.map(indexed.overall_mapping_status).eq("ACCEPTED")
    reasons = []
    for _, row in mapping.loc[mapping.overall_mapping_status.ne("ACCEPTED")].iterrows():
        reasons.append({"inferencex_model": row.inferencex_model, "overall_mapping_status": row.overall_mapping_status, "reason": row.notes})
    return {"total_inferencex_model_labels": int(len(mapping)), "accepted_epoch_mappings": int(mapping.epoch_mapping_status.eq("ACCEPTED").sum()),
            "accepted_aa_mappings": int(mapping.aa_mapping_status.eq("ACCEPTED").sum()), "accepted_mappings_to_both": int(mapping.overall_mapping_status.eq("ACCEPTED").sum()),
            "rows_represented_by_accepted_aa_mappings": int(aa.sum()), "configs_represented_by_accepted_aa_mappings": int(cohort.loc[aa, "config_id"].nunique(dropna=False)),
            "rows_represented_across_all_three_sources": int(both.sum()), "configs_represented_across_all_three_sources": int(cohort.loc[both, "config_id"].nunique(dropna=False)),
            "unresolved_or_missing": reasons}


def _temporal_audit(audit: dict[str, Any]) -> dict[str, Any]:
    fields = []
    for record in audit["record_fields"]:
        path = record["path"]
        if path in {"data[].id", "data[].name", "data[].slug", "data[].model_creator", "data[].model_creator.id", "data[].model_creator.name", "data[].release_date"}:
            status, rule = "static / conditionally safely retroactive", "Identity only; source release metadata does not prove availability at each InferenceX run."
        elif path.startswith("data[].evaluations"):
            status, rule = "current benchmark result", "Post-hoc outcome; never historical predictor feature."
        elif path.startswith("data[].performance"):
            status, rule = "provider-dependent current metric", "Post-hoc API/model observation; never historical predictor feature."
        elif path.startswith("data[].pricing") or path.startswith("data[].artificial_analysis_intelligence_index_cost"):
            status, rule = "pricing that may have changed", "Post-hoc business metadata; never historical predictor feature."
        else:
            status, rule = "unknown as-of status", "Response metadata; contextual only."
        fields.append({"field": path, "temporal_status": status, "rule": rule})
    fields.extend([
        {"field": "Epoch Parameters", "temporal_status": "static / requires release and source-provenance review", "rule": "Potential future descriptor only after exact model and as-of proof."},
        {"field": "Epoch Parameters notes and Numerical format", "temporal_status": "unknown as-of status", "rule": "Unstructured or absent for this cohort; not a current predictor feature."},
    ])
    return {"inferencex_snapshot": "approximately 2026-07-20", "aa_snapshot": "2026-08-25", "finding": "The AA snapshot is post-hoc for the frozen InferenceX cohort.", "fields": fields}


def _worked_example(view: pd.DataFrame, epoch: pd.DataFrame, aa_records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = view.loc[(view.inferencex_model.eq("minimaxm2.5")) & (view.overall_mapping_status.eq("ACCEPTED")) & (view.hardware_mapping_status.eq("accepted"))].copy()
    if candidates.empty:
        return {"status": "NO_ELIGIBLE_WORKED_EXAMPLE"}
    row = candidates.sort_values(["config_id", "benchmark_type", "isl", "osl", "conc"], kind="stable").iloc[0]
    display = ["config_model", "config_hardware", "isl", "osl", "conc", "config_framework", "config_precision", "config_prefill_tp", "config_prefill_ep", "config_prefill_num_workers", "config_decode_tp", "config_decode_ep", "config_decode_num_workers", "config_num_prefill_gpu", "config_num_decode_gpu", TARGET, "metrics_median_ttft", "metrics_median_e2el"]
    deployment = {field: _json(row[field]) for field in display if field in row}
    hardware = {field: _json(row[field]) for field in ["canonical_hardware_id", "epoch_hardware_name", "epoch_memory_capacity_bytes", "epoch_memory_bandwidth_bytes_per_second", "epoch_tensor_fp16_bf16_peak_flops", "epoch_fp8_peak_flops", "epoch_fp4_peak_flops", "epoch_tdp_watts", "epoch_intranode_bandwidth_bytes_per_second"] if field in row}
    model = epoch.loc[epoch.Model.eq("MiniMax-M2.5"), ["Model", "Organization", "Publication date", "Parameters", "Parameters notes", "Domain", "Task"]].iloc[0].to_dict()
    aa = next(record for record in aa_records if record["name"] == "MiniMax-M2.5")
    return {"status": "OK", "selection": "Deterministic first accepted MiniMax-M2.5 InferenceX row with accepted Epoch hardware and AA mapping.",
            "inferencex_deployment": deployment, "epoch_hardware": hardware,
            "epoch_model_metadata": {key: _json(value) for key, value in model.items()}, "artificial_analysis_model_observation": _aa_flat_record(aa),
            "combined_value": "It relates one measured physical deployment to a documented exact model identity, accelerator descriptors, and separate current AA capability/provider observations. It does not turn AA observations into per-GPU outcomes or causal explanations."}


def _markdown_source_audit(audit: dict[str, Any]) -> str:
    rows = "\n".join(f"| `{item['path']}` | {', '.join(item['data_type'])} | {item['missing_count']} ({item['missing_percentage']}%) | {item['classification']} | {item['prediction_eligibility']} |" for item in audit["record_fields"])
    return "# Artificial Analysis Free endpoint audit\n\nThis audit reads only the verified 2026-08-25 frozen Free-endpoint snapshot. It does not call the API. The response is a current model-record catalog, not an InferenceX deployment table.\n\n" + "| Field | Type | Missing | Class | Throughput-prediction treatment |\n|---|---|---:|---|---|\n" + rows + "\n\nAA contains no physical architecture descriptor fields needed for unseen-model transfer. Provider speed, latency, capability indices, and prices stay source-specific.\n"


def _markdown_metric_contract(contract: list[dict[str, str]]) -> str:
    rows = "\n".join(f"| `{r['aa_source_field']}` | {r['aa_unit']} | `{r['related_inferencex_field']}` | {r['comparability']} | {r['caveat']} |" for r in contract)
    return "# Artificial Analysis metric contract\n\n| AA field | AA unit/entity | InferenceX relation | Classification | Caveat |\n|---|---|---|---|---|\n" + rows + "\n\n`aa_median_output_tokens_per_second` never overwrites or aliases `metrics_tput_per_gpu`; AA latency never overwrites or aliases deployment latency.\n"


def _markdown_coverage(coverage: pd.DataFrame) -> str:
    rows = "\n".join(f"| {r.descriptor} | {r.combined_coverage_models}/11 | {r.primary_physical_throughput_eligibility} | {r.notes} |" for r in coverage.itertuples())
    return "# Model descriptor coverage\n\n| Descriptor | Actual combined coverage | Primary-feature status | Notes |\n|---|---:|---|---|\n" + rows + "\n\nOnly total parameter count has a structured value for any accepted model (6/11); that is not an adequate unseen-model physical representation.\n"


def _markdown_summary(verification: dict[str, Any], overlap: dict[str, Any], coverage: pd.DataFrame, example: dict[str, Any]) -> str:
    accepted = ", ".join(["MiniMax-M2.5", "MiniMax-M3"])
    return f"""# Three-source model-side integration summary

InferenceX provides measured physical deployment operating points and `metrics_tput_per_gpu`. Epoch hardware provides physical accelerator descriptors. Epoch model data provides explicit model identity and a sparse static descriptor set, principally total parameters. Artificial Analysis Free provides 200 current model records with identity, release date, capability indices, API/model speed and latency observations, and pricing; the payload does not declare a provider identity or aggregation context. It provides no architecture descriptors.

The verified snapshot is `{verification['snapshot']}` (200 records, SHA-256 `{verification['sha256']}`). Exact three-source mappings are {accepted}. The three-source overlap is {overlap['rows_represented_across_all_three_sources']} rows / {overlap['configs_represented_across_all_three_sources']} configs. Other labels are unresolved because of version/family ambiguity or have no exact AA candidate; details are in `three_source_model_overlap.md`.

AA output speed is not per-GPU throughput, AA latency is not InferenceX deployment latency, capability indices are outcomes, and pricing is business metadata. None is eligible as a primary physical throughput feature. A real worked example is stored in `three_source_worked_example.json`; it combines sources without conflating their measurements.

The transfer decision is **B. PARTIALLY SUFFICIENT — NEED ADDITIONAL MODEL-SPEC SOURCE FIRST**. {coverage.loc[coverage.descriptor.eq('total_parameters'), 'combined_coverage_models'].iloc[0]}/11 models have a structured total-parameter value, while all MoE, active-parameter, layer, hidden-size, attention, and KV-cache descriptors are absent as structured fields. No leave-one-model-out pilot was run.

Next source: an immutable, exact-version model-spec ingestion from official Hugging Face `config.json`, official model cards, or vendor technical reports with release/as-of provenance and manual canonical mapping.
"""


def _markdown_overlap(overlap: dict[str, Any]) -> str:
    unresolved = "\n".join(f"- `{row['inferencex_model']}` — {row['reason']}" for row in overlap["unresolved_or_missing"])
    return f"# Three-source model overlap\n\n- InferenceX labels: {overlap['total_inferencex_model_labels']}\n- Accepted Epoch mappings: {overlap['accepted_epoch_mappings']}\n- Accepted AA mappings: {overlap['accepted_aa_mappings']}\n- Accepted mappings to both: {overlap['accepted_mappings_to_both']}\n- AA-represented InferenceX: {overlap['rows_represented_by_accepted_aa_mappings']} rows / {overlap['configs_represented_by_accepted_aa_mappings']} configs\n- Three-source: {overlap['rows_represented_across_all_three_sources']} rows / {overlap['configs_represented_across_all_three_sources']} configs\n\n## Unresolved or absent\n\n{unresolved}\n"


def _markdown_policy(policy: dict[str, Any]) -> str:
    rows = "\n".join(f"| {r['source']} | `{r['field']}` | {r['category']} | {r['primary_feature_eligibility']} |" for r in policy["records"])
    return "# Model feature policy\n\nOnly pre-deployment, model-intrinsic static descriptors may be considered for a future physical throughput representation. The currently normalized candidate is Epoch `Parameters`; it is insufficient on its own.\n\n| Source | Field | Category | Primary predictor treatment |\n|---|---|---|---|\n" + rows + "\n\nAll AA capability indices, API performance, latency, pricing, and cost fields are excluded.\n"


def _markdown_example(example: dict[str, Any]) -> str:
    if example.get("status") != "OK":
        return "# Three-source worked example\n\nNo eligible exact three-source operating point was available.\n"
    def table(values: dict[str, Any]) -> str:
        return "\n".join(f"| `{key}` | {value} |" for key, value in values.items())
    return "# Three-source worked example\n\n" + example["selection"] + "\n\n## InferenceX deployment\n\n| Field | Value |\n|---|---|\n" + table(example["inferencex_deployment"]) + "\n\n## Epoch hardware\n\n| Field | Value |\n|---|---|\n" + table(example["epoch_hardware"]) + "\n\n## Epoch model metadata\n\n| Field | Value |\n|---|---|\n" + table(example["epoch_model_metadata"]) + "\n\n## Artificial Analysis model/provider observation\n\n| Field | Value |\n|---|---|\n" + table(example["artificial_analysis_model_observation"]) + "\n\n## What the connection adds\n\n" + example["combined_value"] + "\n"


def _cross_source_comparison(overlap: dict[str, Any]) -> dict[str, Any]:
    return {"status": "NOT SCIENTIFICALLY_DEFENSIBLE_YET", "mapped_model_count": overlap["accepted_mappings_to_both"],
            "decision": "No Spearman correlation or rank comparison was calculated.",
            "reason": "Only two exact three-source models are available, and AA output speed is a provider/API aggregate rather than per-GPU throughput. A common controlled InferenceX subset would not cure the inadequate overlap.",
            "invariant": "AA output speed is NOT per-GPU throughput; this analysis would be descriptive only even with adequate overlap."}


def _docs() -> dict[Path, str]:
    return {
        ROOT / "docs/integration/canonical-schema.md": """# Canonical entity and observation schema

`CanonicalHardware` is one resolved Epoch accelerator record, retaining vendor, type, release/provenance, memory capacity, memory bandwidth, FP16/BF16/FP8/FP4/INT8/INT4 peaks, TDP, and valid interconnect descriptors. These are descriptors, not measured InferenceX outcomes. `DeploymentOperatingPoint` remains the established `config_id × benchmark_type × isl × osl × conc` InferenceX aggregate with original hardware/model identity, workload, configuration, and measured outcome fields.

`CanonicalModel` represents a verified exact model/version only. `EpochModelMetadata` retains Epoch identity, static descriptors, training/development metadata, and provenance. `ArtificialAnalysisModelObservation` retains AA identity, release, capability-index, and pricing observations. `ArtificialAnalysisProviderObservation` retains AA API/model speed and latency observations and explicitly records that the current Free payload supplies no provider identity or aggregation context. These classes are not flattened into universal measurements.

```text
DeploymentOperatingPoint              CanonicalHardware
        |                                      |
   CanonicalModel ---- EpochModelMetadata   Epoch physical capabilities
       /      \
InferenceX    ArtificialAnalysisModelObservation
                 + ArtificialAnalysisProviderObservation
                         |
               task-specific modeling views
                         |
          observed / predicted / unsupported

time-varying token demand -> required GPU count -> GPU cluster capacity
-> datacenter capacity -> power / cost
```

Cluster and datacenter information remains downstream and is never a per-GPU throughput predictor. AA API measurements and pricing retain the `aa_` prefix in the model-enriched view and cannot replace InferenceX fields.
""",
        ROOT / "docs/integration/model-spec-source-gap.md": """# Model specification source gap

The current sources are partially sufficient for identity and a sparse total-parameter field, but insufficient for scientific unseen-model throughput transfer. Missing structured exact-version descriptors are active parameters/token, dense/MoE type, expert and active-expert counts, layers, hidden size, attention/KV-head configuration, context length, vocabulary, attention architecture, precision, and KV-cache-relevant characteristics.

Recommended next source class: immutable exact-version copies of official Hugging Face `config.json`, official model cards, vendor technical reports, or official architecture repositories. Each ingestion should store the source URL/identifier, retrieval timestamp, release/as-of provenance, raw immutable snapshot hash, normalized fields with units, and an explicit canonical-model mapping. Do not scrape uncontrolled web pages.
""",
        ROOT / "docs/integration/model-support-novelty.md": """# Future model support and novelty framework

An unseen model should eventually be labelled **SUPPORTED**, **WEAK SUPPORT**, or **UNSUPPORTED** from uncalibrated diagnostics, not invented thresholds. Candidate dimensions are total-parameter distance, active-parameter distance, dense/MoE mismatch, architecture-family mismatch, expert-count and active-expert novelty, layer and hidden-size distance, context-range violation, attention-architecture mismatch, and descriptor missingness pattern.

The current sources cannot calibrate this framework: all dimensions except sparse total parameters are missing as structured exact-version data. Missingness itself must not fingerprint a model identity or become a misleading feature.
""",
        ROOT / "docs/dcbench/model-side-expansion-plan.md": """# DC Bench model-side expansion plan

The real AA Free response supports source-grounded tasks on exact model identity resolution, dense-vs-MoE *missingness* reasoning, total-versus-active parameter distinction, API speed versus GPU throughput distinction, capability score versus serving performance distinction, price/performance interpretation, provider-level versus physical deployment metrics, temporal leakage, and unsupported cross-source comparisons.

It does not yet support architecture-field retrieval for the eleven InferenceX labels or physical unseen-model transfer. DC Bench v0.1 remains frozen; this is a future task plan only.
""",
    }


def build_artifacts(data_dir: str | None = None) -> dict[str, Any]:
    payload, verification = verify_frozen_snapshot()
    audit = source_audit(payload, verification)
    epoch = pd.read_csv(EPOCH_MODELS, low_memory=False)
    cohort, _ = load_existing_throughput_cohort(data_dir or str(default_data_dir(load_data_manifest())))
    cohort["config_model"] = cohort["config_model"].astype(str).str.strip().str.lower()
    mapping = build_three_source_mapping(cohort.config_model.tolist(), epoch, payload["data"])
    coverage = descriptor_coverage(mapping, epoch)
    policy = model_feature_policy(epoch)
    overlap = _overlap(mapping, cohort)
    view, validation = build_model_enrichment(mapping, epoch, payload["data"])
    example = _worked_example(view, epoch, payload["data"])
    contract = _metric_contract()
    temporal = _temporal_audit(audit)
    comparison = _cross_source_comparison(overlap)
    report = INTEGRATION / "reports"
    _write_json(report / "artificial_analysis/source_audit.json", audit)
    _write_json(report / "artificial_analysis/metric_contract.json", {"records": contract})
    _write_json(report / "three_source_model_overlap.json", overlap)
    _write_json(report / "model_feature_policy.json", policy)
    _write_json(report / "model_enrichment_validation.json", validation)
    _write_json(report / "temporal_model_feature_eligibility.json", temporal)
    _write_json(report / "three_source_worked_example.json", example)
    _write_json(report / "cross_source_model_performance_comparison.json", comparison)
    _write_json(report / "inferencex_model_vocabulary.json", {"snapshot": IX_SNAPSHOT_ID, "models": _model_vocabulary(cohort)})
    (report / "artificial_analysis/source_audit.md").write_text(_markdown_source_audit(audit), encoding="utf-8")
    (report / "three_source_model_overlap.md").write_text(_markdown_overlap(overlap), encoding="utf-8")
    (report / "model_feature_policy.md").write_text(_markdown_policy(policy), encoding="utf-8")
    (report / "three_source_worked_example.md").write_text(_markdown_example(example), encoding="utf-8")
    (report / "cross_source_model_performance_comparison.md").write_text("# Exploratory cross-source comparison\n\n**NOT SCIENTIFICALLY DEFENSIBLE YET.** Only two exact three-source models map. AA output speed is a provider/API aggregate, not per-GPU throughput; no correlation or rank statistic was calculated.\n", encoding="utf-8")
    mapping_path = INTEGRATION / "mappings/three_source_model_mapping_v1.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False)
    coverage.to_csv(report / "model_descriptor_coverage.csv", index=False)
    view.to_csv(INTEGRATION / "views/inferencex_throughput_epoch_hardware_model_aa_v1.csv", index=False)
    (ROOT / "docs/integration/artificial-analysis-integration.md").write_text(_markdown_source_audit(audit), encoding="utf-8")
    (ROOT / "docs/integration/artificial-analysis-metric-contract.md").write_text(_markdown_metric_contract(contract), encoding="utf-8")
    (ROOT / "docs/integration/model-descriptor-coverage.md").write_text(_markdown_coverage(coverage), encoding="utf-8")
    (report / "three_source_integration_summary.md").write_text(_markdown_summary(verification, overlap, coverage, example), encoding="utf-8")
    for path, content in _docs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return {"verification": verification, "overlap": overlap, "validation": validation, "worked_example": example,
            "decision": "B. PARTIALLY SUFFICIENT — NEED ADDITIONAL MODEL-SPEC SOURCE FIRST", "view_rows": len(view)}
