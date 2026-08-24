"""Deterministic seed builder and scorer for the DC Bench inference track."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "data/derived/integration/views/inferencex_throughput_epoch_hardware_v1.csv"
VALIDATION = ROOT / "data/derived/integration/reports/validation/controlled_hardware_validation.json"
TRANSFER = ROOT / "data/derived/integration/reports/validation/held_out_hardware_transfer.json"
TARGET = "metrics_tput_per_gpu"
VERSION = "inference_v0_1"

FIXED = ["benchmark_type","isl","osl","conc","config_model","config_framework","config_precision","config_prefill_tp","config_prefill_ep","config_decode_tp","config_decode_ep","config_disagg","config_is_multinode"]

def _safe(v: Any) -> Any:
    v = v.item() if hasattr(v, "item") else v
    return None if isinstance(v, float) and math.isnan(v) else v

def _row_evidence(row: pd.Series) -> dict[str, Any]:
    return {"source_path": str(VIEW.relative_to(ROOT)), "row": {k: _safe(row[k]) for k in ["config_id","config_hardware",*FIXED,TARGET] if k in row}}

def _question(qid, category, difficulty, question, answer, answer_type, mode, evidence, derivation, **extra):
    return {"question_id": qid, "track": "inference_performance", "category": category,
      "difficulty": difficulty, "question": question, "gold_answer": str(answer), "answer_type": answer_type,
      "gold_numeric_value": extra.pop("numeric", None), "gold_numeric_tolerance": extra.pop("tolerance", None),
      "choices": extra.pop("choices", None), "required_reasoning": extra.pop("reasoning", ["retrieve"]),
      "required_sources": extra.pop("sources", ["InferenceX July 20 aggregate"]), "answerability_mode": mode,
      "evidence": evidence, "derivation": derivation, "source_snapshot": "InferenceX db-dump/2026-07-20",
      "artifact_version": VERSION, "notes": extra.pop("notes", ""), **extra}

def _pair(view: pd.DataFrame) -> pd.DataFrame:
    eligible = view[view.hardware_mapping_status.eq("accepted")].copy()
    eligible = eligible.sort_values([*FIXED, "config_hardware", "config_id"])
    for _, group in eligible.groupby(FIXED, dropna=False, sort=True):
        if group.config_hardware.nunique() >= 2 and group[TARGET].notna().all():
            return group.drop_duplicates("config_hardware").head(3)
    raise ValueError("No controlled hardware comparison found")

def build_questions() -> list[dict[str, Any]]:
    view = pd.read_csv(VIEW, low_memory=False)
    view = view[view[TARGET].notna()].copy()
    p = _pair(view); a,b = p.iloc[0],p.iloc[1]
    base = ", ".join(f"{c.replace('config_','')}={a[c]}" for c in FIXED[:6])
    av,bv=float(a[TARGET]),float(b[TARGET]); high,low=(a,b) if av>=bv else (b,a)
    pct=(float(high[TARGET])/float(low[TARGET])-1)*100
    rows=view.sort_values(["config_id", "isl","osl","conc"]).iloc[[0, 137]]
    q=[]
    for i,(_,row) in enumerate(rows.iterrows(),1):
      q.append(_question(f"{VERSION}_{i:03d}","direct_retrieval","easy",f"For the measured operating point with config_id {row.config_id}, hardware {row.config_hardware}, ISL {row.isl}, OSL {row.osl}, and concurrency {row.conc}, what throughput per GPU was reported?",f"{float(row[TARGET]):.6f} tokens/s/GPU","numeric","table_only",[_row_evidence(row)],["retrieve the exact aggregate row"],numeric=float(row[TARGET]),tolerance=1e-3,reasoning=["retrieve","read metric"]))
    q += [
      _question(f"{VERSION}_003","controlled_comparison","medium",f"Holding {base} fixed, which of {a.config_hardware} and {b.config_hardware} has higher measured throughput per GPU?",high.config_hardware,"categorical","multirow_reasoning",[_row_evidence(a),_row_evidence(b)],["verify fixed variables","compare throughput"],reasoning=["filter","compare"]),
      _question(f"{VERSION}_004","quantitative_multirow_reasoning","medium",f"For the same controlled pair ({a.config_hardware} versus {b.config_hardware}; {base}), by what percent is the higher measured throughput per GPU greater than the lower?",f"{pct:.3f}%","numeric","multirow_reasoning",[_row_evidence(a),_row_evidence(b)],["(higher/lower - 1) * 100"],numeric=pct,tolerance=0.05,reasoning=["filter","calculate","compare"]),
      _question(f"{VERSION}_005","configuration_tradeoff_reasoning","medium",f"Among these two measured candidates with {base}, select the deployment maximizing throughput per GPU: {a.config_hardware} ({av:.3f}) or {b.config_hardware} ({bv:.3f}) tokens/s/GPU.",high.config_hardware,"categorical","multirow_reasoning",[_row_evidence(a),_row_evidence(b)],["compare supplied measured candidates"],reasoning=["retrieve","optimize"]),
      _question(f"{VERSION}_006","planning_constrained_decision","hard",f"A demand of 10,000 output tokens/s must use one of two measured candidates with {base}: {a.config_hardware} ({av:.3f} tokens/s/GPU) or {b.config_hardware} ({bv:.3f} tokens/s/GPU). Assuming throughput scales linearly only for this bookkeeping calculation, which requires fewer GPUs: ceil(demand / measured throughput per GPU)?",high.config_hardware,"categorical","multirow_reasoning",[_row_evidence(a),_row_evidence(b)],["ceil(10000/a)","ceil(10000/b)","choose smaller"],reasoning=["calculate","constrained_select"],notes="This is a measured-candidate sizing calculation, not a scaling-law claim."),
      _question(f"{VERSION}_007","research_result_interpretation","hard","In the controlled July Epoch-resolved cohort, did adding Epoch capabilities to hardware identity materially improve ordinary interpolation?", "No; the difference was small and not material.","short_text","project_analysis",[{"source_path":str(VALIDATION.relative_to(ROOT)),"json_path":"results.A_identity_only.random_forest and B_identity_plus_capabilities.random_forest"}],["compare fixed-fold A and B","apply recorded interpretation"],sources=["controlled hardware validation"],reasoning=["interpret_experiment"]),
      _question(f"{VERSION}_008","research_result_interpretation","hard","What does near-equal capability-only and identity-only Random Forest accuracy support?", "Physical capability descriptors preserve much of the predictive information; it does not prove reconstruction of GPU identity.","short_text","project_analysis",[{"source_path":str(VALIDATION.relative_to(ROOT)),"json_path":"results.C_capabilities_only.random_forest"}],["compare A/C","respect claim boundary"],sources=["controlled hardware validation"],reasoning=["interpret_experiment","avoid_overclaim"]),
    ]
    t=json.loads(TRANSFER.read_text()); records=t["records"]
    by={r["held_out_sku"]:r for r in records}
    best=min(records,key=lambda r:r["metrics"]["capability_random_forest"]["mae"]); weak=max(records,key=lambda r:r["metrics"]["capability_random_forest"]["median_absolute_percentage_error"])
    for qid,rec,prompt,answer in [("009",best,"Which held-out SKU had the lowest capability-RF MAE?",best["held_out_sku"]),("010",weak,"Which held-out SKU had the highest capability-RF MdAPE?",weak["held_out_sku"])]:
      q.append(_question(f"{VERSION}_{qid}","generalization_transfer_reasoning","hard",prompt,answer,"categorical","project_analysis",[{"source_path":str(TRANSFER.relative_to(ROOT)),"held_out_hardware":answer}],["compare held-out metrics"],sources=["held-out hardware transfer"],reasoning=["retrieve","rank"] ))
    q += [
      _question(f"{VERSION}_011","support_uncertainty_reasoning","hard","Does the seven-SKU capability-distance versus MdAPE association establish a calibrated support threshold for future hardware?","No; it is descriptive across only seven SKUs and does not define a threshold.","short_text","project_analysis",[{"source_path":str(TRANSFER.relative_to(ROOT)),"field":"support_relationship"}],["interpret descriptive Spearman evidence"],sources=["held-out hardware transfer"],reasoning=["interpret_statistics","avoid_overclaim"]),
      _question(f"{VERSION}_012","planning_constrained_decision","hard","A planner wants a hardware-transfer case with an interpolation-like descriptor profile and the lower of the two such MdAPEs. Which SKU should it choose: GB200 or MI325X?","MI325X","categorical","project_analysis",[{"source_path":str(TRANSFER.relative_to(ROOT)),"held_out_sku":"GB200"},{"source_path":str(TRANSFER.relative_to(ROOT)),"held_out_sku":"MI325X"}],["identify interpolation-like records","compare MdAPE"],sources=["held-out hardware transfer"],reasoning=["filter","compare","constrained_select"]),
    ]
    return q

def score(question: dict[str,Any], response: str) -> float:
    answer=str(response).strip().casefold(); gold=str(question["gold_answer"]).strip().casefold()
    if question["answer_type"] == "numeric":
        import re
        found=re.search(r"[-+]?\d*\.?\d+",answer)
        return float(found is not None and abs(float(found.group())-float(question["gold_numeric_value"]))<=float(question["gold_numeric_tolerance"]))
    return float(answer == gold)
