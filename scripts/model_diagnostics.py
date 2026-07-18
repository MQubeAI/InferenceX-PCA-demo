"""Bounded aggregate diagnostics for target, transform, context, and split selection."""
from __future__ import annotations
import argparse, json, sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from apps import inferencex_pca_demo as app
from modeling.comparison import evaluate_models, prepare_model_frame
from modeling.diagnostics import known_config_folds

TARGETS=("metrics_p99_itl","metrics_median_itl","metrics_mean_itl","metrics_median_ttft","metrics_median_tpot","metrics_mean_e2el","metrics_tput_per_gpu")

def artifact_seed_is_compatible(artifact: dict, seed: int) -> bool:
 """Reject --resume when it would mix experiments from different samples."""
 prior = artifact.get("controls", {}).get("seed")
 return prior is None or int(prior) == int(seed)

def main() -> None:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--data-dir",default=app.DEFAULT_DATA_DIR); p.add_argument("--max-rows",type=int,default=1024); p.add_argument("--folds",type=int,default=3); p.add_argument("--seed",type=int,default=42)
 p.add_argument("--targets",default=",".join(TARGETS)); p.add_argument("--models",default="random_forest,catboost"); p.add_argument("--tabfm-context-strategies",default="random"); p.add_argument("--tabfm-context-sizes",default="512"); p.add_argument("--include-log1p",action="store_true"); p.add_argument("--known-config",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--refresh",action="store_true",help="Recompute requested experiment keys while preserving other artifact entries."); p.add_argument("--output",required=True)
 a=p.parse_args(); out=Path(a.output); existing=json.loads(out.read_text()) if a.resume and out.exists() else {"experiments":{}}
 if not artifact_seed_is_compatible(existing, a.seed): raise ValueError(f"Refusing to resume {out}: artifact seed differs from requested seed {a.seed}.")
 source=app.data_source_status(a.data_dir)[1]; manifest=app.build_dataset_manifest(source); _,_,joined,_=app.load_joined_data(a.data_dir,manifest["fingerprint"]); frame,meta=app.build_analysis_frame(joined,"Median aggregate per config/workload/concurrency"); meta["dataset_fingerprint"]=manifest["fingerprint"]
 models=[x for x in a.models.split(",") if x]; strategies=[x for x in a.tabfm_context_strategies.split(",") if x]; sizes=[int(x) for x in a.tabfm_context_sizes.split(",") if x]
 for target in [x for x in a.targets.split(",") if x]:
  if target not in frame or frame[target].notna().sum()<20:
   existing.setdefault("skipped",{})[target]="unavailable or fewer than 20 usable rows"; continue
  numeric,categorical=app.default_target_features(frame,target); features=numeric+categorical
  transforms=["raw"] + (["log1p"] if (frame[target].dropna()>=0).all() and a.include_log1p else [])
  for transform in transforms:
   key=f"unseen:{target}:{transform}:baselines"
   baseline_models=[m for m in models if m != "tabfm"]
   if baseline_models:
    if a.refresh: existing["experiments"].pop(key,None)
    if key not in existing["experiments"]:
     existing["experiments"][key]=evaluate_models(frame,features,target,baseline_models,a.max_rows,a.seed,a.folds,target_transform=transform)
   if "tabfm" in models:
    for strategy in strategies:
     for size in sizes:
      key=f"unseen:{target}:{transform}:tabfm:{strategy}:{size}"
      if a.refresh: existing["experiments"].pop(key,None)
      if key not in existing["experiments"]:
       existing["experiments"][key]=evaluate_models(frame,features,target,["tabfm"],a.max_rows,a.seed,a.folds,size,transform,strategy)
  if a.known_config:
   work,_=prepare_model_frame(frame,features,target,a.max_rows,a.seed); folds=known_config_folds(work,a.folds,a.seed)
   key=f"known_config:{target}:raw:baselines"
   if a.refresh: existing["experiments"].pop(key,None)
   if key not in existing["experiments"]:
    existing["experiments"][key]=evaluate_models(frame,features,target,[m for m in models if m != "tabfm"],a.max_rows,a.seed,a.folds,folds_override=folds,evaluation_label="known_config_interpolation")
 existing.update({"generation_timestamp_utc":datetime.now(UTC).isoformat(),"dataset":manifest,"controls":{"max_rows":a.max_rows,"folds":a.folds,"seed":a.seed,"models":models,"targets":a.targets,"strategies":strategies,"sizes":sizes},"aggregate_only":True})
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(existing,indent=2,sort_keys=True,default=str)+"\n")
 print(f"Wrote {out} with {len(existing['experiments'])} aggregate experiments")
if __name__=="__main__": main()
