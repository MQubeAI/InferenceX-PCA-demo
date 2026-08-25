import argparse,json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.dcbench import score
p=argparse.ArgumentParser(); p.add_argument("responses"); p.add_argument("--questions",default="data/derived/dcbench/questions/inference_v0_1.jsonl"); a=p.parse_args()
qs={q["question_id"]:q for q in map(json.loads,open(a.questions))}; rs=[json.loads(x) for x in open(a.responses)]
out=[]
for r in rs:
 q=qs[r["question_id"]]; s=score(q,str(r["response"])); out.append({**r,"parsed_answer":str(r["response"]),"correct":bool(s),"score":s,"error_type":None if s else "incorrect_answer"})
print(json.dumps({"scored":len(out),"mean_score":sum(x["score"] for x in out)/len(out),"results":out},indent=2))
