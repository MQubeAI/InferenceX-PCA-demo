# DC Bench — inference/performance seed track

`inference_v0_1` is a small, auditable seed set for testing inference and
datacenter performance reasoning. It uses frozen July 20 InferenceX aggregate
observations plus completed Epoch validation artifacts. It does not contain
arbitrary prediction requests, Artificial Analysis data, facility planning, or
unresolved H100/H200 physical specifications.

Build with `python scripts/build_dcbench_inference_questions.py`; score a JSONL
response file with `python scripts/evaluate_dcbench_inference.py responses.jsonl`.
