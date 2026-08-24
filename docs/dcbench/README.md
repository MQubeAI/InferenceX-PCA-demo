# DC Bench — inference/performance seed track

`inference_v0_1` is a small, auditable seed set for testing inference and
datacenter performance reasoning. It uses frozen July 20 InferenceX aggregate
observations plus completed Epoch validation artifacts. It does not contain
arbitrary prediction requests, Artificial Analysis data, facility planning, or
unresolved H100/H200 physical specifications.

Clean CI rebuilds the benchmark from the tracked immutable 23-row operating
point slice at `data/derived/dcbench/source_slices/`, whose manifest records
the SHA-256 and the excluded full-view provenance. This is an exact subset of
the validated July enriched view, not a synthetic fixture and not a gold-answer
table. The approximately 14 MB full enriched view remains a local integration
artifact; use it only to regenerate the source slice or run an explicit
`--source full` provenance check.

Build with `python scripts/build_dcbench_inference_questions.py`; rebuild the
slice from a local full view with `python scripts/build_dcbench_source_slice.py`;
and score a JSONL response file with
`python scripts/evaluate_dcbench_inference.py responses.jsonl`.
