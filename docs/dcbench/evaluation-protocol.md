# Evaluation protocol

Run each model in a fresh session with identical wording and one context mode:
closed-book, exact evidence supplied, or future repository/tool access. Preserve
raw output; score with the same local evaluator at low/deterministic temperature
where possible. Planned comparison: GPT, Claude, Qwen, and Kimi. No result is
claimed until response JSONL exists.

Errors: retrieval failure, wrong filter, uncontrolled comparison, arithmetic or
ranking error, metric-semantic confusion, hallucinated measurement, unsupported
extrapolation, causal overclaim, failure to abstain, and project-result
misunderstanding. Numeric answers use item tolerances; categorical/short text
are canonical exact matches in v0.1.
