# Artificial Analysis integration boundary

Artificial Analysis data is intentionally absent from this prototype. Raw
exports belong in `data/external/artificial_analysis/raw/`; no external API was
called and no values were fabricated.

When an approved export is available, the safe initial contract filename is
`artificial_analysis_observations.csv`. Required fields are:

`canonical_model_id, source_model_id, model_name, provider, source_timestamp`.

Optional source-specific fields are `capability_index`,
`context_window_tokens`, `parameter_count`, `median_output_speed`,
`median_output_speed_unit`, `ttft`, `ttft_unit`, `end_to_end_latency`,
`end_to_end_latency_unit`, input/output token prices, and currency. The loader
is non-fatal when the directory/file is absent and rejects an export missing the
required provenance fields.

The dashboard will show **Artificial Analysis data not loaded** when absent.
When a contract-compliant file is supplied, it can display rows only for an
accepted canonical model ID. It preserves provider and source timestamp.

AA output speed is not InferenceX throughput; AA TTFT/end-to-end latency are
not automatically comparable to InferenceX TTFT/e2el. They remain separate
BenchmarkObservation-style source records unless a future protocol establishes
measurement comparability. No AA performance metric is used as an InferenceX
throughput feature or target.
