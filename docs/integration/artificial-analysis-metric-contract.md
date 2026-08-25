# Artificial Analysis metric contract

| AA field | AA unit/entity | InferenceX relation | Classification | Caveat |
|---|---|---|---|---|
| `performance.median_output_tokens_per_second` | tokens/s, not per GPU | `metrics_tput_per_gpu` | NOT_DIRECTLY_COMPARABLE | Hardware, workload, concurrency, aggregation, and denominator differ; provider context is absent. Never overwrite or alias. |
| `performance.median_time_to_first_token_seconds` | seconds | `metrics_median_ttft` | NOT_DIRECTLY_COMPARABLE | No matched request, server, provider, or measurement protocol. |
| `performance.median_time_to_first_answer_token_seconds` | seconds | `metrics_median_ttft` | NOT_DIRECTLY_COMPARABLE | AA answer-token definition and provider context are source-specific. |
| `performance.median_end_to_end_response_time_seconds` | seconds | `metrics_median_e2el` | NOT_DIRECTLY_COMPARABLE | AA aggregate and absent provider context must remain source-specific. |
| `evaluations.artificial_analysis_*_index` | index points | `none` | NOT_DIRECTLY_COMPARABLE | Capability outcome, not physical serving descriptor. |
| `pricing.price_1m_*_tokens and intelligence_index_cost` | price per 1M tokens; currency not supplied | `none` | NOT_DIRECTLY_COMPARABLE | Business metadata can change and is not a measured InferenceX metric. |

`aa_median_output_tokens_per_second` never overwrites or aliases `metrics_tput_per_gpu`; AA latency never overwrites or aliases deployment latency.
