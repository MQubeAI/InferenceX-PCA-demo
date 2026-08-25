# Artificial Analysis Free endpoint audit

This audit reads only the verified 2026-08-25 frozen Free-endpoint snapshot. It does not call the API. The response is a current model-record catalog, not an InferenceX deployment table.

| Field | Type | Missing | Class | Throughput-prediction treatment |
|---|---|---:|---|---|
| `data[].artificial_analysis_intelligence_index_cost` | dict | 155 (77.5%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].artificial_analysis_intelligence_index_cost.cost_per_task` | dict | 155 (77.5%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].artificial_analysis_intelligence_index_cost.cost_per_task.total_cost` | float, int | 155 (77.5%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].artificial_analysis_intelligence_index_cost.total_cost` | float, int | 155 (77.5%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].evaluations` | dict | 0 (0.0%) | C. benchmark/capability outcome | leakage-prone; excluded |
| `data[].evaluations.artificial_analysis_agentic_index` | float, int | 149 (74.5%) | C. benchmark/capability outcome | leakage-prone; excluded |
| `data[].evaluations.artificial_analysis_coding_index` | float, int | 130 (65.0%) | C. benchmark/capability outcome | leakage-prone; excluded |
| `data[].evaluations.artificial_analysis_intelligence_index` | float, int | 3 (1.5%) | C. benchmark/capability outcome | leakage-prone; excluded |
| `data[].id` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].model_creator` | dict | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].model_creator.id` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].model_creator.name` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].name` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].performance` | dict | 0 (0.0%) | D. API/provider serving-performance observation | semantically incompatible; excluded |
| `data[].performance.median_end_to_end_response_time_seconds` | float, int | 95 (47.5%) | D. API/provider serving-performance observation | semantically incompatible; excluded |
| `data[].performance.median_output_tokens_per_second` | float, int | 95 (47.5%) | D. API/provider serving-performance observation | semantically incompatible; excluded |
| `data[].performance.median_time_to_first_answer_token_seconds` | float, int | 95 (47.5%) | D. API/provider serving-performance observation | semantically incompatible; excluded |
| `data[].performance.median_time_to_first_token_seconds` | float, int | 95 (47.5%) | D. API/provider serving-performance observation | semantically incompatible; excluded |
| `data[].pricing` | dict | 0 (0.0%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].pricing.price_1m_cache_hit_tokens` | float, int | 137 (68.5%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].pricing.price_1m_cache_write_tokens` | float | 182 (91.0%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].pricing.price_1m_input_tokens` | float, int | 64 (32.0%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].pricing.price_1m_output_tokens` | float, int | 64 (32.0%) | E. pricing/business metadata | leakage-prone; excluded |
| `data[].release_date` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |
| `data[].slug` | str | 0 (0.0%) | A. identity/provenance | contextual/display only |

AA contains no physical architecture descriptor fields needed for unseen-model transfer. Provider speed, latency, capability indices, and prices stay source-specific.
