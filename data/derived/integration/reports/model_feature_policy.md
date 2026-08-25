# Model feature policy

Only pre-deployment, model-intrinsic static descriptors may be considered for a future physical throughput representation. The currently normalized candidate is Epoch `Parameters`; it is insufficient on its own.

| Source | Field | Category | Primary predictor treatment |
|---|---|---|---|
| Epoch all_ai_models.csv | `Model` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Domain` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Task` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Organization` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Authors` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Publication date` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Reference` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Link` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Citations` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Notability criteria` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Notability criteria notes` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Parameters` | A. STATIC / PRE-DEPLOYMENT MODEL DESCRIPTOR | conditional: only normalized, exact-version static values; primary currently only Parameters |
| Epoch all_ai_models.csv | `Parameters notes` | A. STATIC / PRE-DEPLOYMENT MODEL DESCRIPTOR | conditional: only normalized, exact-version static values; primary currently only Parameters |
| Epoch all_ai_models.csv | `Training compute (FLOP)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training compute notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training dataset size (total)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Dataset size notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training time (hours)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training time notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training hardware` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Approach` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Confidence` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Abstract` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Epochs` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `WikiText and Penn Treebank data` | B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME | contextual/display only |
| Epoch all_ai_models.csv | `Model accessibility` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Country (of organization)` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Base model` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Finetune compute (FLOP)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Finetune compute notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Hardware quantity` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Hardware utilization (MFU)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Last modified` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Training cloud compute vendor` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training data center` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Archived links` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Batch size` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Batch size notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Organization categorization` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Foundation model` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Training compute lower bound` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training compute upper bound` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training chip-hours` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training code accessibility` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Accessibility notes` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Possibly over 1e23 FLOP` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Training compute cost (2023 USD)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Utilization notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Numerical format` | A. STATIC / PRE-DEPLOYMENT MODEL DESCRIPTOR | conditional: only normalized, exact-version static values; primary currently only Parameters |
| Epoch all_ai_models.csv | `Frontier model` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Training power draw (W)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Training compute estimation method` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Hugging Face developer id` | E. IDENTITY / PROVENANCE | contextual/display only |
| Epoch all_ai_models.csv | `Post-training compute (FLOP)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Post-training compute notes` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Hardware utilization (HFU)` | E. IDENTITY / PROVENANCE | excluded: training/development metadata is not a serving-throughput descriptor |
| Epoch all_ai_models.csv | `Open model weights?` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_model_id` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_model_name` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_slug` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_model_creator_id` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_model_creator_name` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_release_date` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_artificial_analysis_intelligence_index` | B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_artificial_analysis_coding_index` | B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_artificial_analysis_agentic_index` | B. POST-RELEASE BENCHMARK / CAPABILITY OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_median_output_tokens_per_second` | C. API / PROVIDER PERFORMANCE OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_median_time_to_first_token_seconds` | C. API / PROVIDER PERFORMANCE OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_median_time_to_first_answer_token_seconds` | C. API / PROVIDER PERFORMANCE OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_median_end_to_end_response_time_seconds` | C. API / PROVIDER PERFORMANCE OUTCOME | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_price_1m_cache_hit_tokens` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_price_1m_cache_write_tokens` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_price_1m_input_tokens` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_price_1m_output_tokens` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_intelligence_index_cost_total_cost` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_intelligence_index_cost_per_task_total_cost` | D. PRICING / BUSINESS METADATA | excluded from primary physical-throughput representation |
| Artificial Analysis Free endpoint | `aa_source` | E. IDENTITY / PROVENANCE | contextual/display only |
| Artificial Analysis Free endpoint | `aa_source_snapshot` | E. IDENTITY / PROVENANCE | contextual/display only |

All AA capability indices, API performance, latency, pricing, and cost fields are excluded.
