# Model descriptor coverage

| Descriptor | Actual combined coverage | Primary-feature status | Notes |
|---|---:|---|---|
| total_parameters | 6/11 | conditional | Direct numeric field; six accepted Epoch mappings are non-null. |
| active_parameters_per_token | 0/11 | not eligible | Only unstructured notes for GLM-5.2 and MiniMax-M2.5; not normalized. |
| dense_vs_moe | 0/11 | not eligible | No structured source field. |
| number_of_experts | 0/11 | not eligible | No structured source field. |
| active_experts_per_token | 0/11 | not eligible | No structured source field. |
| architecture_family | 0/11 | not eligible | No structured source field. |
| layer_count | 0/11 | not eligible | No structured source field. |
| hidden_size | 0/11 | not eligible | No structured source field. |
| attention_head_count | 0/11 | not eligible | No structured source field. |
| kv_head_count | 0/11 | not eligible | No structured source field. |
| context_length | 0/11 | not eligible | No structured source field; MiniMax-M3 abstract mentions 1M context but is unstructured. |
| vocabulary_size | 0/11 | not eligible | No structured source field. |
| attention_architecture | 0/11 | not eligible | No structured source field; MiniMax-M3 abstract mentions MSA but is unstructured. |
| parameter_precision | 0/11 | not eligible | Column exists but has no accepted-model value. |
| kv_cache_relevant_characteristics | 0/11 | not eligible | No structured source field. |

Only total parameter count has a structured value for any accepted model (6/11); that is not an adequate unseen-model physical representation.
