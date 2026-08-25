# Three-source model overlap

- InferenceX labels: 11
- Accepted Epoch mappings: 7
- Accepted AA mappings: 2
- Accepted mappings to both: 2
- AA-represented InferenceX: 2406 rows / 339 configs
- Three-source: 2406 rows / 339 configs

## Unresolved or absent

- `dsr1` — InferenceX shorthand lacks release/version; AA additionally contains base and distill variants.
- `dsv4` — InferenceX V4 shorthand cannot select an Epoch Pro/Flash/dated variant; no AA V4 record.
- `glm5` — Epoch exact match. AA distinguishes a reasoning offering and Turbo offering, while InferenceX does not.
- `glm5.1` — Exact Epoch match; no GLM-5.1 AA record in the frozen Free response.
- `glm5.2` — Exact Epoch match; no GLM-5.2 AA record in the frozen Free response.
- `gptoss120b` — Exact Epoch match; no exact gpt-oss-120b AA record in the frozen Free response.
- `kimik2.5` — Exact Epoch match; AA has K2.6 offerings, not K2.5, so there is no exact AA candidate.
- `llama70b` — Family/size shorthand does not identify Llama release or fine-tune; AA candidates are derivative offerings.
- `qwen3.5` — InferenceX family label omits size, dense/MoE choice, mode, and Omni variant.
