# Entity mapping decisions

All mapping decisions are explicit in the derived CSVs. A non-empty candidate
name is not an accepted join: only `mapping_status=accepted` is used to enrich
the throughput view.

## Hardware

Accepted mappings are `b200 → NVIDIA B200`, `b300 → NVIDIA B300 (Blackwell
Ultra)`, `gb200 → NVIDIA GB200`, `gb300 → NVIDIA GB300 (Blackwell Ultra)`,
`mi300x → AMD Instinct MI300X`, `mi325x → AMD Instinct MI325X`, and `mi355x →
AMD Instinct MI355X`. Epoch notes indicate that the B/GB performance figures
are individual-GPU descriptors, avoiding an accidental system-level capability
interpretation.

`h100` remains unresolved among Epoch `NVIDIA H100 NVL`, `NVIDIA H100 PCIe`,
and `NVIDIA H100 SXM5 80GB`. `h200` remains unresolved despite one current
Epoch `NVIDIA H200 SXM` candidate because InferenceX’s bare label does not
prove the SXM form factor. Neither receives a canonical hardware ID or an
Epoch capability join.

See `data/derived/integration/mappings/hardware_mapping.csv` for confidence,
evidence, source snapshot, and notes for every InferenceX label. The mapping
validator rejects duplicate InferenceX decisions, a mapping with more than one
accepted canonical hardware entity, and any accepted ambiguous decision.

## Models

Accepted exact version mappings are `glm5`, `glm5.1`, `glm5.2`, `gptoss120b`,
`kimik2.5`, `minimaxm2.5`, and `minimaxm3`. The following mappings remain
unresolved and are not silently attached to Epoch model metadata or benchmarks:

- `dsr1`: Epoch contains both `DeepSeek-R1` and `DeepSeek-R1 (May 2025)`.
- `dsv4`: Pro, Flash, and dated V4 variants exist.
- `llama70b`: Llama 2/3/3.1/3.3 and fine-tuned 70B variants exist.
- `qwen3.5`: dense, MoE, hosted, Omni, and multiple size variants exist.

The full candidate lists and evidence are in
`data/derived/integration/mappings/model_mapping.csv`. A resolved model ID is
currently provenance for future model-side integration, not a throughput
feature.
