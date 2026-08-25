# Three-source model-side integration summary

InferenceX provides measured physical deployment operating points and `metrics_tput_per_gpu`. Epoch hardware provides physical accelerator descriptors. Epoch model data provides explicit model identity and a sparse static descriptor set, principally total parameters. Artificial Analysis Free provides 200 current model records with identity, release date, capability indices, API/model speed and latency observations, and pricing; the payload does not declare a provider identity or aggregation context. It provides no architecture descriptors.

The verified snapshot is `data/external/artificial_analysis/raw/artificial_analysis_language_models_free_20260825T030353569731Z.json` (200 records, SHA-256 `6f588e1f8bd7dad6074544e5652e821bc5944170e647f6f92c914fc03cdc0f49`). Exact three-source mappings are MiniMax-M2.5, MiniMax-M3. The three-source overlap is 2406 rows / 339 configs. Other labels are unresolved because of version/family ambiguity or have no exact AA candidate; details are in `three_source_model_overlap.md`.

AA output speed is not per-GPU throughput, AA latency is not InferenceX deployment latency, capability indices are outcomes, and pricing is business metadata. None is eligible as a primary physical throughput feature. A real worked example is stored in `three_source_worked_example.json`; it combines sources without conflating their measurements.

The transfer decision is **B. PARTIALLY SUFFICIENT — NEED ADDITIONAL MODEL-SPEC SOURCE FIRST**. 6/11 models have a structured total-parameter value, while all MoE, active-parameter, layer, hidden-size, attention, and KV-cache descriptors are absent as structured fields. No leave-one-model-out pilot was run.

Next source: an immutable, exact-version model-spec ingestion from official Hugging Face `config.json`, official model cards, or vendor technical reports with release/as-of provenance and manual canonical mapping.
