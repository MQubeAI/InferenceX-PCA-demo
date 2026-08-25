# Model specification source gap

The current sources are partially sufficient for identity and a sparse total-parameter field, but insufficient for scientific unseen-model throughput transfer. Missing structured exact-version descriptors are active parameters/token, dense/MoE type, expert and active-expert counts, layers, hidden size, attention/KV-head configuration, context length, vocabulary, attention architecture, precision, and KV-cache-relevant characteristics.

Recommended next source class: immutable exact-version copies of official Hugging Face `config.json`, official model cards, vendor technical reports, or official architecture repositories. Each ingestion should store the source URL/identifier, retrieval timestamp, release/as-of provenance, raw immutable snapshot hash, normalized fields with units, and an explicit canonical-model mapping. Do not scrape uncontrolled web pages.
