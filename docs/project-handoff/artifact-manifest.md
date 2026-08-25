# Artifact manifest

| Path | Purpose | Source/derived | Reproducible | Git? | Notes |
|---|---|---|---|---|---|
| `.data/inferencex-db-dump-2026-07-20/` | active InferenceX source | source | bootstrap | no | local snapshot |
| `data/external/epoch/raw/` | Epoch source CSV/ZIP | source immutable | download | no | raw/large |
| `data/external/epoch/extracted/` | benchmark extraction | source derivative | unzip | no | raw-derived/large |
| `data/derived/integration/mappings/` | auditable entities | derived | build script | yes | compact |
| `data/derived/integration/views/canonical_hardware_capabilities.csv` | Epoch descriptors | derived | build script | no | local regenerated view; not required by clean CI |
| `data/derived/integration/views/inferencex_throughput_epoch_hardware_v1.csv` | enriched rows | derived | build script | no | 14 MB; rebuild locally |
| `data/derived/integration/reports/` | validation/conclusions | derived | validation build | yes | compact JSON/MD |
| `artifacts/`, `reports/` | structural/throughput conclusions | derived | documented scripts | existing tracked contract | canonical research outputs |
| `data/derived/dcbench/source_slices/inference_v0_1_v0_2_source_slice.csv` | exact 23-row operating-point evidence subset for v0.1/v0.2 | derived from validated July enriched view | source-slice script when full view is local | yes | immutable committed slice; not synthetic and contains no gold-answer fields |
| `data/derived/dcbench/source_slices/manifest.json` | slice provenance, schema, and SHA-256 | derived | source-slice script | yes | records excluded full-view hash and selection contract |
| `data/derived/dcbench/` | questions, inputs, templates, equivalence report | derived | DC builders | yes | compact clean-CI reproducibility record |
| `paper/sections/` | manuscript text | source | n/a | yes | figures separately ignored |
| `apps/inferencex_pca_demo.py` | dashboard/prototype | source | n/a | yes | app integration view |
| `modeling/integration.py`, `hardware_validation.py`, `dcbench.py` | reusable pipeline | source | n/a | yes | core code |
| relevant `tests/` | invariants | source | n/a | yes | integration/validation/DC Bench |

Artificial Analysis is only an empty future boundary. GPU clusters/datacenters
remain source-layer inputs for future capacity work, never throughput-row features.
