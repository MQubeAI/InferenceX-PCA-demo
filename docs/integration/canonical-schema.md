# Canonical entity and observation schema

This integration uses canonical shared entities plus source-specific
observations. It does not create a universal all-source table.

## CanonicalHardware

`data/derived/integration/views/canonical_hardware_capabilities.csv` has one
row per Epoch `Hardware name` and uses IDs of the form
`epoch_hardware:<normalized-name>`. It preserves the Epoch hardware name,
vendor, type, release date, last-modified timestamp, source snapshot,
datasheet provenance, and notes.

| Canonical field | Source field | Unit |
|---|---|---|
| `epoch_memory_capacity_bytes` | `Memory (bytes)` | bytes |
| `epoch_memory_bandwidth_bytes_per_second` | `Memory bandwidth (byte/s)` | bytes/s |
| `epoch_tensor_fp16_bf16_peak_flops` | `Tensor-FP16/BF16 performance (FLOP/s)` | FLOP/s |
| `epoch_fp8_peak_flops`, `epoch_fp4_peak_flops` | Epoch FP8/FP4 fields | FLOP/s |
| `epoch_int8_peak_ops`, `epoch_int4_peak_ops` | Epoch INT8/INT4 fields | OP/s |
| `epoch_tdp_watts` | `TDP (W)` | W |
| `epoch_intranode_bandwidth_bytes_per_second` | `Intranode bandwidth (byte/s)` | bytes/s |
| `epoch_internode_bandwidth_bits_per_second` | `Internode bandwidth (bit/s)` | bits/s |

`Max performance`, `Energy efficiency`, total processing performance, and
price-performance remain source fields but are not current primary model
features. Epoch energy efficiency is not InferenceX joules/output-token.

## CanonicalModel

A CanonicalModel identifier has the form `epoch_model:<normalized-version>`.
Its conceptual fields are family, exact version, organization, release date,
total parameters, optional active parameters per token, architecture type,
base model, source identifiers, source snapshot, and provenance. The available
Epoch table currently supports exact model/version, organization, publication
date, total parameters, base model, Hugging Face developer ID, training fields,
and links. It does not reliably provide active parameters or architecture type
for all models, so those values are not invented.

`model_mapping.csv` is an entity-resolution decision table, not a request to
inject model training variables into throughput prediction.

## DeploymentOperatingPoint

An InferenceX deployment operating point is retained at its established median
aggregate grain:

`config_id × benchmark_type × isl × osl × conc`.

It contains canonical model/hardware IDs only when mappings are accepted;
InferenceX model/hardware identity, framework, precision, speculative method,
disaggregation, multinode status, prefill/decode TP/EP/DP-attention/worker
allocation, GPU allocation, workload fields, measured outcomes, representative
date, source snapshot, and row-count provenance are retained. Its throughput
outcome is `metrics_tput_per_gpu` in tokens/s/GPU. It is not a hardware-spec
row, benchmark-score row, cluster row, or datacenter row.

## BenchmarkObservation

Epoch benchmark observations are long-form and retain at minimum:

`source, source_snapshot, source_file, benchmark_name, model_version,
canonical_model_id (only if resolved), metric_name, metric_value, metric_unit,
evaluation_date, provider/harness/prompt context, score uncertainty, raw
source identifier, provenance`.

The extracted CSVs are not normalized into a lossy universal wide schema.

## GPUCluster and DataCenter

GPUCluster records hold cluster identity, owner/users, hardware composition,
chip count, operational/decommissioning dates, power, cost, location, sources,
and uncertainty. DataCenter records hold site identity, owner/users, current
chip types/H100 equivalents, power, capital cost, location, construction and
energy parties, and sources. These are separate downstream objects. A future
capacity model can consume a predicted workload/resource requirement and then
relate it to these entities; they never directly enrich an InferenceX row based
only on accelerator name.
