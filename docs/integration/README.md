# Multi-Source Inference Integration

## Objective

Integrate SemiAnalysis InferenceX with Epoch AI and Artificial Analysis while preserving the distinct semantics and units of observation of each source.

The integration should use canonical shared entities and task-specific modeling views rather than concatenating all sources into one universal table.

## Canonical entities

- Model
- Hardware
- Workload
- Serving Configuration
- Deployment Operating Point
- Benchmark Observation
- GPU Cluster
- Data Center
- Data Source / Provenance

## First integration experiment

Compare:

1. Existing categorical-hardware predictor
2. Hardware identity plus Epoch capability descriptors
3. Epoch capability descriptors with hardware identity removed
4. Leave-one-hardware-SKU-out prediction

## Epoch ML Hardware

Candidate descriptors:

- memory capacity
- memory bandwidth
- BF16 / FP16 compute
- FP8 compute
- FP4 compute
- INT8 compute
- INT4 compute
- TDP
- intranode bandwidth
- internode bandwidth

## Epoch AI Models

Use for model-side enrichment after canonical model resolution.

Do not silently map broad model-family names onto specific model variants.

## Epoch Benchmark Data

Represent benchmark results as source-specific observations rather than forcing heterogeneous benchmark schemas into one wide table.

## Epoch GPU Clusters

Treat as infrastructure-level information.

Do not join cluster records directly to InferenceX rows solely through accelerator identity.

## Epoch Data Centers

Treat as facility-level information.

Do not use datacenter power, cost, or installed capacity as direct per-configuration throughput predictors without a valid relationship.

## Artificial Analysis

Use primarily for model/provider enrichment.

Artificial Analysis output-speed and latency measurements must remain distinct from InferenceX measurements unless explicit harmonization establishes comparability.
