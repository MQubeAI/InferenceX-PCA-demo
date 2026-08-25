# Canonical entity and observation schema

`CanonicalHardware` is one resolved Epoch accelerator record, retaining vendor, type, release/provenance, memory capacity, memory bandwidth, FP16/BF16/FP8/FP4/INT8/INT4 peaks, TDP, and valid interconnect descriptors. These are descriptors, not measured InferenceX outcomes. `DeploymentOperatingPoint` remains the established `config_id × benchmark_type × isl × osl × conc` InferenceX aggregate with original hardware/model identity, workload, configuration, and measured outcome fields.

`CanonicalModel` represents a verified exact model/version only. `EpochModelMetadata` retains Epoch identity, static descriptors, training/development metadata, and provenance. `ArtificialAnalysisModelObservation` retains AA identity, release, capability-index, and pricing observations. `ArtificialAnalysisProviderObservation` retains AA API/model speed and latency observations and explicitly records that the current Free payload supplies no provider identity or aggregation context. These classes are not flattened into universal measurements.

```text
DeploymentOperatingPoint              CanonicalHardware
        |                                      |
   CanonicalModel ---- EpochModelMetadata   Epoch physical capabilities
       /      InferenceX    ArtificialAnalysisModelObservation
                 + ArtificialAnalysisProviderObservation
                         |
               task-specific modeling views
                         |
          observed / predicted / unsupported

time-varying token demand -> required GPU count -> GPU cluster capacity
-> datacenter capacity -> power / cost
```

Cluster and datacenter information remains downstream and is never a per-GPU throughput predictor. AA API measurements and pricing retain the `aa_` prefix in the model-enriched view and cannot replace InferenceX fields.
