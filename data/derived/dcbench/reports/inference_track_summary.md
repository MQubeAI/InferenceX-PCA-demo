# DC Bench inference track summary

The seed track turns verified InferenceX observations and Epoch validation into
12 independently scoreable reasoning tasks. It differs from querying
InferenceX because it tests controlled filtering, arithmetic, bounded selection,
and interpretation of project-only transfer evidence. Gold answers are built
deterministically from the frozen derived view and validation JSON.

The strongest examples are controlled hardware comparison, demand-to-GPU
bookkeeping over measured candidates, and the support-boundary question about
the seven-SKU Spearman result. Epoch enables identity-versus-physical-description
and held-out-SKU questions. It cannot yet test provider/model capability,
facility power/capacity, temporal demand, or arbitrary prediction. Artificial
Analysis can later add source-specific model/provider tasks; cluster/datacenter
data can add capacity/power planning; synchronized demand can add dynamic tasks.

Progression: heterogeneous InferenceX observations → structural analysis →
within-system TabFM signal → Epoch physical descriptors → capability-only
ordinary accuracy → held-out measured-SKU evidence → verified DC Bench tasks.
