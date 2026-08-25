# Inference research handoff

Inference performance spans hardware, model, workload, framework, precision,
parallelism, serving architecture, and allocation; lookup cannot cover the
combinatorial design space. InferenceX supplies measured deployment operating
points, joined by `config_id` and median-aggregated by config/workload/concurrency.
Structural work finds meaningful configuration structure: PCA, deterministic AE,
cross-method clustering, UMAP/t-SNE local structure, and workload/configuration
ablations corroborate it; no simple throughput, TPOT, or energy gradient explains
the map. Group-held UMAP degrades, reinforcing its descriptive—not predictive—role.

Historical grouped TabFM on a deterministic 4,096-row compute-budget sample
reproduced R² 0.961979 ± 0.008605 and MAE 338.540 tokens/s/GPU. It supports
within-measured-system interpolation, not future hardware transfer.

Multi-source design uses canonical entities, source-specific observations, and
task-specific views—not one giant table. InferenceX is measured behavior; Epoch
is hardware/model/benchmark/infrastructure metadata; AA is future provider/model
observations; clusters/datacenters are downstream capacity layers.

Epoch maps B200/B300/GB200/GB300/MI300X/MI325X/MI355X; H100/H200 variants remain
unresolved. The July aggregate has 8,239 rows; the accepted descriptor cohort
has 7,168 rows/1,243 configs with row-count preservation. A/B/C/D RF R²/MAE:
0.8377/784.23, 0.8382/767.36, 0.8375/768.20, 0.8349/780.86. Outcome B: capabilities
do not materially improve interpolation but capability-only preserves accuracy.

Leave-one-SKU-out capability RF beats global/workload/analogue/ridge MAE for all
seven SKUs. Transfer is uneven: MI300X and MI355X have weak relative error;
distance–MdAPE Spearman about 0.82 across seven points is descriptive only.
Supported: transferable physical information and measured-SKU transfer. Unsupported:
unreleased GPUs, H100/H200 transfer, arbitrary what-if, causal claims, or calibrated thresholds.

DC Bench now turns these frozen observations into verified retrieval, comparison,
arithmetic, decision, interpretation, transfer, and support tasks. Future AA adds
model-side tasks; future demand→GPU→cluster/datacenter work adds capacity/power.
