# Representation Analysis Stage 4: Research Validation

Status: completed bounded methodological validation  
Artifact schema: `representation-validation-stage4-v1`  
Snapshot: `db-dump/2026-07-20`  
Created: 2026-07-24 UTC

## Fixed protocol and invariants

Stage 4 asks whether the Stage 3 interpretation survives stricter evaluation. It is not an
architecture or hyperparameter search.

- Cohort: 8,063 eligible `single_turn` aggregate groups and 1,354 configurations.
- Inputs: the frozen 19 configuration/workload source features.
- Outcomes: excluded from preprocessing, representation fitting, and selection.
- Validation: three `config_id`-grouped folds with zero train/validation configuration overlap.
- Independent partition assignments: seeds 17, 29, and 43; three folds per assignment.
- Neural seeds: 42, 123, and 2026.
- AE: fixed 51→64→32→15→32→64→51 architecture.
- VAE: fixed corresponding architecture and beta 0.1.
- PCA: the published artifact was not refit or rewritten. Fold-local PCA models exist only for
  strict evaluation.

The PCA artifact SHA-256 remained
`7857c1e0e3d29ee7dc1c7027a6fa2872d276dda2ad56a265cc1b9f0dd28951c8`
before and after Stage 4. Existing Stage 3 artifacts remain readable.

## Remaining methodological weaknesses entering Stage 4

Stage 3 fit its unsupervised preprocessing on the complete eligible cohort before applying grouped
evaluation. Its 51 encoded columns also gave multi-category source variables more reconstruction
weight than single-column variables. One group partition could understate partition sensitivity.
Neural feature importance had not been assessed across partitions, and independently favorable
clustering scores did not establish that methods found the same benchmark families.

Stage 4 directly tests those weaknesses. External generalization remains unresolved.

## Leakage analysis

For each fold, the numeric and categorical imputers, standardization, and one-hot category mapping
were fit on training rows only. Validation rows were transformed with that same object. PCA, AE,
and VAE used the identical fold-specific matrix. Some folds produced 50 rather than 51 columns
because a rare category did not occur in training; that validation-only category was treated as
unknown for all methods.

| Method | Stage 3 MSE | Strict train-only MSE | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| PCA-15 | 0.033197 | 0.034832 | +0.001635 | +4.9% |
| AE-15 | 0.012669 | 0.011641 | -0.001028 | -8.1% |
| VAE-15 | 0.046865 | 0.049129 | +0.002264 | +4.8% |

These are matched seed-42 fold means, not the all-seed Stage 3 headline means; holding the seed
fixed isolates the preprocessing change.

Evidence: leakage-free preprocessing causes small-to-moderate numerical shifts but preserves the
reconstruction ordering.

Interpretation: Stage 3's full-cohort preprocessing did not create its qualitative conclusion.
The strict estimates should nevertheless be preferred for publication.

Unresolved: the preserved dashboard PCA basis remains the historical complete-cohort basis. It
must be described separately from fold-local evaluation PCA.

## Source-feature reconstruction

Encoded-column MAE and MSE remain available. The additional score gives each of the 19 source
variables one equal vote. Numeric and boolean errors are measured on their standardized fold
scale; categorical error is one minus exact decoded-category accuracy.

| Method | Balanced source MAE | SD | Balanced source MSE | SD |
|---|---:|---:|---:|---:|
| PCA-15 | 0.111514 | 0.002613 | 0.065999 | 0.002922 |
| AE-15 | 0.049866 | 0.004805 | 0.016154 | 0.004064 |
| VAE-15 | 0.145865 | 0.008942 | 0.099963 | 0.015836 |

### Feature-type summary

| Method | Type | Mean balanced MSE | Exact accuracy | Top-2 accuracy |
|---|---|---:|---:|---:|
| PCA | numeric | 0.017493 | — | — |
| PCA | boolean | 0.016476 | 1.000 | — |
| PCA | categorical | 0.202630 | 0.797 | 0.886 |
| AE | numeric | 0.013313 | — | — |
| AE | boolean | 0.007483 | 1.000 | — |
| AE | categorical | 0.028774 | 0.971 | 0.983 |
| VAE | numeric | 0.052749 | — | — |
| VAE | boolean | 0.017611 | 0.999 | — |
| VAE | categorical | 0.260274 | 0.740 | 0.872 |

The hardest categorical reconstructions were model/hardware for PCA, hardware/model for AE, and
hardware/model for VAE. Per-feature MAE, MSE, rank, exact accuracy, top-two accuracy, and sparse
confusion counts are retained in the artifact and dashboard.

Evidence: the AE advantage is not an artifact of the 51-column weighting; it becomes clearer when
each source variable contributes equally.

Interpretation: the deterministic AE preserves category identity particularly well. This is a
descriptive reconstruction result, not evidence of causal or downstream superiority.

## Partition robustness

The table aggregates nine folds for PCA and 27 fold-seed runs for each neural method.

| Method | Encoded MSE mean ± run SD | Between-partition SD | Between-seed SD | Source-balanced MSE |
|---|---:|---:|---:|---:|
| PCA-15 | 0.035306 ± 0.000614 | 0.000301 | — | 0.066986 |
| AE-15 | 0.012102 ± 0.002138 | 0.000652 | 0.000722 | 0.016836 |
| VAE-15 | 0.047241 ± 0.004052 | 0.000637 | 0.000877 | 0.096001 |

AE partition means ranged from 0.011393 to 0.012678; VAE partition means ranged from 0.046689 to
0.047938. AE aligned geometry stability was 0.780 overall, compared with 0.640 for VAE. Within a
fixed seed across partitions, AE stability was approximately 0.802–0.806; VAE was
approximately 0.622–0.666.

Evidence: between-partition variation is smaller than total fold/run variation for both neural
methods. The original group assignment is not driving the reconstruction conclusion.

Interpretation: AE geometry is more reproducible than VAE geometry under both seed and partition
changes.

## Bounded AE robustness intervention

Five-percent independent input masking was evaluated with the unchanged AE architecture and seed
42 on the current three grouped folds.

| AE variant | Encoded MSE | Encoded MAE | Aligned stability |
|---|---:|---:|---:|
| Baseline | 0.011641 ± 0.002169 | 0.059777 ± 0.002688 | 0.7931 |
| 5% denoising | 0.017457 ± 0.007506 | 0.073660 ± 0.011000 | 0.7836 |

Evidence: denoising worsened both validation reconstruction and variability and did not improve
aligned stability.

Interpretation: retain the Stage 3 AE. There is no methodological justification for adding this
regularizer.

## Bounded VAE anti-collapse intervention

A linear 50-epoch KL warm-up to the fixed beta of 0.1 was evaluated with seed 42 and the current
three folds.

| VAE variant | Encoded MSE | Encoded MAE | Active dimensions | Aligned stability |
|---|---:|---:|---:|---:|
| Baseline | 0.049129 ± 0.006359 | 0.120485 ± 0.003884 | 5, 7, 6 | 0.5806 |
| KL warm-up | 0.041550 ± 0.005632 | 0.109498 ± 0.006372 | 7, 7, 7 | 0.6595 |

Warm-up validation KL was 0.4641 ± 0.0258. Mean per-dimension KL by fold was 0.4698, 0.4704,
and 0.4406; summed KL was 7.0469, 7.0555, and 6.6085. Mean latent variance was 0.3939, 0.3933,
and 0.3702. Decoder sensitivity maxima were 0.1519, 0.1054, and 0.1141, and no fold's decoder
ignored latent changes.

Evidence: warm-up improves reconstruction, active-dimension consistency, and stability without
complete collapse.

Interpretation: warm-up is a defensible anti-collapse improvement for a future versioned VAE
artifact. Seven active dimensions out of 15 still constitutes partial use of the nominal latent
space, so the VAE should not be described as fully resolved or universally competitive.

## Feature-importance consistency

PCA used 20 group-bootstrap samples per partition. Its most frequent top-five source features were
`isl` (100%), `config_prefill_ep` (98%), `osl` (82%), `config_prefill_tp` (57%), and
`config_decode_tp` (55%).

AE decoder-sensitivity rankings had mean pairwise Spearman correlation 0.461. Its most repeated
top-five features were `isl` (89%), `config_prefill_tp` (67%), `config_disagg` (56%), and
`config_prefill_ep` (56%). VAE rankings were more internally consistent (0.890), repeatedly
emphasizing `isl`, `config_disagg`, `config_is_multinode`, and
`config_prefill_dp_attention` (all 100%), followed by `config_prefill_ep` (89%).

Evidence: `isl` and `config_prefill_ep` recur across all three methods. The neural methods also
repeatedly expose distributed-layout variables.

Interpretation: repeated appearance supports descriptive structural relevance, but decoder
sensitivity and PCA loadings are different estimands and should not be numerically equated.

## Cross-method cluster consistency

All methods used the shared k-means procedure. Labels were aligned optimally before reporting
overlap.

| Pair | ARI mean ± SD | NMI mean | Row overlap | Configuration ARI | Configuration overlap |
|---|---:|---:|---:|---:|---:|
| PCA ↔ AE | 0.654 ± 0.127 | 0.643 | 0.850 | 0.805 | 0.860 |
| PCA ↔ VAE | 0.735 ± 0.024 | 0.700 | 0.885 | 0.832 | 0.882 |
| AE ↔ VAE | 0.786 ± 0.157 | 0.721 | 0.878 | 0.803 | 0.894 |

Evidence: the methods recover substantially overlapping benchmark families despite different
objectives and geometry. Agreement variability, especially for AE↔VAE, remains material.

Interpretation: shared structure is more defensible evidence than visual separation or a small
silhouette difference. The agreement does not imply that any partition is uniquely natural.

## Bounded feature-family ablations

Configuration-only uses 16 source variables; workload-only uses `isl`, `osl`, and `conc`.
PCA uses all three folds. To honor the compute bound, neural ablations use only seed 42 and fixed
fold 0 and are therefore exploratory.

| Features | Method | Best k | Silhouette | TPOT R² | Throughput R² | Energy R² |
|---|---|---:|---:|---:|---:|---:|
| Configuration only | PCA | 5 | 0.409 | 0.139 | 0.179 | 0.155 |
| Configuration only | AE | 2 | 0.449 | 0.113 | 0.240 | 0.121 |
| Configuration only | VAE | 2 | 0.233 | 0.057 | 0.268 | 0.123 |
| Workload only | PCA | 7 | 0.891 | 0.043 | 0.168 | 0.005 |
| Workload only | AE | 7 | 0.878 | 0.094 | 0.283 | 0.073 |
| Workload only | VAE | 6 | 0.830 | 0.156 | 0.302 | 0.122 |

Evidence: the three workload variables form sharply separated repeated workload families, while
configuration variables retain more energy and PCA TPOT signal. Workload-only probes can preserve
throughput signal but largely lose observed-energy signal.

Interpretation: visual/cluster structure is strongly workload-driven, while configuration
variables contribute complementary outcome information. High workload-only silhouette should not
be read as a better representation. Reconstruction is not compared across ablations because the
encoded input dimensions differ and PCA-15 is overcomplete for the three-variable workload case.

Unresolved: the neural ablations need full folds/seeds before publication-grade uncertainty claims.

## External validation proposal

The local June rollback export contains 7,462 eligible groups. The July cohort has 601 additional
groups. A retrospective temporal test can fit preprocessing and representations on June only,
freeze every choice, and score only the 601 new July groups. Because July already informed the
current methodology, the stronger prospective test should reserve the next untouched official
snapshot as external validation. No external-validation model was fit in Stage 4.

## Computational cost

The primary Stage 4 run used 186.1 seconds of neural training and 240.1 seconds wall time. A
diagnostic-only repeat of the three fixed warm-up folds added 6.4 seconds to retain per-dimension
KL, latent-variance, and decoder-response arrays. Total neural work was 192.5 seconds—within the
approximate two-times-Stage-3 bound and without a search. The Stage 4 JSON artifact is about
2.6 MiB; no duplicate row-level embedding companion was required.

## Updated publication discussion

### Evidence from this experiment

Strict preprocessing, three independent group partitions, and equal-weight source reconstruction
do not reverse the Stage 3 conclusions. AE remains strongest on reconstruction and more stable
than VAE. PCA remains the most directly interpretable and cheapest representation. KL warm-up
materially improves the VAE but does not make all 15 dimensions active. Cross-method cluster
agreement shows that the representations recover overlapping benchmark families.

### Interpretation

The comparison is now substantially more defensible as an internal-validation study. Its main
claims should be evidence-specific: AE for reconstruction, PCA for transparent global directions
and compute cost, and VAE warm-up for a more regularized stochastic representation that still
carries partial-collapse limitations. There is no universal winner.

### Remaining publication weaknesses

- No untouched external benchmark snapshot has been evaluated.
- Neural ablations have only one seed/fold.
- The cohort is observational and unevenly covers configurations and energy measurements.
- K-means imposes convex clusters and does not prove a natural taxonomy.
- Categorical decoding is evaluated by one-hot argmax rather than separate categorical heads.
- Neural sensitivity is descriptive, local, and non-causal.
- The preserved historical PCA basis and fold-local evaluation PCA must remain clearly labeled.

Recommendation: methodology is publication-ready for a carefully scoped internal-validation or
workshop-style analysis, provided the limitations are explicit. A stronger archival claim should
wait for the planned temporal/external validation and full-uncertainty ablation confirmation.
