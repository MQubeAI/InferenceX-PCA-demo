# Representation Analysis Stage 3 Final Experiment

Status: completed bounded experiment  
Snapshot: `db-dump/2026-07-20`  
Final seeds: 42, 123, 2026  
Latent dimension: 15

## Fixed protocol

Stage 3 retained the exact 8,063-row cohort, 1,354 configurations, 19 source features, 51 encoded
columns, preprocessing state, row order, cohort hash, row-key hash, and three `config_id`
`GroupKFold` partitions from Stages 1–2. Each method reports zero group overlap. Outcome metrics
were excluded from preprocessing, representation fitting, beta selection, and architecture
selection.

The final neural architecture remained `51 → 64 → 32 → 15 → 32 → 64 → 51`, with ReLU, Adam,
learning rate 0.001, batch size 256, early-stopping patience 12, and minimum improvement 0.00001.
No broad search was run.

## Autoencoder convergence decision

The AE maximum epoch cap increased from 150 to 250, a 67% increase. Architecture and patience were
unchanged. One of nine runs early-stopped at epoch 222; eight reached the 250-epoch cap. The final
mean validation MSE improved from the Stage 2 seed-42 value of 0.016306 to
**0.012460 ± 0.003290** across all folds and seeds. Mean MAE was
**0.060560 ± 0.003965**. Seed-level mean MSEs were 0.012669, 0.012553, and 0.012157, so the
improvement was not driven by one seed.

| Seed | Fold | MSE | MAE | Epochs | Best epoch | Early stopped | Train–validation gap | Runtime s |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 42 | 0 | 0.016714 | 0.064320 | 250 | 248 | No | 0.009529 | 3.082 |
| 42 | 1 | 0.009584 | 0.058074 | 250 | 248 | No | 0.001530 | 3.225 |
| 42 | 2 | 0.011708 | 0.061867 | 250 | 250 | No | 0.003134 | 2.985 |
| 123 | 0 | 0.017758 | 0.067462 | 222 | 210 | Yes | 0.009674 | 2.676 |
| 123 | 1 | 0.009693 | 0.055144 | 250 | 250 | No | 0.001829 | 3.031 |
| 123 | 2 | 0.010208 | 0.056425 | 250 | 250 | No | 0.002684 | 3.228 |
| 2026 | 0 | 0.015641 | 0.062696 | 250 | 250 | No | 0.008719 | 3.129 |
| 2026 | 1 | 0.009968 | 0.058304 | 250 | 250 | No | 0.002114 | 3.228 |
| 2026 | 2 | 0.010864 | 0.060750 | 250 | 250 | No | 0.002635 | 3.154 |

The positive mean train–validation reconstruction gap was 0.004650 ± 0.003535. Fold 0 was
consistently harder than folds 1–2, while seed-level means remained close. This is treated as fold
composition uncertainty, not seed instability.

## VAE beta diagnostic and decision

Seed 42 used the same three folds and 150-epoch cap for beta 0.1 and 0.5. The fully comparable
Stage 2 beta-1.0 result was reused.

| Beta | MSE mean | MAE mean | KL mean | Total objective | Active dimensions mean | Minimum active | Stability | Best silhouette |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.046865 | 0.119245 | 0.429011 | 0.089767 | 6.33 | 6 | 0.630 | 0.222 |
| 0.5 | 0.126764 | 0.189913 | 0.132765 | 0.193146 | 3.33 | 2 | 0.585 | 0.658 |
| 1.0 | 0.192879 | 0.230490 | 0.060652 | 0.253531 | 2.67 | 2 | 0.551 | 0.748 |

Beta 0.1 was selected because it passed the fixed minimum of five active dimensions in every fold,
raised the minimum activity from 2 to 6 dimensions, improved reconstruction, and improved
cross-fold stability. Outcome probes were not part of this decision.

The declining silhouette as latent activity improves is important evidence: the high beta-1.0
silhouette was coupled to low activity and was not evidence of a superior representation.

## Final VAE results

Beta 0.1 was run across all three seeds and folds. Final MSE was
**0.045563 ± 0.003920**, and MAE was **0.116615 ± 0.004510**.

| Seed | Fold | MSE | MAE | Epochs | Best epoch | Early stopped | Active dimensions | Runtime s |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 42 | 0 | 0.050955 | 0.123182 | 139 | 127 | Yes | 7 | 2.128 |
| 42 | 1 | 0.045619 | 0.116989 | 150 | 149 | No | 6 | 2.139 |
| 42 | 2 | 0.044022 | 0.117564 | 150 | 148 | No | 6 | 2.251 |
| 123 | 0 | 0.047672 | 0.115821 | 150 | 150 | No | 6 | 2.286 |
| 123 | 1 | 0.040372 | 0.110681 | 150 | 142 | No | 6 | 2.186 |
| 123 | 2 | 0.045625 | 0.119108 | 150 | 150 | No | 5 | 2.127 |
| 2026 | 0 | 0.050789 | 0.120937 | 150 | 143 | No | 5 | 2.278 |
| 2026 | 1 | 0.039850 | 0.109020 | 150 | 142 | No | 6 | 2.211 |
| 2026 | 2 | 0.045163 | 0.116234 | 150 | 143 | No | 6 | 2.285 |

No run completely collapsed or ignored latent perturbations. However, only 5–7 of 15 dimensions
were active. This remains a material **partial posterior-collapse limitation**. The VAE is retained
as a bounded regularized baseline, not presented as competitive with the AE on reconstruction.

## Matched 15-dimensional reconstruction

| Representation | Validation MSE | Validation MAE |
|---|---:|---:|
| PCA-15 | 0.033197 ± 0.001115 | 0.103729 ± 0.001388 |
| AE-15 | **0.012460 ± 0.003290** | **0.060560 ± 0.003965** |
| VAE-15, beta 0.1 | 0.045563 ± 0.003920 | 0.116615 ± 0.004510 |

PCA uses the preserved full-cohort basis and scores the same grouped validation rows; it was not
refitted per fold. AE and VAE parameters were fitted only on each fold's training configurations.

The largest AE source-feature MSEs were `config_prefill_tp` (0.0359), `conc` (0.0296),
`config_decode_tp` (0.0226), `config_hardware` (0.0168), and `config_model` (0.0165). The largest
VAE errors were `config_prefill_tp` (0.1011), `config_spec_method` (0.0809),
`config_hardware` (0.0705), `config_decode_tp` (0.0582), and `config_model` (0.0549).

## Stability

| Representation | Stability evidence | Score | Distance Spearman | Neighbor overlap | Procrustes similarity | Cluster ARI |
|---|---|---:|---:|---:|---:|---:|
| PCA | Mean June–July PC1–PC5 loading cosine | **0.993** | n/a | n/a | n/a | 1.000 |
| AE | Cross-seed representative-fold geometry | 0.771 | 0.912–0.922 | 0.549–0.579 | 0.819–0.858 | 0.856 |
| VAE | Cross-seed representative-fold geometry | 0.606 | 0.811–0.861 | 0.429–0.525 | 0.424–0.613 | 0.870 |

Neural axes were centered and Procrustes-aligned before coordinate comparison. Raw axes were not
compared directly.

## Clustering

K-means used k=2 through 10 and `n_init=20`; selected k maximized mean silhouette across seeds.

| Representation | Dimension | Selected k | Silhouette | Davies–Bouldin | Calinski–Harabasz | Size balance | Cluster ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA-15 | 15 | 3 | 0.349 ± 0.003 | 1.371 | 2,190.2 | 0.171 | 1.000 |
| AE-15 | 15 | 2 | **0.465 ± 0.005** | **0.994** | **6,602.4** | 0.401 | 0.856 |
| VAE-15 | 15 | 2 | 0.226 ± 0.008 | 1.923 | 1,842.1 | **0.500** | 0.870 |
| PCA-5 compact reference | 5 | 5 | 0.503 ± 0.003 | 0.792 | 4,296.3 | 0.009 | 1.000 |

PCA-5 is shown only as a compact clustering reference. Its reconstruction is not compared against
the 15-dimensional neural methods. Its very low size balance indicates one or more small clusters.

## Grouped ridge outcome probes

R² values are mean ± standard deviation across grouped folds and, for neural methods, seeds.

| Outcome | PCA-15 | AE-15 | VAE-15 |
|---|---:|---:|---:|
| Median TPOT | **0.171 ± 0.051** | 0.144 ± 0.052 | 0.096 ± 0.042 |
| Throughput/GPU | 0.274 ± 0.097 | 0.266 ± 0.080 | **0.279 ± 0.047** |
| Observed energy | 0.161 ± 0.006 | **0.175 ± 0.019** | 0.141 ± 0.018 |

Differences are modest relative to fold uncertainty. Outcome probes are post-hoc and cannot define
a universal winner.

## Interpretability

PCA retains signed loadings and June–July direction stability. AE and VAE artifacts contain
decoder sensitivity, source-feature reconstruction error, -1 to +1 latent traversal summaries,
and representative observations from high and low latent regions. Neural traversal changes are
descriptive and are not causal feature effects.

## Computational cost and storage

| Method | Parameters | Total runtime | Training runtime | Transform time | JSON | Parquet | Weights |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE | 11,842 | 42.2 s | 27.7 s | 0.011 s | about 3.1 MiB | about 2.5 MiB | 462,269 B |
| VAE | 12,337 | 33.9 s | 19.9 s | 0.012 s | about 3.0 MiB | about 2.5 MiB | 486,227 B |

The Stage 2 JSON files were about 9.5 MiB because they stored 8,063 row-level embeddings inline.
Final v2 artifacts keep metadata, histories, splits, diagnostics, and summaries in JSON while
moving three representative seed embeddings, row keys, fold identifiers, cluster labels, and
outcome overlay columns into Zstandard-compressed Parquet. This uses the existing PyArrow
dependency. Stage 2 artifacts were not silently migrated or invalidated.

## Evidence, interpretation, and unresolved questions

Evidence from this experiment:

- AE-15 has the lowest matched reconstruction error.
- PCA has the strongest temporal/loading stability and direct interpretability.
- AE-15 has the strongest matched-15 k-means separation metrics.
- VAE beta 0.1 is substantially healthier than beta 1.0 but remains partially collapsed.
- Outcome-probe advantages depend on the outcome and are small relative to fold variability.

Interpretation:

The data support nonlinear deterministic structure for reconstruction, but do not support a claim
that variational complexity improves this representation experiment. PCA remains the most
defensible interpretation-first method; AE adds reconstruction value at modest computational and
interpretability cost.

Unresolved questions and limitations:

- External-snapshot neural stability has not been tested.
- The established full-cohort preprocessing is intentionally preserved rather than fit within
  each fold.
- Joint MSE does not enforce valid exclusive categorical decoder outputs.
- Only representative fold-0 embeddings are stored for each seed; cross-fold geometry is retained
  as summary metrics.
- K-means assumes convex clusters.
- Observed energy has narrower support than TPOT and throughput.
- The AE still often reaches its enlarged cap; a further cap change was not made because the
  bounded protocol already improved and remained stable.

No universal winner is declared.
