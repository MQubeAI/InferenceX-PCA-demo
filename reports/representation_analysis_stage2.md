# Representation Analysis Stage 2 Screening Report

Status: completed screening; superseded for final estimates by Stage 3  
Run date: 2026-07-23  
Snapshot: `db-dump/2026-07-20`

## Decision record

- Framework: PyTorch in the isolated Python 3.11 research environment; Streamlit does not import it.
- Categorical reconstruction: joint MSE on the established one-hot matrix for the first experiment.
- Architecture: `51 → 64 → 32 → latent → 32 → 64 → 51`, with ReLU.
- Dimensions screened: 2, 5, 10, and 15.
- Validation: three deterministic `GroupKFold` folds by `config_id`; zero group overlap.
- Screening seed: 42.
- Optimizer: Adam, learning rate 0.001, batch size 256.
- Early stopping: 12 epochs without 0.00001 validation-objective improvement; 150-epoch cap.
- AE loss: mean squared reconstruction error.
- VAE loss: mean squared reconstruction error plus beta 1.0 times mean KL divergence.
- Selection: smallest dimension within 1% of minimum mean grouped-validation reconstruction MSE.
- Clustering: k-means, k=2 through 10, `n_init=20`.
- Stability: centered Procrustes coordinate similarity, pairwise-distance Spearman correlation, and
  nearest-neighbor overlap.
- Evaluation probe: grouped ridge regression with alpha 1.0.
- Artifact format: versioned JSON metadata/results plus a versioned PyTorch state dictionary.

Outcome metrics did not participate in fitting or architecture selection.

## Screening results

### Matched-dimension validation reconstruction

| Dimension | PCA MAE | AE MAE | VAE MAE | PCA MSE | AE MSE | VAE MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.248364 | 0.157316 | 0.326431 | 0.199492 | 0.093198 | 0.342060 |
| 5 | 0.205720 | 0.120281 | 0.327514 | 0.121695 | 0.046250 | 0.342023 |
| 10 | 0.145676 | 0.086660 | 0.264767 | 0.063677 | 0.023065 | 0.244396 |
| 15 | 0.103730 | 0.071829 | 0.230490 | 0.033197 | 0.016306 | 0.192879 |

PCA values score grouped validation rows using the preserved full-cohort PCA basis; the basis was
not refit. Neural values come from models fitted only on each fold's training configuration groups.

### Autoencoder

The selection rule chose latent dimension 15. It has 11,842 parameters. Mean validation MSE was
0.016306 and MAE was 0.071829. The preliminary aligned fold-stability score was 0.7993. All three
folds reached the 150-epoch cap (best epochs 150, 150, and 149), so the cap—not patience—bounded
this screen. Total bounded-grid runtime was 34.1 seconds; representative full-cohort transform
time was recorded in the artifact.

### Variational autoencoder

The selection rule also chose latent dimension 15. It has 12,337 parameters. Mean validation MSE
was 0.192879 and MAE was 0.230490. Preliminary aligned fold stability was 0.5514. The three folds
stopped after 121, 75, and 55 epochs, with best epochs 109, 63, and 43. Total bounded-grid runtime
was 22.5 seconds.

The binary collapse rule did not flag any selected fold: active dimensions were 2, 2, and 4 of 15,
and decoder sensitivity remained above 0.001. However, only 13%–27% of latent dimensions were
active. This is a material **partial posterior-collapse concern**, even though complete collapse
was not detected. Beta 1.0 should not advance unchanged to a multi-seed final run without a
bounded beta diagnostic.

## Clustering

| Space | Dimension | Best k | Best silhouette |
|---|---:|---:|---:|
| Standardized original features | 51 | 3 | 0.3083 |
| Preserved PCA | 5 | 5 | 0.5005 |
| Autoencoder representative fold | 15 | 2 | 0.4762 |
| VAE representative fold | 15 | 8 | 0.7481 |

The VAE silhouette is not treated as superiority. Its low active-dimension count may concentrate
the embedding into separated modes, and cluster stability across seeds has not been measured.

## Descriptive outcome evaluation

### Median TPOT

| Method | Strongest dimension | Pearson | Spearman | Neighbor consistency | Ridge R² | Ridge MAE |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 5 | -0.120 | -0.210 | 0.703 | 0.073 | 0.0178 |
| Autoencoder | 7 | 0.173 | 0.337 | 0.661 | 0.108 | 0.0172 |
| VAE | 3 | 0.198 | 0.195 | 0.541 | 0.061 | 0.0179 |

### Throughput per GPU

| Method | Strongest dimension | Pearson | Spearman | Neighbor consistency | Ridge R² | Ridge MAE |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 5 | -0.345 | -0.408 | 0.805 | 0.182 | 2,043.1 |
| Autoencoder | 11 | 0.401 | 0.369 | 0.767 | 0.288 | 2,002.5 |
| VAE | 7 | -0.280 | -0.278 | 0.718 | 0.115 | 2,309.1 |

### Observed joules per output token

| Method | Strongest dimension | Pearson | Spearman | Neighbor consistency | Ridge R² | Ridge MAE |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 3 | 0.288 | 0.600 | 0.680 | 0.137 | 4.571 |
| Autoencoder | 3 | 0.147 | 0.332 | 0.643 | 0.160 | 4.590 |
| VAE | 7 | 0.141 | 0.278 | 0.572 | 0.003 | 5.228 |

These are post-fit descriptive overlays and evaluation probes, not selection criteria or causal
claims.

## Compute and artifact sizes

| Method | Grid runtime | Parameters | JSON artifact | Weights |
|---|---:|---:|---:|---:|
| Autoencoder | 34.1 s | 11,842 | about 9.5 MiB | 52,489 bytes |
| Variational autoencoder | 22.5 s | 12,337 | about 9.7 MiB | 55,057 bytes |

The PCA JSON remains 228,522 bytes and retains SHA-256
`7857c1e0e3d29ee7dc1c7027a6fa2872d276dda2ad56a265cc1b9f0dd28951c8`.

## Decision before Stage 3

Do not launch Stage 3 yet.

The provisional AE architecture is the fixed 64/32 network with latent dimension 15. A final run
would use seeds 42, 123, and 2026 across the same three folds. Because the seed-42 selected
dimension already exists, two additional AE seeds should require roughly 15–25 seconds of model
training on the current CPU, plus evaluation and serialization.

The VAE needs a bounded one-seed beta diagnostic at latent dimension 15 for beta 0.1 and 0.5 before
choosing its final beta. That diagnostic is estimated at 10–20 seconds. If one beta passes the
active-dimension and sensitivity checks, two additional seeds at the chosen beta should require
roughly 10–20 seconds. Total remaining model compute is estimated at 35–65 seconds, excluding
dashboard validation.

Stage 3 decisions still required:

1. Approve AE latent dimension 15 despite every fold reaching the epoch cap, or modestly extend
   only the selected architecture's cap.
2. Approve the bounded VAE beta diagnostic because beta 1.0 shows partial collapse.
3. Confirm three total seeds (42, 123, 2026).
4. Keep k-means as the sole shared clustering method for the first publication experiment.
5. Decide whether the approximately 10 MiB JSON embeddings are acceptable or should move to a
   versioned columnar companion artifact before final publication packaging.

## Stage 3 resolution

The bounded Stage 3 experiment is complete. AE retained latent dimension 15 with a 250-epoch cap.
The beta diagnostic selected VAE beta 0.1, which improved activity but retained partial collapse.
Final embeddings moved to versioned Zstandard-compressed Parquet companions; this Stage 2 artifact
remains unchanged and readable. See `reports/representation_analysis_stage3.md` for final estimates.

Stage 4 subsequently tested preprocessing leakage, independent group partitions, source-balanced
reconstruction, bounded robustness interventions, feature consistency, and cross-method cluster
agreement without changing this artifact. See `reports/representation_analysis_stage4.md`.
