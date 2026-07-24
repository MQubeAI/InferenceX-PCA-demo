# Representation Analysis Protocol

Status: fixed before interpretation of neural screening results  
Snapshot: `db-dump/2026-07-20`  
Schema: `representation-analysis-v1`

## Research questions and hypotheses

This experiment asks whether modest nonlinear representations of inference configuration space
improve reconstruction, neighborhood structure, or descriptive outcome organization enough to
justify their added complexity relative to the established PCA basis.

The preregistered working hypotheses are:

1. PCA will remain the most interpretable and temporally stable representation.
2. A deterministic autoencoder may improve matched-dimension reconstruction if configuration
   interactions are materially nonlinear.
3. A VAE may produce a smoother latent space, but beta 1.0 may trade reconstruction quality for
   regularization or produce posterior collapse on this modest dataset.
4. Apparent separation in a two-dimensional plot will not be treated as evidence of superiority.

No winner is declared until grouped, multi-seed evidence exists.

## Dataset and exclusions

The source is the active cumulative snapshot at
`/tmp/inferencex-dump-comparison/db-dump-2026-07-20/`, corresponding to the official
`db-dump/2026-07-20` release. It contains 81,851 raw benchmark rows and 8,239 median aggregate
groups. The canonical representation cohort contains exactly 8,063 `single_turn` groups and
1,354 unique `config_id` values. Source observations span 2025-09-29 through 2026-07-18.

The 176 `agentic_traces` aggregate groups are excluded because they lack compatible ISL/OSL
semantics. The canonical aggregate key is `config_id`, `benchmark_type`, `isl`, `osl`, and `conc`.
The ordered key sequence is hashed and stored with every neural artifact.

## Shared inputs and preprocessing

All methods use this exact source-feature order:

1. `isl`
2. `osl`
3. `conc`
4. `config_prefill_tp`
5. `config_prefill_ep`
6. `config_prefill_dp_attention`
7. `config_prefill_num_workers`
8. `config_decode_tp`
9. `config_decode_ep`
10. `config_decode_dp_attention`
11. `config_decode_num_workers`
12. `config_num_prefill_gpu`
13. `config_hardware`
14. `config_framework`
15. `config_model`
16. `config_precision`
17. `config_spec_method`
18. `config_disagg`
19. `config_is_multinode`

The 14 numeric/boolean fields use median imputation followed by standard scaling. Boolean-like
dump values are normalized to 0/1 before numeric conversion. The five categorical fields use
most-frequent imputation and dense one-hot encoding with unknown values ignored and a maximum of
30 categories per source feature. The resulting established matrix has 51 encoded columns.

For this first experiment, AE and VAE decoders jointly reconstruct the 51-column matrix with mean
squared error. This is the simplest defensible implementation and exactly reuses the PCA input
matrix. It does not guarantee that reconstructed one-hot values form an exclusive categorical
choice; unknown categories also map to an all-zero vector. Consequently categorical
reconstruction is reported in encoded space and aggregated by source feature. Separate
categorical decoder heads are deferred unless this limitation materially affects conclusions.

`metrics_median_tpot`, `metrics_tput_per_gpu`, and
`metrics_joules_per_output_token`—and every other metric, latency, throughput, power, or energy
field—are excluded from representation preprocessing and fitting.

The established PCA transform fits preprocessing on the complete unsupervised cohort. To preserve
the PCA basis and supply the exact same matrix to all methods, the neural experiments reuse that
canonical transform; neural parameter fitting is restricted to training configuration groups in
each fold. This protocol choice is explicit because fold-specific preprocessing would no longer
be the same matrix as the preserved PCA basis.

## Grouped validation

Three deterministic `GroupKFold` folds use `config_id` as the group. No configuration can appear
in both train and validation partitions within a fold. Artifacts store row-ID membership, group
counts, and zero-overlap checks for every split. Seed 42 is used for Stage 2 screening. Seeds 42,
123, and 2026 are reserved for the final selected architecture.

## PCA method

The existing July PCA basis is preserved byte-for-byte and is not refit by the artifact builder.
Its explained variance, cumulative variance, components needed for 90%, June–July alignment,
loadings, component quantiles, and three outcome overlays remain authoritative. Reconstruction is
reported at latent dimensions 2, 5, 10, and 15 by truncating the preserved basis and scoring rows
assigned to each grouped validation fold. This scoring does not claim that PCA was fitted only on
fold training rows.

## Autoencoder method

The deterministic baseline uses:

`51 → 64 → 32 → latent → 32 → 64 → 51`

ReLU activations are used between linear layers. Candidate latent dimensions are 2, 5, 10, and
15. Adam uses learning rate 0.001 and batch size 256. The loss is mean squared reconstruction
error. Training stops after 12 epochs without at least 0.00001 validation-objective improvement,
with a maximum of 150 epochs.

Stage 2 screens the bounded dimension grid with seed 42. Selection uses reconstruction,
stability, and complexity only: choose the smallest dimension within 1% of the minimum mean
grouped-validation MSE. Outcome metrics cannot affect selection. Stage 3 will use at least three
seeds only after the screening report and compute estimate are reviewed.

## Variational autoencoder method

The VAE uses the same hidden widths and candidate latent dimensions. The encoder emits a mean and
log variance, sampling uses reparameterization, and the symmetric decoder reconstructs the shared
matrix. The objective is mean squared reconstruction error plus beta times mean KL divergence.
The initial fixed beta is 1.0. Values 0.1 and 0.5 will be considered only if beta 1.0 shows a
specific collapse or reconstruction problem; no broad beta search is allowed.

Posterior-collapse diagnostics include average KL per dimension, active dimensions using KL
greater than 0.01, variance of encoder means, and mean decoder-output sensitivity to a one-unit
latent perturbation. A run is flagged when it has no active dimensions or maximum decoder
sensitivity below 0.001.

## Reconstruction protocol

Validation MSE, MAE, source-feature error, feature-type error, and fold-level error are reported.
Matched comparisons are strictly labeled at dimensions 2, 5, 10, and 15. Results at unequal
dimensions are never presented as direct reconstruction comparisons.

## Clustering protocol

K-means is shared across PCA, AE, VAE, and the standardized original feature space. The bounded
range is k=2 through 10 with `n_init=20` and seed 42. Metrics are silhouette score,
Davies–Bouldin index, Calinski–Harabasz score, minimum/maximum cluster sizes, and size balance.
Cluster stability will use aligned/resampled representations in Stage 3. Visual separation alone
is not an evaluation metric. HDBSCAN is excluded from the initial experiment.

## Outcome-overlay and probe protocol

After representation fitting, the three outcomes are evaluated descriptively with dimension-level
Pearson and Spearman correlations, strongest associated dimension, target summaries by cluster
and latent quantile, and nearest-neighbor target consistency.

Ridge regression with alpha 1.0 is the sole supervised evaluation probe. It uses the same three
grouped `config_id` folds for every representation. Probe results are evaluation evidence only;
they never update representations and are not the project's main supervised models. TabFM and
all unrelated supervised models remain untouched.

## Interpretability protocol

PCA uses signed encoded/source-feature loadings and positive/negative directions. Neural methods
use decoder sensitivity, latent perturbations, representative observations in low/high latent
regions, and reconstruction error by source feature. Neural axes are aligned before direct
cross-run comparison and are not assigned causal meanings.

## Stability protocol

PCA retains June–July loading and subspace stability; grouped-fold stability may be added without
changing the saved basis. Neural runs are compared using orthogonal Procrustes alignment,
Spearman correlation of pairwise distances, nearest-neighbor overlap, and cluster agreement.
Raw neural axes are never compared across seeds without alignment.

Stage 2 fold-trained models provide preliminary stability evidence. Stage 3 requires seeds 42,
123, and 2026 for the selected architecture.

## Computational-cost protocol

Artifacts report parameter count, per-fold runtime, total screening runtime, transformation time,
weights size, JSON artifact size, epoch count, and software versions. Peak memory is optional
because it is not reliably available in the current environment.

## Stopping rules

Stage 2 stops after one seed across the four fixed latent dimensions for each neural method. Early
stopping bounds each fold. A VAE beta adjustment is allowed only in response to explicit
posterior-collapse evidence. Stage 3 cannot begin until Stage 2 results, compute estimate,
architecture and latent-dimension choices, seed count, and collapse concerns are reported.

Training occurs only through scripts in the Python 3.11 research environment. Streamlit reads
JSON artifacts and never imports PyTorch during ordinary startup.

## Stage 3 fixed amendment

After the Stage 2 stop report, the bounded final amendment fixed latent dimension 15 and seeds 42,
123, and 2026. The AE epoch cap increased from 150 to 250 while patience, optimizer, batch size,
learning rate, depth, and width remained unchanged. The VAE diagnostic was limited to beta 0.1,
0.5, and the fully comparable existing beta-1.0 result. Before the diagnostic, five active
dimensions in every fold was fixed as the minimum defensible activity gate.

Final v2 artifacts store summaries and histories in JSON. Representative fold-0 embeddings for
each seed, row keys, fold identifiers, cluster labels, and overlay columns use Zstandard-compressed
Parquet. Cohort, feature-order, row-key, seed, row-count, and checksum compatibility are validated
before dashboard rendering. Stage 2 v1 artifacts remain unchanged.

## Stage 4 validation amendment

Stage 4 does not alter the cohort, source features, architecture, nominal latent dimension, or
outcome-exclusion rules. For every grouped fold, one shared preprocessing object fits numeric
imputation, categorical imputation, standardization, and categorical mappings on training rows
only. PCA, AE, and VAE receive that same fold matrix; validation-only categories are represented
as unknown. Three independent shuffled group assignments (seeds 17, 29, and 43) each contain
three config_id-disjoint folds. Neural uncertainty continues to use seeds 42, 123, and 2026.

Encoded-column reconstruction remains reported. A second source-balanced score gives each of the
19 source variables one vote: standardized reconstruction error for numeric and boolean fields,
and classification error for categorical fields. Categorical exact accuracy, top-two accuracy,
and sparse confusion counts are retained.

The only robustness interventions are 5% input corruption for the fixed AE and a 50-epoch linear
KL warm-up to the fixed VAE beta of 0.1. The two feature-family ablations are configuration-only
and workload-only. To stay within the fixed compute bound, neural ablations are exploratory
seed-42/fold-0 checks; PCA uses all three folds. Reconstruction scores are not compared across
ablations with different input dimensions. Cross-method cluster agreement uses ARI, NMI, optimal
label overlap, and configuration-level majority assignments under the shared k-means procedure.

Stage 4 preserves all Stage 3 artifacts and writes the new schema
`representation-validation-stage4-v1`.

## Limitations

- The snapshot is observational and unevenly covers configurations and workloads.
- Outcomes are descriptive overlays; associations and ridge probes are not causal.
- The published PCA artifact preserves the complete-cohort historical basis; Stage 4 additionally
  fits fold-local PCA models solely for leakage-free evaluation without changing that artifact.
- Joint MSE does not enforce valid categorical decoder outputs.
- One Stage 2 seed is insufficient for final stability or publication claims.
- Fold-specific latent coordinate systems require alignment.
- K-means favors approximately convex clusters and does not establish a natural cluster count.
- Energy outcomes exist only for a measured subset with narrow support.
