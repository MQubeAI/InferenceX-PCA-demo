# Stage 5 Manifold and Neighborhood Analysis Protocol

Status: frozen infrastructure protocol; execution and interpretation are deferred  
Snapshot: `db-dump/2026-07-20`  
Future artifact schema: `manifold-analysis-stage5-v1`

## Scope and safeguards

Stage 5 asks whether nonlinear neighborhood analysis can reveal stable local or
multiscale structure in the established structural benchmark space, while
distinguishing neighborhood evidence from an attractive two-dimensional figure.
It is not a clustering exercise, a model-selection exercise, or an outcome-driven
optimization.  No result or claim is made by this protocol or its infrastructure.

The canonical cohort is the completed July 20 study cohort: 8,063 eligible
`single_turn` aggregate rows, 1,354 `config_id` groups, stable row IDs, and the
same semantic/cohort hashes.  The ordered 19 frozen source features are encoded
with the established preprocessing and form the primary approximately 51-column
structural matrix.  All `metrics_*` fields and every latency, throughput, power,
energy, or other outcome are excluded from fit, tuning, selection, and ranking.
Every Stage 5 fit or held-out transform carries the audited encoded feature order,
which must map entirely back to those source variables; unlabelled matrices are
not accepted by the frozen fitting path.

## 1. Structural fitting

The primary metric is Euclidean distance on the canonical encoded structural
matrix.  Reference neighborhood scales are k = 10, 30, and 50; k = 100 is a
sensitivity scale.  Grouped uncertainty samples `config_id` values, retaining all
of each sampled configuration's rows.  Future headline uncertainty uses 200
grouped bootstrap replicates with seed 42; individual benchmark rows are never
bootstrapped independently.

UMAP is research-only and is not imported by the Streamlit application.

* UMAP-2 is the primary nonlinear neighborhood visualization.  Its frozen grid
  is `n_neighbors` = 15, 50, 100; `min_dist` = 0.0, 0.1, 0.5; seeds 42, 123,
  2026; Euclidean metric.  The canonical display is `(15, 0.1, 42)`.  The 27-run
  grid is a sensitivity grid, never a beauty contest.
* UMAP-15 uses the same 27 settings.  It is only a candidate quantitative
  representation.  It may be included in the explicitly row-aligned
  neighborhood-consensus sensitivity, but cannot be presented as a replacement
  for PCA/AE/VAE or be promoted unless later evidence covers fidelity, seed and
  parameter stability, grouped held-out transform behavior, distance sensitivity,
  and incremental value.  Held-out UMAP work reuses the Stage 4 preprocessor,
  fitting it on training configurations only before fitting UMAP and transforming
  validation rows.
* t-SNE-2 is fitted directly to the primary encoded matrix, not PCA-15 or
  PCA-30.  Its frozen grid is perplexity 15, 30, 50 and seeds 42, 123, 2026.
  It uses the installed scikit-learn implementation with PCA initialization,
  automatic learning rate, 1,000 maximum iterations, early exaggeration 12,
  Euclidean metric, and Barnes-Hut method.  `(perplexity=30, seed=42)` is its
  canonical display.  t-SNE is descriptive only: it is not transformed to held
  out configurations, clustered, probed, or included in quantitative consensus.

## 2. Neighborhood-fidelity evaluation

Every manifold candidate is evaluated against exact/reference original-space kNN
graphs.  Evidence is reported per row and in aggregate using neighbor recall,
neighbor Jaccard, local shared-neighbor rank agreement, and scikit-learn
trustworthiness at the fixed k values.  Recall remains separate from rank
agreement; rows lacking enough shared neighbors are explicitly ineligible for the
rank statistic.  Cross-seed evidence is neighbor-set based, so arbitrary rotation
or reflection of coordinates is irrelevant.  Cross-method consensus may compare
PCA-15, AE-15, VAE-15, and optionally UMAP-15, but excludes t-SNE by default.
All such comparisons align unique semantic row IDs before graph construction.  A
consensus neighbor is one recurring in at least two included methods; the complete
per-row method count is retained rather than treating an absent neighbor as rank
agreement.

## 3. Hierarchy analysis

The hypothesis to test later, not a conclusion, is that exact workloads
`(isl, osl, conc)` may organize broad neighborhoods while configurations organize
finer structure within comparable workloads.  The analysis reports workload
neighborhood purity and, within eligible workload cells, categorical configuration
homophily or numeric/discrete absolute differences.  A cell is eligible at k only
when it contains at least k + 1 rows.  Reports must retain eligible rows/cells,
excluded rows/cells, and exclusion reasons.  Null tests permute configuration
values only within exact workload cells with a deterministic seed.

## 4. Visualization

Two-dimensional maps support descriptive inspection only.  Axes have no inherent
meaning; visual islands and global distances are not physical distance evidence.
No UMAP-2 or t-SNE-2 setting is selected because it appears cleaner.  UMAP-2 is
the primary visual map and t-SNE-2 is a robustness/sensitivity check.

## 5. Post-hoc outcome overlays

Only after structural fitting and neighborhood evaluation are frozen may outcome
metrics be joined as descriptive overlays.  They cannot affect a parameter,
method, seed, representation, or map choice.  This infrastructure deliberately
does not create overlays or outcome artifacts.

## 6. Later external/generalization testing

Before a candidate representation can be promoted, later work must execute the
prepared grouped held-out UMAP transforms and evaluate an untouched external or
later snapshot.  The prepared source-level mixed-distance sensitivity gives each
original source variable one equal contribution (numeric absolute difference
divided by its finite reference range and capped at one, or
categorical/boolean mismatch), without using outcomes.  A constant numeric source
contributes zero when both values are present; two missing values contribute zero
and a one-sided missing value contributes one.  A dense precomputed mixed distance
matrix is an explicit, memory-accounted operation only; normal analysis uses
chunked kNN construction.

## Future artifact identity contract

Any later Stage 5 result must record source dump, cohort and semantic identities,
row-key identity, feature order, an explicit empty target-input list, software
versions, parameter grid, seeds, neighborhood k values, method, runtime, and
projection row IDs.  It remains exploratory until reviewed evidence is produced
in a later results PR; it is not added to the promoted artifact index here.
