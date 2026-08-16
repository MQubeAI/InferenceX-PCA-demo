# Stage 5 manifold and neighborhood results

## 1. Executive result

The frozen July 20 structural cohort (8,063 single-turn aggregate rows from 1,354
configurations) was evaluated without outcome inputs.  The canonical UMAP-2
projection retained a substantial fraction of the nearest original-space neighbors
at `k=10` (recall 0.695; Jaccard 0.565), but preservation declined at broader
neighborhood sizes (`k=50`: 0.395 and 0.272) and varied appreciably across the
frozen UMAP grid.  Thus it is a descriptive local visualization, not a uniquely
determined map.

Exact workload-cell purity was modest in the original structural graph (0.148--0.178
across `k=10/30/50`) and lower in the canonical UMAP-2 graph (0.086--0.092).  In
contrast, all 16 configuration variables had local relationships in the predicted
direction relative to their workload-conditioned permutation null in the canonical
map.  The evidence therefore supports configuration organization within comparable
workloads, but gives only limited support for a strong workload-first hierarchy.

The source-balanced mixed metric changes a material minority of original-space
neighborhoods, especially beyond `k=10`, so the conclusions are only partially
metric-robust.  t-SNE has stronger local fidelity throughout its frozen nine-run
grid, but remains a non-transforming visualization and supplies no global-geometry
or clustering claim.  UMAP-15 has mixed evidence: it is reasonably seed-stable but
has lower held-out neighbor recovery and no automatic promotion is justified.

## 2. Cohort and protocol compliance

The analysis used snapshot `db-dump/2026-07-20`, 8,063 canonical eligible
`single_turn` rows, and 1,354 `config_id` groups.  Its 19 frozen structural source
fields were `isl`, `osl`, `conc`, and the 16 `config_*` fields recorded in the
structural artifact; the authoritative encoded feature order and semantic identity
were validated before fitting.  `target_metrics_in_inputs` is an empty list in both
Stage 5 JSON artifacts.

The outcome-free structural boundary is
`artifacts/manifold-analysis-stage5-structural-db-dump-2026-07-20.json`, SHA-256
`33441444687c6fdf30dcca5885c31a30b155e3cb634f73d3e7d6c0efc5757fce`.
It references the 362,835-row projection companion with SHA-256
`9f877cee2e5492a7ab4f208b8554a5063fc874baec1dba75fa150ab2202a7f39`.
The final artifact references the same structural SHA.  The frozen counts are
27 UMAP-2, 9 t-SNE-2, 3 workload-only UMAP-2, 3 configuration-only UMAP-2,
3 mixed-distance UMAP-2, 27 UMAP-15, and 27 grouped held-out UMAP-15 fits.

## 3. UMAP-2 neighborhood fidelity

For the fixed canonical display (`n_neighbors=15`, `min_dist=0.1`, seed 42),
trustworthiness / reference-neighbor recall / neighbor Jaccard / local rank
agreement were:

| k | Trustworthiness | Recall | Jaccard | Local rank agreement |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.973 | 0.695 | 0.565 | 0.340 |
| 30 | 0.907 | 0.475 | 0.338 | 0.522 |
| 50 | 0.875 | 0.395 | 0.272 | 0.546 |
| 100 | 0.824 | 0.318 | 0.213 | 0.589 |

Across all 27 frozen settings, recall ranged 0.574--0.700 at `k=10`,
0.463--0.591 at `k=30`, and 0.387--0.604 at `k=50`; Jaccard ranged
0.440--0.572, 0.328--0.440, and 0.267--0.450 respectively.  Matched-parameter
cross-seed Jaccard was 0.447--0.624 (`k=10`), 0.425--0.652 (`k=30`), and
0.353--0.712 (`k=50`).  These are neighborhood comparisons, not coordinate-axis
correlations.  The canonical display lies within, rather than being selected from,
these sensitivity ranges.

## 4. Workload/configuration hierarchy

### Broad workload structure

Exact `(isl, osl, conc)` purity in the original structural neighbor graph was
0.178 (95% grouped-bootstrap interval 0.170--0.187), 0.170 (0.165--0.176), and
0.148 (0.145--0.153) at `k=10/30/50`.  Canonical UMAP-2 purity was lower at
0.086 (0.082--0.091), 0.092 (0.089--0.096), and 0.086 (0.084--0.089).
These exact-cell results do not support a claim that workload dominates local
neighborhood membership.

### Configuration within workload

Within eligible workload cells (at `k=10`: 45 cells, 7,424 rows, 74,240 directed
neighbor pairs; 292 cells and 639 rows excluded explicitly), every one of the 16
configuration variables was shifted in the preregistered direction against 200
within-cell permutations.  Categorical examples were higher local homophily for
`config_is_multinode` (0.948 observed vs 0.711 null), `config_disagg` (0.945 vs
0.711), `config_framework` (0.496 vs 0.265), and `config_model` (0.353 vs 0.174).
Numeric examples had smaller local absolute differences for `config_num_prefill_gpu`
(6.585 vs 13.881), `config_decode_ep` (1.422 vs 2.699), and `config_prefill_ep`
(0.905 vs 1.927).  Each empirical tail value is 1/201 = 0.00498; these are
directional permutation summaries, not a post-hoc multiple-testing declaration.

The same directional pattern persists under the bounded mixed-distance UMAP-2
sensitivity.  Taken together, configuration organization conditional on workload
is supported, whereas the broad workload half is weak by exact local purity.  The
full workload-to-configuration hierarchy is therefore **partially supported**.

## 5. Workload/configuration ablations

With only `isl`, `osl`, and `conc`, canonical-seed UMAP-2 recall was only 0.112,
0.199, and 0.224 at `k=10/30/50`; its mean cross-seed neighbor Jaccard was 0.112.
The workload-only map is consequently not a reproducible substitute for the full
structural geometry.

Removing workload and retaining the 16 configuration fields produced recalls of
0.651, 0.621, and 0.475 and mean cross-seed Jaccard 0.488.  This is consistent with
substantial configuration structure, while not making its two-dimensional axes
comparable to those of the full map.

## 6. Mixed-distance sensitivity

Before fitting mixed-metric UMAP, the source-balanced and encoded-Euclidean
original-space graphs had recall/Jaccard 0.859/0.796 at `k=10`, 0.749/0.626 at
`k=30`, and 0.744/0.615 at `k=50`.  The fraction of rows with less than half
neighborhood overlap was 14.2%, 23.6%, and 25.2%, respectively.  The explicit
float32 distance matrix was 260,047,876 bytes and was retained only long enough for
the three cached mixed-distance fits; it was not written as an artifact.

The canonical mixed-distance UMAP-2 achieved recall 0.677/0.403/0.322 and Jaccard
0.543/0.274/0.209 at `k=10/30/50`.  Its workload purity was 0.058/0.066/0.063,
and its within-workload configuration effects retained their expected directions.
The conclusion is **partially robust**: conditional configuration organization
survives, but the identity of many broader local neighborhoods is metric-sensitive.

## 7. t-SNE robustness check

Canonical t-SNE (`perplexity=30`, seed 42) had trustworthiness 1.000/0.997/0.994
and recall 0.808/0.664/0.644 at `k=10/30/50`; Jaccard was 0.708/0.518/0.496.
Across all nine frozen fits, recall varied only 0.797--0.808 (`k=10`),
0.659--0.670 (`k=30`), and 0.618--0.655 (`k=50`).  Its final KL value for the
canonical run was 0.519 after 999 iterations; this is an optimization diagnostic
within the frozen configuration, not a global parameter-selection criterion.

t-SNE therefore corroborates the presence of recoverable *local* structural
neighborhoods.  It has no held-out transform, was excluded from the quantitative
consensus calculation, was not clustered, and does not justify interpretation of
island spacing, area, or apparent regimes.

## 8. Post-hoc performance/energy overlays

Outcomes were joined by canonical row ID only after the structural artifact hash was
frozen, and only on the already-fixed canonical UMAP-2 coordinates.  These are
descriptive associations, not causal effects and not map-selection criteria.

Median TPOT had near-zero raw coordinate correlations (Pearson x/y 0.005/-0.019;
Spearman 0.008/0.026); within exact workloads the y correlations were
Pearson -0.136 and Spearman -0.192.  Throughput/GPU had raw Pearson x/y
0.026/0.113 and Spearman 0.054/0.053; its within-workload y correlations were
-0.086/-0.156.  Joules/output-token was available for 2,766 rows and had raw
Pearson x/y -0.063/-0.049 and Spearman -0.076/-0.104; within-workload values
remained small (absolute Spearman at most 0.095).  None of these descriptive
gradients establishes that a representation or configuration causes an outcome.

## 9. UMAP-15 quantitative evidence

Canonical UMAP-15 had recall 0.718/0.542/0.491 and Jaccard 0.589/0.397/0.350 at
`k=10/30/50`.  Across the frozen 27-run grid, recall ranged 0.659--0.735,
0.535--0.655, and 0.473--0.669.  Matched-parameter cross-seed Jaccard ranged
0.603--0.684, 0.560--0.751, and 0.516--0.799, respectively.

| Evidence category | Assessment | Evidence |
| --- | --- | --- |
| Local fidelity | mixed | Stronger than PCA/VAE at `k=10` in some comparisons, but canonical recall trails AE-15 and trails PCA-15 at `k=30/50`. |
| Seed robustness | supportive | Matched-parameter Jaccard means: 0.653, 0.685, 0.709 at `k=10/30/50`. |
| Parameter robustness | mixed | `k=50` recall spans 0.473--0.669 over the fixed grid. |
| Grouped held-out behavior | mixed | 27 zero-overlap fits; mean held-out recall 0.482/0.428/0.410 at `k=10/30/50`, lower than in-cohort canonical values. |
| Metric sensitivity | mixed | The source-level reference sensitivity changes 14--25% of rows materially; no extra unfrozen UMAP-15 metric grid was run. |
| Incremental/complementary information | mixed | It overlaps AE-15 most at `k=10` (Jaccard 0.556) but has much lower local-rank agreement with the frozen methods than their core pairwise agreement. |

UMAP-15 remains non-promoted and human-gated.  No permanent UMAP-15 embedding
companion was added to the promoted artifact contract.

## 10. Cross-method consensus

The PCA/AE/VAE core had pairwise Jaccard 0.451--0.594 at `k=10`, 0.295--0.598 at
`k=30`, and 0.288--0.584 at `k=50`; PCA--VAE was the closest pair.  Their mean
recurring-neighbor count was 8.29, 25.37, and 42.60, with a properly bounded
recurrence fraction (recurring neighbors divided by the per-row candidate-union
size) of 0.542, 0.501, and 0.498.

Adding canonical UMAP-15 is a sensitivity analysis, not a four-method leaderboard.
It changes the bounded recurrence fraction to 0.567, 0.497, and 0.476 and the raw
recurring count to 9.96, 30.38, and 50.00.  UMAP-15 disagreement is greatest in
rank ordering: at `k=10`, local rank agreement with PCA/AE/VAE is
0.275/0.374/0.291, versus 0.869--0.890 among core methods.  The modest exact
workload purity also means the observed agreement should not be described as merely
workload-cell identity.

## 11. Negative results and claim boundaries

- UMAP-2 does not preserve broad original-space neighborhoods particularly well;
  canonical recall declines from 0.695 at `k=10` to 0.318 at `k=100`.
- UMAP-2 local neighborhoods have only moderate cross-seed agreement and appreciable
  frozen-parameter sensitivity; no map was selected as “best.”
- Exact workload-cell purity is modest rather than dominant, so the strong
  workload-first half of the proposed hierarchy is unsupported.
- Source-balanced similarity changes a material minority of neighborhoods, making
  broad local conclusions metric-sensitive.
- UMAP-15 does not have uniformly superior local fidelity, and grouped held-out
  recovery is materially lower than in-cohort recovery.
- t-SNE offers no transform, no quantitative consensus role, and no evidence for
  globally meaningful islands or natural regimes.
- Outcome overlays are weak-to-modest descriptive coordinate associations and do
  not imply performance or energy causation.

## 12. Human decisions required

1. Decide whether the canonical UMAP-2 map merits a descriptive dashboard or
   paper role given its local-fidelity and sensitivity limits.
2. Decide whether t-SNE is main-text context or supplement-only; its role should
   remain local robustness context.
3. Review the mixed UMAP-15 evidence before any quantitative promotion decision.
4. Decide whether a clearly experimental permanent UMAP-15 companion is warranted;
   none is required by this PR.
5. Decide whether the results are sufficient to authorize PR 3 dashboard
   integration.  Stage 5 is intentionally absent from the active artifact index.
6. Decide whether validation on an untouched future snapshot should precede any
   stronger publication claim.
