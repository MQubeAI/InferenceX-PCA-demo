# InferenceX PCA Demo: Current-State Audit

**Audited:** 2026-07-17
**Code reviewed:** `apps/inferencex_pca_demo.py` at `917689a`
**Scope:** the current local checkout and its available, ignored CSV export. No application behavior was changed.

## Executive state

This is a one-module Streamlit application for descriptive configuration/workload analysis of InferenceX benchmark results. It loads `benchmark_results.csv` and `configs.csv`, performs a validated many-to-one join on `config_id`, lets the user choose an analysis unit, and exposes PCA plus a Random Forest regression baseline. The primary CSV workflow ran successfully with the available local data.

The current default analysis is **Median aggregate per config/workload/concurrency**. It reduced 79,830 joined benchmark rows to 7,462 rows. The default PCA and grouped Random Forest completed with the local data, but their current results do not match the fixed results in the README. The biggest correctness concern is that both modeling UIs allow unsafe user-selected columns despite guidance that says otherwise; the metric-predictor checkbox does not enforce its stated behavior.

## Architecture and data flow

```text
local CSVs (preferred) or JSON fallback
  -> normalize/prefix configs -> validated left join on config_id
  -> selected analysis unit (raw/latest/median/one per config)
  -> lazy dashboard section
       -> data understanding / PCA / Random Forest / session-state summaries / downloads
```

- `load_joined_data` uses CSV when both required files exist. Otherwise it searches the selected directory, the default dump name, other `inferencex-dump-*` directories, and `DUMP_DIR` for the two JSON files.
- Config `id` is renamed to `config_id`; non-key config fields receive `config_` prefixes. The join uses `validate="many_to_one"`, which correctly rejects duplicate config keys.
- Optional small tables can be loaded only after a checkbox action and only when under 10 MiB. They are summarized by table/row/column count, not joined into the analysis.
- The app is one 3,523-line module. Each dashboard section is selected by a radio control and renders only when selected.

## File inventory

| Path | State and purpose |
| --- | --- |
| `apps/inferencex_pca_demo.py` | All loading, aggregation, analysis, UI, exports, and narrative logic. |
| `README.md` | Setup, data contract, methodology, historical result claims, and future ideas. Several claims are stale; see below. |
| `requirements-streamlit.txt` | Five minimum-version runtime dependencies; no lockfile or development/test tooling. |
| `.gitignore` | Excludes local data, dumps, exports, environments, caches, secrets, and common generated files. |
| `inferencex-pca-data/` | Present locally and ignored. It was not staged or modified. |
| `docs/` | Added by this audit only. |

There are no committed tests, test configuration, CI workflow, formatting/lint configuration, packaging metadata, or project scripts. The tracked project initially contained four files.

## Local data observed

| CSV | Rows | Columns | App use |
| --- | ---: | ---: | --- |
| `benchmark_results.csv` | 79,830 | 55 | Required; loaded. |
| `configs.csv` | 1,662 | 18 | Required; loaded and joined. |
| `availability.csv` | 6,239 | 9 | Optional, count-only when requested. |
| `eval_results.csv` | 2,083 | 16 | Optional, count-only when requested; not joined. |
| `run_stats.csv` | 7,067 | 6 | Optional, count-only when requested. |
| `workflow_runs.csv` | 857 | 12 | Optional, count-only when requested. |
| `changelog_entries.csv` | 627 | 8 | Optional, count-only when requested. |

The required join produced 79,830 rows and 72 columns, with no unmatched benchmark config rows and no duplicate config IDs in `configs.csv`. Of the 1,662 configurations in the config export, 1,197 appear in benchmark results. The available data has one `benchmark_type`, no non-null `error` values, and 77,212 rows that belong to repeated `config_id`/`benchmark_type`/`isl`/`osl`/`conc` keys. Several telemetry/energy metrics are almost entirely missing (for example, temperature metrics are 99.99% missing).

### Analysis-unit behavior

| UI choice | Observed rows | Behavior |
| --- | ---: | --- |
| Raw benchmark rows | 79,830 | Adds `row_count_in_group = 1`. |
| Latest row per config/workload/concurrency | 7,462 | Selects latest parsed `date`; uses final stable-order row if no recognized timestamp field. |
| Median aggregate per config/workload/concurrency | 7,462 | Median for metric numeric fields; mode/first or median for other columns; default. |
| One row per config | 1,197 | Aggregates all workload coverage for a config into one synthetic row. |

## Implemented and working

- CSV-first loading and JSON-record/JSONL fallback, nested JSON normalization, path/status UI, and visible missing-data error handling.
- Config prefixing and a validated `benchmark_results.config_id = configs.id` many-to-one left join.
- The four analysis-unit choices, row-count metadata, repeat-group reporting, and date-aware latest-row selection.
- Feature dictionary, data-quality summary, numeric/categorical summaries, coverage matrices, workload tables/charts, and five data-understanding downloads.
- Default PCA preprocessing: median imputation and standardization for numeric features; most-frequent imputation and one-hot encoding for categoricals; up to 30 encoded categories per source feature.
- PCA scatter, explained-variance table/chart, encoded and source-feature contribution summaries, signed loading cards, target correlations, and persisted session-state artifacts.
- Random Forest regression with a preprocessing pipeline, deterministic seed, grouped `config_id` split when selected, holdout R2/MAE, and held-out permutation importance.
- Findings and Sales Pitch sections, including CSV/Markdown downloads. Downloads are generated in-browser; the app does not write export files to the repository.
- Lazy section rendering and cached joined data, analysis frames, and optional-table reads.

## PCA audit

### Current default run (observed)

Settings: median analysis unit, seed 42, all 7,462 analysis rows, default feature choices, 19 source features, and 50 encoded dimensions.

- PC1–PC5 explained variance: **28.25%, 13.51%, 8.44%, 7.72%, 6.87%** (64.78% cumulative).
- The top source-feature contributions across those five PCs were disaggregated serving, multinode serving, decode tensor parallelism, prefill expert parallelism, prefill tensor parallelism, and prefill GPU allocation.
- The PCA completed and the section rendered with no Streamlit exception.

### Correct implementation details

- Outcome metrics are not in the default selection and are used as color/correlation overlays by default.
- Numeric preprocessing is fitted on the selected PCA sample; no target is used to fit PCA.
- The contribution calculation sums squared loading times explained-variance ratio across the first five fitted components, then normalizes across those retained components. It is a retained-PC structural ranking, not a full-spectrum feature-importance measure.
- One-hot encoded columns are regrouped to their source feature for the source-feature contribution table.

### Risks and limitations

1. **Unsafe PCA selections are permitted.** The PCA numeric and categorical widgets offer all qualifying columns. Metric/outcome selection merely produces a warning; IDs and other numeric metadata can also be selected. This contradicts the stated invariant that PCA inputs are configuration and workload fields only.
2. **Feature-group contributions are encoding-sensitive.** Numeric fields are standardized, while categorical fields contribute multiple unscaled 0/1 columns. One source feature's contribution is the sum of its encoded levels, so categorical cardinality and the `max_categories=30` bucketing rule affect rankings.
3. **Only the first five components are considered for the displayed contribution ranking.** The label should not be read as a contribution to all input variance.
4. **No stability analysis exists.** There is no resampling, bootstrap, sensitivity grid across analysis units/feature selections, component alignment, or uncertainty reporting. Changing the sampled rows, category frequencies, or analysis unit can change loadings and rankings.
5. **Signed component narratives are unstable by construction.** PCA component signs may reverse without changing the solution. The sales-axis naming code uses signs and hard-coded feature patterns, so “high side/low side” and named axes should be treated as presentation labels, not stable empirical findings.
6. **Session-state artifacts can be stale.** Findings does not validate that a prior PCA result matches the current controls. Sales validates only analysis-unit name and row count, not feature selection, seed, or row limit.

## Supervised-model audit

### Current default run (observed)

Settings: median analysis unit, target `metrics_p99_itl`, default 19 predictors, seed 42, 150 trees, `min_samples_leaf=2`, grouped split by `config_id`, 5 permutation repeats.

- Training/test rows: **5,717 / 1,745**.
- Training/test config groups: **897 / 300**; group overlap was **0**.
- Test R2: **0.691**; test MAE: **0.534**.
- Leading held-out permutation importances were `conc`, `config_framework`, `config_spec_method`, `config_hardware`, `isl`, and `config_precision`.

The preprocessing is inside the fitted `Pipeline`, so imputation, scaling, and categorical encoding are learned from training rows only. The grouped split genuinely holds out complete `config_id` groups for the observed default run.

### Risks and limitations

1. **The safety control does not enforce safety.** “Allow other metric-like columns as predictors” changes neither widget options nor the effective default predictor list. Other outcome metrics remain selectable even when unchecked. Selecting correlated outcome metrics creates direct target leakage risk.
2. **IDs and metadata can be selected.** The predictor widgets expose all numeric/categorical candidates other than the selected target, including numeric identifiers and provenance-like fields. The feature dictionary describes these as excluded, but the training UI does not enforce that classification.
3. **Grouped validation is optional.** The UI offers a random split, which can mix repeated or near-identical configurations between train and test. The default grouped option is safer, but users can change it.
4. **A single group holdout is not a robust performance estimate.** There is no repeated/grouped cross-validation, confidence interval, baseline comparison, target transformation/outlier review, or per-workload evaluation.
5. **Permutation importance is descriptive and correlation-sensitive.** Correlated configuration and workload predictors can share or suppress importance; negative values are displayed without interpretation. The model uses only one selected target and one holdout.
6. **Sales auto-model differs from the target tab.** It recomputes a default 150-tree model with three permutation repeats and `n_jobs=1`; the Target-Aware tab uses the user tree count, five repeats, and `n_jobs=-1`. A similarly named result can therefore differ.
7. **One-row-per-config changes the prediction question.** Its target aggregates across unequal workload coverage, while workload fields collapse to medians/modes; it should not be interpreted as a benchmark-row prediction result.

## UI, controls, charts, caches, exports, and error states

| Section | Controls and outputs | Observed runtime state |
| --- | --- | --- |
| Data Preview | Counts, column summary, first 100 joined rows. | Rendered; 4 dataframes. |
| Data Understanding | Optional side-table loader, family filters, coverage filters, summaries, quality report, five downloads. | Rendered; 17 dataframes, 5 charts, 5 downloads. Side tables are not integrated. |
| PCA Explorer | Numeric/categorical feature multiselects, color overlay, correlation target, loadings component. | Rendered; 7 dataframes, 5 charts. Saves session-state PCA artifacts. |
| Target-Aware Feature Value | Target/predictor widgets, split mode, tree-count slider; metrics and permutation chart. | Rendered; 3 dataframes and 1 chart. Saves session-state target artifacts. |
| Findings | Session-state summary, overlap, six conditional downloads. | Rendered; 6 dataframes and 6 downloads after PCA/model run. |
| Sales Pitch Visuals | Reuses PCA, selects overlay metric, adds clipped color map, may auto-compute target model, five downloads. | Rendered; 6 dataframes, 2 charts, 5 downloads; correctly warns that color is not a PCA input. |
| Notes | Static methodology/deployment text. | Rendered. Its statement that JSON is the default source is inaccurate. |

Global sidebar controls are data directory, analysis unit, maximum PCA/model rows (500–100,000; default 20,000), and seed. Missing required CSV and fallback JSON files generate a visible error and return before tabs. PCA has visible “no features”, too-few-row, preprocessing, and too-few-dimensions states. The model has no-metric, no-predictor, too-few-target-row, split fallback, fitting-error, and duplicate-predictor states. These were statically reviewed; the normal-data path for every section was executed.

### Runtime and maintenance findings

- Headless Streamlit started successfully and `/_stcore/health` returned `ok`.
- Streamlit's in-process harness executed all seven normal-data sections with no app exceptions or UI errors.
- The runtime emitted repeated Streamlit 1.58 deprecation warnings: `use_container_width` should be replaced before its removal. This audit does not refactor it.
- Data Understanding triggered an Arrow conversion fallback for a mixed-type dataframe column. Streamlit recovered automatically; it is a serialization/performance warning, not a rendered UI failure.
- CSV/data loading and optional data reads are cached by the supplied path string, not file content or modification time. Replacing local CSVs at the same path during a server process can leave stale cache results.
- Loading retains raw benchmarks, configs, and joined data concurrently; analysis functions copy frames. PCA uses a dense encoded matrix, bounded only by user row limit and category cap. The defaults were practical for the observed data, but the 100,000-row maximum has no memory estimate or guard.
- The module mixes data access, schema rules, computations, UI, prose, and export construction. That makes regression testing and change review harder.

## Implemented but undocumented

- JSONL parsing and several JSON mapping shapes; search of local dump directories and `DUMP_DIR`.
- One-row-per-config analysis unit and latest-row timestamp fallback behavior.
- Feature dictionary, data-quality reporting, coverage matrices, optional-table size guard, and all generated export names.
- Caching, sampling, row-limit, seed, category cap, and current session-state reuse rules.
- Sales Pitch Visuals and its automatic target-model execution are not described in the README layout/workflow.

## Documented but not implemented (or not operational in this app)

- README describes optional side tables as supported later; they can load within the size guard but are neither displayed as records nor joined into analysis.
- README's cloud mounting, cost/token valuation, coverage-gap analysis, quality/performance tradeoff work, and TabFM/TabPFN/TabICL comparison are future items, not current features.
- `eval_results.csv` is not joined, so the stated quality/performance analysis is not available.
- The README's fixed “Current PCA Findings,” PC interpretations, and target-model numbers are not generated, pinned, or reproducible from the present default code/data.

## README versus code

- **Correct:** CSV-first setup, required files, core join, default median unit, ignored local data, PCA/model purpose, grouped-split rationale, and high-level limitations broadly match code.
- **Stale numeric claims:** README says PC1–PC4 are about 28.8%, 16.1%, 11.2%, and 8.3%, first five 70.8%, and a p99 ITL RF result of R2 0.783/MAE 0.151. The current default run produced 28.25%, 13.51%, 8.44%, 7.72%, first five 64.78%, and R2 0.691/MAE 0.534.
- **Stale substantive claims:** the current top PCA source features begin with disaggregation and multinode fields, rather than the README's fixed ordering; current top model predictors include framework and hardware more prominently than the README claims.
- **Contradiction in code:** `render_notes` and one generated-findings limitation say the app reads JSON by default/only, while the actual loader and README use CSV first.
- **Incomplete layout:** README does not mention the actual seven-section dashboard, `docs/`, the current data-understanding layer, or the sales visual layer.

## Prioritized issues

1. **High — enforce feature allowlists in PCA and supervised widgets.** Do not merely warn when outcome metrics, IDs, provenance, or metadata are selected. Make “allow other metrics” actually control availability.
2. **High — replace README's fixed results and interpretations with a versioned, reproducible run record or remove them.** Current claims are inaccurate for the available data/code.
3. **High — add grouped, repeated validation and leakage tests before treating RF values as stable.** Keep `config_id` group isolation mandatory for comparative reporting.
4. **Medium — add PCA stability/sensitivity reporting.** At minimum compare samples, analysis units, and feature sets with component alignment; do not present hard-coded axis narratives as stable facts.
5. **Medium — define and validate schemas and join coverage.** Fail clearly for missing key/type fields and report unmatched configs/benchmark rows, rather than only relying on merge validation.
6. **Medium — prevent stale session-state cross-tab results.** Key artifacts by source fingerprint, analysis unit, row limit, seed, features, target, and model settings.
7. **Medium — modularize computations and add fixture-based tests.** Separate I/O/schema, aggregation, PCA, RF, and UI/export functions.
8. **Low — resolve Streamlit deprecations and Arrow mixed-type warning; add dependency pinning/lock strategy.**

## Clean future insertion points

### TabFM

Add a model adapter beside the existing `render_target_feature_value` pipeline, not inside PCA. Reuse the analysis-frame builder, the enforced feature allowlist, same target eligibility, and the same `config_id` split object. Return the existing target-analysis contract (target, feature manifest, split manifest, predictions, metrics, held-out importance/equivalent explanation) so Findings can compare models without conflating PCA structure with supervised performance. Keep it optional and do not change Random Forest baseline behavior until a planned model-comparison pass.

### Conditional residual VAE

Add only after target/schema contracts and grouped evaluation are stable. Place a residual-generation adapter after a baseline prediction is trained on a group-isolated training fold: condition on approved configuration/workload fields and model residuals, then evaluate only on withheld config groups. Store generated/imputed values, uncertainty, conditioning manifest, seed, and fold identity separately from benchmark facts. Integrate through a new future analysis section and export contract; do not feed generated targets into PCA or overwrite measured outcomes.

## Validation performed

- Inspected Git state/history, tracked files, ignored rules, worktrees, source, README, requirements, local data filenames/sizes/schemas, and absence of tests/scripts.
- Loaded the actual CSVs and verified source selection, join cardinality, join coverage, data-quality counts, all analysis-unit counts, default PCA, and default grouped RF run.
- Ran Python syntax compilation and module import using the existing `.venv-streamlit` (Python 3.14.5; Streamlit 1.58.0; pandas 3.0.3; NumPy 2.5.0; scikit-learn 1.9.0; Plotly 6.8.0).
- Started Streamlit headlessly, verified health, and exercised every dashboard section with Streamlit's in-process test harness.
- No committed automated tests exist, so there was no test suite to run. Browser-driven visual inspection was not performed because the in-app browser control surface was unavailable in this session; the section harness and health endpoint are the runtime evidence reported here.
