# InferenceX official dump comparison

Audit date: 2026-07-22

Project baseline: local `inferencex-pca-data/` export corresponding to `db-dump/2026-06-29`

Comparison release: `db-dump/2026-07-20`

## Decision

**B. Newer dump exists, but the previous stop condition still applies.**

The official July 20 dump adds 2,021 benchmark rows, 171 benchmarked configurations, 1,381 energy-measured rows, and 54 net-new energy-measured configuration IDs. It does **not** broaden the energy workload domain: measured energy remains limited to `single_turn`, OSL 1024, and ISL 1024 or 8192. Hardware, framework, model, speculative-method, disaggregation, multinode, and concurrency value coverage for energy is otherwise unchanged; only BF16 is newly represented, with just 20 rows from 2 configurations.

The earlier statement that the target formula and measurement boundary were undocumented can now be narrowed. The current official benchmark implementation explicitly defines the load-window boundary and formulas, and the implementation has been present since 2026-05-26, before the first energy row. However, the dump does not pin a metric-code version on each benchmark row, and the measured sample remains too narrow and too concentrated to support the intended general estimator. No model was trained.

## Authoritative source and release

The authoritative source is the official InferenceX site and its linked GitHub release repository:

- The [official InferenceX About page](https://inferencex.semianalysis.com/about) says that weekly full-database snapshots are published as public GitHub Releases and links directly to the release feed.
- That link resolves to the [official `SemiAnalysisAI/InferenceX-app` database-dump releases](https://github.com/SemiAnalysisAI/InferenceX-app/releases?q=db-dump).
- The newest release returned by GitHub's official releases API on 2026-07-22 was [DB Dump 2026-07-20](https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20): release ID `356485166`, tag `db-dump/2026-07-20`, release commit `13230c5`, published `2026-07-20T03:12:26Z`.
- The official benchmark repository states that only `SemiAnalysisAI/InferenceX` contains official results. No mirror or third-party dataset was used.

The July 20 release is newer than the June 29 project baseline.

### Download and integrity

The assets were downloaded to the untracked temporary directory:

`/tmp/inferencex-dump-comparison/db-dump-2026-07-20/`

| Asset | Bytes | Published SHA-256 | Verification |
|---|---:|---|---|
| `inferencex-2026-07-20.dump.zst.part00` | 1,992,294,400 | `0f1c5f4631cdc8d69de2b31ae1abeadb9ceba9c01408559cfa3eafcfabf0719e` | OK |
| `inferencex-2026-07-20.dump.zst.part01` | 1,882,650,124 | `672390222e46826cc66b90ea5603168196fb2da4f57d19af201117065ecdb935` | OK |
| `SHA256SUMS` | 208 | `39792272da94662490c062dd0b306219ca0571bf5b2ff716d5a195325ec65425` | OK |

The two parts reconstructed a 47,116,627,455-byte PostgreSQL custom dump. `pg_restore --list` successfully read a PostgreSQL 17.10 archive created on 2026-07-20 at 00:42 UTC. The archive contained the expected `configs` and `benchmark_results` tables. Those two tables alone were restored into an isolated PostgreSQL 17 Docker database for this audit:

- `configs`: 1,935 rows, 18 columns
- `benchmark_results`: 81,851 rows
- every benchmark row joined to a configuration; no missing configuration join was found
- all 81,851 `metrics` values parsed as JSON objects

The configuration schema is unchanged. The new raw benchmark table retains the expected identity, workload, date, image, error, and JSONB metrics fields and adds `offload_mode` and `trace_replay_id`; these do not affect the project's aggregation keys. No project data was overwritten.

For audit reproducibility, the temporary CSV extractions have these hashes:

- `benchmark_results_raw.csv`: `cab2f167e4b63a29ff84032413054244b5005ead1242a75c1c42d4ee9b0aaa91`
- `configs.csv`: `827fad4296c602ce010259b2ab7540420b8900ff869fb2578097293f328e29c4`

The local baseline inputs were left unchanged:

- `inferencex-pca-data/benchmark_results.csv`: `c75cd533970eb11603763973c7c6d273654f7b6fa67719e474db818b515a0b85`
- `inferencex-pca-data/configs.csv`: `38146e652ba71a23c0abe95e109a3408615471f26f231fa95e8bb71e0c9b391f`

## Comparison method

The comparison follows the dashboard's current loading and aggregation conventions:

1. Flatten the new dump's `metrics` JSONB object to `metrics_*` columns, matching the existing CSV export.
2. Rename configuration `id` to `config_id` and prefix all other configuration fields with `config_`.
3. Join benchmark rows to configurations on `config_id` with a many-to-one validation.
4. Group by `config_id`, `benchmark_type`, `isl`, `osl`, and `conc`, retaining null workload keys.
5. Use median for metric columns, the existing numeric configuration reducer for numeric configuration fields, mode/first for categorical fields, and first non-null for metadata.
6. For the energy-specific aggregate, first exclude null, non-finite, zero, and negative `metrics_joules_per_output_token` values, then apply the identical grouping and median policy.

This reproduced all documented baseline counts before the new dump was trusted: 79,830 raw rows, 1,197 benchmarked configurations, 7,462 aggregate rows, 3,794 usable energy rows, 2,135 energy aggregate groups, and 251 energy-measured configurations.

## Overall old-versus-new summary

| Measure | 2026-06-29 baseline | 2026-07-20 dump | Change |
|---|---:|---:|---:|
| Raw benchmark rows | 79,830 | 81,851 | +2,021 |
| Median config/workload/concurrency rows | 7,462 | 8,239 | +777 |
| Benchmarked `config_id` values | 1,197 | 1,368 | +171 |
| Configuration-table rows | 1,662 | 1,935 | +273 |
| Benchmark date range | 2025-09-29–2026-06-26 | 2025-09-29–2026-07-19 | +23 days at end |
| Benchmark types | 1 (`single_turn`) | 2 (`single_turn`, `agentic_traces`) | +`agentic_traces` |
| Models | 10 | 11 | +`glm5.2` |
| Hardware values | 9 | 9 | unchanged |
| Frameworks | 11 | 12 | +`llmd-vllm` |
| Precisions | 4 | 4 | unchanged |

All 79,830 baseline benchmark IDs are still present. No baseline IDs were removed, and all 3,794 pre-existing output-energy values are byte-equivalent numerically in the new dump. The apparent integer/float representation difference for ISL/OSL after CSV parsing does not change values or grouping.

The 2,021 new rows consist of:

- 216 `agentic_traces` rows with null ISL/OSL
- 867 `single_turn` 1024/1024 rows
- 938 `single_turn` 8192/1024 rows

The overall corpus adds `glm5.2` and `llmd-vllm`, but neither has an energy measurement in this dump. No new hardware or precision value was added overall.

## Energy comparison

| Measure | 2026-06-29 baseline | 2026-07-20 dump | Change |
|---|---:|---:|---:|
| Usable output joules/token rows | 3,794 | 5,175 | +1,381 (+36.4%) |
| Average-power rows | 3,794 | 5,175 | +1,381 |
| Input joules/token rows | 50 | 50 | unchanged |
| Total joules/token rows | 3,794 | 5,175 | +1,381 |
| Median energy aggregate groups | 2,135 | 2,766 | +631 |
| Energy-measured `config_id` values | 251 | 305 | +54 (+21.5%) |
| Measured share of benchmarked configs | 20.97% | 22.30% | +1.33 percentage points |
| Energy date range | 2026-05-27–2026-06-26 | 2026-05-27–2026-07-18 | +22 days at end |
| Energy benchmark types | `single_turn` | `single_turn` | unchanged |
| Energy workload shapes | 1024/1024, 8192/1024 | 1024/1024, 8192/1024 | unchanged |
| Energy concurrency values | 15 values, 1–21,504 | same 15 values | unchanged |
| Energy models | 9 | 9 | unchanged |
| Energy hardware values | 8 | 8 | unchanged |
| Energy frameworks | 7 | 7 | unchanged |
| Energy precisions | FP4, FP8, INT4 | BF16, FP4, FP8, INT4 | +BF16 only |
| Speculative methods | `none`, `mtp` | `none`, `mtp` | unchanged |
| Disaggregation | false/true | false/true | unchanged |
| Multinode | false/true | false/true | unchanged |

The measured energy domain remains exactly:

- `benchmark_type = single_turn`
- `osl = 1024`
- `isl ∈ {1024, 8192}`
- concurrency values `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 21504`

The grouped support is almost evenly split by workload: 1,391 groups / 280 configs at 1024/1024 and 1,375 groups / 282 configs at 8192/1024. A configuration has a median of 9 measured workload/concurrency groups, but the range is 1–22.

### What the added energy rows cover

The 1,381 added energy rows span 121 configuration IDs in the added rows, but the net measured-config set grows by 54: 35 previously benchmarked configuration IDs gain energy coverage, and 19 newly benchmarked IDs have energy. The added rows are all single-node, non-disaggregated runs.

Added energy rows by major category:

| Category | Added rows | Config IDs represented in added rows |
|---|---:|---:|
| MiniMax M3 | 578 | 44 |
| DeepSeek V4 | 406 | 36 |
| Qwen 3.5 | 240 | 25 |
| Kimi K2.5 | 101 | 8 |
| DeepSeek R1 | 36 | 6 |
| GLM 5.1 | 20 | 2 |
| MI355X | 558 | 45 |
| B200 | 349 | 35 |
| B300 | 255 | 25 |
| H200 | 117 | 8 |
| vLLM | 873 | 71 |
| SGLang | 398 | 42 |
| ATOM | 110 | 8 |

BF16 is the sole new energy category value and is represented by only 20 rows and 2 configurations. No new energy-measured model, hardware, framework, workload shape, concurrency, speculative method, disaggregation state, or multinode state appears.

Energy coverage remains concentrated:

- vLLM: 72.04% of measured rows (184 configurations)
- MiniMax M3: 54.94% of measured rows (112 configurations)
- MI355X: 36.41% of measured rows; B200 25.04%; B300 21.01%
- FP8: 51.05%; FP4: 46.38%; INT4: 2.18%; BF16: 0.39%
- non-disaggregated and single-node: 5,125/5,175 rows each (99.03%)

Topology fields contain sparse levels that cannot support meaningful train/test coverage. Examples include prefill worker counts 2, 4, 8, 10, and 12 with one configuration each; decode TP/EP 12 with one configuration; prefill GPU counts 32, 40, and 48 with one configuration each; decode GPU count 12 with one configuration; and concurrency 21,504 with one configuration. A row-wise increase does not fix these grouped-support gaps.

## Target-definition and integrity audit

### Official definition

The [official `utils/aggregate_power.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/utils/aggregate_power.py) defines:

- `avg_power_w` as mean per-GPU power during the benchmark load window delimited by the benchmark start/end Unix timestamps
- total system energy as `avg_power_w * num_gpus * duration_seconds`
- output joules/token as total system energy divided by actual total output tokens
- total joules/token as total system energy divided by actual input plus output tokens

Git history shows the file was introduced at commit `7a8a5ab133f20911b7f43d4e70787890e5dfa293` on 2026-05-26 and its metric formula has remained in place through the audit checkout. This precedes the first measured energy date, 2026-05-27. Thus output and total energy are mechanically related, but they are not duplicate targets. The source definition resolves the earlier formula ambiguity; per-row code-version provenance is still absent from the two restored analysis tables.

### Values and relationships

| Statistic | Baseline | New dump |
|---|---:|---:|
| Nonpositive output target rows | 0 | 0 |
| Minimum output J/token | 0.078524 | 0.078524 |
| Median | 3.144111 | 3.115596 |
| P90 | 16.914642 | 15.468512 |
| P95 | 27.768971 | 25.882398 |
| P99 | 70.484538 | 56.660668 |
| Maximum | 152.098103 | 152.098103 |
| Output/total overlap | 3,794 | 5,175 |
| Exactly equal output/total rows | 0 | 0 |
| Within 1% of equality | 0 | 0 |
| Output-total correlation | 0.7996 | 0.7926 |

All available output, total, input, and average-power values are positive. In the new dump, total J/token ranges 0.020516–75.671105, average power ranges 223.835–1,212.036 W, and the 50 input J/token values range 0.076238–2.196511.

Using nominal ISL/OSL, `total/output` remains close to `osl / (isl + osl)` for most rows:

- 1024/1024: median ratio 0.499887 versus nominal 0.5; maximum absolute deviation 0.005800
- 8192/1024: median ratio 0.111229 versus nominal 0.111111; P99 deviation 0.074209; maximum deviation 0.439563
- across all energy rows, median absolute deviation is 0.000717; 59 rows exceed 0.01, 48 exceed 0.05, and 14 exceed 0.1

These deviations remain notable, but the official formula uses actual aggregate token counts rather than nominal ISL/OSL. The restored benchmark rows do not carry actual token totals for the affected energy rows, so this audit cannot deterministically reconcile each outlier. The deviations are not evidence that output and total energy are duplicates.

Pearson correlations with output J/token in the new measured subset are: total J/token 0.7926, average power -0.2667, total throughput/GPU -0.2736, input throughput/GPU -0.2433, output throughput/GPU -0.2624, concurrency -0.1179, and ISL 0.0765. Input J/token correlation is 0.9425 but is based on only 50 rows and is not reliable evidence for modeling. Power, throughput, latency, and other energy fields remain leakage-prone and must not be model inputs.

### Temporal behavior

| Month | Energy rows | Config IDs | Median output J/token | P90 | Range |
|---|---:|---:|---:|---:|---:|
| 2026-05 | 524 | 67 | 2.181585 | 8.917111 | 0.100753–56.944418 |
| 2026-06 | 3,387 | 206 | 3.319405 | 18.038598 | 0.078524–152.098103 |
| 2026-07 | 1,264 | 110 | 3.044566 | 13.011116 | 0.120370–86.995097 |

No invalid-value or formula-regime break appears in July. The upstream aggregation formula predates the full measured period. Monthly distribution shifts are material enough to require date-aware evaluation, but they are confounded by changing configuration composition and do not by themselves show an instrumentation failure.

A July temporal holdout is mechanically possible (1,264 rows, 110 configurations), but only 56 July configurations also occur before July (50.9%). July covers 6 of 9 measured models, 7 of 8 hardware values, only 3 of 7 frameworks, and introduces the two-config BF16 subgroup. This is useful for a tightly scoped robustness check, not a broad temporal generalization claim.

## Modeling-support assessment

| Question | Finding |
|---|---|
| Grouped validation by `config_id` feasible? | Mechanically yes with 305 groups, but many categorical/topology levels are too sparse for stable subgroup evaluation. |
| More workload shapes? | No. Energy still covers only two ISL/OSL shapes and one benchmark type. |
| Broader date coverage? | Yes, by 22 days; still only 53 calendar days and three partial months. |
| Broader energy hardware/framework/model domain? | No new values in any of those fields. |
| Less concentration? | Not materially: 72% vLLM, 55% MiniMax M3, 99% single-node/non-disaggregated. |
| Temporal holdout feasible? | Only as a narrow July stress test; framework and repeat-configuration support are incomplete. |
| Input-field support sufficient? | No. Several worker, TP/EP, GPU-count, precision, and high-concurrency values have one to a few config groups. |
| Target formula defined? | Yes in current official source; historical per-row implementation provenance remains implicit. |
| Leakage risk resolved? | No. Power, throughput, latency, token-derived energy, and related metrics remain forbidden inputs. |

The extra rows improve precision for configurations already close to the dominant measured domain, but they do not resolve the external-validity problem. A grouped tree-model comparison would answer only whether the model can interpolate within two fixed single-turn workloads and a heavily concentrated configuration population. It would not justify the dashboard estimator originally proposed.

## Recommendation and remaining blockers

Keep energy prediction blocked and retain observed-only dashboard behavior.

The binding blockers are:

1. no measured energy outside `single_turn`, OSL 1024, and ISL 1024/8192;
2. no newly measured hardware, framework, or model category despite the larger dump;
3. extreme concentration in vLLM, MiniMax M3, single-node, and non-disaggregated runs;
4. sparse grouped support for several serving-topology values and the new BF16 category;
5. only a short 53-day energy window, with limited repeated-configuration and framework coverage in a July holdout;
6. notable nominal output/total-ratio deviations that cannot be resolved per row without actual token totals or linked raw artifacts;
7. no evidence yet that grouped predictive performance is stable—training was deliberately not run because the support audit fails first.

Revisit the stop condition only after an official dump adds genuinely new measured workload shapes and broader, repeated configuration support across frameworks/hardware/topologies. The newly discovered official formula should be cited in future target documentation, and the data pipeline should preserve the benchmark code identity and actual input/output token totals needed to audit every energy row.

## Commands used

All shell commands were run with the repository-required `rtk` prefix. The principal reproducible commands were:

```bash
rtk proxy curl -fsSL https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/releases/latest

rtk proxy mkdir -p /tmp/inferencex-dump-comparison/db-dump-2026-07-20
rtk proxy curl -fL --retry 3 --continue-at - -O \
  https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/2026-07-20/SHA256SUMS
rtk proxy curl -fL --retry 3 --continue-at - -O \
  https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/2026-07-20/inferencex-2026-07-20.dump.zst.part00
rtk proxy curl -fL --retry 3 --continue-at - -O \
  https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/2026-07-20/inferencex-2026-07-20.dump.zst.part01
rtk proxy shasum -a 256 -c SHA256SUMS
rtk proxy sh -c 'cat inferencex-2026-07-20.dump.zst.part00 inferencex-2026-07-20.dump.zst.part01 | zstd -d --long=27 -o inferencex-2026-07-20.dump'

rtk proxy docker run -d --name inferencex-dump-audit-20260720 \
  -e POSTGRES_PASSWORD=audit-only \
  -v /tmp/inferencex-dump-comparison/db-dump-2026-07-20:/dump postgres:17
rtk proxy docker exec inferencex-dump-audit-20260720 pg_restore --list \
  /dump/inferencex-2026-07-20.dump
rtk proxy docker exec inferencex-dump-audit-20260720 createdb -U postgres inferencex_audit
rtk proxy docker exec inferencex-dump-audit-20260720 pg_restore -U postgres \
  --exit-on-error --no-owner --no-privileges --jobs=4 \
  --dbname=inferencex_audit --table=configs --table=benchmark_results \
  /dump/inferencex-2026-07-20.dump
rtk proxy docker exec inferencex-dump-audit-20260720 psql -U postgres \
  -d inferencex_audit -c "\\copy configs TO '/dump/configs.csv' CSV HEADER"
rtk proxy docker exec inferencex-dump-audit-20260720 psql -U postgres \
  -d inferencex_audit -c "\\copy benchmark_results TO '/dump/benchmark_results_raw.csv' CSV HEADER"

rtk proxy git clone --depth=1 --filter=blob:none \
  https://github.com/SemiAnalysisAI/InferenceX.git \
  /tmp/inferencex-dump-comparison/InferenceX-official
rtk git -C /tmp/inferencex-dump-comparison/InferenceX-official fetch --unshallow
rtk git -C /tmp/inferencex-dump-comparison/InferenceX-official log --follow \
  --format='%H %cI %s' -- utils/aggregate_power.py

rtk proxy .venv-streamlit/bin/python - <<'PY'
# Loaded the old flattened CSVs and the restored JSONB dump; flattened metrics
# with pandas.json_normalize; invoked the project join/grouping conventions;
# compared IDs, domains, grouped medians, distributions, correlations,
# formula residuals, category concentrations, and temporal support.
PY
```

Validation commands run after report creation are recorded in the final task response.
