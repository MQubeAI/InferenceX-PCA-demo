# Automated InferenceX snapshot refresh

## Scope and safety boundary

The active dashboard checkpoint is always the root [`data-manifest.json`](../data-manifest.json).
It is immutable in normal use. A newly published upstream dump is only a **candidate** until a
reviewed pointer-change PR is merged. Discovery, candidate construction, release publication, and
promotion are separate operations.

The preserved July baseline is represented twice on purpose:

- `data-manifest.json` is the backward-compatible active runtime contract;
- `data/manifests/2026-07-20.json` is the historical versioned manifest.

Older manifests and releases are never overwritten. `data/artifact-index.json` points the dashboard
at the matching promoted research artifacts. July uses committed repository artifacts; future
snapshots may use verified release bundles installed below ignored `.artifacts/`.

## Normal CI

`.github/workflows/ci.yml` is the permanent known-good control. It starts with no `.data`, installs
the locked Python 3.11 runtime, bootstraps the active bundle, verifies the ZIP and each CSV hash,
runs all unit/semantic/PCA/representation tests, runs the July semantic audit, and finally performs
the clean-bootstrap AppTest plus headless Streamlit health check. It never trains a model.

## Discovery

`scripts/check_inferencex_release.py` queries only the public
`SemiAnalysisAI/InferenceX-app` releases API. It accepts non-draft, non-prerelease tags matching
`db-dump/YYYY-MM-DD` and returns exactly one of `NO_NEW_RELEASE`, `NEW_RELEASE_AVAILABLE`, or
`INVALID_RELEASE_METADATA`.

`.github/workflows/check-inferencex-release.yml` runs Monday, Wednesday, and Friday. A same-release
result is a cheap successful no-op. A new release starts no expensive work unless the repository
Actions variable `INFERENCEX_AUTOMATED_REFRESH_ENABLED` is explicitly set to `true`.

## Candidate ingestion

`scripts/build_snapshot_candidate.py` is the candidate-only entry point. It:

1. finds the exact official release and downloads `SHA256SUMS` plus all dump parts;
2. validates each upstream part before restore;
3. requires PostgreSQL 17 server/client tools, zstd, and a configurable free-disk floor (60 GiB
   by default); it starts a private loopback PostgreSQL 17 cluster below the scratch directory
   when an explicit isolated `--database-url` is not supplied;
4. streams concatenated parts through `zstd --long=27 -d -c` directly into `pg_restore`;
5. obtains a TOC first and writes a minimal restore list containing only schemas and the
   `benchmark_results`/`configs` table objects; if PostgreSQL reports an exact missing user
   type/domain/function/extension, it adds only the matching TOC dependency and retries the
   disposable database. It never falls back to a full restore;
6. discovers restored schema/table names and exports ordered `COPY (SELECT * ... ORDER BY id)` CSVs;
7. validates metrics JSON, table primary IDs, unique configs, benchmark-to-config joins, required
   schema, row counts, and full hashes;
8. creates a root-only deterministic ZIP and verifies extracted hashes; and
9. writes a candidate manifest, candidate-vs-active JSON, and Markdown report outside Git.

No decompressed dump file is ever written. A restore failure is terminal: inspect the TOC or extend
the narrowly selected dependencies; do not restore the full database as a workaround.

## Data/schema audit and policy

`modeling.snapshot_refresh.audit_candidate_against_active` compares stable benchmark/config IDs and
normalizes metrics JSON before detecting added, missing, and changed existing IDs. It records row
counts, date/workload/configuration coverage, metric names, target missingness, and required-field
type changes. Schema changes are classified as `SAFE_ADDITIVE`, `REVIEW_REQUIRED`, or
`INCOMPATIBLE`; removal or semantic type change of a required field is incompatible.

`refresh-policy.json` and `evaluate_refresh_policy` reject bad checksums, invalid metrics, broken
joins, leakage, failed artifact construction, non-finite metrics, and failed clean-clone/Streamlit
checks. Performance is reported as `PASS`, `REVIEW_REQUIRED`, or `FAIL`; there is deliberately no
unreviewed numeric threshold that auto-promotes scientific drift.

## Research refresh

`scripts/build_snapshot_pca_artifact.py` creates a snapshot-named PCA artifact using the frozen 19
configuration/workload inputs and deterministic canonical row ordering. Outcomes remain overlays.

`scripts/run_snapshot_research.py --plan` is safe everywhere and records the fixed protocol.
`--execute` is for the dedicated runner: it runs the selected final AE/VAE architecture with seeds
42/123/2026, fixed VAE beta inherited from the promoted protocol (not a new model search), comparison,
and Stage 4. All outputs use `db-dump-YYYY-MM-DD` names. The historical July artifacts are never
rewritten or retrained.

The selected TabFM contract is raw `metrics_tput_per_gpu`, full-context grouped `config_id`
evaluation. This repository currently has evaluation artifacts, not a persisted serving checkpoint.
The runner must supply an approved `INFERENCEX_TABFM_COMMAND` that writes the snapshot-named TabFM
evaluation and uncertainty artifacts. A missing/failed command is a promotion-blocking failure, not
a fallback to July results.

## Candidate release and review PR

The resource-gated workflow is `.github/workflows/refresh-inferencex-candidate.yml`. It runs only
on `[self-hosted, linux, x64, inferencex-refresh]` with PostgreSQL 17 server tools, zstd, 60+ GiB
free space, and the approved TabFM environment. It runs an isolated candidate clean-clone bootstrap
against locally finalized bundles before publishing any candidate asset, then creates immutable
candidate data/research release assets and opens `refresh/db-dump-YYYY-MM-DD` as a **draft** PR
containing only versioned manifest, artifact-index fragment, and reports—not raw CSVs or databases.
It does not alter `data-manifest.json`.

Candidate releases use `dashboard-data-candidate/YYYY-MM-DD`; they are distinct from promoted
`dashboard-data/YYYY-MM-DD` releases and never overwrite an existing tag.

## Human promotion and rollback

After review, run `.github/workflows/promote-inferencex-snapshot.yml` with the candidate date. The
protected `inferencex-promotion` environment (if configured) provides an additional approval gate.
It opens a second `promote/db-dump-YYYY-MM-DD` PR that explicitly changes only the active manifest
and artifact pointer, marking the versioned manifest/index entry promoted. Merging that PR is the
only promotion action.

Rollback is the same explicit pointer operation: select a previously promoted
`data/manifests/YYYY-MM-DD.json`, verify its matching promoted index entry, and prepare a reviewed
pointer PR with `scripts/promote_snapshot.py --snapshot-date YYYY-MM-DD --apply`. It does not delete
or mutate any release.

`post-promotion-verify.yml` runs a clean bootstrap/AppTest/health check after a pointer change on
main. It never silently falls back to the old checkpoint.

## July replay fixture

`scripts/replay_july_snapshot.py --plan` confirms the official July release remains discoverable and
that it matches the active baseline. `--execute --output-root ...` runs the actual
candidate ingestion path against `db-dump/2026-07-20`, creates a new candidate-only package, and
requires exact semantic PCA cohort identity with the active July data. Raw bytes may differ because
future exports are explicitly ordered; the published July release remains untouched.

`.github/workflows/replay-july-snapshot.yml` runs that same ingestion-only fixture on the approved
resource runner. It intentionally does not invoke PCA/AE/VAE/TabFM training, publish an asset, open
a PR, or change a pointer.

## Failure handling and retry

Inspect the workflow artifact and generated `candidate-vs-active.json` first. A checksum, schema,
join, metrics, artifact, TabFM, or health failure leaves the active checkpoint unchanged. Correct the
runner/configuration or intentionally mark the candidate `rejected` in a review PR. Retrying uses the
same source tag and cannot overwrite its release tag. The discovery workflow is safe to rerun because
it only dispatches when a newer release exists and concurrency prevents duplicate source-tag builds.
