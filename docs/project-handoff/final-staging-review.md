# Final staging review

## Approved

All paths in `exact-staging-manifest.txt` are reviewed source, compact derived
artifact, test, or documentation files for the Epoch validation/DC Bench package.

## Rejected

`data/derived/integration/views/canonical_hardware_capabilities.csv` is compact
and reproducible, but it contains source-text trailing whitespace that prevents
a clean staged whitespace check. It remains local and is rebuilt by the
integration script; mapping and validated report artifacts remain staged.

## Needs human review

Broader pre-existing paper, dashboard, comparison, AGENTS, and paper-helper work
is intentionally absent because it is unrelated or ownership-ambiguous.

## Excluded local/raw assets

Epoch raw/extracted data, `.data`, environments, caches, logs, legacy local
snapshots, and `inferencex_throughput_epoch_hardware_v1.csv` remain unstaged.

## Large-file check

No approved file exceeds 5 MB. Excluded reproducible views are 59 KB (canonical
capabilities) and 14 MB (enriched operating points). No approved file exceeds
25 MB or 50 MB.

## Secret check

PASS. No credentials or nonportable absolute `/Users/...` path was found in
approved paths.
