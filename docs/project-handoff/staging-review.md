# Staging review

## Include now

The exact manifest lists reviewed integration/DC Bench code, tests, docs,
compact mappings/reports, and two standalone paper sections.

## Exclude raw/local

Exclude `data/external/**/raw`, extracted vendor data, ZIPs, `.data`, local
snapshots, environments, caches, logs, and reproducible capability/enriched
views. The latter include source-text whitespace and/or duplicate benchmark rows.

## Defer

Pre-existing broader paper work, paper helper/export code, dashboard edits,
comparison changes, and `AGENTS.md` need ownership review before staging.

## Investigate

No credentials were found in proposed paths. The only secret-scan match was
ordinary documentation wording, not a secret.
