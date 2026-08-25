# Git packaging plan

Stage source, tests, docs, paper `.tex`, compact mappings/views/reports/JSONL,
and deterministic fixtures. Do **not** stage raw/extracted Epoch data, `.data`,
legacy local CSV snapshots, environments, caches, logs, or ignored paper figures.

Recommended later staging roots: `AGENTS.md`, `.gitignore`, `apps/`, relevant
`modeling/`, `scripts/`, `docs/integration`, `docs/dcbench`,
`docs/project-handoff`, compact `data/derived/integration`, compact
`data/derived/dcbench`, relevant `tests/`, and the two paper sections. Review
each with `git diff --cached`; no staging occurred in this pass.
