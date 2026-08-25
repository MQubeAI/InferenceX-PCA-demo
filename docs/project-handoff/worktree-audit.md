# Worktree audit — 2026-08-23

| Path/group | Category | Git? | Reproducible / required | Local-only risk |
|---|---|---|---|---|
| `AGENTS.md`, dashboard/comparison edits | existing project / integration | yes | source instructions and UI/model plumbing | none |
| `modeling/integration.py`, build script, integration test | Epoch integration | yes | rebuild compact views | none |
| hardware validation source/tests, `docs/integration/`, compact integration reports | hardware validation | yes | scientific record | raw Epoch excluded |
| DC Bench source/tests/docs and compact derived artifacts | DC Bench | yes | deterministic seed/evaluation record | synthetic fixtures labelled |
| `paper/`, paper helper/export/test | existing paper work | yes except ignored figures | manuscript source | rendered binaries ignored |
| `data/external/**/raw`, extracted archives | raw external data | no | retain locally | large immutable download |
| `.data/`, legacy local CSV, `.venv*/`, caches/logs | runtime/source snapshot | no | local recovery only | must remain excluded |

No file was deleted. Keep compact mappings, validation JSON, reports, and DC
Bench JSONL; exclude raw/extracted external source material.
