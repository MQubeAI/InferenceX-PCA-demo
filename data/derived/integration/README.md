# Derived Multi-Source Integration Data

All files in this directory must be reproducible from raw source snapshots.

Build the first Epoch hardware prototype with:

```bash
.venv-streamlit/bin/python scripts/build_hardware_integration.py \
  --data-dir .data/inferencex-db-dump-2026-07-20
```

The builder verifies raw Epoch SHA-256 identities before and after execution.
It never writes under `data/external/epoch/raw/`.

## mappings

Canonical source-to-entity mapping tables.

## views

Task-specific modeling datasets.

## reports

Audits, overlap reports, schema diagnostics, coverage reports, and validation outputs.

## prototypes

Data artifacts used by early multi-source what-if prototypes.

Raw external-source files do not belong here.
