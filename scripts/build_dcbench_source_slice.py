"""Extract the immutable compact operating-point slice behind DC Bench v0.1/v0.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.dcbench import build_questions  # noqa: E402
from modeling.dcbench_source import (  # noqa: E402
    FROZEN_SOURCE_MANIFEST,
    FROZEN_SOURCE_SLICE,
    FULL_ENRICHED_VIEW,
    SOURCE_COLUMNS,
    SOURCE_SLICE_VERSION,
    sha256_file,
)
from modeling.dcbench_v0_2_candidates import build_candidates  # noqa: E402

GENERATION_TIMESTAMP_UTC = "2026-08-24T00:00:00Z"


def _source_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for evidence in item["evidence"]:
            if "row" in evidence:
                rows.append(evidence["row"])
            rows.extend(evidence.get("candidate_rows", []))
    return rows


def _match_row(frame: pd.DataFrame, identity: dict[str, Any]) -> pd.DataFrame:
    matches = pd.Series(True, index=frame.index)
    for column, value in identity.items():
        if column not in frame.columns:
            continue
        if value is None:
            matches &= frame[column].isna()
        else:
            matches &= frame[column].eq(value)
    return frame.loc[matches]


def extract_source_slice() -> dict[str, Any]:
    """Write the exact 23-row input union selected by full-data builders."""
    if not FULL_ENRICHED_VIEW.is_file():
        raise FileNotFoundError(f"Full enriched view required for extraction: {FULL_ENRICHED_VIEW}")
    full = pd.read_csv(FULL_ENRICHED_VIEW, low_memory=False)
    missing = [column for column in SOURCE_COLUMNS if column not in full.columns]
    if missing:
        raise ValueError(f"Full enriched view lacks required source-slice columns: {missing}")

    records = _source_rows(build_questions(source_mode="full"))
    records.extend(_source_rows(build_candidates(source_mode="full")))
    indexes: set[int] = set()
    for record in records:
        matches = _match_row(full, record)
        if len(matches) != 1:
            raise ValueError(f"Evidence row matched {len(matches)} full-view rows: {record}")
        indexes.add(int(matches.index[0]))
    subset = full.loc[sorted(indexes), SOURCE_COLUMNS].copy()
    subset = subset.sort_values(
        ["benchmark_type", "config_id", "config_hardware", "isl", "osl", "conc"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    FROZEN_SOURCE_SLICE.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(FROZEN_SOURCE_SLICE, index=False, lineterminator="\n")
    manifest = {
        "artifact": FROZEN_SOURCE_SLICE.name,
        "version": SOURCE_SLICE_VERSION,
        "generation_timestamp_utc": GENERATION_TIMESTAMP_UTC,
        "extraction_script": "scripts/build_dcbench_source_slice.py",
        "source_artifact": str(FULL_ENRICHED_VIEW.relative_to(ROOT)),
        "source_snapshot": "InferenceX db-dump/2026-07-20",
        "source_view_sha256": sha256_file(FULL_ENRICHED_VIEW),
        "source_view_rows": int(len(full)),
        "source_view_columns": int(len(full.columns)),
        "slice_rows": int(len(subset)),
        "slice_columns": int(len(subset.columns)),
        "selected_columns": SOURCE_COLUMNS,
        "slice_sha256": sha256_file(FROZEN_SOURCE_SLICE),
        "benchmark_versions_supported": ["inference_v0_1", "inference_v0_2_candidate"],
        "selection": "Exact union of operating-point evidence records emitted by the full-data v0.1 and v0.2 builders.",
        "gold_answer_columns_included": False,
    }
    FROZEN_SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = extract_source_slice()
    print(json.dumps({"slice": str(FROZEN_SOURCE_SLICE), "rows": result["slice_rows"], "sha256": result["slice_sha256"]}))
