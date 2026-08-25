"""Immutable operating-point source boundary for DC Bench question builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_ENRICHED_VIEW = ROOT / "data/derived/integration/views/inferencex_throughput_epoch_hardware_v1.csv"
SLICE_DIRECTORY = ROOT / "data/derived/dcbench/source_slices"
FROZEN_SOURCE_SLICE = SLICE_DIRECTORY / "inference_v0_1_v0_2_source_slice.csv"
FROZEN_SOURCE_MANIFEST = SLICE_DIRECTORY / "manifest.json"
SOURCE_SLICE_VERSION = "dcbench-source-slice-v1"

# This is the union of every original view column required to reproduce the
# frozen v0.1 and v0.2 operating-point evidence.  It intentionally excludes
# physical capability columns, raw Epoch records, outcomes other than the
# benchmark target, and all gold-answer fields.
SOURCE_COLUMNS = [
    "config_id",
    "benchmark_type",
    "isl",
    "osl",
    "conc",
    "workers",
    "offload_mode",
    "metrics_tput_per_gpu",
    "config_hardware",
    "config_framework",
    "config_model",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_dp_attention",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_dp_attention",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_num_decode_gpu",
    "hardware_mapping_status",
]

# v0.1 direct-retrieval and controlled-pair anchors are historical selection
# identities, not outcome values.  They permit the frozen slice to reproduce
# the original question choices without requiring unrelated rows from the full
# 8,239-row view.
V0_1_DIRECT_ANCHORS = (
    {"config_id": 1, "config_hardware": "h200", "benchmark_type": "single_turn", "isl": 1024.0, "osl": 1024.0, "conc": 4},
    {"config_id": 9, "config_hardware": "mi355x", "benchmark_type": "single_turn", "isl": 8192.0, "osl": 1024.0, "conc": 8},
)
V0_1_CONTROLLED_ANCHORS = (
    {"config_id": 878, "config_hardware": "b300", "benchmark_type": "agentic_traces", "isl": None, "osl": None, "conc": 1},
    {"config_id": 992, "config_hardware": "mi355x", "benchmark_type": "agentic_traces", "isl": None, "osl": None, "conc": 1},
)


class DcBenchSourceError(RuntimeError):
    """The requested DC Bench operating-point source is unavailable or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    try:
        return json.loads(FROZEN_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DcBenchSourceError(
            "Frozen DC Bench source-slice manifest is missing. Run "
            "scripts/build_dcbench_source_slice.py from a full integration environment."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DcBenchSourceError("Frozen DC Bench source-slice manifest is invalid JSON.") from exc


def _validate_slice(path: Path, manifest: dict[str, Any]) -> None:
    if not path.is_file():
        raise DcBenchSourceError(f"Frozen DC Bench source slice is missing: {path}")
    if manifest.get("slice_sha256") != sha256_file(path):
        raise DcBenchSourceError("Frozen DC Bench source-slice SHA-256 does not match its manifest.")
    if manifest.get("selected_columns") != SOURCE_COLUMNS:
        raise DcBenchSourceError("Frozen DC Bench source-slice column contract is incompatible.")


def load_operating_point_source(source_mode: str = "frozen") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load either the immutable slice or an explicitly requested full view.

    ``frozen`` is the default for benchmark rebuilds and clean CI.  ``full`` is
    opt-in for local provenance regeneration and fails clearly if the local
    enriched view is absent; there is no silent source fallback.
    """
    if source_mode == "frozen":
        manifest = _load_manifest()
        _validate_slice(FROZEN_SOURCE_SLICE, manifest)
        frame = pd.read_csv(FROZEN_SOURCE_SLICE, low_memory=False)
        if list(frame.columns) != SOURCE_COLUMNS or len(frame) != manifest.get("slice_rows"):
            raise DcBenchSourceError("Frozen DC Bench source slice does not match its row/column contract.")
        return frame, {
            "kind": "frozen_source_slice",
            "path": str(FROZEN_SOURCE_SLICE.relative_to(ROOT)),
            "sha256": manifest["slice_sha256"],
            "source_snapshot": manifest["source_snapshot"],
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    if source_mode == "full":
        if not FULL_ENRICHED_VIEW.is_file():
            raise DcBenchSourceError(
                "Full enriched DC Bench view is unavailable. Use source_mode='frozen' for the committed benchmark."
            )
        frame = pd.read_csv(FULL_ENRICHED_VIEW, low_memory=False)
        missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
        if missing:
            raise DcBenchSourceError("Full enriched view is missing required DC Bench columns: " + ", ".join(missing))
        return frame, {
            "kind": "full_enriched_view",
            "path": str(FULL_ENRICHED_VIEW.relative_to(ROOT)),
            "sha256": sha256_file(FULL_ENRICHED_VIEW),
            "source_snapshot": "InferenceX db-dump/2026-07-20",
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    raise ValueError("source_mode must be 'frozen' or 'full'")


def anchored_rows(frame: pd.DataFrame, anchors: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    """Resolve exact operating-point anchors and fail on missing or duplicate rows."""
    rows = []
    for anchor in anchors:
        matches = pd.Series(True, index=frame.index)
        for column, value in anchor.items():
            if value is None:
                matches &= frame[column].isna()
            else:
                matches &= frame[column].eq(value)
        selected = frame.loc[matches]
        if len(selected) != 1:
            raise DcBenchSourceError(f"DC Bench anchor matched {len(selected)} rows: {anchor}")
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)
