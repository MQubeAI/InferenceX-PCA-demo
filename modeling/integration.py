"""Auditable, task-specific multi-source integration helpers.

This module intentionally keeps source observations separate.  Its only joined
view is the first approved task view: InferenceX throughput operating points
enriched with one resolved Epoch hardware descriptor record.  It never joins
Epoch cluster/datacenter tables into an operating-point view and it never turns
Artificial Analysis performance into an InferenceX outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EPOCH_ROOT = REPOSITORY_ROOT / "data" / "external" / "epoch"
EPOCH_RAW_DIR = EPOCH_ROOT / "raw"
EPOCH_EXTRACTED_BENCHMARK_DIR = EPOCH_ROOT / "extracted" / "benchmark_data"
EPOCH_HASH_MANIFEST = EPOCH_ROOT / "SHA256SUMS.txt"
ARTIFICIAL_ANALYSIS_RAW_DIR = (
    REPOSITORY_ROOT / "data" / "external" / "artificial_analysis" / "raw"
)
INTEGRATION_ROOT = REPOSITORY_ROOT / "data" / "derived" / "integration"

INFERENCE_SNAPSHOT_RELEASE = "db-dump/2026-07-20"
INFERENCE_SNAPSHOT_DATE = "2026-07-20"
TARGET = "metrics_tput_per_gpu"
VIEW_VERSION = "v1"

CAPABILITY_SOURCE_COLUMNS = {
    "epoch_memory_capacity_bytes": "Memory (bytes)",
    "epoch_memory_bandwidth_bytes_per_second": "Memory bandwidth (byte/s)",
    "epoch_tensor_fp16_bf16_peak_flops": "Tensor-FP16/BF16 performance (FLOP/s)",
    "epoch_fp8_peak_flops": "FP8 performance (FLOP/s)",
    "epoch_fp4_peak_flops": "FP4 performance (FLOP/s)",
    "epoch_int8_peak_ops": "INT8 performance (OP/s)",
    "epoch_int4_peak_ops": "INT4 performance (OP/s)",
    "epoch_tdp_watts": "TDP (W)",
    "epoch_intranode_bandwidth_bytes_per_second": "Intranode bandwidth (byte/s)",
    "epoch_internode_bandwidth_bits_per_second": "Internode bandwidth (bit/s)",
}
CAPABILITY_FEATURE_COLUMNS = tuple(CAPABILITY_SOURCE_COLUMNS)

BASE_NUMERIC_FEATURES = (
    "isl",
    "osl",
    "conc",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_num_decode_gpu",
)
BASE_CATEGORICAL_FEATURES = (
    "benchmark_type",
    "config_hardware",
    "config_framework",
    "config_model",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
)
STRUCTURAL_MATCH_COLUMNS = (
    "benchmark_type",
    "isl",
    "osl",
    "conc",
    "config_model",
    "config_framework",
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
)
ANALOGUE_NUMERIC_COLUMNS = (
    "isl",
    "osl",
    "conc",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_num_decode_gpu",
)
ANALOGUE_CATEGORICAL_COLUMNS = (
    "benchmark_type",
    "config_hardware",
    "config_model",
    "config_framework",
    "config_precision",
    "config_spec_method",
    "config_disagg",
    "config_is_multinode",
)

AA_REQUIRED_COLUMNS = (
    "canonical_model_id",
    "source_model_id",
    "model_name",
    "provider",
    "source_timestamp",
)
AA_OPTIONAL_COLUMNS = (
    "capability_index",
    "context_window_tokens",
    "parameter_count",
    "median_output_speed",
    "median_output_speed_unit",
    "ttft",
    "ttft_unit",
    "end_to_end_latency",
    "end_to_end_latency_unit",
    "input_token_price",
    "output_token_price",
    "price_currency",
)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "unknown"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_safe) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_epoch_raw_hashes(manifest_path: Path = EPOCH_HASH_MANIFEST) -> dict[Path, str]:
    """Read the checked-in local manifest; it is a guard, not an external refresh."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Epoch raw hash manifest is missing: {manifest_path}")
    expected: dict[Path, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})\s{2}(.+)$", line.strip())
        if match:
            expected[REPOSITORY_ROOT / match.group(2)] = match.group(1)
    if not expected:
        raise ValueError("Epoch raw hash manifest has no SHA-256 entries.")
    return expected


def verify_epoch_raw_hashes(manifest_path: Path = EPOCH_HASH_MANIFEST) -> dict[str, str]:
    """Verify that the immutable Epoch raw snapshots still match their manifest."""
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for path, expected in expected_epoch_raw_hashes(manifest_path).items():
        if not path.exists():
            mismatches.append(f"missing: {path.relative_to(REPOSITORY_ROOT)}")
            continue
        actual = sha256_file(path)
        observed[str(path.relative_to(REPOSITORY_ROOT))] = actual
        if actual != expected:
            mismatches.append(
                f"{path.relative_to(REPOSITORY_ROOT)} expected {expected}, got {actual}"
            )
    if mismatches:
        raise RuntimeError("Epoch raw snapshot immutability check failed: " + "; ".join(mismatches))
    return observed


def epoch_snapshot_date(manifest_path: Path = EPOCH_HASH_MANIFEST) -> str:
    """Return the local raw-snapshot manifest date, without pretending it is an as-of archive."""
    if not manifest_path.exists():
        return "unknown"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^Generated:\s+(\d{4}-\d{2}-\d{2})", line)
        if match:
            return match.group(1)
    return "unknown"


def dataframe_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "missing_count": int(frame[column].isna().sum()),
            "missing_percentage": float(frame[column].isna().mean() * 100),
            "unique_non_null": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]


def audit_epoch_sources() -> dict[str, Any]:
    """Return a complete machine-readable inventory while keeping observations separate."""
    raw_names = (
        "ml_hardware.csv",
        "all_ai_models.csv",
        "gpu_clusters.csv",
        "data_centers.csv",
    )
    raw_sources: dict[str, Any] = {}
    for name in raw_names:
        path = EPOCH_RAW_DIR / name
        frame = pd.read_csv(path, low_memory=False)
        identity_column = {
            "ml_hardware.csv": "Hardware name",
            "all_ai_models.csv": "Model",
            "gpu_clusters.csv": "Name",
            "data_centers.csv": "Name",
        }[name]
        raw_sources[name] = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": dataframe_schema(frame),
            "identity_column": identity_column,
            "unique_identity_count": int(frame[identity_column].nunique(dropna=True)),
            "duplicate_non_null_identity_count": int(
                frame.loc[frame[identity_column].notna(), identity_column].duplicated().sum()
            ),
        }

    benchmark_files = sorted(EPOCH_EXTRACTED_BENCHMARK_DIR.rglob("*.csv"))
    file_inventory: list[dict[str, Any]] = []
    schemas: dict[tuple[str, ...], int] = {}
    common_columns: set[str] | None = None
    total_rows = 0
    for path in benchmark_files:
        frame = pd.read_csv(path, low_memory=False)
        columns = tuple(str(column) for column in frame.columns)
        schemas[columns] = schemas.get(columns, 0) + 1
        common_columns = set(columns) if common_columns is None else common_columns & set(columns)
        total_rows += len(frame)
        file_inventory.append(
            {
                "path": str(path.relative_to(EPOCH_EXTRACTED_BENCHMARK_DIR)),
                "rows": int(len(frame)),
                "columns": list(columns),
                "sha256": sha256_file(path),
                "model_identifier_columns": [
                    name for name in ("Model version", "Model name", "Name", "id") if name in frame
                ],
                "date_columns": [
                    name
                    for name in frame.columns
                    if "date" in str(name).lower() or "started" in str(name).lower()
                ],
            }
        )
    eci_path = EPOCH_EXTRACTED_BENCHMARK_DIR / "epoch_capabilities_index.csv"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "epoch_source_snapshot": epoch_snapshot_date(),
        "raw_sources": raw_sources,
        "benchmark_observations": {
            "extracted_path": str(EPOCH_EXTRACTED_BENCHMARK_DIR.relative_to(REPOSITORY_ROOT)),
            "file_count": len(benchmark_files),
            "total_rows_across_files": int(total_rows),
            "unique_schemas": len(schemas),
            "schema_groups": [
                {"file_count": count, "columns": list(columns)}
                for columns, count in sorted(schemas.items(), key=lambda item: (-item[1], item[0]))
            ],
            "common_columns_across_all_files": sorted(common_columns or set()),
            "epoch_capabilities_index": {
                "present": eci_path.exists(),
                "schema": dataframe_schema(pd.read_csv(eci_path, low_memory=False))
                if eci_path.exists()
                else [],
            },
            "files": file_inventory,
        },
        "layering": {
            "ml_hardware.csv": "CanonicalHardware source; eligible for an explicit many-to-one hardware descriptor join.",
            "all_ai_models.csv": "CanonicalModel metadata source; not a current throughput predictor source.",
            "benchmark_data": "Long-form BenchmarkObservation sources with heterogeneous schemas; no universal wide join.",
            "gpu_clusters.csv": "GPUCluster infrastructure layer; not joined to InferenceX operating points.",
            "data_centers.csv": "DataCenter infrastructure layer; not joined to InferenceX operating points.",
        },
    }


def canonical_hardware_capabilities(hardware: pd.DataFrame) -> pd.DataFrame:
    """Normalize Epoch hardware descriptors without imputing any physical capability."""
    required = {"Hardware name", "Manufacturer", "Release date", "Last modified", *CAPABILITY_SOURCE_COLUMNS.values()}
    missing = sorted(required - set(hardware.columns))
    if missing:
        raise ValueError("Epoch hardware source has unexpected missing columns: " + ", ".join(missing))
    result = pd.DataFrame(
        {
            "canonical_hardware_id": hardware["Hardware name"].map(lambda value: f"epoch_hardware:{_slug(value)}"),
            "epoch_hardware_name": hardware["Hardware name"],
            "vendor": hardware["Manufacturer"],
            "epoch_hardware_type": hardware.get("Type"),
            "epoch_release_date": hardware["Release date"],
            "epoch_last_modified": hardware["Last modified"],
            "source": "Epoch AI ml_hardware.csv",
            "source_snapshot": epoch_snapshot_date(),
            "provenance": hardware.get("Link to datasheet"),
            "source_notes": hardware.get("Notes"),
        }
    )
    for derived, source in CAPABILITY_SOURCE_COLUMNS.items():
        # Explicit numeric coercion preserves absent values as NaN; it never uses zero.
        result[derived] = pd.to_numeric(hardware[source], errors="coerce")
    if result["canonical_hardware_id"].duplicated().any():
        raise ValueError("Canonical hardware IDs are not unique.")
    return result


def _hardware_mapping_specs() -> list[dict[str, Any]]:
    return [
        {"inferencex_hardware": "b200", "epoch_hardware_name": "NVIDIA B200", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact normalized accelerator label; Epoch note states performance is per individual GPU.", "notes": "Accepted as the Epoch B200 individual-GPU record."},
        {"inferencex_hardware": "b300", "epoch_hardware_name": "NVIDIA B300 (Blackwell Ultra)", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact B300 family label; Epoch identifies the individual GPU rather than a system aggregate.", "notes": "Accepted as the Epoch B300 individual-GPU record."},
        {"inferencex_hardware": "gb200", "epoch_hardware_name": "NVIDIA GB200", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact normalized GB200 label; Epoch note records per-GPU capabilities.", "notes": "Accepted as the Epoch GB200 per-GPU descriptor."},
        {"inferencex_hardware": "gb300", "epoch_hardware_name": "NVIDIA GB300 (Blackwell Ultra)", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact GB300 family label; Epoch distinguishes the system name from per-GPU figures.", "notes": "Accepted as the Epoch GB300 per-GPU descriptor."},
        {"inferencex_hardware": "mi300x", "epoch_hardware_name": "AMD Instinct MI300X", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact normalized MI300X accelerator name.", "notes": "Accepted."},
        {"inferencex_hardware": "mi325x", "epoch_hardware_name": "AMD Instinct MI325X", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact normalized MI325X accelerator name.", "notes": "Accepted."},
        {"inferencex_hardware": "mi355x", "epoch_hardware_name": "AMD Instinct MI355X", "match_type": "exact_hardware_label", "match_confidence": "high", "ambiguity": False, "evidence": "Exact normalized MI355X accelerator name.", "notes": "Accepted."},
        {"inferencex_hardware": "h100", "epoch_hardware_name": "NVIDIA H100 NVL; NVIDIA H100 PCIe; NVIDIA H100 SXM5 80GB", "match_type": "unresolved_variant", "match_confidence": "unresolved", "ambiguity": True, "evidence": "InferenceX records only `h100`; Epoch exposes materially distinct NVL, PCIe, and SXM5 records.", "notes": "No per-row join until repository provenance resolves form factor and memory/NVLink variant."},
        {"inferencex_hardware": "h200", "epoch_hardware_name": "NVIDIA H200 SXM", "match_type": "unresolved_variant", "match_confidence": "unresolved", "ambiguity": True, "evidence": "InferenceX records only `h200`; its configuration provenance does not establish the Epoch SXM form factor.", "notes": "A single current Epoch candidate does not prove that every InferenceX H200 operating point is SXM."},
    ]


def hardware_mapping(capabilities: pd.DataFrame, inferencex_hardware: Iterable[str]) -> pd.DataFrame:
    """Create the explicit, manually reviewed hardware mapping table."""
    name_to_id = capabilities.set_index("epoch_hardware_name")["canonical_hardware_id"].to_dict()
    expected = sorted({str(value).strip().lower() for value in inferencex_hardware if pd.notna(value)})
    records: list[dict[str, Any]] = []
    specs = {record["inferencex_hardware"]: record for record in _hardware_mapping_specs()}
    for label in expected:
        spec = dict(specs.get(label, {
            "inferencex_hardware": label,
            "epoch_hardware_name": "",
            "match_type": "unmapped",
            "match_confidence": "unresolved",
            "ambiguity": True,
            "evidence": "No manually reviewed Epoch mapping specification exists.",
            "notes": "Unmapped; no descriptor join is permitted.",
        }))
        accepted = not spec["ambiguity"] and spec["epoch_hardware_name"] in name_to_id
        spec["canonical_hardware_id"] = name_to_id.get(spec["epoch_hardware_name"]) if accepted else pd.NA
        spec["mapping_status"] = "accepted" if accepted else "unresolved"
        spec["source"] = "InferenceX config_hardware -> Epoch AI ml_hardware.csv"
        spec["source_snapshot"] = epoch_snapshot_date()
        records.append(spec)
    result = pd.DataFrame(records)
    validate_hardware_mapping(result)
    return result[
        [
            "inferencex_hardware", "canonical_hardware_id", "epoch_hardware_name", "match_type",
            "match_confidence", "mapping_status", "ambiguity", "evidence", "notes", "source",
            "source_snapshot",
        ]
    ]


def validate_hardware_mapping(mapping: pd.DataFrame) -> None:
    if mapping["inferencex_hardware"].duplicated().any():
        raise ValueError("Every InferenceX hardware label must have one mapping decision.")
    accepted = mapping.loc[mapping["mapping_status"].eq("accepted")]
    if accepted["canonical_hardware_id"].isna().any():
        raise ValueError("An accepted hardware mapping lacks a canonical entity.")
    if accepted.groupby("inferencex_hardware")["canonical_hardware_id"].nunique().gt(1).any():
        raise ValueError("One InferenceX hardware label maps to multiple accepted entities.")
    ambiguous = mapping["ambiguity"].astype(bool)
    if mapping.loc[ambiguous, "mapping_status"].eq("accepted").any():
        raise ValueError("Ambiguous mappings cannot be accepted for production enrichment.")


def _model_mapping_specs() -> list[dict[str, Any]]:
    return [
        {"inferencex_model": "dsr1", "epoch_model_name": "DeepSeek-R1; DeepSeek-R1 (May 2025)", "status": "unresolved", "ambiguity": True, "evidence": "Two Epoch DeepSeek-R1 records encode distinct releases; InferenceX shorthand has no version/date.", "notes": "Do not choose a DeepSeek-R1 release."},
        {"inferencex_model": "dsv4", "epoch_model_name": "DeepSeek-V4-Pro; DeepSeek-V4-Flash; DeepSeek-V4-Pro-0813; DeepSeek V4 Flash 0731", "status": "unresolved", "ambiguity": True, "evidence": "The InferenceX V4 shorthand does not identify Pro versus Flash or dated variant.", "notes": "Do not join model metadata or benchmark observations."},
        {"inferencex_model": "glm5", "epoch_model_name": "GLM-5", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized version label with one Epoch record.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "glm5.1", "epoch_model_name": "GLM-5.1", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized version label with one Epoch record.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "glm5.2", "epoch_model_name": "GLM-5.2", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized version label with one Epoch record.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "gptoss120b", "epoch_model_name": "gpt-oss-120b", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized model name and parameter suffix.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "kimik2.5", "epoch_model_name": "Kimi K2.5", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized model-version label.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "llama70b", "epoch_model_name": "Llama 2-70B; Llama 3-70B; Llama 3.1-70B; Llama 3.3 70B", "status": "unresolved", "ambiguity": True, "evidence": "Model family/size shorthand does not identify a Llama release or fine-tune.", "notes": "No default Llama version is defensible."},
        {"inferencex_model": "minimaxm2.5", "epoch_model_name": "MiniMax-M2.5", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized model-version label.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "minimaxm3", "epoch_model_name": "MiniMax-M3", "status": "accepted", "ambiguity": False, "evidence": "Exact normalized model-version label.", "notes": "Accepted exact model-version mapping."},
        {"inferencex_model": "qwen3.5", "epoch_model_name": "Qwen3.5-0.8B; Qwen3.5-2B; Qwen3.5-4B; Qwen3.5-9B; Qwen3.5-27B; Qwen3.5-35B-A3B; Qwen3.5-122B-A10B; Qwen3.5 397B-A17B; Qwen3.5-Omni-Flash; Qwen3.5-Omni-Plus", "status": "unresolved", "ambiguity": True, "evidence": "Qwen3.5 is a family shorthand with dense, MoE, hosted, and multimodal variants in Epoch.", "notes": "No implicit variant mapping."},
    ]


def model_mapping(epoch_models: pd.DataFrame, inferencex_models: Iterable[str]) -> pd.DataFrame:
    known_epoch_models = set(epoch_models["Model"].astype(str))
    specs = {record["inferencex_model"]: record for record in _model_mapping_specs()}
    records: list[dict[str, Any]] = []
    for label in sorted({str(value).strip().lower() for value in inferencex_models if pd.notna(value)}):
        spec = dict(specs.get(label, {
            "inferencex_model": label,
            "epoch_model_name": "",
            "status": "unresolved",
            "ambiguity": True,
            "evidence": "No manually reviewed Epoch mapping specification exists.",
            "notes": "Unmapped; model metadata is not joined.",
        }))
        accepted = spec["status"] == "accepted" and spec["epoch_model_name"] in known_epoch_models
        spec["canonical_model_id"] = (
            f"epoch_model:{_slug(spec['epoch_model_name'])}" if accepted else pd.NA
        )
        spec["mapping_status"] = "accepted" if accepted else "unresolved"
        spec["source"] = "InferenceX config_model -> Epoch AI all_ai_models.csv"
        spec["source_snapshot"] = epoch_snapshot_date()
        records.append(spec)
    result = pd.DataFrame(records)
    if result["inferencex_model"].duplicated().any():
        raise ValueError("Every InferenceX model label must have one mapping decision.")
    if result.loc[result["ambiguity"].astype(bool), "mapping_status"].eq("accepted").any():
        raise ValueError("Ambiguous model mappings cannot be accepted.")
    return result[
        [
            "inferencex_model", "canonical_model_id", "epoch_model_name", "mapping_status",
            "ambiguity", "evidence", "notes", "source", "source_snapshot",
        ]
    ]


def load_existing_throughput_cohort(data_dir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the established July aggregate builder instead of reconstructing a cohort."""
    from scripts.build_july_pca_artifact import load_aggregate

    _raw, aggregate, metadata = load_aggregate(data_dir)
    if TARGET not in aggregate:
        raise ValueError(f"Established throughput target is absent: {TARGET}")
    if "config_id" not in aggregate:
        raise ValueError("Established throughput cohort has no config_id grouping field.")
    return aggregate.copy(), metadata


def _temporal_status(view: pd.DataFrame) -> pd.Series:
    status = pd.Series("unmapped_or_ambiguous", index=view.index, dtype="string")
    accepted = view["hardware_mapping_status"].eq("accepted")
    release = pd.to_datetime(view["epoch_release_date"], errors="coerce", utc=True)
    observed = pd.to_datetime(view.get("date"), errors="coerce", utc=True)
    status.loc[accepted & release.isna()] = "release_date_unknown"
    status.loc[accepted & release.notna() & observed.isna()] = "benchmark_date_unknown"
    status.loc[accepted & release.notna() & observed.notna() & release.le(observed)] = (
        "release_date_precedes_benchmark"
    )
    status.loc[accepted & release.notna() & observed.notna() & release.gt(observed)] = (
        "release_after_benchmark"
    )
    return status


def build_throughput_integration_view(
    throughput: pd.DataFrame,
    hardware_map: pd.DataFrame,
    capabilities: pd.DataFrame,
    model_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join one accepted canonical hardware record per operating point, with validation."""
    before = throughput.copy()
    before_target = pd.to_numeric(before[TARGET], errors="coerce").to_numpy()
    before_rows = len(before)
    before_configs = int(before["config_id"].nunique(dropna=False))

    view = before.copy()
    view["inferencex_hardware"] = view["config_hardware"].astype("string").str.strip().str.lower()
    view["inferencex_model"] = view["config_model"].astype("string").str.strip().str.lower()
    view = view.merge(hardware_map, on="inferencex_hardware", how="left", validate="many_to_one")
    view = view.merge(
        model_map[["inferencex_model", "canonical_model_id", "mapping_status"]].rename(
            columns={"mapping_status": "model_mapping_status"}
        ),
        on="inferencex_model",
        how="left",
        validate="many_to_one",
    )
    capability_columns = [
        "canonical_hardware_id", "epoch_hardware_name", "vendor", "epoch_hardware_type",
        "epoch_release_date", "epoch_last_modified", "source", "source_snapshot", "provenance",
        "source_notes", *CAPABILITY_FEATURE_COLUMNS,
    ]
    view = view.merge(
        capabilities[capability_columns].rename(
            columns={
                "source": "epoch_capability_source",
                "source_snapshot": "epoch_capability_source_snapshot",
                "provenance": "epoch_capability_provenance",
                "source_notes": "epoch_capability_source_notes",
            }
        ),
        on="canonical_hardware_id",
        how="left",
        validate="many_to_one",
    )
    view["hardware_mapping_status"] = view["mapping_status"].fillna("unresolved")
    view["integration_view_version"] = VIEW_VERSION
    view["inferencex_source"] = "SemiAnalysis InferenceX"
    view["inferencex_snapshot_release"] = INFERENCE_SNAPSHOT_RELEASE
    view["epoch_source"] = "Epoch AI ml_hardware.csv"
    view["epoch_source_snapshot"] = epoch_snapshot_date()
    view["epoch_temporal_descriptor_status"] = _temporal_status(view)
    key_columns = ["config_id", "benchmark_type", "isl", "osl", "conc"]
    view["integration_row_id"] = view[key_columns].astype("string").fillna("<missing>").agg("|".join, axis=1)

    after_target = pd.to_numeric(view[TARGET], errors="coerce").to_numpy()
    target_unchanged = bool(
        len(before_target) == len(after_target)
        and np.array_equal(before_target, after_target, equal_nan=True)
    )
    invariant_results = {
        "row_count_preserved": {"pass": len(view) == before_rows, "before": before_rows, "after": len(view)},
        "config_id_count_preserved": {
            "pass": int(view["config_id"].nunique(dropna=False)) == before_configs,
            "before": before_configs,
            "after": int(view["config_id"].nunique(dropna=False)),
        },
        "target_values_unchanged": {"pass": target_unchanged, "target": TARGET},
        "unique_integration_row_id": {
            "pass": not view["integration_row_id"].duplicated().any(),
            "duplicates": int(view["integration_row_id"].duplicated().sum()),
        },
        "no_infrastructure_columns": {
            "pass": not any("cluster" in column.lower() or "data_center" in column.lower() for column in view.columns),
        },
    }
    if not all(item["pass"] for item in invariant_results.values()):
        raise AssertionError("Throughput integration invariant failed: " + json.dumps(invariant_results))
    coverage = (
        view.groupby(["config_hardware", "hardware_mapping_status"], dropna=False)
        .agg(rows=("config_id", "size"), configs=("config_id", "nunique"))
        .reset_index()
        .sort_values("config_hardware")
        .to_dict(orient="records")
    )
    validation = {
        "view": "views/inferencex_throughput_epoch_hardware_v1.csv",
        "target": TARGET,
        "invariant_results": invariant_results,
        "hardware_mapping_coverage": coverage,
        "accepted_mapping_row_coverage": float(view["hardware_mapping_status"].eq("accepted").mean()),
        "capability_missingness": [
            {
                "column": column,
                "missing_count": int(view[column].isna().sum()),
                "missing_percentage": float(view[column].isna().mean() * 100),
            }
            for column in CAPABILITY_FEATURE_COLUMNS
        ],
        "target_distribution_before": pd.Series(before_target).describe().to_dict(),
        "target_distribution_after": pd.Series(after_target).describe().to_dict(),
        "temporal_descriptor_status": view["epoch_temporal_descriptor_status"].value_counts(dropna=False).to_dict(),
        "temporal_caveat": (
            "Epoch raw snapshot provenance post-dates the InferenceX July cutoff. Release dates are used "
            "as a necessary record-level eligibility check, but this snapshot alone cannot prove field-level "
            "historical availability. The resulting experiment is a post-hoc descriptor study, not an as-of claim."
        ),
    }
    return view, validation


def capability_eligible_cohort(view: pd.DataFrame) -> pd.DataFrame:
    eligible = view.loc[
        view["hardware_mapping_status"].eq("accepted")
        & view["epoch_temporal_descriptor_status"].eq("release_date_precedes_benchmark")
        & pd.to_numeric(view[TARGET], errors="coerce").notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No capability-mapped, release-date-eligible throughput rows are available.")
    return eligible


def run_hardware_representation_experiment(view: pd.DataFrame) -> dict[str, Any]:
    """Evaluate identity and capability representations under the existing grouped RF protocol."""
    from modeling.comparison import evaluate_models

    cohort = capability_eligible_cohort(view)
    numeric = [column for column in BASE_NUMERIC_FEATURES if column in cohort]
    categorical = [column for column in BASE_CATEGORICAL_FEATURES if column in cohort]
    # Preserve a wholly absent descriptor in the capability view, but do not pass
    # an all-missing column to a learner: there is no training-fold median to fit.
    capabilities = [
        column for column in CAPABILITY_FEATURE_COLUMNS
        if column in cohort and cohort[column].notna().any()
    ]
    representations = {
        "A_identity": numeric + categorical,
        "B_identity_plus_capabilities": numeric + capabilities + categorical,
        "C_capabilities_only": numeric + capabilities + [
            column for column in categorical if column != "config_hardware"
        ],
    }
    output: dict[str, Any] = {
        "study_name": "post_hoc_hardware_descriptor_representation",
        "target": TARGET,
        "cohort": {
            "rows": int(len(cohort)),
            "configs": int(cohort["config_id"].nunique(dropna=False)),
            "hardware_labels": sorted(cohort["config_hardware"].dropna().astype(str).unique()),
            "filter": "accepted hardware mapping, release date precedes observed benchmark date, non-null target",
        },
        "protocol": {
            "point_model": "RandomForestRegressor(n_estimators=150, min_samples_leaf=2)",
            "validation": "deterministic three-fold GroupKFold by config_id",
            "preprocessing": "fit per fold: numeric median plus missing indicator; categorical __MISSING__",
            "target_leakage_policy": "Only configuration/workload fields and Epoch physical descriptors are features; no InferenceX outcome, Epoch composite score, or visualization coordinate is a feature.",
            "temporal_status": "Post-hoc descriptor study. It is not an as-of historical prediction claim because the available Epoch snapshot is newer than the InferenceX snapshot.",
            "uncertainty": "Not evaluated; no prior conformal interval is reused.",
        },
        "representations": {},
        "excluded_all_missing_capabilities": [
            column for column in CAPABILITY_FEATURE_COLUMNS
            if column in cohort and not cohort[column].notna().any()
        ],
    }
    for name, features in representations.items():
        result = evaluate_models(
            cohort,
            features,
            TARGET,
            ["random_forest"],
            max_rows=len(cohort),
            seed=42,
            n_splits=3,
            compute_permutation_importance=False,
            subgroup_columns=["config_hardware"],
        )
        model = result["models"]["random_forest"]
        output["representations"][name] = {
            "feature_columns": features,
            "hardware_identity_included": "config_hardware" in features,
            "epoch_capabilities_included": any(column in features for column in capabilities),
            "metrics": model["metrics"],
            "folds": model["folds"],
            "hardware_subgroups": model["subgroups"].get("config_hardware", []),
            "runtime_seconds": model["runtime_seconds"],
        }
    return output


def _metric_dict(truth: pd.Series, prediction: np.ndarray) -> dict[str, float | None]:
    from modeling.comparison import prediction_metrics

    return prediction_metrics(truth, prediction)


def run_hardware_holdout_study(view: pd.DataFrame, minimum_test_rows: int = 30) -> dict[str, Any]:
    """Test capability representation transfer to a fully withheld observed hardware SKU."""
    from modeling.comparison import FoldPreprocessor, _rf_fit_predict_without_importance

    cohort = capability_eligible_cohort(view)
    features = [column for column in BASE_NUMERIC_FEATURES if column in cohort] + [
        column for column in CAPABILITY_FEATURE_COLUMNS
        if column in cohort and cohort[column].notna().any()
    ] + [
        column
        for column in BASE_CATEGORICAL_FEATURES
        if column in cohort and column != "config_hardware"
    ]
    records: list[dict[str, Any]] = []
    for sku in sorted(cohort["config_hardware"].dropna().astype(str).unique()):
        test = cohort.loc[cohort["config_hardware"].astype(str).eq(sku)].copy()
        train = cohort.loc[~cohort["config_hardware"].astype(str).eq(sku)].copy()
        record: dict[str, Any] = {
            "hardware_sku": sku,
            "test_rows": int(len(test)),
            "test_configs": int(test["config_id"].nunique(dropna=False)),
            "train_rows": int(len(train)),
            "train_configs": int(train["config_id"].nunique(dropna=False)),
            "group_overlap": int(
                len(set(train["config_id"].astype(str)) & set(test["config_id"].astype(str)))
            ),
            "status": "evaluated" if len(test) >= minimum_test_rows else "insufficient_test_support",
            "minimum_test_rows": int(minimum_test_rows),
            "model": "capabilities_only_random_forest",
            "features": features,
            "capability_metrics": None,
            "training_median_baseline_metrics": None,
            "note": "Hardware identity is intentionally absent. Categorical identity-only is not a semantic unseen-SKU baseline.",
        }
        if record["group_overlap"]:
            raise AssertionError(f"Hardware holdout leaked config_id groups for {sku}.")
        if len(test) >= minimum_test_rows:
            processor = FoldPreprocessor.fit(train, features)
            transformed_train = processor.transform(train)
            transformed_test = processor.transform(test)
            y_train = pd.to_numeric(train[TARGET], errors="coerce")
            y_test = pd.to_numeric(test[TARGET], errors="coerce")
            prediction, _ = _rf_fit_predict_without_importance(
                transformed_train, transformed_test, y_train, 42
            )
            record["capability_metrics"] = _metric_dict(y_test, prediction)
            baseline = np.repeat(float(y_train.median()), len(y_test))
            record["training_median_baseline_metrics"] = _metric_dict(y_test, baseline)
        records.append(record)
    unresolved = (
        view.loc[view["hardware_mapping_status"].ne("accepted"), "config_hardware"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return {
        "study_name": "held_out_hardware_sku_transfer",
        "target": TARGET,
        "protocol": {
            "withholding": "Every row for one accepted hardware SKU is excluded from training.",
            "features": "Capability representation C: workload/configuration plus Epoch physical descriptors, without config_hardware.",
            "validation": "No held-out config_id appears in training.",
            "comparison": "Training-median baseline only. A categorical identity-only model cannot semantically represent an unseen category and is not reported as a comparable transfer baseline.",
            "claim_boundary": "This is held-out hardware SKU transfer over observed hardware, not unreleased-hardware prediction.",
            "temporal_status": "Post-hoc descriptor study; current Epoch snapshot is newer than the InferenceX cutoff.",
        },
        "skipped_unresolved_hardware": unresolved,
        "records": records,
    }


def artificial_analysis_status(raw_dir: Path = ARTIFICIAL_ANALYSIS_RAW_DIR) -> dict[str, Any]:
    """Safely expose the future ingest boundary without fabricating Artificial Analysis values."""
    contract = raw_dir / "artificial_analysis_observations.csv"
    if not raw_dir.exists():
        return {
            "status": "not_loaded",
            "message": "Artificial Analysis data not loaded.",
            "raw_directory": _display_path(raw_dir),
            "expected_contract": "artificial_analysis_observations.csv",
        }
    files = sorted(path.name for path in raw_dir.iterdir() if path.is_file())
    if not contract.exists():
        return {
            "status": "not_loaded",
            "message": "Artificial Analysis data not loaded.",
            "raw_directory": _display_path(raw_dir),
            "files_detected": files,
            "expected_contract": "artificial_analysis_observations.csv",
        }
    columns = list(pd.read_csv(contract, nrows=0).columns)
    missing = [column for column in AA_REQUIRED_COLUMNS if column not in columns]
    if missing:
        return {
            "status": "incompatible_contract",
            "message": "Artificial Analysis export found but required source/provenance columns are missing.",
            "raw_directory": _display_path(raw_dir),
            "missing_required_columns": missing,
            "columns": columns,
        }
    return {
        "status": "available",
        "message": "Artificial Analysis source-specific observations are available.",
        "path": _display_path(contract),
        "columns": columns,
        "performance_semantics": "AA speed/TTFT/end-to-end values remain source-specific and are never treated as InferenceX throughput.",
    }


def load_artificial_analysis_contract_export(raw_dir: Path = ARTIFICIAL_ANALYSIS_RAW_DIR) -> pd.DataFrame:
    """Load only a validated future contract export; no metric conversion is performed."""
    status = artificial_analysis_status(raw_dir)
    if status["status"] != "available":
        return pd.DataFrame(columns=[*AA_REQUIRED_COLUMNS, *AA_OPTIONAL_COLUMNS])
    return pd.read_csv(raw_dir / "artificial_analysis_observations.csv", low_memory=False)


def observed_counterfactual(
    view: pd.DataFrame, selected: pd.Series, target_hardware: str
) -> dict[str, Any]:
    """Return an exact counterpart only when all stored structural fields agree."""
    target = str(target_hardware).strip().lower()
    selected_hardware = str(selected.get("config_hardware", "")).strip().lower()
    if target == selected_hardware:
        return {
            "status": "observed",
            "observed": True,
            "rows": 1,
            "throughput": float(selected[TARGET]),
            "message": "Observed reference operating point.",
        }
    candidates = view.loc[view["config_hardware"].astype("string").str.lower().eq(target)].copy()
    for column in STRUCTURAL_MATCH_COLUMNS:
        if column not in candidates or column not in selected.index:
            continue
        value = selected[column]
        if pd.isna(value):
            candidates = candidates.loc[candidates[column].isna()]
        else:
            candidates = candidates.loc[candidates[column].astype(str).eq(str(value))]
    values = pd.to_numeric(candidates.get(TARGET), errors="coerce").dropna()
    if not values.empty:
        return {
            "status": "observed",
            "observed": True,
            "rows": int(len(values)),
            "throughput": float(values.median()),
            "message": "Exact structural counterpart observed in InferenceX; median shown if repeated configurations match.",
        }
    return {
        "status": "unsupported",
        "observed": False,
        "rows": 0,
        "throughput": None,
        "message": "Unsupported: no exact InferenceX counterpart and no reusable validated point-prediction artifact is loaded.",
    }


def nearest_analogues(
    view: pd.DataFrame,
    requested: pd.Series,
    count: int = 5,
) -> pd.DataFrame:
    """Find descriptive neighbours in original workload/configuration space, never UMAP/t-SNE."""
    work = view.loc[pd.to_numeric(view.get(TARGET), errors="coerce").notna()].copy()
    distances = pd.Series(0.0, index=work.index)
    terms = 0
    for column in ANALOGUE_NUMERIC_COLUMNS:
        if column not in work or column not in requested.index:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        value = pd.to_numeric(pd.Series([requested[column]]), errors="coerce").iloc[0]
        scale = values.quantile(0.75) - values.quantile(0.25)
        if not np.isfinite(scale) or scale <= 0:
            scale = values.std()
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        distances += (values.fillna(values.median()).sub(value).abs() / float(scale)).clip(upper=10)
        terms += 1
    for column in ANALOGUE_CATEGORICAL_COLUMNS:
        if column not in work or column not in requested.index:
            continue
        expected = "<missing>" if pd.isna(requested[column]) else str(requested[column])
        distances += work[column].astype("string").fillna("<missing>").ne(expected).astype(float)
        terms += 1
    if not terms:
        return pd.DataFrame()
    work["configuration_distance"] = distances / terms
    differences: list[str] = []
    display_columns = ("config_hardware", "config_model", "isl", "osl", "conc", "config_framework", "config_precision")
    for _, row in work.iterrows():
        changed = []
        for column in display_columns:
            if column in row and column in requested.index and str(row[column]) != str(requested[column]):
                changed.append(f"{column.replace('config_', '')}: {requested[column]} → {row[column]}")
        differences.append("; ".join(changed) if changed else "Exact structural match")
    work["major_differences"] = differences
    columns = [
        column
        for column in (
            "integration_row_id", "config_hardware", "config_model", "isl", "osl", "conc",
            "config_framework", "config_precision", TARGET, "configuration_distance", "major_differences",
        )
        if column in work
    ]
    return work.sort_values("configuration_distance").head(count).loc[:, columns].reset_index(drop=True)


def run_integration_pipeline(
    data_dir: str,
    output_root: Path = INTEGRATION_ROOT,
    run_experiments: bool = True,
) -> dict[str, Any]:
    """Build all derived Phase-1 integration artifacts; raw sources are hash-checked twice."""
    raw_hashes_before = verify_epoch_raw_hashes()
    output_root = Path(output_root)
    for name in ("mappings", "views", "reports", "reports/hardware_holdout", "prototypes"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    epoch_audit = audit_epoch_sources()
    hardware_raw = pd.read_csv(EPOCH_RAW_DIR / "ml_hardware.csv", low_memory=False)
    model_raw = pd.read_csv(EPOCH_RAW_DIR / "all_ai_models.csv", low_memory=False)
    throughput, metadata = load_existing_throughput_cohort(data_dir)
    capabilities = canonical_hardware_capabilities(hardware_raw)
    hmap = hardware_mapping(capabilities, throughput["config_hardware"])
    mmap = model_mapping(model_raw, throughput["config_model"])
    view, validation = build_throughput_integration_view(throughput, hmap, capabilities, mmap)

    hmap.to_csv(output_root / "mappings" / "hardware_mapping.csv", index=False)
    mmap.to_csv(output_root / "mappings" / "model_mapping.csv", index=False)
    capabilities.to_csv(output_root / "views" / "canonical_hardware_capabilities.csv", index=False)
    view.to_csv(output_root / "views" / "inferencex_throughput_epoch_hardware_v1.csv", index=False)
    epoch_audit["inferencex_source"] = {
        "snapshot_release": INFERENCE_SNAPSHOT_RELEASE,
        "snapshot_date": INFERENCE_SNAPSHOT_DATE,
        "data_dir": str(Path(data_dir)),
        "cohort_builder": "scripts.build_july_pca_artifact.load_aggregate -> apps.build_analysis_frame median aggregate",
        "aggregate_rows": int(len(throughput)),
        "aggregate_configs": int(throughput["config_id"].nunique(dropna=False)),
        "throughput_target": TARGET,
        "throughput_non_null_rows": int(pd.to_numeric(throughput[TARGET], errors="coerce").notna().sum()),
        "feature_protocol": {"numeric": list(BASE_NUMERIC_FEATURES), "categorical": list(BASE_CATEGORICAL_FEATURES)},
        "dataset_manifest": metadata.get("manifest", {}),
    }
    write_json(output_root / "reports" / "epoch_source_audit.json", epoch_audit)
    write_json(output_root / "reports" / "inferencex_throughput_enrichment_validation.json", validation)

    experiment: dict[str, Any] | None = None
    holdout: dict[str, Any] | None = None
    if run_experiments:
        experiment = run_hardware_representation_experiment(view)
        write_json(output_root / "reports" / "hardware_representation_experiment.json", experiment)
        holdout = run_hardware_holdout_study(view)
        write_json(output_root / "reports" / "hardware_holdout" / "index.json", holdout)
        for record in holdout["records"]:
            write_json(
                output_root / "reports" / "hardware_holdout" / f"{_slug(record['hardware_sku'])}.json",
                record,
            )
    aa_status = artificial_analysis_status()
    prototype = {
        "prototype_version": VIEW_VERSION,
        "throughput_view": "views/inferencex_throughput_epoch_hardware_v1.csv",
        "prediction_status": "unsupported_without_a_reproducible_point_model_artifact",
        "prediction_reason": "The historical TabFM result is aggregate-only and no reusable predictor is present.",
        "observed_lookup": "Exact InferenceX measurements may be shown as observed; exact structural cross-hardware counterparts may be shown as observed what-if evidence.",
        "artificial_analysis": aa_status,
        "nearest_analogues": "Original configuration/workload mixed-distance only; no UMAP/t-SNE coordinates.",
    }
    write_json(output_root / "prototypes" / "prototype_manifest.json", prototype)
    raw_hashes_after = verify_epoch_raw_hashes()
    if raw_hashes_before != raw_hashes_after:
        raise AssertionError("Epoch raw files changed during the integration pipeline.")
    result = {
        "raw_hashes_verified": True,
        "hardware_mapping_rows": int(len(hmap)),
        "hardware_mappings_accepted": int(hmap["mapping_status"].eq("accepted").sum()),
        "hardware_mappings_unresolved": hmap.loc[hmap["mapping_status"].ne("accepted"), "inferencex_hardware"].tolist(),
        "model_mapping_rows": int(len(mmap)),
        "model_mappings_accepted": int(mmap["mapping_status"].eq("accepted").sum()),
        "model_mappings_unresolved": mmap.loc[mmap["mapping_status"].ne("accepted"), "inferencex_model"].tolist(),
        "view_rows": int(len(view)),
        "view_configs": int(view["config_id"].nunique(dropna=False)),
        "experiment_ran": experiment is not None,
        "holdout_ran": holdout is not None,
        "artificial_analysis_status": aa_status["status"],
    }
    write_json(output_root / "reports" / "integration_run_summary.json", result)
    return result
