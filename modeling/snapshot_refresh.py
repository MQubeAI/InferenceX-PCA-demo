"""Versioned, fail-closed primitives for InferenceX snapshot refreshes.

This module deliberately does not know how to promote a candidate.  It builds
and audits immutable candidates; an explicit reviewed change is responsible for
making a candidate active.  The standard library implementation keeps release
discovery and ordinary tests independent of PostgreSQL, Streamlit, or a model
training runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from modeling.dataset_checkpoint import (
    DATA_MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    DatasetManifestError,
    load_data_manifest,
    sha256_file,
    validate_data_manifest,
)
from modeling.pca_target_analysis import PCA_FEATURES


OFFICIAL_PROJECT = "SemiAnalysisAI/InferenceX-app"
OFFICIAL_RELEASES_API = f"https://api.github.com/repos/{OFFICIAL_PROJECT}/releases"
RELEASE_TAG_PATTERN = re.compile(r"^db-dump/(\d{4}-\d{2}-\d{2})$")
VERSIONED_MANIFEST_SCHEMA_VERSION = "inferencex-snapshot-manifest-v1"
REFRESH_POLICY_SCHEMA_VERSION = "inferencex-refresh-policy-v1"
REQUIRED_BENCHMARK_COLUMNS = ("id", "config_id", "benchmark_type", "isl", "osl", "conc", "metrics")
REQUIRED_CONFIG_COLUMNS = (
    "id",
    "hardware",
    "framework",
    "model",
    "precision",
    "spec_method",
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_prefill_gpu",
)
REQUIRED_TARGET_METRICS = ("median_tpot", "tput_per_gpu", "joules_per_output_token")
SNAPSHOT_STATUSES = {"candidate", "promoted", "rejected"}
SCHEMA_SAFE_ADDITIVE = "SAFE_ADDITIVE"
SCHEMA_REVIEW_REQUIRED = "REVIEW_REQUIRED"
SCHEMA_INCOMPATIBLE = "INCOMPATIBLE"


class SnapshotRefreshError(RuntimeError):
    """A candidate cannot be trusted or is not ready to advance."""


@dataclass(frozen=True)
class OfficialRelease:
    """The minimal verified release metadata needed by candidate ingestion."""

    tag: str
    snapshot_date: str
    html_url: str
    assets: tuple[dict[str, Any], ...]
    published_at: str | None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_scalar(value: Any) -> str:
    if value is None:
        return "<NA>"
    if isinstance(value, float) and math.isnan(value):
        return "<NA>"
    return str(value)


def parse_official_release(payload: dict[str, Any], *, allow_prerelease: bool = False) -> OfficialRelease | None:
    """Parse one official GitHub release, ignoring unrelated releases.

    A malformed ``db-dump/`` tag is an error because it looks like an intended
    upstream snapshot but cannot safely be ordered or replayed.  Other release
    families are intentionally ignored.
    """

    tag = payload.get("tag_name")
    if not isinstance(tag, str):
        return None
    if payload.get("draft") or (payload.get("prerelease") and not allow_prerelease):
        return None
    if not tag.startswith("db-dump/"):
        return None
    match = RELEASE_TAG_PATTERN.fullmatch(tag)
    if not match:
        raise SnapshotRefreshError(f"Invalid official db-dump tag: {tag}")
    snapshot_date = match.group(1)
    try:
        datetime.strptime(snapshot_date, "%Y-%m-%d")
    except ValueError as exc:
        raise SnapshotRefreshError(f"Invalid official snapshot date: {snapshot_date}") from exc
    url = payload.get("html_url")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        raise SnapshotRefreshError(f"Official release {tag} has no immutable GitHub URL.")
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise SnapshotRefreshError(f"Official release {tag} has invalid asset metadata.")
    assets = tuple(asset for asset in raw_assets if isinstance(asset, dict))
    return OfficialRelease(
        tag=tag,
        snapshot_date=snapshot_date,
        html_url=url,
        assets=assets,
        published_at=payload.get("published_at") if isinstance(payload.get("published_at"), str) else None,
    )


def fetch_official_releases(
    *,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    allow_prerelease: bool = False,
) -> list[OfficialRelease]:
    """Fetch only the official upstream release API and return usable db dumps."""

    request = urllib.request.Request(
        OFFICIAL_RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "InferenceX-refresh"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotRefreshError(f"Could not read official release metadata: {exc}") from exc
    if not isinstance(payload, list):
        raise SnapshotRefreshError("Official release API returned a non-list payload.")
    releases = [
        release
        for item in payload
        if (release := parse_official_release(item, allow_prerelease=allow_prerelease)) is not None
    ]
    return sorted(releases, key=lambda release: release.snapshot_date)


def discover_new_release(
    releases: Iterable[OfficialRelease | dict[str, Any]],
    active_release: str,
    *,
    allow_prerelease: bool = False,
) -> dict[str, Any]:
    """Classify official upstream metadata without downloading a dump."""

    active_match = RELEASE_TAG_PATTERN.fullmatch(active_release)
    if not active_match:
        return {"status": "INVALID_RELEASE_METADATA", "reason": "active manifest source.release is malformed"}
    try:
        parsed = [
            item if isinstance(item, OfficialRelease) else parse_official_release(item, allow_prerelease=allow_prerelease)
            for item in releases
        ]
    except SnapshotRefreshError as exc:
        return {"status": "INVALID_RELEASE_METADATA", "reason": str(exc)}
    available = [item for item in parsed if item is not None]
    if not available:
        return {"status": "INVALID_RELEASE_METADATA", "reason": "no valid official db-dump release was supplied"}
    newest = max(available, key=lambda release: release.snapshot_date)
    active_date = active_match.group(1)
    if newest.snapshot_date <= active_date:
        return {
            "status": "NO_NEW_RELEASE",
            "active_release": active_release,
            "newest_release": newest.tag,
        }
    return {
        "status": "NEW_RELEASE_AVAILABLE",
        "active_release": active_release,
        "newest_release": newest.tag,
        "release_url": newest.html_url,
        "snapshot_date": newest.snapshot_date,
        "assets": list(newest.assets),
    }


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise SnapshotRefreshError(f"CSV header is missing: {path.name}")
    if len(header) != len(set(header)):
        raise SnapshotRefreshError(f"CSV header contains duplicate names: {path.name}")
    return header


def _infer_scalar_type(value: str) -> str:
    normalized = value.strip()
    if normalized == "":
        return "null"
    if normalized.lower() in {"t", "f", "true", "false"}:
        return "boolean"
    try:
        int(normalized)
    except ValueError:
        try:
            float(normalized)
        except ValueError:
            return "string"
        return "number"
    return "integer"


def _column_types(path: Path, header: list[str], *, sample_limit: int = 10_000) -> dict[str, str]:
    observed: dict[str, set[str]] = {column: set() for column in header}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= sample_limit:
                break
            for column in header:
                observed[column].add(_infer_scalar_type(row.get(column, "")))
    result: dict[str, str] = {}
    for column, values in observed.items():
        nonnull = values - {"null"}
        result[column] = "null" if not nonnull else "+".join(sorted(nonnull))
    return result


def _metric_payload(value: str, row_number: int) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SnapshotRefreshError(f"Invalid metrics JSON at benchmark row {row_number}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SnapshotRefreshError(f"metrics is not an object at benchmark row {row_number}")
    return parsed


def inspect_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Validate raw tables and produce deterministic candidate metadata."""

    root = Path(directory)
    benchmark_path = root / "benchmark_results_raw.csv"
    configs_path = root / "configs.csv"
    if not benchmark_path.is_file() or not configs_path.is_file():
        raise SnapshotRefreshError("Checkpoint requires benchmark_results_raw.csv and configs.csv.")
    benchmark_header = _read_csv_header(benchmark_path)
    configs_header = _read_csv_header(configs_path)
    missing_benchmark = sorted(set(REQUIRED_BENCHMARK_COLUMNS) - set(benchmark_header))
    missing_configs = sorted(set(REQUIRED_CONFIG_COLUMNS) - set(configs_header))
    if missing_benchmark or missing_configs:
        raise SnapshotRefreshError(
            "Required schema is missing: "
            + "; ".join(
                part
                for part in (
                    f"benchmark={','.join(missing_benchmark)}" if missing_benchmark else "",
                    f"configs={','.join(missing_configs)}" if missing_configs else "",
                )
                if part
            )
        )

    config_ids: set[str] = set()
    config_duplicates = 0
    config_categories: dict[str, set[str]] = {
        column: set() for column in configs_header if column != "id"
    }
    with configs_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            config_id = row.get("id", "")
            if not config_id:
                raise SnapshotRefreshError("configs.csv contains an empty id.")
            if config_id in config_ids:
                config_duplicates += 1
            config_ids.add(config_id)
            for column in config_categories:
                if row.get(column, ""):
                    config_categories[column].add(row[column])
    if config_duplicates:
        raise SnapshotRefreshError(f"configs.csv contains {config_duplicates} duplicate id values.")

    benchmark_ids: set[str] = set()
    benchmark_duplicates = 0
    benchmark_config_ids: set[str] = set()
    metric_names: set[str] = set()
    missingness = Counter({metric: 0 for metric in REQUIRED_TARGET_METRICS})
    benchmark_types: Counter[str] = Counter()
    workload_values: dict[str, set[str]] = {column: set() for column in ("isl", "osl", "conc")}
    dates: list[str] = []
    raw_rows = 0
    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_rows, row in enumerate(csv.DictReader(handle), 1):
            benchmark_id = row.get("id", "")
            if not benchmark_id:
                raise SnapshotRefreshError(f"benchmark_results_raw.csv has an empty id at row {raw_rows}.")
            if benchmark_id in benchmark_ids:
                benchmark_duplicates += 1
            benchmark_ids.add(benchmark_id)
            config_id = row.get("config_id", "")
            if not config_id:
                raise SnapshotRefreshError(f"benchmark row {raw_rows} has an empty config_id.")
            benchmark_config_ids.add(config_id)
            metrics = _metric_payload(row.get("metrics", ""), raw_rows)
            metric_names.update(metrics)
            for metric in REQUIRED_TARGET_METRICS:
                if metrics.get(metric) is None:
                    missingness[metric] += 1
            benchmark_types[row.get("benchmark_type", "<NA>")] += 1
            for column in workload_values:
                workload_values[column].add(row.get(column, "<NA>"))
            if row.get("date"):
                dates.append(row["date"])
    if benchmark_duplicates:
        raise SnapshotRefreshError(f"benchmark_results_raw.csv contains {benchmark_duplicates} duplicate id values.")
    unresolved = sorted(benchmark_config_ids - config_ids)
    if unresolved:
        raise SnapshotRefreshError(
            f"{len(unresolved)} benchmark config_id values do not resolve in configs.csv."
        )

    def file_entry(path: Path, header: list[str], rows: int) -> dict[str, Any]:
        return {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "physical_rows": rows,
            "header": header,
            "column_types": _column_types(path, header),
        }

    return {
        "files": {
            "benchmark_results_raw.csv": file_entry(benchmark_path, benchmark_header, raw_rows),
            "configs.csv": file_entry(configs_path, configs_header, len(config_ids)),
        },
        "row_counts": {
            "benchmark_results": raw_rows,
            "configs_physical": len(config_ids),
            "benchmark_referenced_configs": len(benchmark_config_ids),
        },
        "integrity": {
            "metrics_json_valid": True,
            "config_ids_unique": True,
            "benchmark_ids_unique": True,
            "benchmark_config_join_missing": 0,
        },
        "coverage": {
            "date_range": [min(dates), max(dates)] if dates else [None, None],
            "benchmark_types": dict(sorted(benchmark_types.items())),
            "workloads": {key: sorted(values) for key, values in workload_values.items()},
            "config_categories": {key: sorted(values) for key, values in config_categories.items()},
            "metric_columns": sorted(metric_names),
            "target_missingness": {
                metric: {"missing_rows": missingness[metric], "present_rows": raw_rows - missingness[metric]}
                for metric in REQUIRED_TARGET_METRICS
            },
            "energy_coverage": {
                "rows": raw_rows - missingness["joules_per_output_token"],
                "configurations": None,
            },
        },
    }


def deterministic_csv_export(
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
    destination: str | Path,
    *,
    order_columns: tuple[str, ...],
) -> Path:
    """Write a CSV in a stable cross-machine order and newline format."""

    if not order_columns or any(column not in fieldnames for column in order_columns):
        raise SnapshotRefreshError("Deterministic export order must use declared CSV columns.")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(stable_scalar(row.get(column)) for column in order_columns),
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(ordered)
    return output


def create_checkpoint_bundle(directory: str | Path, output: str | Path) -> dict[str, Any]:
    """Create a deterministic, root-only ZIP and verify its extracted contents."""

    root = Path(directory)
    output_path = Path(output)
    required = ("benchmark_results_raw.csv", "configs.csv")
    for name in required:
        if not (root / name).is_file():
            raise SnapshotRefreshError(f"Cannot package missing checkpoint file: {name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in required:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (root / name).read_bytes())
    with zipfile.ZipFile(output_path) as archive:
        names = archive.namelist()
        if names != list(required):
            raise SnapshotRefreshError("Checkpoint ZIP must contain exactly the two root CSVs.")
        extracted_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in required
        }
    source_hashes = {name: sha256_file(root / name) for name in required}
    if extracted_hashes != source_hashes:
        raise SnapshotRefreshError("Checkpoint ZIP extraction did not reproduce source file hashes.")
    return {
        "filename": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "files": source_hashes,
    }


def _normalized_rows(path: Path, *, key: str, normalize_metrics: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            identifier = row.get(key, "")
            if not identifier:
                raise SnapshotRefreshError(f"{path.name} has an empty {key} at row {row_number}.")
            if normalize_metrics and "metrics" in row:
                row["metrics"] = json.dumps(_metric_payload(row["metrics"], row_number), sort_keys=True, separators=(",", ":"))
            result[identifier] = canonical_json(row).decode("utf-8")
    return result


def classify_schema_difference(active: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify schema changes; removal/type change of required fields is fatal."""

    tables = {}
    overall = SCHEMA_SAFE_ADDITIVE
    for filename, required in (
        ("benchmark_results_raw.csv", set(REQUIRED_BENCHMARK_COLUMNS)),
        ("configs.csv", set(REQUIRED_CONFIG_COLUMNS)),
    ):
        active_file = active["files"][filename]
        candidate_file = candidate["files"][filename]
        active_columns = set(active_file["header"])
        candidate_columns = set(candidate_file["header"])
        removed = sorted(active_columns - candidate_columns)
        added = sorted(candidate_columns - active_columns)
        missing_required = sorted(required - candidate_columns)
        type_changes = sorted(
            column
            for column in active_columns & candidate_columns
            if active_file["column_types"].get(column) != candidate_file["column_types"].get(column)
            and column in required
        )
        if missing_required or type_changes:
            classification = SCHEMA_INCOMPATIBLE
            overall = SCHEMA_INCOMPATIBLE
        elif removed:
            classification = SCHEMA_REVIEW_REQUIRED
            if overall == SCHEMA_SAFE_ADDITIVE:
                overall = SCHEMA_REVIEW_REQUIRED
        else:
            classification = SCHEMA_SAFE_ADDITIVE
        tables[filename] = {
            "classification": classification,
            "added_columns": added,
            "removed_columns": removed,
            "missing_required_columns": missing_required,
            "required_field_type_changes": type_changes,
        }
    return {"classification": overall, "tables": tables}


def audit_candidate_against_active(active_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    """Compare data/schema/coverage by stable identifiers, not source order."""

    active = inspect_checkpoint(active_dir)
    candidate = inspect_checkpoint(candidate_dir)
    benchmark_active = _normalized_rows(Path(active_dir) / "benchmark_results_raw.csv", key="id", normalize_metrics=True)
    benchmark_candidate = _normalized_rows(Path(candidate_dir) / "benchmark_results_raw.csv", key="id", normalize_metrics=True)
    config_active = _normalized_rows(Path(active_dir) / "configs.csv", key="id")
    config_candidate = _normalized_rows(Path(candidate_dir) / "configs.csv", key="id")

    def delta(left: dict[str, str], right: dict[str, str]) -> dict[str, int]:
        left_ids, right_ids = set(left), set(right)
        return {
            "added_ids": len(right_ids - left_ids),
            "missing_prior_ids": len(left_ids - right_ids),
            "changed_existing_ids": sum(left[item] != right[item] for item in left_ids & right_ids),
        }

    schema = classify_schema_difference(active, candidate)
    benchmark_delta = delta(benchmark_active, benchmark_candidate)
    config_delta = delta(config_active, config_candidate)
    classification = "HARD_INTEGRITY_FAILURE" if schema["classification"] == SCHEMA_INCOMPATIBLE else "EXPECTED_DATASET_DRIFT"
    if classification != "HARD_INTEGRITY_FAILURE" and (
        benchmark_delta["missing_prior_ids"]
        or benchmark_delta["changed_existing_ids"]
        or config_delta["missing_prior_ids"]
        or config_delta["changed_existing_ids"]
    ):
        classification = "REVIEW_REQUIRED"
    return {
        "schema_version": "inferencex-candidate-vs-active-audit-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "schema": schema,
        "row_count_delta": {
            key: candidate["row_counts"][key] - active["row_counts"][key]
            for key in active["row_counts"]
        },
        "benchmark_ids": benchmark_delta,
        "config_ids": config_delta,
        "active": active,
        "candidate": candidate,
        "classification": classification,
    }


def candidate_manifest(
    *,
    checkpoint_dir: str | Path,
    source_release: OfficialRelease,
    parent_manifest: dict[str, Any],
    bundle: dict[str, Any],
    generation: dict[str, Any],
    status: str = "candidate",
) -> dict[str, Any]:
    """Create, but never activate, an immutable versioned candidate manifest."""

    if status not in SNAPSHOT_STATUSES:
        raise SnapshotRefreshError(f"Unknown snapshot status: {status}")
    inspection = inspect_checkpoint(checkpoint_dir)
    date = source_release.snapshot_date
    files = inspection["files"]
    return {
        "schema_version": VERSIONED_MANIFEST_SCHEMA_VERSION,
        "dataset_id": f"inferencex-db-dump-{date}",
        "status": status,
        "parent_active_checkpoint": parent_manifest["dataset_id"],
        "source": {
            "project": OFFICIAL_PROJECT,
            "release": source_release.tag,
            "release_url": source_release.html_url,
            "published_at": source_release.published_at,
            "assets": [
                {key: asset[key] for key in ("name", "size", "browser_download_url") if key in asset}
                for asset in source_release.assets
            ],
        },
        "snapshot_date": date,
        "format": "raw_csv",
        "dashboard_compatibility_version": f"{date}-v1",
        "local_directory": f".data/inferencex-db-dump-{date}",
        "files": {
            name: {
                "size_bytes": metadata["size_bytes"],
                "sha256": metadata["sha256"],
                "expected": {
                    "rows": metadata["physical_rows"],
                    "required_columns": REQUIRED_BENCHMARK_COLUMNS if name.startswith("benchmark") else REQUIRED_CONFIG_COLUMNS,
                },
                "physical_rows": metadata["physical_rows"],
                "header": metadata["header"],
            }
            for name, metadata in files.items()
        },
        "row_counts": inspection["row_counts"],
        "required_schema": {
            "benchmark_results_raw.csv": list(REQUIRED_BENCHMARK_COLUMNS),
            "configs.csv": list(REQUIRED_CONFIG_COLUMNS),
        },
        "bundle": {
            "filename": bundle["filename"],
            "download_url": bundle.get("download_url"),
            "sha256": bundle["sha256"],
            "size_bytes": bundle["size_bytes"],
            "archive_format": "zip",
            "archive_root": "",
        },
        "generation": generation,
        "integrity": inspection["integrity"],
        "coverage": inspection["coverage"],
    }


def active_manifest_from_versioned(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project a promoted versioned manifest into legacy active-manifest shape."""

    if manifest.get("status") != "promoted":
        raise SnapshotRefreshError("Only a promoted versioned manifest may become active.")
    active = {
        key: manifest[key]
        for key in (
            "dataset_id",
            "source",
            "snapshot_date",
            "format",
            "dashboard_compatibility_version",
            "local_directory",
            "files",
            "bundle",
        )
    }
    active["schema_version"] = MANIFEST_SCHEMA_VERSION
    active["upstream_recovery"] = manifest.get("upstream_recovery", {"normal_dashboard_use": "not required"})
    validate_data_manifest(active)
    return active


def snapshot_artifact_names(snapshot_date: str) -> dict[str, str]:
    """The snapshot-specific names required by the fixed research protocol."""

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_date):
        raise SnapshotRefreshError("Snapshot artifact names require YYYY-MM-DD.")
    suffix = f"db-dump-{snapshot_date}"
    return {
        "pca": f"pca-{suffix}.json",
        "ae_json": f"representation-ae-final-{suffix}.json",
        "ae_embeddings": f"representation-ae-final-{suffix}.parquet",
        "ae_weights": f"representation-ae-final-{suffix}.pt",
        "vae_json": f"representation-vae-final-{suffix}.json",
        "vae_embeddings": f"representation-vae-final-{suffix}.parquet",
        "vae_weights": f"representation-vae-final-{suffix}.pt",
        "comparison": f"representation-comparison-final-{suffix}.json",
        "stage4": f"representation-validation-stage4-{suffix}.json",
        "tabfm_evaluation": f"tabfm-throughput-{suffix}.json",
        "uncertainty": f"throughput-uncertainty-{suffix}.json",
        "model_summary": f"model-research-summary-{suffix}.json",
    }


def evaluate_refresh_policy(
    audit: dict[str, Any],
    *,
    research: dict[str, Any] | None = None,
    clean_clone_passed: bool = False,
) -> dict[str, Any]:
    """Evaluate explicit integrity gates without inventing performance cutoffs."""

    integrity = audit["candidate"]["integrity"]
    hard_failures: list[str] = []
    if audit["schema"]["classification"] == SCHEMA_INCOMPATIBLE:
        hard_failures.append("incompatible required schema")
    for key, expected in (
        ("metrics_json_valid", True),
        ("config_ids_unique", True),
        ("benchmark_ids_unique", True),
    ):
        if integrity.get(key) is not expected:
            hard_failures.append(f"integrity gate failed: {key}")
    if integrity.get("benchmark_config_join_missing") != 0:
        hard_failures.append("benchmark/config join integrity failed")
    if not clean_clone_passed:
        hard_failures.append("clean-clone validation not passed")
    research = research or {}
    missing_research = [
        name
        for name in ("pca", "ae", "vae", "comparison", "stage4", "tabfm", "uncertainty")
        if research.get(name, {}).get("status") != "PASS"
    ]
    if missing_research:
        hard_failures.append("mandatory research gate incomplete: " + ", ".join(missing_research))
    return {
        "schema_version": REFRESH_POLICY_SCHEMA_VERSION,
        "hard_failures": hard_failures,
        "research_gate_status": research,
        "promotion_status": "FAIL" if hard_failures else "REVIEW_REQUIRED",
        "note": (
            "A passing candidate is review-required by design; no score threshold auto-promotes a snapshot."
            if not hard_failures
            else "Candidate cannot be promoted until every hard failure is resolved."
        ),
    }


def refresh_report_markdown(candidate_manifest: dict[str, Any], audit: dict[str, Any], policy: dict[str, Any]) -> str:
    """Generate a small human-readable companion to machine audit JSON."""

    lines = [
        f"# InferenceX candidate refresh: {candidate_manifest['source']['release']}",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{candidate_manifest['dataset_id']}`",
        f"- Parent active checkpoint: `{candidate_manifest['parent_active_checkpoint']}`",
        f"- Official release: {candidate_manifest['source']['release_url']}",
        f"- Bundle SHA-256: `{candidate_manifest['bundle']['sha256']}`",
        "",
        "## Data and schema",
        "",
        f"- Schema: **{audit['schema']['classification']}**",
        f"- Benchmark rows delta: {audit['row_count_delta']['benchmark_results']:+d}",
        f"- Physical configs delta: {audit['row_count_delta']['configs_physical']:+d}",
        f"- Referenced configs delta: {audit['row_count_delta']['benchmark_referenced_configs']:+d}",
        f"- Added benchmark IDs: {audit['benchmark_ids']['added_ids']}",
        f"- Missing prior benchmark IDs: {audit['benchmark_ids']['missing_prior_ids']}",
        f"- Changed existing benchmark IDs: {audit['benchmark_ids']['changed_existing_ids']}",
        f"- Added config IDs: {audit['config_ids']['added_ids']}",
        f"- Missing prior config IDs: {audit['config_ids']['missing_prior_ids']}",
        f"- Changed existing config IDs: {audit['config_ids']['changed_existing_ids']}",
        f"- Candidate date range: {audit['candidate']['coverage']['date_range']}",
        f"- Candidate benchmark types: {', '.join(audit['candidate']['coverage']['benchmark_types'])}",
        f"- Candidate metric columns: {', '.join(audit['candidate']['coverage']['metric_columns'])}",
        f"- Candidate energy rows: {audit['candidate']['coverage']['energy_coverage']['rows']}",
        "",
        "## Fixed research evidence",
        "",
        *[
            f"- {name}: **{state.get('status', 'UNKNOWN')}**"
            for name, state in sorted(policy.get("research_gate_status", {}).items())
        ],
        "",
        "## Promotion gate",
        "",
        f"- Status: **{policy['promotion_status']}**",
        *[f"- Hard failure: {item}" for item in policy["hard_failures"]],
        "",
        "This report is candidate-only. Merging a reviewed promotion change is the only way to alter the active checkpoint.",
    ]
    return "\n".join(lines) + "\n"


def load_active_manifest(repository_root: str | Path) -> dict[str, Any]:
    return load_data_manifest(Path(repository_root) / DATA_MANIFEST_FILENAME)


def copy_promoted_manifest(versioned_manifest_path: str | Path, active_manifest_path: str | Path) -> dict[str, Any]:
    """Write the active pointer only from an explicitly promoted manifest."""

    source = Path(versioned_manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    active = active_manifest_from_versioned(manifest)
    destination = Path(active_manifest_path)
    destination.write_text(json.dumps(active, indent=2) + "\n", encoding="utf-8")
    return active
