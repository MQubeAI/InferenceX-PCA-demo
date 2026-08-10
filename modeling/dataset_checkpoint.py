"""Portable frozen-dataset identity and bootstrap support for the dashboard.

The dashboard's research artifacts are tied to one immutable CSV checkpoint.  This
module deliberately keeps machine-local details (paths and mtimes) out of that
identity and is importable without Streamlit or the neural-training environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


DATA_MANIFEST_FILENAME = "data-manifest.json"
MANIFEST_SCHEMA_VERSION = "inferencex-dashboard-data-manifest-v1"
IDENTITY_SCHEMA_VERSION = "inferencex-dataset-identity-v2"

VERIFIED_CHECKPOINT = "VERIFIED CHECKPOINT"
CUSTOM_DATASET = "CUSTOM DATASET"
LEGACY_DATASET = "LEGACY DATASET"
MISSING = "MISSING"
CHECKSUM_FAILURE = "CHECKSUM FAILURE"
ARTIFACT_INCOMPATIBLE = "ARTIFACT INCOMPATIBLE"


class DatasetManifestError(ValueError):
    """The committed manifest is invalid or cannot prove an expected property."""


class DatasetBootstrapError(RuntimeError):
    """The frozen checkpoint could not be installed safely."""


class DatasetPublicationRequiredError(DatasetBootstrapError):
    """The project has not yet published the pinned dashboard-data bundle."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    return repository_root() / DATA_MANIFEST_FILENAME


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a full-file SHA-256 digest without retaining file bytes in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetManifestError(f"{label} must be a non-empty string.")
    return value


def required_file_entries(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise DatasetManifestError("files must be a non-empty object keyed by filename.")
    entries: list[tuple[str, dict[str, Any]]] = []
    for name, metadata in files.items():
        _require_string(name, "file name")
        if Path(name).name != name:
            raise DatasetManifestError("dataset file names must not contain directories.")
        if not isinstance(metadata, dict):
            raise DatasetManifestError(f"files.{name} must be an object.")
        size = metadata.get("size_bytes")
        if not isinstance(size, int) or size < 0:
            raise DatasetManifestError(f"files.{name}.size_bytes must be a non-negative integer.")
        digest = metadata.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower())
        ):
            raise DatasetManifestError(f"files.{name}.sha256 must be a lowercase SHA-256 or null.")
        entries.append((name, metadata))
    return sorted(entries)


def validate_data_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise DatasetManifestError("Data manifest must be a JSON object.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DatasetManifestError("Data manifest schema version is incompatible.")
    _require_string(manifest.get("dataset_id"), "dataset_id")
    _require_string(manifest.get("snapshot_date"), "snapshot_date")
    _require_string(manifest.get("format"), "format")
    _require_string(manifest.get("dashboard_compatibility_version"), "dashboard_compatibility_version")
    local_directory = _require_string(manifest.get("local_directory"), "local_directory")
    local_path = Path(local_directory)
    if local_path.is_absolute() or ".." in local_path.parts:
        raise DatasetManifestError("local_directory must be a repository-relative path.")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise DatasetManifestError("source must be an object.")
    for key in ("project", "release", "release_url"):
        _require_string(source.get(key), f"source.{key}")
    required_file_entries(manifest)
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict):
        raise DatasetManifestError("bundle must be an object.")
    _require_string(bundle.get("filename"), "bundle.filename")
    if bundle.get("download_url") is not None:
        _require_string(bundle.get("download_url"), "bundle.download_url")
    if bundle.get("sha256") is not None:
        digest = bundle["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise DatasetManifestError("bundle.sha256 must be a SHA-256 or null.")
    if bundle.get("archive_format") not in {"zip"}:
        raise DatasetManifestError("Only a zip dashboard-data bundle is currently supported.")
    return manifest


def load_data_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else default_manifest_path()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetManifestError(f"Data manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"Data manifest is invalid JSON: {manifest_path}") from exc
    validate_data_manifest(manifest)
    return manifest


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash manifest metadata so a verification record identifies its exact contract."""

    validate_data_manifest(manifest)
    return _sha256_bytes(_canonical_json(manifest))


def default_data_dir(manifest: dict[str, Any] | None = None) -> Path:
    resolved_manifest = manifest or load_data_manifest()
    return repository_root() / resolved_manifest["local_directory"]


def resolve_data_dir(data_dir: str | Path) -> Path:
    value = Path(data_dir).expanduser()
    return value if value.is_absolute() else repository_root() / value


def format_required_files(manifest: dict[str, Any]) -> tuple[str, ...]:
    return tuple(name for name, _metadata in required_file_entries(manifest))


def file_identity(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "name": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def dataset_identity(
    directory: str | Path,
    file_names: Iterable[str],
    *,
    active_mode: str,
) -> dict[str, Any]:
    """Build content-only identity for a directory of active source files.

    The serialised payload intentionally contains no path, mtime, owner, or other
    machine-local attribute.  Identical bytes therefore have identical identity on
    every machine.
    """

    data_dir = Path(directory)
    files = [file_identity(data_dir / name) for name in sorted(file_names)]
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "active_mode": active_mode,
        "files": files,
    }
    return {**payload, "fingerprint": _sha256_bytes(_canonical_json(payload))}


def files_match_manifest(directory: str | Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify complete file hashes, sizes, and required file presence."""

    data_dir = Path(directory)
    errors: list[str] = []
    for name, expected in required_file_entries(manifest):
        path = data_dir / name
        if not path.is_file():
            errors.append(f"missing required file: {name}")
            continue
        actual_size = path.stat().st_size
        if actual_size != expected["size_bytes"]:
            errors.append(
                f"size mismatch for {name}: expected {expected['size_bytes']}, got {actual_size}"
            )
            continue
        expected_hash = expected.get("sha256")
        if not expected_hash:
            errors.append(f"manifest has no published SHA-256 for {name}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch for {name}")
    return not errors, errors


def verify_basic_schema(directory: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Check the inexpensive, committed CSV schema and row-count contract."""

    errors: list[str] = []
    data_dir = Path(directory)
    for name, expected in required_file_entries(manifest):
        schema = expected.get("expected")
        if not schema:
            continue
        path = data_dir / name
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    errors.append(f"CSV header is missing: {name}")
                    continue
                required_columns = schema.get("required_columns", [])
                missing_columns = sorted(set(required_columns) - set(header))
                if missing_columns:
                    errors.append(
                        f"CSV schema mismatch for {name}; missing columns: {', '.join(missing_columns)}"
                    )
                row_count = sum(1 for _row in reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            errors.append(f"could not read {name}: {exc}")
            continue
        expected_rows = schema.get("rows")
        if expected_rows is not None and row_count != expected_rows:
            errors.append(
                f"row-count mismatch for {name}: expected {expected_rows}, got {row_count}"
            )
        minimum_rows = schema.get("minimum_rows")
        if minimum_rows is not None and row_count < minimum_rows:
            errors.append(
                f"row-count mismatch for {name}: expected at least {minimum_rows}, got {row_count}"
            )
    return errors


def checkpoint_status(
    directory: str | Path,
    *,
    active_mode: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify an active source without ever treating custom data as verified."""

    expected = manifest or load_data_manifest()
    data_dir = Path(directory)
    required_names = format_required_files(expected)
    present = [name for name in required_names if (data_dir / name).is_file()]
    expected_dir = default_data_dir(expected).resolve()
    is_default_location = data_dir.resolve() == expected_dir

    if active_mode == "missing":
        return {"status": MISSING, "reason": "required data files are not available"}
    if active_mode != "Raw CSV":
        return {
            "status": LEGACY_DATASET,
            "reason": f"{active_mode} is a developer/backward-compatibility source, not the frozen raw checkpoint",
        }
    if len(present) != len(required_names):
        return {
            "status": CHECKSUM_FAILURE if is_default_location else CUSTOM_DATASET,
            "reason": "raw CSV checkpoint is incomplete",
        }
    valid, errors = files_match_manifest(data_dir, expected)
    if valid:
        return {
            "status": VERIFIED_CHECKPOINT,
            "reason": "full-file SHA-256 and schema contract verified",
            "dataset_id": expected["dataset_id"],
            "source_release": expected["source"]["release"],
            "manifest_fingerprint": manifest_fingerprint(expected),
        }
    if is_default_location:
        return {"status": CHECKSUM_FAILURE, "reason": "; ".join(errors)}
    return {
        "status": CUSTOM_DATASET,
        "reason": "; ".join(errors),
    }


def active_dataset_manifest(
    directory: str | Path,
    *,
    active_mode: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return semantic identity plus verification state for an active dataset."""

    expected = manifest or load_data_manifest()
    if active_mode == "missing":
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "active_mode": active_mode,
            "files": [],
            "fingerprint": "",
            "verification": checkpoint_status(directory, active_mode=active_mode, manifest=expected),
        }
    if active_mode == "Raw CSV":
        names = format_required_files(expected)
    elif active_mode == "CSV":
        names = ("benchmark_results.csv", "configs.csv")
    else:
        names = ("benchmark_results.json", "configs.json")
    identity = dataset_identity(directory, names, active_mode=active_mode)
    verification = checkpoint_status(directory, active_mode=active_mode, manifest=expected)
    if verification["status"] == VERIFIED_CHECKPOINT:
        identity.update(
            {
                "dataset_id": verification["dataset_id"],
                "source_release": verification["source_release"],
                "manifest_fingerprint": verification["manifest_fingerprint"],
            }
        )
    identity["verification"] = verification
    return identity


def _artifact_file_hashes(artifact_dataset: dict[str, Any]) -> list[dict[str, Any]]:
    raw_files = artifact_dataset.get("files", [])
    if isinstance(raw_files, dict):
        raw_files = [dict(metadata, name=name) for name, metadata in raw_files.items()]
    return [entry for entry in raw_files if isinstance(entry, dict) and entry.get("sha256")]


def dataset_matches_artifact(
    active_dataset: dict[str, Any],
    artifact_dataset: dict[str, Any],
) -> bool:
    """Fail closed unless a dataset can be semantically tied to an artifact.

    New artifacts may carry full file hashes.  Legacy July artifacts only contain
    a snapshot release and machine-local manifest; they are accepted only after the
    active data has independently passed the committed full-file checkpoint hashes.
    """

    if active_dataset.get("verification", {}).get("status") != VERIFIED_CHECKPOINT:
        return False
    if not isinstance(artifact_dataset, dict):
        return False
    artifact_id = artifact_dataset.get("dataset_id")
    artifact_release = (
        artifact_dataset.get("source_release")
        or artifact_dataset.get("version")
        or artifact_dataset.get("source_dump")
        or artifact_dataset.get("release")
    )
    if artifact_id and artifact_id != active_dataset.get("dataset_id"):
        return False
    if artifact_release and artifact_release != active_dataset.get("source_release"):
        return False
    hashes = _artifact_file_hashes(artifact_dataset)
    if hashes:
        active_hashes = {
            entry["name"]: (entry["size_bytes"], entry["sha256"])
            for entry in active_dataset.get("files", [])
        }
        return all(
            active_hashes.get(entry.get("name"))
            == (entry.get("size_bytes"), entry.get("sha256"))
            for entry in hashes
        )
    # Legacy artifacts must at least identify their source release.  The active
    # checkpoint's independently verified full hashes provide the missing safety.
    return bool(artifact_id or artifact_release)


def artifact_matches_active_dataset(active_dataset: dict[str, Any], artifact: dict[str, Any]) -> bool:
    """Extract standard dataset metadata from the project's artifact generations."""

    if not isinstance(artifact, dict):
        return False
    if isinstance(artifact.get("dataset"), dict):
        return dataset_matches_artifact(active_dataset, artifact["dataset"])
    if isinstance(artifact.get("dump"), dict):
        return dataset_matches_artifact(active_dataset, artifact["dump"])
    if artifact.get("source_dump"):
        return dataset_matches_artifact(
            active_dataset, {"source_release": artifact["source_dump"]}
        )
    return False


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = destination / member.filename
            if not member_path.resolve().is_relative_to(destination.resolve()):
                raise DatasetBootstrapError("Bundle contains an unsafe archive path.")
        bundle.extractall(destination)


def _download(url: str, output: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, output.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise DatasetBootstrapError(f"Dataset download failed: {exc}") from exc


def _bundle_ready(manifest: dict[str, Any]) -> tuple[str, str]:
    bundle = manifest["bundle"]
    url = bundle.get("download_url")
    digest = bundle.get("sha256")
    file_hashes_ready = all(metadata.get("sha256") for _name, metadata in required_file_entries(manifest))
    if not url or not digest or not file_hashes_ready:
        filename = bundle["filename"]
        raise DatasetPublicationRequiredError(
            "The frozen dashboard-data bundle is not published yet. Expected release asset: "
            f"{filename}. Publish its pinned URL, archive SHA-256, and full SHA-256 values for "
            "each required CSV in data-manifest.json; do not use the multi-gigabyte upstream dump "
            "for normal dashboard startup."
        )
    return url, digest


def _candidate_directory(extracted_root: Path, manifest: dict[str, Any]) -> Path:
    archive_root = manifest["bundle"].get("archive_root", "")
    candidate = extracted_root / archive_root if archive_root else extracted_root
    if not candidate.is_dir():
        raise DatasetBootstrapError("Dataset bundle does not contain the configured archive root.")
    return candidate


def bootstrap_dataset(
    *,
    manifest_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    offline: bool = False,
) -> Path:
    """Install a verified checkpoint atomically, or explain why it cannot be obtained."""

    manifest = load_data_manifest(manifest_path)
    destination = resolve_data_dir(data_dir) if data_dir else default_data_dir(manifest)
    valid, errors = files_match_manifest(destination, manifest)
    if valid and not verify_basic_schema(destination, manifest):
        return destination
    if offline:
        detail = "; ".join(errors) if errors else "schema validation failed"
        raise DatasetBootstrapError(f"Dataset is not verified and offline mode is enabled: {detail}")

    url, expected_archive_hash = _bundle_ready(manifest)
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bootstrap-", dir=destination_parent) as temporary:
        staging = Path(temporary)
        archive = staging / manifest["bundle"]["filename"]
        _download(url, archive)
        actual_archive_hash = sha256_file(archive)
        if actual_archive_hash != expected_archive_hash:
            raise DatasetBootstrapError(
                "Dataset archive checksum mismatch; the downloaded bundle was not installed."
            )
        extracted = staging / "extracted"
        extracted.mkdir()
        try:
            _safe_extract_zip(archive, extracted)
        except zipfile.BadZipFile as exc:
            raise DatasetBootstrapError("Dataset download is not a valid zip archive.") from exc
        candidate = _candidate_directory(extracted, manifest)
        candidate_valid, candidate_errors = files_match_manifest(candidate, manifest)
        candidate_errors.extend(verify_basic_schema(candidate, manifest))
        if not candidate_valid or candidate_errors:
            raise DatasetBootstrapError(
                "Extracted dataset failed verification: " + "; ".join(candidate_errors)
            )
        verification_record = {
            "dataset_id": manifest["dataset_id"],
            "manifest_fingerprint": manifest_fingerprint(manifest),
            "verified_at_utc": datetime.now(UTC).isoformat(),
        }
        (candidate / ".checkpoint-verification.json").write_text(
            json.dumps(verification_record, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            backup = destination.with_name(
                f"{destination.name}.invalid-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            )
            os.replace(destination, backup)
        os.replace(candidate, destination)
    return destination
