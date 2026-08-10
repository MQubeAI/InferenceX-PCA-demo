"""Versioned research-artifact bootstrap for promoted dashboard checkpoints.

July artifacts stay in Git for backward compatibility.  Future snapshot artifacts
can live in immutable release bundles so repeated refreshes do not grow normal
Git history.  The active data manifest and artifact index must agree exactly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from modeling.dataset_checkpoint import DatasetBootstrapError, repository_root, sha256_file


ARTIFACT_INDEX_PATH = repository_root() / "data" / "artifact-index.json"
REQUIRED_ARTIFACT_KEYS = ("pca", "ae", "vae", "vae_beta", "comparison", "stage4", "model_summary")


class ArtifactIndexError(ValueError):
    """The active artifact pointer is missing or cannot be trusted."""


def load_artifact_index(path: str | Path | None = None) -> dict[str, Any]:
    index_path = Path(path) if path else ARTIFACT_INDEX_PATH
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIndexError(f"Artifact index is unreadable: {index_path}") from exc
    if index.get("schema_version") != "inferencex-artifact-index-v1":
        raise ArtifactIndexError("Artifact index schema is incompatible.")
    if not isinstance(index.get("snapshots"), dict):
        raise ArtifactIndexError("Artifact index snapshots must be an object.")
    return index


def _active_entry(manifest: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    dataset_id = manifest.get("dataset_id")
    if index.get("active_dataset_id") != dataset_id:
        raise ArtifactIndexError("Active artifact index does not match data-manifest.json.")
    entry = index["snapshots"].get(dataset_id)
    if not isinstance(entry, dict) or entry.get("status") != "promoted":
        raise ArtifactIndexError("No promoted artifact entry exists for the active checkpoint.")
    if entry.get("source_release") != manifest.get("source", {}).get("release"):
        raise ArtifactIndexError("Active artifact source release differs from data-manifest.json.")
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, dict) or set(REQUIRED_ARTIFACT_KEYS) - set(artifacts):
        raise ArtifactIndexError("Active artifact entry is incomplete.")
    return entry


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = destination / member.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise DatasetBootstrapError("Artifact bundle contains an unsafe archive path.")
        bundle.extractall(destination)


def _download(url: str, output: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=120) as response, output.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise DatasetBootstrapError(f"Research artifact download failed: {exc}") from exc


def _artifact_filename(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("filename"), str):
        return value["filename"]
    raise ArtifactIndexError("Artifact entry must name a filename.")


def active_artifact_paths(
    manifest: dict[str, Any],
    *,
    index_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the promoted artifact set without falling back to another snapshot."""

    index = load_artifact_index(index_path)
    entry = _active_entry(manifest, index)
    storage = entry.get("storage")
    root = repository_root()
    if storage == "repository":
        paths = {key: root / _artifact_filename(value) for key, value in entry["artifacts"].items()}
    elif storage == "release_bundle":
        base = Path(artifact_root) if artifact_root else root / ".artifacts" / manifest["dataset_id"]
        paths = {key: base / _artifact_filename(value) for key, value in entry["artifacts"].items()}
    else:
        raise ArtifactIndexError("Active artifact storage must be repository or release_bundle.")
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if not paths[key].is_file()]
    if missing:
        raise ArtifactIndexError("Active research artifacts are missing: " + ", ".join(missing))
    return {key: paths[key] for key in REQUIRED_ARTIFACT_KEYS}


def bootstrap_active_artifacts(
    manifest: dict[str, Any],
    *,
    index_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    offline: bool = False,
) -> dict[str, Path]:
    """Install a release-bundled promoted artifact set atomically when required."""

    index = load_artifact_index(index_path)
    entry = _active_entry(manifest, index)
    if entry.get("storage") == "repository":
        return active_artifact_paths(manifest, index_path=index_path, artifact_root=artifact_root)
    bundle = entry.get("bundle")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("download_url"), str):
        raise ArtifactIndexError("Release-bundled active artifacts require a pinned download URL.")
    expected_hash = bundle.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ArtifactIndexError("Release-bundled active artifacts require a full SHA-256.")
    target = Path(artifact_root) if artifact_root else repository_root() / ".artifacts" / manifest["dataset_id"]
    try:
        return active_artifact_paths(manifest, index_path=index_path, artifact_root=target)
    except ArtifactIndexError:
        if offline:
            raise DatasetBootstrapError("Active research artifacts are missing and offline mode is enabled.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".artifact-bootstrap-", dir=target.parent) as temporary:
        staging = Path(temporary)
        archive = staging / bundle.get("filename", "research-artifacts.zip")
        _download(bundle["download_url"], archive)
        if sha256_file(archive) != expected_hash:
            raise DatasetBootstrapError("Research artifact archive checksum mismatch.")
        extracted = staging / "extracted"
        extracted.mkdir()
        try:
            _safe_extract(archive, extracted)
        except zipfile.BadZipFile as exc:
            raise DatasetBootstrapError("Research artifact download is not a valid ZIP.") from exc
        for key, descriptor in entry["artifacts"].items():
            source = extracted / _artifact_filename(descriptor)
            if not source.is_file():
                raise DatasetBootstrapError(f"Research artifact bundle is missing {key}.")
            if isinstance(descriptor, dict):
                if source.stat().st_size != descriptor.get("size_bytes") or sha256_file(source) != descriptor.get("sha256"):
                    raise DatasetBootstrapError(f"Research artifact checksum mismatch: {key}.")
        if target.exists():
            shutil.rmtree(target)
        extracted.replace(target)
    return active_artifact_paths(manifest, index_path=index_path, artifact_root=target)
