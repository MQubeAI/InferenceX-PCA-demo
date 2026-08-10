from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from apps import inferencex_pca_demo as app
from modeling.dataset_checkpoint import (
    CHECKSUM_FAILURE,
    DatasetBootstrapError,
    DatasetPublicationRequiredError,
    VERIFIED_CHECKPOINT,
    active_dataset_manifest,
    artifact_matches_active_dataset,
    bootstrap_dataset,
    checkpoint_status,
    sha256_file,
)


PCA_ARTIFACT = Path("artifacts/pca-db-dump-2026-07-20.json")


def write_raw_checkpoint(directory: Path, *, changed: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metrics = '{"tput_per_gpu": 10.0}' if not changed else '{"tput_per_gpu": 11.0}'
    (directory / "benchmark_results_raw.csv").write_text(
        f"config_id,metrics\n1,\"{metrics.replace(chr(34), chr(34) * 2)}\"\n",
        encoding="utf-8",
    )
    (directory / "configs.csv").write_text(
        "id,hardware\n1,gpu\n",
        encoding="utf-8",
    )


def fixture_manifest(directory: Path, *, bundle_url: str | None = None, bundle_sha256: str | None = None) -> dict:
    return {
        "schema_version": "inferencex-dashboard-data-manifest-v1",
        "dataset_id": "inferencex-db-dump-2026-07-20",
        "source": {
            "project": "SemiAnalysisAI/InferenceX-app",
            "release": "db-dump/2026-07-20",
            "release_url": "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-20",
        },
        "snapshot_date": "2026-07-20",
        "format": "raw_csv",
        "dashboard_compatibility_version": "test-v1",
        "local_directory": ".data/inferencex-db-dump-2026-07-20",
        "files": {
            name: {
                "size_bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
                "expected": {
                    "rows": 1,
                    "required_columns": columns,
                },
            }
            for name, columns in {
                "benchmark_results_raw.csv": ["config_id", "metrics"],
                "configs.csv": ["id"],
            }.items()
        },
        "bundle": {
            "filename": "fixture-dashboard-data.zip",
            "download_url": bundle_url,
            "sha256": bundle_sha256,
            "archive_format": "zip",
            "archive_root": "",
        },
    }


class PortableIdentityTests(unittest.TestCase):
    def test_same_bytes_in_different_directories_match_july_pca_despite_mtime_changes(self) -> None:
        artifact = json.loads(PCA_ARTIFACT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alice = root / "alice" / "checkpoint"
            bob = root / "bob" / "checkpoint"
            write_raw_checkpoint(alice)
            shutil.copytree(alice, bob)
            os.utime(bob / "benchmark_results_raw.csv", (1_000_000_000, 1_000_000_000))
            manifest = fixture_manifest(alice)

            alice_identity = active_dataset_manifest(alice, active_mode="Raw CSV", manifest=manifest)
            bob_identity = active_dataset_manifest(bob, active_mode="Raw CSV", manifest=manifest)

            self.assertEqual(alice_identity["fingerprint"], bob_identity["fingerprint"])
            self.assertEqual(alice_identity["files"], bob_identity["files"])
            self.assertEqual(alice_identity["verification"]["status"], VERIFIED_CHECKPOINT)
            self.assertEqual(bob_identity["verification"]["status"], VERIFIED_CHECKPOINT)
            self.assertTrue(artifact_matches_active_dataset(alice_identity, artifact))
            self.assertTrue(artifact_matches_active_dataset(bob_identity, artifact))

    def test_one_changed_file_fails_compatibility(self) -> None:
        artifact = json.loads(PCA_ARTIFACT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            changed = root / "changed"
            write_raw_checkpoint(original)
            write_raw_checkpoint(changed, changed=True)
            manifest = fixture_manifest(original)
            identity = active_dataset_manifest(changed, active_mode="Raw CSV", manifest=manifest)

            self.assertNotEqual(identity["verification"]["status"], VERIFIED_CHECKPOINT)
            self.assertFalse(artifact_matches_active_dataset(identity, artifact))

    def test_current_manifest_without_published_hashes_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            write_raw_checkpoint(checkpoint)
            state = checkpoint_status(
                checkpoint,
                active_mode="Raw CSV",
                manifest=app.ACTIVE_DATA_MANIFEST,
            )
            self.assertEqual(state["status"], "CUSTOM DATASET")


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_downloads_verifies_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_raw_checkpoint(source)
            archive = root / "fixture-dashboard-data.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                for name in ("benchmark_results_raw.csv", "configs.csv"):
                    output.write(source / name, name)
            manifest = fixture_manifest(
                source,
                bundle_url=archive.as_uri(),
                bundle_sha256=sha256_file(archive),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            destination = root / "installed"

            self.assertEqual(
                bootstrap_dataset(manifest_path=manifest_path, data_dir=destination),
                destination,
            )
            self.assertEqual(
                checkpoint_status(destination, active_mode="Raw CSV", manifest=manifest)["status"],
                VERIFIED_CHECKPOINT,
            )
            manifest["bundle"]["download_url"] = "file:///not-used-after-verification.zip"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                bootstrap_dataset(manifest_path=manifest_path, data_dir=destination),
                destination,
            )

    def test_checksum_failure_never_installs_partial_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_raw_checkpoint(source)
            archive = root / "fixture-dashboard-data.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name in ("benchmark_results_raw.csv", "configs.csv"):
                    output.write(source / name, name)
            manifest = fixture_manifest(
                source,
                bundle_url=archive.as_uri(),
                bundle_sha256="0" * 64,
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            destination = root / "installed"

            with self.assertRaisesRegex(DatasetBootstrapError, "checksum mismatch"):
                bootstrap_dataset(manifest_path=manifest_path, data_dir=destination)
            self.assertFalse(destination.exists())

    def test_unpublished_bundle_is_explicit(self) -> None:
        with self.assertRaises(DatasetPublicationRequiredError):
            bootstrap_dataset(offline=False)


if __name__ == "__main__":
    unittest.main()
