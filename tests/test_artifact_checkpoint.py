from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from modeling.artifact_checkpoint import (
    OPTIONAL_STAGE5_ARTIFACT_KEYS,
    REQUIRED_ARTIFACT_KEYS,
    ArtifactIndexError,
    active_artifact_paths,
    bootstrap_active_artifacts,
    load_artifact_index,
    optional_stage5_artifact_paths,
)
from modeling.dataset_checkpoint import load_data_manifest


class ArtifactCheckpointTests(unittest.TestCase):
    def test_promoted_july_repository_artifacts_resolve(self) -> None:
        paths = bootstrap_active_artifacts(load_data_manifest(), offline=True)
        self.assertEqual(set(paths), set(REQUIRED_ARTIFACT_KEYS))
        self.assertTrue(all(path.is_file() for path in paths.values()))
        stage5_paths = optional_stage5_artifact_paths(load_data_manifest())
        self.assertIsNotNone(stage5_paths)
        self.assertEqual(set(stage5_paths or {}), set(OPTIONAL_STAGE5_ARTIFACT_KEYS))
        self.assertTrue(all(path.is_file() for path in (stage5_paths or {}).values()))

    def test_active_data_and_artifact_pointers_must_match(self) -> None:
        manifest = load_data_manifest()
        index = copy.deepcopy(load_artifact_index())
        index["active_dataset_id"] = "other"
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "artifact-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(ArtifactIndexError):
                active_artifact_paths(manifest, index_path=index_path)

    def test_valid_future_snapshot_without_stage5_keeps_core_artifacts_available(self) -> None:
        manifest = load_data_manifest()
        index = copy.deepcopy(load_artifact_index())
        for key in OPTIONAL_STAGE5_ARTIFACT_KEYS:
            del index["snapshots"][manifest["dataset_id"]]["artifacts"][key]
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "artifact-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(set(active_artifact_paths(manifest, index_path=index_path)), set(REQUIRED_ARTIFACT_KEYS))
            self.assertIsNone(optional_stage5_artifact_paths(manifest, index_path=index_path))

    def test_partial_stage5_capability_fails_closed(self) -> None:
        manifest = load_data_manifest()
        index = copy.deepcopy(load_artifact_index())
        del index["snapshots"][manifest["dataset_id"]]["artifacts"]["manifold_projections"]
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "artifact-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(set(active_artifact_paths(manifest, index_path=index_path)), set(REQUIRED_ARTIFACT_KEYS))
            with self.assertRaisesRegex(ArtifactIndexError, "Stage 5 manifold artifact capability is incomplete"):
                optional_stage5_artifact_paths(manifest, index_path=index_path)

    def test_corrupt_complete_stage5_capability_does_not_invalidate_core_resolution(self) -> None:
        manifest = load_data_manifest()
        index = copy.deepcopy(load_artifact_index())
        index["snapshots"][manifest["dataset_id"]]["artifacts"]["manifold"] = "artifacts/not-present.json"
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "artifact-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual(set(active_artifact_paths(manifest, index_path=index_path)), set(REQUIRED_ARTIFACT_KEYS))
            with self.assertRaisesRegex(ArtifactIndexError, "Stage 5 manifold artifacts are missing"):
                optional_stage5_artifact_paths(manifest, index_path=index_path)


if __name__ == "__main__":
    unittest.main()
