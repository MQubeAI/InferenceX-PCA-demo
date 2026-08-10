from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from modeling.artifact_checkpoint import ArtifactIndexError, active_artifact_paths, bootstrap_active_artifacts, load_artifact_index
from modeling.dataset_checkpoint import load_data_manifest


class ArtifactCheckpointTests(unittest.TestCase):
    def test_promoted_july_repository_artifacts_resolve(self) -> None:
        paths = bootstrap_active_artifacts(load_data_manifest(), offline=True)
        self.assertEqual(set(paths), {"pca", "ae", "vae", "vae_beta", "comparison", "stage4", "model_summary"})
        self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_active_data_and_artifact_pointers_must_match(self) -> None:
        manifest = load_data_manifest()
        index = copy.deepcopy(load_artifact_index())
        index["active_dataset_id"] = "other"
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "artifact-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(ArtifactIndexError):
                active_artifact_paths(manifest, index_path=index_path)


if __name__ == "__main__":
    unittest.main()
