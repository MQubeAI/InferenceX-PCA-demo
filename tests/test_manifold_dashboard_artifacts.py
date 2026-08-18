from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from modeling.manifold_dashboard_artifacts import (
    STAGE5_COHORT_ROWS,
    STAGE5_PROJECTION_ROWS,
    Stage5ArtifactError,
    aligned_projection_frame,
    load_stage5_dashboard_artifacts,
)


ARTIFACTS = Path("artifacts")
STRUCTURAL = ARTIFACTS / "manifold-analysis-stage5-structural-db-dump-2026-07-20.json"
FINAL = ARTIFACTS / "manifold-analysis-stage5-db-dump-2026-07-20.json"
PROJECTIONS = ARTIFACTS / "manifold-projections-stage5-db-dump-2026-07-20.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifoldDashboardArtifactTests(unittest.TestCase):
    def _load(self, structural: Path = STRUCTURAL, final: Path = FINAL, projections: Path = PROJECTIONS):
        return load_stage5_dashboard_artifacts(
            final_path=final,
            structural_path=structural,
            projection_path=projections,
        )

    def _copied_artifacts(self, directory: Path) -> tuple[Path, Path, Path]:
        structural = directory / STRUCTURAL.name
        final = directory / FINAL.name
        projections = directory / PROJECTIONS.name
        shutil.copy2(STRUCTURAL, structural)
        shutil.copy2(FINAL, final)
        shutil.copy2(PROJECTIONS, projections)
        return structural, final, projections

    @staticmethod
    def _rewrite_final(final: Path, *, structural_sha: str, projection_sha: str) -> None:
        payload = json.loads(final.read_text(encoding="utf-8"))
        payload["structural_artifact"]["sha256"] = structural_sha
        payload["projection_artifact"]["sha256"] = projection_sha
        final.write_text(json.dumps(payload), encoding="utf-8")

    def test_promoted_stage5_artifacts_validate_complete_contract(self) -> None:
        artifacts = self._load()
        self.assertEqual(artifacts.structural_sha256, "33441444687c6fdf30dcca5885c31a30b155e3cb634f73d3e7d6c0efc5757fce")
        self.assertEqual(len(artifacts.projections), STAGE5_PROJECTION_ROWS)
        self.assertEqual(artifacts.projections["parameter_id"].nunique(), 45)
        self.assertTrue(
            artifacts.projections.groupby("parameter_id")["row_id"].nunique().eq(STAGE5_COHORT_ROWS).all()
        )
        self.assertFalse(artifacts.projections.duplicated(["row_id", "parameter_id"]).any())

    def test_loader_validates_exact_frozen_grid_and_active_snapshot(self) -> None:
        # A successful load cross-checks all frozen UMAP/t-SNE/ablation IDs to
        # the 45 JSON-backed projection parameter identities.
        self._load()
        with self.assertRaisesRegex(Stage5ArtifactError, "Active dashboard snapshot"):
            load_stage5_dashboard_artifacts(
                final_path=FINAL,
                structural_path=STRUCTURAL,
                projection_path=PROJECTIONS,
                active_source_dump="db-dump/other",
            )
        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(structural.read_text(encoding="utf-8"))
            payload["umap_2"]["runs"][0]["config"]["min_dist"] = 0.2
            structural.write_text(json.dumps(payload), encoding="utf-8")
            structural_sha = _sha256(structural)
            self._rewrite_final(final, structural_sha=structural_sha, projection_sha=_sha256(projections))
            with patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha):
                with self.assertRaisesRegex(Stage5ArtifactError, "UMAP-2 parameter grid"):
                    self._load(structural, final, projections)
        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(structural.read_text(encoding="utf-8"))
            payload["umap_2"]["runs"][0]["parameter_id"] = "unexpected-parameter-id"
            structural.write_text(json.dumps(payload), encoding="utf-8")
            structural_sha = _sha256(structural)
            self._rewrite_final(final, structural_sha=structural_sha, projection_sha=_sha256(projections))
            with patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha):
                with self.assertRaisesRegex(Stage5ArtifactError, "projection parameter IDs"):
                    self._load(structural, final, projections)

    def test_loader_rejects_wrong_snapshot_and_structural_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(structural.read_text(encoding="utf-8"))
            payload["source_dump"] = "other-snapshot"
            structural.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage5ArtifactError, "snapshot"):
                self._load(structural, final, projections)
        with patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", "0" * 64):
            with self.assertRaisesRegex(Stage5ArtifactError, "SHA-256"):
                self._load()

    def test_loader_rejects_incomplete_counts_and_structural_outcome_leakage(self) -> None:
        for mutation, expected_error in (
            (lambda payload: payload["run_counts"].update({"umap_2": 26}), "run counts"),
            (lambda payload: payload.update({"outcome_overlays": {"metric": 1}}), "outcome overlays"),
        ):
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                structural, final, projections = self._copied_artifacts(Path(temporary))
                payload = json.loads(structural.read_text(encoding="utf-8"))
                mutation(payload)
                structural.write_text(json.dumps(payload), encoding="utf-8")
                structural_sha = _sha256(structural)
                projection_sha = _sha256(projections)
                self._rewrite_final(final, structural_sha=structural_sha, projection_sha=projection_sha)
                with patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha):
                    with self.assertRaisesRegex(Stage5ArtifactError, expected_error):
                        self._load(structural, final, projections)

    def test_loader_rejects_invalid_phase_outcome_boundary_and_held_out_overlap(self) -> None:
        for mutation, expected_error in (
            (lambda payload: payload.update({"phase": "final"}), "structural phase"),
            (lambda payload: payload.update({"target_metrics_in_inputs": ["metrics_tpot"]}), "outcome leakage"),
        ):
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                structural, final, projections = self._copied_artifacts(Path(temporary))
                payload = json.loads(structural.read_text(encoding="utf-8"))
                mutation(payload)
                structural.write_text(json.dumps(payload), encoding="utf-8")
                structural_sha = _sha256(structural)
                self._rewrite_final(final, structural_sha=structural_sha, projection_sha=_sha256(projections))
                with patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha):
                    with self.assertRaisesRegex(Stage5ArtifactError, expected_error):
                        self._load(structural, final, projections)

        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["phase"] = "structural_freeze"
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage5ArtifactError, "final phase"):
                self._load(structural, final, projections)

        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["outcome_overlays"]["descriptive_only"] = False
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage5ArtifactError, "descriptive only"):
                self._load(structural, final, projections)

        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["umap_15"]["grouped_held_out"][0]["group_overlap"] = 1
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage5ArtifactError, "configuration overlap"):
                self._load(structural, final, projections)

    def test_loader_rejects_projection_hash_mismatch_and_duplicate_projection_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["projection_artifact"]["sha256"] = "0" * 64
            final.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage5ArtifactError, "projection SHA-256"):
                self._load(structural, final, projections)

        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            frame = pd.read_parquet(projections)
            parameter_id = frame["parameter_id"].iloc[0]
            positions = frame.index[frame["parameter_id"].eq(parameter_id)].tolist()
            frame.loc[positions[-1], "row_id"] = frame.loc[positions[0], "row_id"]
            frame.to_parquet(projections, index=False)
            projection_sha = _sha256(projections)
            payload = json.loads(structural.read_text(encoding="utf-8"))
            payload["projection_artifact"]["sha256"] = projection_sha
            structural.write_text(json.dumps(payload), encoding="utf-8")
            structural_sha = _sha256(structural)
            self._rewrite_final(final, structural_sha=structural_sha, projection_sha=projection_sha)
            with (
                patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha),
                patch("modeling.manifold_dashboard_artifacts.STAGE5_PROJECTION_SHA256", projection_sha),
            ):
                with self.assertRaisesRegex(Stage5ArtifactError, "duplicate"):
                    self._load(structural, final, projections)

        with tempfile.TemporaryDirectory() as temporary:
            structural, final, projections = self._copied_artifacts(Path(temporary))
            frame = pd.read_parquet(projections)
            parameter_id = frame["parameter_id"].iloc[0]
            position = frame.index[frame["parameter_id"].eq(parameter_id)].tolist()[-1]
            frame.loc[position, "row_id"] = "misaligned-semantic-row-id"
            frame.to_parquet(projections, index=False)
            projection_sha = _sha256(projections)
            payload = json.loads(structural.read_text(encoding="utf-8"))
            payload["projection_artifact"]["sha256"] = projection_sha
            structural.write_text(json.dumps(payload), encoding="utf-8")
            structural_sha = _sha256(structural)
            self._rewrite_final(final, structural_sha=structural_sha, projection_sha=projection_sha)
            with (
                patch("modeling.manifold_dashboard_artifacts.STAGE5_STRUCTURAL_SHA256", structural_sha),
                patch("modeling.manifold_dashboard_artifacts.STAGE5_PROJECTION_SHA256", projection_sha),
            ):
                with self.assertRaisesRegex(Stage5ArtifactError, "semantic row identities"):
                    self._load(structural, final, projections)

    def test_projection_metadata_join_uses_row_identity_not_position(self) -> None:
        artifacts = self._load()
        parameter_id = artifacts.projections["parameter_id"].iloc[0]
        row_ids = artifacts.projections.loc[
            artifacts.projections["parameter_id"].eq(parameter_id), "row_id"
        ].astype(str).tolist()
        reversed_ids = list(reversed(row_ids))
        cohort = pd.DataFrame({"row_marker": range(len(reversed_ids))})
        aligned = aligned_projection_frame(
            artifacts,
            parameter_id=parameter_id,
            canonical_row_ids=reversed_ids,
            canonical_cohort=cohort,
        )
        self.assertEqual(aligned["row_id"].tolist(), reversed_ids)
        self.assertEqual(aligned["row_marker"].tolist(), list(range(len(reversed_ids))))


if __name__ == "__main__":
    unittest.main()
