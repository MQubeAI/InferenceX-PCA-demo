from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from modeling.snapshot_ingestion import minimal_restore_toc, parse_sha256sums, verify_upstream_parts
from modeling.snapshot_refresh import (
    OfficialRelease,
    SCHEMA_INCOMPATIBLE,
    SCHEMA_SAFE_ADDITIVE,
    SnapshotRefreshError,
    audit_candidate_against_active,
    active_manifest_from_versioned,
    candidate_manifest,
    classify_schema_difference,
    create_checkpoint_bundle,
    deterministic_csv_export,
    discover_new_release,
    evaluate_refresh_policy,
    inspect_checkpoint,
    parse_official_release,
    refresh_report_markdown,
    snapshot_artifact_names,
)


def release(date: str) -> OfficialRelease:
    return OfficialRelease(
        tag=f"db-dump/{date}",
        snapshot_date=date,
        html_url=f"https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/{date}",
        assets=(),
        published_at=None,
    )


def write_checkpoint(directory: Path, *, add_config_column: bool = False, missing_config: bool = False, invalid_metrics: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    benchmark_header = ["id", "config_id", "benchmark_type", "date", "isl", "osl", "conc", "metrics"]
    metrics = "not-json" if invalid_metrics else json.dumps({"median_tpot": 0.1, "tput_per_gpu": 20.0, "joules_per_output_token": 0.2})
    with (directory / "benchmark_results_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_header, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"id": "2", "config_id": "999" if missing_config else "2", "benchmark_type": "single_turn", "date": "2026-07-20", "isl": "1024", "osl": "1024", "conc": "8", "metrics": metrics})
        writer.writerow({"id": "1", "config_id": "1", "benchmark_type": "single_turn", "date": "2026-07-19", "isl": "2048", "osl": "1024", "conc": "16", "metrics": metrics})
    config_header = [
        "id", "hardware", "framework", "model", "precision", "spec_method", "disagg", "is_multinode",
        "prefill_tp", "prefill_ep", "prefill_dp_attention", "prefill_num_workers", "decode_tp", "decode_ep",
        "decode_dp_attention", "decode_num_workers", "num_prefill_gpu", "num_decode_gpu",
    ]
    if add_config_column:
        config_header.append("new_unused_field")
    with (directory / "configs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config_header, lineterminator="\n")
        writer.writeheader()
        for identifier in ("1", "2"):
            row = {column: "1" for column in config_header}
            row.update({"id": identifier, "hardware": "h200", "framework": "trt", "model": "llama", "precision": "fp8", "spec_method": "none", "disagg": "f", "is_multinode": "f", "prefill_dp_attention": "f", "decode_dp_attention": "f"})
            writer.writerow(row)


class SnapshotRefreshTests(unittest.TestCase):
    def test_official_release_tag_parsing_and_prerelease_filtering(self) -> None:
        parsed = parse_official_release({"tag_name": "db-dump/2026-07-20", "html_url": release("2026-07-20").html_url, "assets": []})
        self.assertEqual(parsed, release("2026-07-20"))
        self.assertIsNone(parse_official_release({"tag_name": "other/v1", "html_url": "https://github.com/x/y", "assets": []}))
        self.assertIsNone(parse_official_release({"tag_name": "db-dump/2026-07-21", "html_url": release("2026-07-21").html_url, "assets": [], "prerelease": True}))

    def test_newer_and_same_release_detection(self) -> None:
        self.assertEqual(discover_new_release([release("2026-07-20")], "db-dump/2026-07-20")["status"], "NO_NEW_RELEASE")
        result = discover_new_release([release("2026-07-20"), release("2026-07-27")], "db-dump/2026-07-20")
        self.assertEqual(result["status"], "NEW_RELEASE_AVAILABLE")
        self.assertEqual(result["newest_release"], "db-dump/2026-07-27")

    def test_malformed_release_is_rejected(self) -> None:
        self.assertEqual(discover_new_release([{"tag_name": "db-dump/not-a-date", "html_url": "https://github.com/a/b", "assets": []}], "db-dump/2026-07-20")["status"], "INVALID_RELEASE_METADATA")

    def test_upstream_checksum_parser_rejects_bad_and_accepts_known(self) -> None:
        values = parse_sha256sums("a" * 64 + "  sample.dump.zst.part00\n")
        self.assertEqual(values["sample.dump.zst.part00"], "a" * 64)
        with self.assertRaises(SnapshotRefreshError):
            parse_sha256sums("bad checksum")

    def test_pre_downloaded_parts_are_reverified_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, contents = "inferencex-2026-07-27.dump.zst.part00", b"verified-part"
            (root / name).write_bytes(contents)
            (root / "SHA256SUMS").write_text(hashlib.sha256(contents).hexdigest() + f"  {name}\n", encoding="utf-8")
            official = OfficialRelease(
                tag="db-dump/2026-07-27",
                snapshot_date="2026-07-27",
                html_url="https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-07-27",
                assets=(
                    {"name": "SHA256SUMS", "browser_download_url": "https://example.invalid/sums"},
                    {"name": name, "browser_download_url": "https://example.invalid/part"},
                ),
                published_at=None,
            )
            self.assertEqual(verify_upstream_parts(official, root), [root / name])
            (root / name).write_bytes(b"corrupt")
            with self.assertRaisesRegex(SnapshotRefreshError, "checksum mismatch"):
                verify_upstream_parts(official, root)

    def test_minimal_restore_toc_refuses_full_restore_fallback(self) -> None:
        toc = [
            "1; 0 0 SCHEMA - public postgres",
            "2; 0 0 TYPE public example_type postgres",
            "3; 1259 1 TABLE public benchmark_results postgres",
            "4; 0 1 TABLE DATA public benchmark_results postgres",
            "5; 1259 2 TABLE public configs postgres",
            "6; 0 2 TABLE DATA public configs postgres",
            "7; 2606 3 FK CONSTRAINT public benchmark_results benchmark_results_other_fkey postgres",
            "8; 1259 4 TABLE public unrelated postgres",
        ]
        selected = minimal_restore_toc(toc)
        self.assertFalse(any("unrelated" in row for row in selected))
        self.assertFalse(any("FK CONSTRAINT" in row for row in selected))
        self.assertFalse(any("TYPE" in row for row in selected))
        selected_with_type = minimal_restore_toc(toc, dependency_names={"example_type"})
        self.assertTrue(any("TYPE" in row for row in selected_with_type))

    def test_deterministic_csv_export_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = deterministic_csv_export([{"id": "2", "value": "b"}, {"id": "1", "value": "a"}], ["id", "value"], root / "one.csv", order_columns=("id",))
            two = deterministic_csv_export([{"id": "1", "value": "a"}, {"id": "2", "value": "b"}], ["id", "value"], root / "two.csv", order_columns=("id",))
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_checkpoint_integrity_and_manifest_are_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate"
            write_checkpoint(checkpoint)
            first, second = inspect_checkpoint(checkpoint), inspect_checkpoint(checkpoint)
            self.assertEqual(first["files"], second["files"])
            bundle = create_checkpoint_bundle(checkpoint, root / "bundle.zip")
            manifest = candidate_manifest(checkpoint_dir=checkpoint, source_release=release("2026-07-27"), parent_manifest={"dataset_id": "inferencex-db-dump-2026-07-20"}, bundle=bundle, generation={"test": True})
            self.assertEqual(manifest["status"], "candidate")
            self.assertEqual(manifest["row_counts"]["benchmark_referenced_configs"], 2)
            self.assertEqual(manifest["parent_active_checkpoint"], "inferencex-db-dump-2026-07-20")

    def test_active_manifest_is_preserved_until_explicit_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "candidate")
            bundle = create_checkpoint_bundle(root / "candidate", root / "bundle.zip")
            manifest = candidate_manifest(checkpoint_dir=root / "candidate", source_release=release("2026-07-27"), parent_manifest={"dataset_id": "inferencex-db-dump-2026-07-20"}, bundle=bundle, generation={"test": True})
            with self.assertRaises(SnapshotRefreshError):
                active_manifest_from_versioned(manifest)
            manifest["status"] = "promoted"
            active = active_manifest_from_versioned(manifest)
            self.assertEqual(active["dataset_id"], "inferencex-db-dump-2026-07-27")

    def test_bundle_contains_only_root_csvs_and_corruption_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "source")
            bundle = root / "bundle.zip"
            create_checkpoint_bundle(root / "source", bundle)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(archive.namelist(), ["benchmark_results_raw.csv", "configs.csv"])
            with zipfile.ZipFile(bundle, "a") as archive:
                archive.writestr("extra.txt", "corruption")
            with zipfile.ZipFile(bundle) as archive:
                self.assertNotEqual(archive.namelist(), ["benchmark_results_raw.csv", "configs.csv"])

    def test_additive_and_required_schema_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "active")
            write_checkpoint(root / "candidate", add_config_column=True)
            self.assertEqual(classify_schema_difference(inspect_checkpoint(root / "active"), inspect_checkpoint(root / "candidate"))["classification"], SCHEMA_SAFE_ADDITIVE)
            broken = root / "broken"
            write_checkpoint(broken)
            configs = broken / "configs.csv"
            with configs.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            header = [column for column in rows[0] if column != "hardware"]
            with configs.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key in header} for row in rows])
            with self.assertRaises(SnapshotRefreshError):
                inspect_checkpoint(broken)

    def test_required_field_type_change_is_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "active")
            write_checkpoint(root / "candidate")
            configs = root / "candidate" / "configs.csv"
            with configs.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["hardware"] = "7"
            rows[1]["hardware"] = "8"
            with configs.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            classification = classify_schema_difference(inspect_checkpoint(root / "active"), inspect_checkpoint(root / "candidate"))
            self.assertEqual(classification["classification"], SCHEMA_INCOMPATIBLE)

    def test_missing_config_join_and_corrupt_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "missing", missing_config=True)
            with self.assertRaisesRegex(SnapshotRefreshError, "do not resolve"):
                inspect_checkpoint(root / "missing")
            write_checkpoint(root / "metrics", invalid_metrics=True)
            with self.assertRaisesRegex(SnapshotRefreshError, "Invalid metrics JSON"):
                inspect_checkpoint(root / "metrics")

    def test_candidate_vs_active_report_and_review_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "active")
            write_checkpoint(root / "candidate", add_config_column=True)
            audit = audit_candidate_against_active(root / "active", root / "candidate")
            self.assertEqual(audit["schema"]["classification"], SCHEMA_SAFE_ADDITIVE)
            research = {name: {"status": "PASS"} for name in ("pca", "ae", "vae", "comparison", "stage4", "tabfm", "uncertainty")}
            policy = evaluate_refresh_policy(audit, research=research, clean_clone_passed=True)
            self.assertEqual(policy["promotion_status"], "REVIEW_REQUIRED")
            bundle = create_checkpoint_bundle(root / "candidate", root / "candidate.zip")
            manifest = candidate_manifest(checkpoint_dir=root / "candidate", source_release=release("2026-07-27"), parent_manifest={"dataset_id": "inferencex-db-dump-2026-07-20"}, bundle=bundle, generation={"test": True})
            self.assertIn("candidate-only", refresh_report_markdown(manifest, audit, policy))

    def test_missing_or_changed_historic_ids_require_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint(root / "active")
            write_checkpoint(root / "candidate")
            benchmark = root / "candidate" / "benchmark_results_raw.csv"
            with benchmark.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = handle.seek(0) or list(csv.DictReader(handle).fieldnames or [])
            rows[0]["conc"] = "32"
            with benchmark.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            audit = audit_candidate_against_active(root / "active", root / "candidate")
            self.assertEqual(audit["classification"], "REVIEW_REQUIRED")
            self.assertEqual(audit["benchmark_ids"]["changed_existing_ids"], 1)

    def test_snapshot_artifact_names_are_versioned(self) -> None:
        names = snapshot_artifact_names("2026-07-27")
        self.assertTrue(all("2026-07-27" in value for value in names.values()))


if __name__ == "__main__":
    unittest.main()
