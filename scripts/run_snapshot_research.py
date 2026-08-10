"""Run the approved, fixed research protocol for one candidate snapshot.

This command is intentionally for the dedicated refresh runner, never ordinary
PR CI.  ``--plan`` is safe everywhere.  ``--execute`` trains only the already
selected PCA/AE/VAE/Stage 4 protocol and requires a separate approved TabFM
command; it fails rather than pretending an evaluation-only TabFM result exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from modeling.final_representation_training import run_final_experiment  # noqa: E402
from modeling.representation_validation import run_stage4_validation  # noqa: E402
from modeling.snapshot_research import fixed_protocol_plan, selected_tabfm_contract  # noqa: E402
from scripts.build_representation_comparison_final import build_final_comparison  # noqa: E402
from scripts.build_snapshot_pca_artifact import build_snapshot_pca_artifact, load_candidate_aggregate  # noqa: E402


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixed_beta(active_artifacts: Path) -> float:
    artifact = json.loads((active_artifacts / "representation-vae-final-db-dump-2026-07-20.json").read_text(encoding="utf-8"))
    beta = artifact.get("hyperparameters", {}).get("beta")
    if not isinstance(beta, (int, float)):
        raise RuntimeError("The approved VAE beta is not recorded in the active final artifact.")
    return float(beta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tabfm-command", help="required command for the approved external TabFM evaluation")
    args = parser.parse_args()
    if args.plan == args.execute:
        parser.error("choose exactly one of --plan or --execute")
    manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
    plan = fixed_protocol_plan(manifest, args.artifact_dir)
    plan["tabfm_contract"] = selected_tabfm_contract(REPOSITORY_ROOT)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0

    names = plan["artifact_names"]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pca = build_snapshot_pca_artifact(data_dir=args.data_dir, snapshot_manifest=manifest)
    pca_path = args.artifact_dir / names["pca"]
    _write(pca_path, pca)
    _raw, aggregate, _metadata = load_candidate_aggregate(args.data_dir)
    source_dump = manifest["source"]["release"]
    ae_path, ae_embeddings, ae_weights = (args.artifact_dir / names[key] for key in ("ae_json", "ae_embeddings", "ae_weights"))
    vae_path, vae_embeddings, vae_weights = (args.artifact_dir / names[key] for key in ("vae_json", "vae_embeddings", "vae_weights"))
    run_final_experiment(aggregate, method="autoencoder", beta=0.0, output_path=ae_path, companion_path=ae_embeddings, weights_path=ae_weights, maximum_epochs=250, source_dump=source_dump, enforce_snapshot_counts=False)
    beta = _fixed_beta(REPOSITORY_ROOT / "artifacts")
    vae_diagnostic_path = args.artifact_dir / f"representation-vae-beta-diagnostic-db-dump-{manifest['snapshot_date']}.json"
    _write(vae_diagnostic_path, {"schema_version": "representation-vae-beta-diagnostic-v2", "source_dump": source_dump, "selection_status": "selected", "selected_beta": beta, "selection_rationale": "Inherited fixed beta from the promoted protocol; no new beta search was run."})
    run_final_experiment(aggregate, method="variational_autoencoder", beta=beta, output_path=vae_path, companion_path=vae_embeddings, weights_path=vae_weights, maximum_epochs=150, source_dump=source_dump, enforce_snapshot_counts=False)
    comparison_path = args.artifact_dir / names["comparison"]
    comparison = build_final_comparison(data_dir=args.data_dir, pca_path=pca_path, ae_path=ae_path, vae_path=vae_path, beta_diagnostic_path=vae_diagnostic_path, source_dump=source_dump, enforce_snapshot_counts=False)
    _write(comparison_path, comparison)
    stage4_path = args.artifact_dir / names["stage4"]
    run_stage4_validation(aggregate, stage3_ae_path=ae_path, stage3_vae_path=vae_path, stage3_comparison_path=comparison_path, pca_artifact_path=pca_path, output_path=stage4_path, source_dump=source_dump, enforce_snapshot_counts=False)
    if not args.tabfm_command:
        raise RuntimeError("TabFM evaluation is mandatory for promotion; provide --tabfm-command on the approved runner.")
    subprocess.run(shlex.split(args.tabfm_command), check=True, cwd=REPOSITORY_ROOT, env={**os.environ, "INFERENCEX_SNAPSHOT_MANIFEST": str(args.snapshot_manifest), "INFERENCEX_SNAPSHOT_ARTIFACT_DIR": str(args.artifact_dir)})
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "build_snapshot_model_summary.py"),
            "--snapshot-manifest",
            str(args.snapshot_manifest),
            "--tabfm",
            str(args.artifact_dir / names["tabfm_evaluation"]),
            "--uncertainty",
            str(args.artifact_dir / names["uncertainty"]),
            "--output",
            str(args.artifact_dir / names["model_summary"]),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    print(json.dumps({"status": "FIXED_RESEARCH_COMPLETE", "plan": plan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
