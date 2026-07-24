#!/usr/bin/env python3
"""Run the bounded Stage 4 methodological validation experiment."""

from __future__ import annotations

import argparse

from apps import inferencex_pca_demo as app
from modeling.representation_validation import (
    augment_stage4_warmup_diagnostics,
    run_stage4_validation,
)
from scripts.build_july_pca_artifact import load_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        default="artifacts/representation-validation-stage4-db-dump-2026-07-20.json",
    )
    parser.add_argument(
        "--augment-warmup-diagnostics",
        action="store_true",
        help="Recreate only the three fixed VAE warm-up runs and retain diagnostics.",
    )
    args = parser.parse_args()
    _raw, aggregate, _metadata = load_aggregate(args.data_dir)
    if args.augment_warmup_diagnostics:
        artifact = augment_stage4_warmup_diagnostics(
            aggregate,
            artifact_path=args.output,
        )
        print(
            f"Updated {args.output}; diagnostic rerun "
            f"{artifact['compute_ledger']['diagnostic_rerun_seconds']:.1f}s"
        )
        return
    artifact = run_stage4_validation(
        aggregate,
        stage3_ae_path="artifacts/representation-ae-final-db-dump-2026-07-20.json",
        stage3_vae_path="artifacts/representation-vae-final-db-dump-2026-07-20.json",
        stage3_comparison_path=(
            "artifacts/representation-comparison-final-db-dump-2026-07-20.json"
        ),
        pca_artifact_path="artifacts/pca-db-dump-2026-07-20.json",
        output_path=args.output,
    )
    print(
        f"Wrote {args.output}; neural training "
        f"{artifact['compute_ledger']['neural_training_seconds']:.1f}s; "
        f"wall {artifact['compute_ledger']['wall_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
