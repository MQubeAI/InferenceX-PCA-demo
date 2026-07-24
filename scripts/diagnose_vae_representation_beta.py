#!/usr/bin/env python3
"""Run the bounded seed-42 VAE beta diagnostic."""

from __future__ import annotations

import argparse

from apps import inferencex_pca_demo as app
from modeling.final_representation_training import run_vae_beta_diagnostic
from scripts.build_july_pca_artifact import load_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--stage2-beta1",
        default="artifacts/representation-vae-db-dump-2026-07-20.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/representation-vae-beta-diagnostic-db-dump-2026-07-20.json",
    )
    parser.add_argument("--maximum-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    _raw, aggregate, _metadata = load_aggregate(args.data_dir)
    artifact = run_vae_beta_diagnostic(
        aggregate,
        stage2_beta1_path=args.stage2_beta1,
        output_path=args.output,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    print(
        f"Wrote {args.output}; selection={artifact['selection_status']}; "
        f"selected beta={artifact['selected_beta']}"
    )


if __name__ == "__main__":
    main()

