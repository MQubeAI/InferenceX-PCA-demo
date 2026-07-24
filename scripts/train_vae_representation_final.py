#!/usr/bin/env python3
"""Run the selected fixed Stage 3 VAE-15 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps import inferencex_pca_demo as app
from modeling.final_representation_training import run_final_experiment
from scripts.build_july_pca_artifact import load_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--beta-diagnostic",
        default="artifacts/representation-vae-beta-diagnostic-db-dump-2026-07-20.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/representation-vae-final-db-dump-2026-07-20.json",
    )
    parser.add_argument(
        "--embeddings",
        default="artifacts/representation-vae-final-db-dump-2026-07-20.parquet",
    )
    parser.add_argument(
        "--weights",
        default="artifacts/representation-vae-final-db-dump-2026-07-20.pt",
    )
    parser.add_argument("--maximum-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    diagnostic = json.loads(Path(args.beta_diagnostic).read_text(encoding="utf-8"))
    if diagnostic.get("selection_status") != "selected":
        raise RuntimeError(
            "The bounded beta diagnostic did not identify a defensible VAE beta; "
            "final VAE training is stopped."
        )
    beta = float(diagnostic["selected_beta"])
    _raw, aggregate, _metadata = load_aggregate(args.data_dir)
    artifact = run_final_experiment(
        aggregate,
        method="variational_autoencoder",
        beta=beta,
        output_path=args.output,
        companion_path=args.embeddings,
        weights_path=args.weights,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    summary = artifact["summary"]
    print(
        f"Wrote {args.output}; beta={beta}; validation MSE "
        f"{summary['validation_mse']['mean']:.6f} +/- "
        f"{summary['validation_mse']['standard_deviation']:.6f}"
    )


if __name__ == "__main__":
    main()

