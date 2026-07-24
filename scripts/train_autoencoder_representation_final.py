#!/usr/bin/env python3
"""Run the fixed Stage 3 AE-15 experiment across three seeds and folds."""

from __future__ import annotations

import argparse

from apps import inferencex_pca_demo as app
from modeling.final_representation_training import run_final_experiment
from scripts.build_july_pca_artifact import load_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        default="artifacts/representation-ae-final-db-dump-2026-07-20.json",
    )
    parser.add_argument(
        "--embeddings",
        default="artifacts/representation-ae-final-db-dump-2026-07-20.parquet",
    )
    parser.add_argument(
        "--weights",
        default="artifacts/representation-ae-final-db-dump-2026-07-20.pt",
    )
    parser.add_argument("--maximum-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    _raw, aggregate, _metadata = load_aggregate(args.data_dir)
    artifact = run_final_experiment(
        aggregate,
        method="autoencoder",
        beta=0.0,
        output_path=args.output,
        companion_path=args.embeddings,
        weights_path=args.weights,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    summary = artifact["summary"]
    print(
        f"Wrote {args.output}; validation MSE "
        f"{summary['validation_mse']['mean']:.6f} +/- "
        f"{summary['validation_mse']['standard_deviation']:.6f}; "
        f"early-stopped runs={summary['early_stopping_runs']}/9"
    )


if __name__ == "__main__":
    main()

