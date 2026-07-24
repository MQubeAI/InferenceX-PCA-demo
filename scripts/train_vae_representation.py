#!/usr/bin/env python3
"""Run the bounded variational-autoencoder representation screening."""

from __future__ import annotations

import argparse

from apps import inferencex_pca_demo as app
from modeling.representation_training import run_screening_experiment
from scripts.build_july_pca_artifact import load_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=app.DEFAULT_DATA_DIR)
    parser.add_argument("--output", default="artifacts/representation-vae-db-dump-2026-07-20.json")
    parser.add_argument("--weights", default="artifacts/representation-vae-db-dump-2026-07-20.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--maximum-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    _raw, aggregate, _metadata = load_aggregate(args.data_dir)
    artifact = run_screening_experiment(
        aggregate,
        method="variational_autoencoder",
        output_path=args.output,
        weights_path=args.weights,
        seed=args.seed,
        beta=args.beta,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    selected = artifact["selected_result"]
    collapse = selected["diagnostics"].get("posterior_collapse", False)
    print(
        f"Wrote {args.output}; selected latent dimension {selected['latent_dimension']}; "
        f"validation MSE {selected['validation_mse_mean']:.6f}; collapse={collapse}"
    )


if __name__ == "__main__":
    main()

