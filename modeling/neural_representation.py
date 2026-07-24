"""Lazy-loaded PyTorch training helpers for representation scripts.

This module intentionally has no top-level neural-framework import so importing
the Streamlit application remains lightweight.
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np


def train_neural_fold(
    matrix: np.ndarray,
    train_indices: list[int],
    validation_indices: list[int],
    *,
    method: str,
    latent_dimension: int,
    seed: int,
    hidden_dimensions: tuple[int, int] = (64, 32),
    beta: float = 1.0,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    maximum_epochs: int = 150,
    patience: int = 12,
    minimum_delta: float = 1e-5,
    input_corruption: float = 0.0,
    weight_decay: float = 0.0,
    kl_warmup_epochs: int = 0,
) -> dict[str, Any]:
    """Train one AE or VAE fold and return metrics plus all-row projections."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if method not in {"autoencoder", "variational_autoencoder"}:
        raise ValueError(f"Unsupported neural representation method: {method}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    input_dimension = int(matrix.shape[1])
    first, second = hidden_dimensions

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dimension, first),
                nn.ReLU(),
                nn.Linear(first, second),
                nn.ReLU(),
                nn.Linear(second, latent_dimension),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dimension, second),
                nn.ReLU(),
                nn.Linear(second, first),
                nn.ReLU(),
                nn.Linear(first, input_dimension),
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            latent = self.encoder(values)
            return self.decoder(latent), latent

    class VariationalAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_body = nn.Sequential(
                nn.Linear(input_dimension, first),
                nn.ReLU(),
                nn.Linear(first, second),
                nn.ReLU(),
            )
            self.mean = nn.Linear(second, latent_dimension)
            self.log_variance = nn.Linear(second, latent_dimension)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dimension, second),
                nn.ReLU(),
                nn.Linear(second, first),
                nn.ReLU(),
                nn.Linear(first, input_dimension),
            )

        def encode(self, values: Any) -> tuple[Any, Any]:
            hidden = self.encoder_body(values)
            return self.mean(hidden), self.log_variance(hidden)

        def forward(self, values: Any) -> tuple[Any, Any, Any, Any]:
            mean, log_variance = self.encode(values)
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mean + torch.randn_like(standard_deviation) * standard_deviation
            return self.decoder(latent), mean, log_variance, latent

    model = Autoencoder() if method == "autoencoder" else VariationalAutoencoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_values = torch.tensor(matrix[train_indices], dtype=torch.float32)
    validation_values = torch.tensor(matrix[validation_indices], dtype=torch.float32)
    all_values = torch.tensor(matrix, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_values),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    def losses(
        model_values: Any,
        target_values: Any,
        *,
        sample: bool,
        beta_value: float,
    ) -> tuple[Any, Any, Any]:
        if method == "autoencoder":
            reconstructed, _latent = model(model_values)
            reconstruction = torch.mean(
                torch.square(reconstructed - target_values)
            )
            kl = torch.zeros((), dtype=target_values.dtype)
        else:
            if sample:
                reconstructed, mean, log_variance, _latent = model(model_values)
            else:
                mean, log_variance = model.encode(model_values)
                reconstructed = model.decoder(mean)
            reconstruction = torch.mean(
                torch.square(reconstructed - target_values)
            )
            kl = torch.mean(-0.5 * (1 + log_variance - mean.square() - log_variance.exp()))
        return reconstruction + beta_value * kl, reconstruction, kl

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        effective_beta = (
            beta * min(1.0, epoch / kl_warmup_epochs)
            if method == "variational_autoencoder" and kl_warmup_epochs > 0
            else beta
        )
        train_total = train_reconstruction = train_kl = 0.0
        batches = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            model_batch = batch
            if method == "autoencoder" and input_corruption > 0:
                keep = torch.rand_like(batch).ge(input_corruption)
                model_batch = batch * keep
            total, reconstruction, kl = losses(
                model_batch,
                batch,
                sample=True,
                beta_value=effective_beta,
            )
            total.backward()
            optimizer.step()
            train_total += float(total.detach())
            train_reconstruction += float(reconstruction.detach())
            train_kl += float(kl.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_total, validation_reconstruction, validation_kl = losses(
                validation_values,
                validation_values,
                sample=False,
                beta_value=beta,
            )
        row = {
            "epoch": epoch,
            "train_total": train_total / batches,
            "train_reconstruction": train_reconstruction / batches,
            "train_kl": train_kl / batches,
            "validation_total": float(validation_total),
            "validation_reconstruction": float(validation_reconstruction),
            "validation_kl": float(validation_kl),
            "effective_training_beta": effective_beta,
        }
        history.append(row)
        score = row["validation_total"]
        if score < best_validation - minimum_delta:
            best_validation = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Early stopping failed to preserve a neural model state.")
    model.load_state_dict(best_state)
    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        if method == "autoencoder":
            all_reconstructed, all_embedding = model(all_values)
            mean = log_variance = None
        else:
            mean, log_variance = model.encode(all_values)
            all_embedding = mean
            all_reconstructed = model.decoder(mean)
    inference_seconds = time.perf_counter() - inference_started
    reconstructed = all_reconstructed.cpu().numpy()
    embedding = all_embedding.cpu().numpy()

    diagnostics: dict[str, Any] = {}
    if method == "variational_autoencoder":
        mean_array = mean.cpu().numpy()
        log_variance_array = log_variance.cpu().numpy()
        kl_by_dimension = np.mean(
            -0.5 * (1 + log_variance_array - np.square(mean_array) - np.exp(log_variance_array)),
            axis=0,
        )
        latent_variance = np.var(mean_array, axis=0)
        active = int(np.sum(kl_by_dimension > 0.01))
        sensitivities = []
        with torch.no_grad():
            baseline = model.decoder(mean).cpu().numpy()
            for dimension in range(latent_dimension):
                changed = mean.clone()
                changed[:, dimension] += 1.0
                shifted = model.decoder(changed).cpu().numpy()
                sensitivities.append(float(np.mean(np.abs(shifted - baseline))))
        diagnostics = {
            "average_kl_per_latent_dimension": kl_by_dimension.tolist(),
            "active_latent_dimensions": active,
            "active_dimension_threshold": 0.01,
            "latent_variance": latent_variance.tolist(),
            "decoder_sensitivity_per_dimension": sensitivities,
            "decoder_ignores_latent_changes": bool(max(sensitivities, default=0.0) < 1e-3),
            "posterior_collapse": bool(active == 0 or max(sensitivities, default=0.0) < 1e-3),
        }
    else:
        baseline = reconstructed
        sensitivities = []
        latent_tensor = torch.tensor(embedding, dtype=torch.float32)
        with torch.no_grad():
            for dimension in range(latent_dimension):
                changed = latent_tensor.clone()
                changed[:, dimension] += 1.0
                shifted = model.decoder(changed).cpu().numpy()
                sensitivities.append(float(np.mean(np.abs(shifted - baseline))))
        diagnostics = {
            "latent_variance": np.var(embedding, axis=0).tolist(),
            "decoder_sensitivity_per_dimension": sensitivities,
        }

    return {
        "embedding": embedding,
        "reconstructed": reconstructed,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "early_stopping_occurred": len(history) < maximum_epochs,
        "stopping_reason": (
            "early_stopping_patience" if len(history) < maximum_epochs else "maximum_epoch_cap"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "diagnostics": diagnostics,
        "state_dict": best_state,
        "torch_version": torch.__version__,
        "regularization": {
            "input_corruption": input_corruption,
            "weight_decay": weight_decay,
            "kl_warmup_epochs": kl_warmup_epochs,
        },
    }
