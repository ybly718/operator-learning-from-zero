"""Metrics that expose more than a single pointwise mean squared error."""

from __future__ import annotations

import torch


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-sample relative L2 error."""

    difference = (prediction - target).flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    numerator = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def boundary_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error at the two boundary nodes."""

    boundary_prediction = torch.stack((prediction[..., 0], prediction[..., -1]), dim=-1)
    boundary_target = torch.stack((target[..., 0], target[..., -1]), dim=-1)
    return torch.mean(torch.abs(boundary_prediction - boundary_target))


def spectral_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative L2 error of real Fourier coefficients."""

    prediction_hat = torch.fft.rfft(prediction, dim=-1)
    target_hat = torch.fft.rfft(target, dim=-1)
    numerator = torch.linalg.vector_norm(prediction_hat - target_hat, dim=-1)
    denominator = torch.linalg.vector_norm(target_hat, dim=-1).clamp_min(eps)
    return (numerator / denominator).mean()


@torch.no_grad()
def metric_bundle(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return the standard metrics used by the starter experiment."""

    return {
        "relative_l2": float(relative_l2(prediction, target).cpu()),
        "boundary_mae": float(boundary_mae(prediction, target).cpu()),
        "spectral_relative_l2": float(spectral_relative_l2(prediction, target).cpu()),
    }
