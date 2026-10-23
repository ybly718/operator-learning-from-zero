"""Shared training and evaluation routines for fair model comparisons."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import metric_bundle, relative_l2


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float]]
    test_metrics: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(
    model: nn.Module,
    coefficients: np.ndarray,
    grid: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.as_tensor(coefficients, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    grid_tensor = _tensor(grid, device)
    predictions = [model(batch[0].to(device), grid_tensor).cpu() for batch in loader]
    return torch.cat(predictions, dim=0)


def fit_model(
    model: nn.Module,
    *,
    train_coefficients: np.ndarray,
    train_solutions: np.ndarray,
    validation_coefficients: np.ndarray,
    validation_solutions: np.ndarray,
    test_coefficients: np.ndarray,
    test_solutions: np.ndarray,
    grid: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> TrainingResult:
    model.to(device)
    grid_tensor = _tensor(grid, device)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_coefficients, dtype=torch.float32),
            torch.as_tensor(train_solutions, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        accumulated_loss = 0.0
        sample_count = 0
        for coefficients, targets in train_loader:
            coefficients = coefficients.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coefficients, grid_tensor)
            loss = relative_l2(predictions, targets)
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.detach()) * coefficients.shape[0]
            sample_count += coefficients.shape[0]
        scheduler.step()

        validation_prediction = predict(
            model,
            validation_coefficients,
            grid,
            device=device,
            batch_size=batch_size,
        )
        validation_target = torch.as_tensor(validation_solutions, dtype=torch.float32)
        history.append(
            {
                "epoch": float(epoch),
                "train_relative_l2": accumulated_loss / sample_count,
                "validation_relative_l2": float(
                    relative_l2(validation_prediction, validation_target)
                ),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )

    test_prediction = predict(
        model,
        test_coefficients,
        grid,
        device=device,
        batch_size=batch_size,
    )
    test_target = torch.as_tensor(test_solutions, dtype=torch.float32)
    return TrainingResult(history=history, test_metrics=metric_bundle(test_prediction, test_target))
