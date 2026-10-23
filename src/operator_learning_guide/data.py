"""Synthetic operator-learning data with a trustworthy numerical reference solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorSplit:
    """A collection of independent input and output function realizations."""

    coefficients: np.ndarray
    solutions: np.ndarray


@dataclass(frozen=True)
class OperatorDataset:
    """Train/validation/test splits sharing one physical grid."""

    grid: np.ndarray
    train: OperatorSplit
    validation: OperatorSplit
    test: OperatorSplit


def _smooth_random_fields(
    rng: np.random.Generator,
    n_samples: int,
    n_grid: int,
    smooth_modes: int,
) -> np.ndarray:
    """Create smooth zero-mean, unit-variance fields using a spectral filter."""

    noise = rng.normal(size=(n_samples, n_grid))
    frequencies = np.fft.rfftfreq(n_grid) * n_grid
    cutoff = max(float(smooth_modes), 1.0)
    spectral_filter = np.exp(-0.5 * (frequencies / cutoff) ** 2)
    fields = np.fft.irfft(np.fft.rfft(noise, axis=-1) * spectral_filter, n=n_grid, axis=-1)
    fields -= fields.mean(axis=-1, keepdims=True)
    fields /= fields.std(axis=-1, keepdims=True) + 1e-8
    return fields


def solve_variable_coefficient_poisson(
    coefficient: np.ndarray,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve -(a u_x)_x = f on [0, 1] with homogeneous Dirichlet boundaries.

    A conservative centered discretization is used. The function accepts one
    coefficient field of shape ``(n_grid,)`` and returns the nodal solution.
    """

    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 1 or coefficient.size < 4:
        raise ValueError("coefficient must be a one-dimensional array with at least 4 points")
    if np.any(coefficient <= 0):
        raise ValueError("elliptic coefficients must be strictly positive")

    n_grid = coefficient.size
    spacing = 1.0 / (n_grid - 1)
    if forcing is None:
        forcing = np.ones(n_grid, dtype=np.float64)
    forcing = np.asarray(forcing, dtype=np.float64)
    if forcing.shape != coefficient.shape:
        raise ValueError("forcing and coefficient must use the same grid")

    face_coefficient = 0.5 * (coefficient[:-1] + coefficient[1:])
    n_inner = n_grid - 2
    matrix = np.zeros((n_inner, n_inner), dtype=np.float64)

    for row, node in enumerate(range(1, n_grid - 1)):
        left = face_coefficient[node - 1] / spacing**2
        right = face_coefficient[node] / spacing**2
        matrix[row, row] = left + right
        if row > 0:
            matrix[row, row - 1] = -left
        if row + 1 < n_inner:
            matrix[row, row + 1] = -right

    solution = np.zeros(n_grid, dtype=np.float64)
    solution[1:-1] = np.linalg.solve(matrix, forcing[1:-1])
    return solution


def _make_split(
    rng: np.random.Generator,
    n_samples: int,
    n_grid: int,
    smooth_modes: int,
    log_coefficient_scale: float,
) -> OperatorSplit:
    log_fields = _smooth_random_fields(rng, n_samples, n_grid, smooth_modes)
    coefficients = np.exp(log_coefficient_scale * log_fields)
    solutions = np.stack(
        [solve_variable_coefficient_poisson(coefficient) for coefficient in coefficients]
    )
    return OperatorSplit(
        coefficients=coefficients.astype(np.float32),
        solutions=solutions.astype(np.float32),
    )


def generate_poisson_dataset(
    *,
    n_train: int,
    n_validation: int,
    n_test: int,
    n_grid: int = 64,
    smooth_modes: int = 8,
    log_coefficient_scale: float = 0.7,
    seed: int = 7,
) -> OperatorDataset:
    """Generate independent function-level splits for a Poisson solution operator."""

    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("all splits must contain at least one function realization")
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    return OperatorDataset(
        grid=grid,
        train=_make_split(
            rng, n_train, n_grid, smooth_modes, log_coefficient_scale
        ),
        validation=_make_split(
            rng, n_validation, n_grid, smooth_modes, log_coefficient_scale
        ),
        test=_make_split(rng, n_test, n_grid, smooth_modes, log_coefficient_scale),
    )
