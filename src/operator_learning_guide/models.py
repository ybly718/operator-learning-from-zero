"""Small reference implementations of DeepONet and the one-dimensional FNO."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must include at least an input and output layer")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DeepONet1D(nn.Module):
    """Branch/trunk DeepONet evaluated on a shared one-dimensional query grid."""

    def __init__(
        self,
        n_sensors: int,
        hidden_channels: int = 128,
        latent_channels: int = 128,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.branch = MLP(n_sensors, hidden_channels, latent_channels, depth)
        self.trunk = MLP(1, hidden_channels, latent_channels, depth)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = math.sqrt(float(latent_channels))

    def forward(self, coefficient: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim == 1:
            grid = grid[:, None]
        branch_features = self.branch(coefficient)
        trunk_features = self.trunk(grid)
        return torch.einsum("bp,np->bn", branch_features, trunk_features) / self.scale + self.bias


class SpectralConv1D(nn.Module):
    """A truncated Fourier multiplier acting on the low-frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        self.n_modes = n_modes
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_hat = torch.fft.rfft(inputs, dim=-1)
        available_modes = input_hat.shape[-1]
        n_modes = min(self.n_modes, available_modes)
        output_hat = torch.zeros(
            inputs.shape[0],
            self.weight.shape[1],
            available_modes,
            device=inputs.device,
            dtype=torch.cfloat,
        )
        output_hat[..., :n_modes] = torch.einsum(
            "bim,iom->bom",
            input_hat[..., :n_modes],
            self.weight[..., :n_modes],
        )
        return torch.fft.irfft(output_hat, n=inputs.shape[-1], dim=-1)


class FNOBlock1D(nn.Module):
    def __init__(self, channels: int, n_modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1D(channels, channels, n_modes)
        self.local = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(inputs) + self.local(inputs))


class FNO1D(nn.Module):
    """A compact FNO for regular one-dimensional grids."""

    def __init__(
        self,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        padding: int = 8,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.lifting = nn.Linear(2, hidden_channels)
        self.blocks = nn.ModuleList(
            [FNOBlock1D(hidden_channels, n_modes) for _ in range(n_layers)]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, coefficient: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch_size, n_grid = coefficient.shape
        if grid.ndim == 1:
            grid = grid[None, :].expand(batch_size, -1)
        features = torch.stack((coefficient, grid), dim=-1)
        features = self.lifting(features).permute(0, 2, 1)
        if self.padding:
            features = F.pad(features, (0, self.padding))
        for block in self.blocks:
            features = block(features)
        if self.padding:
            features = features[..., :n_grid]
        return self.projection(features).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
