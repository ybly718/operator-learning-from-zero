import torch

from operator_learning_guide.models import FNO1D, DeepONet1D, count_parameters


def test_reference_models_preserve_batch_and_grid_shape() -> None:
    coefficients = torch.rand(3, 32)
    grid = torch.linspace(0.0, 1.0, 32)
    models = [
        DeepONet1D(n_sensors=32, hidden_channels=16, latent_channels=12, depth=3),
        FNO1D(hidden_channels=8, n_modes=6, n_layers=2, padding=4),
    ]
    for model in models:
        output = model(coefficients, grid)
        assert output.shape == coefficients.shape
        assert torch.isfinite(output).all()
        assert count_parameters(model) > 0


def test_fno_accepts_batched_coordinates() -> None:
    coefficients = torch.rand(2, 24)
    grid = torch.linspace(0.0, 1.0, 24).repeat(2, 1)
    output = FNO1D(hidden_channels=8, n_modes=5, n_layers=1)(coefficients, grid)
    assert output.shape == (2, 24)
