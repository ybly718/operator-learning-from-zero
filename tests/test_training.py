import torch

from operator_learning_guide.data import generate_poisson_dataset
from operator_learning_guide.models import DeepONet1D
from operator_learning_guide.training import fit_model, seed_everything


def test_one_epoch_training_loop_returns_history_and_metrics() -> None:
    seed_everything(3)
    dataset = generate_poisson_dataset(
        n_train=8,
        n_validation=4,
        n_test=4,
        n_grid=16,
        seed=3,
    )
    result = fit_model(
        DeepONet1D(n_sensors=16, hidden_channels=8, latent_channels=8, depth=2),
        train_coefficients=dataset.train.coefficients,
        train_solutions=dataset.train.solutions,
        validation_coefficients=dataset.validation.coefficients,
        validation_solutions=dataset.validation.solutions,
        test_coefficients=dataset.test.coefficients,
        test_solutions=dataset.test.solutions,
        grid=dataset.grid,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    assert len(result.history) == 1
    assert set(result.test_metrics) == {
        "relative_l2",
        "boundary_mae",
        "spectral_relative_l2",
    }
