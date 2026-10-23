import pytest
import torch

from operator_learning_guide.metrics import (
    boundary_mae,
    metric_bundle,
    relative_l2,
    spectral_relative_l2,
)


def test_metrics_are_zero_for_exact_prediction() -> None:
    target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    assert relative_l2(target, target).item() == pytest.approx(0.0)
    assert boundary_mae(target, target).item() == pytest.approx(0.0)
    assert spectral_relative_l2(target, target).item() == pytest.approx(0.0)
    assert set(metric_bundle(target, target)) == {
        "relative_l2",
        "boundary_mae",
        "spectral_relative_l2",
    }


def test_relative_l2_is_macro_averaged_across_functions() -> None:
    target = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    prediction = torch.tensor([[0.0, 0.0], [9.0, 0.0]])
    assert relative_l2(prediction, target).item() == pytest.approx(0.55)
