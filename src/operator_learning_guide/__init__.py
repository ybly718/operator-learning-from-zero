"""Minimal, tested building blocks for the operator-learning course."""

from .data import OperatorDataset, OperatorSplit, generate_poisson_dataset
from .metrics import metric_bundle, relative_l2
from .models import FNO1D, DeepONet1D, count_parameters

__all__ = [
    "DeepONet1D",
    "FNO1D",
    "OperatorDataset",
    "OperatorSplit",
    "count_parameters",
    "generate_poisson_dataset",
    "metric_bundle",
    "relative_l2",
]
