import numpy as np

from operator_learning_guide.data import (
    generate_poisson_dataset,
    solve_variable_coefficient_poisson,
)


def test_constant_coefficient_poisson_matches_analytic_solution() -> None:
    grid = np.linspace(0.0, 1.0, 65)
    numerical = solve_variable_coefficient_poisson(np.ones_like(grid))
    analytic = 0.5 * grid * (1.0 - grid)
    np.testing.assert_allclose(numerical, analytic, atol=1e-12)


def test_dataset_has_independent_function_level_splits() -> None:
    dataset = generate_poisson_dataset(
        n_train=4,
        n_validation=3,
        n_test=2,
        n_grid=16,
        seed=11,
    )
    assert dataset.train.coefficients.shape == (4, 16)
    assert dataset.validation.solutions.shape == (3, 16)
    assert dataset.test.solutions.shape == (2, 16)
    assert not np.array_equal(dataset.train.coefficients[0], dataset.test.coefficients[0])
    assert np.all(dataset.train.coefficients > 0)
    np.testing.assert_allclose(dataset.train.solutions[:, [0, -1]], 0.0)
