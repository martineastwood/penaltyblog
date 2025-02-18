import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import poisson

import penaltyblog as pb


def test_numba_poisson_logpmf():
    test_cases = [(0, 1.0), (1, 2.0), (5, 3.0), (-1, 1.0)]  # Edge case

    for k, lambda_ in test_cases:
        expected = poisson.logpmf(k, lambda_) if k >= 0 else -np.inf
        result = pb.models.numba_poisson_logpmf(k, lambda_)
        np.testing.assert_almost_equal(result, expected, decimal=6)


def test_numba_poisson_pmf():
    lambda_home, lambda_away = 1.5, 2.0
    max_goals = 5

    home_vector, away_vector = pb.models.numba_poisson_pmf(
        lambda_home, lambda_away, max_goals
    )

    expected_home = np.array([poisson.pmf(i, lambda_home) for i in range(max_goals)])
    expected_away = np.array([poisson.pmf(i, lambda_away) for i in range(max_goals)])

    np.testing.assert_array_almost_equal(home_vector, expected_home)
    np.testing.assert_array_almost_equal(away_vector, expected_away)


def test_numba_rho_correction_llh():
    test_cases = [
        ((0, 0, 1.0, 1.0, 0.1), 1.1),  # Both 0
        ((0, 1, 1.0, 1.0, 0.1), 0.9),  # 0-1
        ((1, 0, 1.0, 1.0, 0.1), 0.9),  # 1-0
        ((1, 1, 1.0, 1.0, 0.1), 1.1),  # Both 1
        ((2, 2, 1.0, 1.0, 0.1), 1.0),  # Other cases
    ]

    for inputs, expected in test_cases:
        result = pb.models.numba_rho_correction_llh(*inputs)
        np.testing.assert_almost_equal(result, expected)


def test_frank_copula_pdf():
    # Test independence case
    u = np.array([0.2, 0.5, 0.8])
    v = np.array([0.3, 0.6, 0.9])
    result = pb.models.frank_copula_pdf(u, v, 0.0)
    np.testing.assert_array_almost_equal(result, np.ones_like(u))

    # Test with non-zero kappa
    kappa = 2.0
    result = pb.models.frank_copula_pdf(u, v, kappa)
    assert np.all(result > 0)  # Density should be positive
    assert np.all(result <= 1)  # Density should be bounded

    # Test edge cases
    edge_cases = [(0, 0, 1.0), (1, 1, 1.0)]
    for u_val, v_val, kappa in edge_cases:
        result = pb.models.frank_copula_pdf(np.array([u_val]), np.array([v_val]), kappa)
        assert np.isfinite(result)
