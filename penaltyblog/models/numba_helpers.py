from math import exp, lgamma, log

import numpy as np
from numba import float64, njit
from numpy.typing import NDArray


@njit
def numba_poisson_logpmf(k, lambda_):
    """Compute log PMF of Poisson manually since Numba doesn't support scipy.stats.poisson"""
    if k < 0:
        return -np.inf  # Log PMF should be negative infinity for invalid k
    return k * log(lambda_) - lambda_ - lgamma(k + 1)


@njit
def numba_poisson_pmf(lambda_home, lambda_away, max_goals):
    """Computes Poisson PMF vectors using Numba for optimization"""
    home_goals_vector = np.zeros(max_goals)
    away_goals_vector = np.zeros(max_goals)

    for g in range(max_goals):
        home_goals_vector[g] = exp(numba_poisson_logpmf(g, lambda_home))
        away_goals_vector[g] = exp(numba_poisson_logpmf(g, lambda_away))

    return home_goals_vector, away_goals_vector


@njit
def numba_rho_correction_llh(goals_home, goals_away, lambda_home, lambda_away, rho):
    goals_home = float64(goals_home)
    goals_away = float64(goals_away)
    lambda_home = float64(lambda_home)
    lambda_away = float64(lambda_away)
    rho = float64(rho)

    return (
        1.0
        + (goals_home == 0 and goals_away == 0) * rho
        - (goals_home == 0 and goals_away == 1) * rho
        - (goals_home == 1 and goals_away == 0) * rho
        + (goals_home == 1 and goals_away == 1) * rho
    )


@njit()
def frank_copula_pdf(u: NDArray, v: NDArray, kappa: float) -> NDArray:
    """
    Computes the Frank copula probability density function with numerical stability.
    """
    # Convert to float64 array for consistent return types
    result = np.ones_like(u, dtype=np.float64)

    if np.abs(kappa) < 1e-5:  # If kappa is close to 0, return independence
        return result

    # Compute exponentials
    exp_neg_kappa = np.exp(-kappa)
    exp_neg_kappa_u = np.exp(-kappa * u)
    exp_neg_kappa_v = np.exp(-kappa * v)
    exp_neg_kappa_uv = np.exp(-kappa * (u + v))

    num = kappa * exp_neg_kappa_uv * (1 - exp_neg_kappa)

    # Compute denominator safely
    denom = (exp_neg_kappa - 1 + (exp_neg_kappa_u - 1) * (exp_neg_kappa_v - 1)) ** 2
    denom = np.maximum(denom, 1e-10)  # Prevent division by zero

    result = num / denom
    result = np.clip(result, 1e-10, 1)

    return result
