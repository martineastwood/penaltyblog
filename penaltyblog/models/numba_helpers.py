from math import exp, lgamma, log

import numpy as np
from numba import float64, njit


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
