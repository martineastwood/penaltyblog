from math import exp, lgamma, log

import numpy as np
import pandas as pd
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
        home_goals_vector[g] = exp(
            numba_poisson_logpmf(g, lambda_home)
        )  # Compute PMF from log PMF
        away_goals_vector[g] = exp(numba_poisson_logpmf(g, lambda_away))

    return home_goals_vector, away_goals_vector


@njit
def numba_rho_correction(goals_home, goals_away, lambda_home, lambda_away, rho):
    # Explicitly define types for Numba
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


def rho_correction_vec(df: pd.DataFrame) -> NDArray:
    dc_adj = np.select(
        [
            (df["goals_home"] == 0) & (df["goals_away"] == 0),
            (df["goals_home"] == 0) & (df["goals_away"] == 1),
            (df["goals_home"] == 1) & (df["goals_away"] == 0),
            (df["goals_home"] == 1) & (df["goals_away"] == 1),
        ],
        [
            1 - (df["home_exp"] * df["away_exp"] * df["rho"]),
            1 + (df["home_exp"] * df["rho"]),
            1 + (df["away_exp"] * df["rho"]),
            1 - df["rho"],
        ],
        default=1,
    )
    return dc_adj


def rho_correction(
    goals_home: int, goals_away: int, home_exp: float, away_exp: float, rho: float
) -> float:
    """
    Applies the dixon and coles correction
    """
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_weights(dates, xi=0.0018, base_date=None) -> NDArray:
    """
    Calculates a decay curve based on the algorithm given by
    Dixon and Coles in their paper

    Parameters
    ----------
    dates : list
        A list or pd.Series of dates to calculate weights for
    x1 : float
        Controls the steepness of the decay curve
    base_date : date
        The base date to start the decay from. If set to None
        then it uses the maximum date
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights
