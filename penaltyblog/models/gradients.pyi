from typing import Any

from numpy.typing import NDArray

def poisson_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]: ...
def dixon_coles_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    rho: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]: ...
def negative_binomial_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    dispersion: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]: ...
def zero_inflated_poisson_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    zero_inflation: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]: ...
def bivariate_poisson_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    correlation: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]: ...
def weibull_copula_gradient(
    attack: NDArray[Any],
    defence: NDArray[Any],
    hfa: float,
    shape: float,
    kappa: float,
    home_idx: NDArray[Any],
    away_idx: NDArray[Any],
    goals_home: NDArray[Any],
    goals_away: NDArray[Any],
    weights: NDArray[Any],
    max_goals: int,
) -> NDArray[Any]: ...
