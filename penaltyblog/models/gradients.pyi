import numpy as np
from numpy.typing import NDArray

def poisson_gradient(
    attack: NDArray[np.float64],
    defence: NDArray[np.float64],
    hfa: float,
    home_idx: NDArray[np.int64],
    away_idx: NDArray[np.int64],
    goals_home: NDArray[np.int64],
    goals_away: NDArray[np.int64],
) -> NDArray[np.float64]: ...
def dixon_coles_gradient(
    attack: NDArray[np.float64],
    defence: NDArray[np.float64],
    hfa: float,
    rho: float,
    home_idx: NDArray[np.int64],
    away_idx: NDArray[np.int64],
    goals_home: NDArray[np.int64],
    goals_away: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def negative_binomial_gradient(
    attack: NDArray[np.float64],
    defence: NDArray[np.float64],
    hfa: float,
    dispersion: float,
    home_idx: NDArray[np.int64],
    away_idx: NDArray[np.int64],
    goals_home: NDArray[np.int64],
    goals_away: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def zero_inflated_poisson_gradient(
    attack: NDArray[np.float64],
    defence: NDArray[np.float64],
    hfa: float,
    zero_inflation: float,
    home_idx: NDArray[np.int64],
    away_idx: NDArray[np.int64],
    goals_home: NDArray[np.int64],
    goals_away: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bivariate_poisson_gradient(
    attack: NDArray[np.float64],
    defence: NDArray[np.float64],
    hfa: float,
    correlation: float,
    home_idx: NDArray[np.int64],
    away_idx: NDArray[np.int64],
    goals_home: NDArray[np.int64],
    goals_away: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]: ...
