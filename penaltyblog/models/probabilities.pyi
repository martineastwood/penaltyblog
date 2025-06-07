import numpy as np
from numpy.typing import NDArray

def compute_poisson_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
def compute_dixon_coles_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    rho: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
def compute_negative_binomial_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    dispersion: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
def compute_zero_inflated_poisson_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    rho: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
def compute_bivariate_poisson_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    rho: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
def compute_weibull_copula_probabilities(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    shape: float,
    kappa: float,
    max_goals: int,
    score_matrix: NDArray[np.float64],
    lambda_home: NDArray[np.float64],
    lambda_away: NDArray[np.float64],
) -> None: ...
