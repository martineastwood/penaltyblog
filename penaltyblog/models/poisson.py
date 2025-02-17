import warnings
from math import exp, lgamma, log

import numpy as np
from numba import njit
from scipy.optimize import minimize

from .football_probability_grid import FootballProbabilityGrid


class PoissonGoalsModel:
    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
        self.goals_home = np.array(goals_home, dtype=int)
        self.goals_away = np.array(goals_away, dtype=int)
        self.teams_home = np.array(teams_home)
        self.teams_away = np.array(teams_away)
        self.weights = np.array(weights)

        try:
            len(self.weights)
        except:
            self.weights = np.ones_like(self.goals_home)

        self.teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))
        self.n_teams = len(self.teams)

        self._params = np.concatenate(
            (
                np.ones(self.n_teams),  # Attack params
                -np.ones(self.n_teams),  # Defense params
                [0.5],  # Home advantage
            )
        )

        self._res = None
        self.loglikelihood = None
        self.aic = None
        self.n_params = None
        self.fitted = False

        # Precompute team index mapping for performance improvement
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}

        # Convert team names to indices for fast lookup
        self.home_idx = np.array([self.team_to_idx[t] for t in self.teams_home])
        self.away_idx = np.array([self.team_to_idx[t] for t in self.teams_away])

    def _neg_log_likelihood(self, params):
        """Negative Log-Likelihood optimized with Numba"""

        return _numba_neg_log_likelihood(
            params,
            self.n_teams,
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
        )

    def fit(self):
        options = {"maxiter": 100, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 3)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._neg_log_likelihood,
                self._params,
                constraints=constraints,
                bounds=bounds,
                options=options,
                method="L-BFGS-B",
            )

        self._params = self._res.x
        self.n_params = len(self._params)
        self.loglikelihood = -self._res.fun
        self.aic = -2 * self.loglikelihood + 2 * self.n_params
        self.fitted = True

    def predict(self, home_team, away_team, max_goals=15):
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Please call `fit()` first."
            )

        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Both teams must have been in the training data.")

        home_idx = self.team_to_idx[home_team]
        away_idx = self.team_to_idx[away_team]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-1]

        lambda_home = np.exp(home_advantage + home_attack + away_defense)
        lambda_away = np.exp(away_attack + home_defense)

        # Use a Numba-accelerated function to compute probability vectors
        home_goals_vector, away_goals_vector = _numba_poisson_pmf(
            lambda_home, lambda_away, max_goals
        )

        # Compute score matrix
        score_matrix = np.outer(home_goals_vector, away_goals_vector)

        # Return FootballProbabilityGrid
        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

    def get_params(self):
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Call `fit()` first."
            )

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defense_" + team for team in self.teams]
                + ["home_advantage"],
                self._params,
            )
        )
        return params


@njit
def poisson_logpmf(k, lambda_):
    """Compute log PMF of Poisson manually since Numba doesn't support scipy.stats.poisson"""
    if k < 0:
        return -np.inf  # Log PMF should be negative infinity for invalid k
    return k * log(lambda_) - lambda_ - lgamma(k + 1)


@njit
def _numba_neg_log_likelihood(
    params, n_teams, home_idx, away_idx, goals_home, goals_away, weights
):
    """Optimized negative log-likelihood function using Numba"""
    attack_params = params[:n_teams]
    defense_params = params[n_teams : 2 * n_teams]
    home_advantage = params[-1]

    n_matches = len(goals_home)
    log_likelihood = 0.0

    for i in range(n_matches):
        lambda_home = exp(
            home_advantage + attack_params[home_idx[i]] + defense_params[away_idx[i]]
        )
        lambda_away = exp(attack_params[away_idx[i]] + defense_params[home_idx[i]])

        log_likelihood += (
            poisson_logpmf(goals_home[i], lambda_home)
            + poisson_logpmf(goals_away[i], lambda_away)
        ) * weights[i]

    return -log_likelihood


@njit
def _numba_poisson_pmf(lambda_home, lambda_away, max_goals):
    """Computes Poisson PMF vectors using Numba for optimization"""
    home_goals_vector = np.zeros(max_goals)
    away_goals_vector = np.zeros(max_goals)

    for g in range(max_goals):
        home_goals_vector[g] = exp(
            poisson_logpmf(g, lambda_home)
        )  # Compute PMF from log PMF
        away_goals_vector[g] = exp(poisson_logpmf(g, lambda_away))

    return home_goals_vector, away_goals_vector
