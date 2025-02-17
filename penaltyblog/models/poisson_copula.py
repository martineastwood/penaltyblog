import warnings

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid


@njit()
def frank_copula_pdf(u, v, kappa):
    """Computes the Frank copula probability density function with numerical stability."""

    if np.abs(kappa) < 1e-5:  # If kappa is close to 0, return independence
        return np.ones_like(u)

    # Compute exponentials
    exp_neg_kappa = np.exp(-kappa)
    exp_neg_kappa_u = np.exp(-kappa * u)
    exp_neg_kappa_v = np.exp(-kappa * v)
    exp_neg_kappa_uv = np.exp(-kappa * (u + v))

    num = kappa * exp_neg_kappa_uv * (1 - exp_neg_kappa)

    # Compute denominator safely
    denom = (exp_neg_kappa - 1 + (exp_neg_kappa_u - 1) * (exp_neg_kappa_v - 1)) ** 2
    denom = np.maximum(denom, 1e-10)  # Prevent division by zero

    copula_density = num / denom

    # Ensure probabilities remain within a valid range
    copula_density = np.clip(copula_density, 1e-10, 1)

    return copula_density


class PoissonCopulaGoalsModel:
    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
        self.goals_home = np.array(goals_home, dtype=int)
        self.goals_away = np.array(goals_away, dtype=int)
        self.teams_home = np.array(teams_home)
        self.teams_away = np.array(teams_away)
        self.weights = np.array(weights)

        self.teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))
        self.n_teams = len(self.teams)

        self._params = np.concatenate(
            (
                [1] * self.n_teams,  # Attack params
                [-1] * self.n_teams,  # Defense params
                [0.5],  # Home advantage
                [0.5],  # Frank copula parameter (kappa)
            )
        )

        self._res = None
        self.loglikelihood = None
        self.aic = None
        self.n_params = None
        self.fitted = False

    def __repr__(self):
        repr_str = "Poisson Copula Goal Model\n"
        repr_str += f"Fitted: {self.fitted}\n"
        if self.fitted:
            repr_str += f"AIC: {self.aic:.3f}\n"
            repr_str += "Parameters:\n"
            for i, team in enumerate(self.teams):
                repr_str += f"{team}: Attack {self._params[i]:.3f}, Defence {self._params[i + self.n_teams]:.3f}\n"
            repr_str += f"Home Advantage: {self._params[-2]:.3f}\n"
            repr_str += f"Kappa: {self._params[-1]:.3f}\n"
        else:
            repr_str += "Model has not been fitted yet.\n"
        return repr_str

    def _neg_log_likelihood(self, params):
        attack_params = params[: self.n_teams]
        defense_params = params[self.n_teams : 2 * self.n_teams]
        home_advantage = params[-2]
        copula_kappa = params[-1]

        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        home_idx = np.vectorize(team_to_idx.get)(self.teams_home)
        away_idx = np.vectorize(team_to_idx.get)(self.teams_away)

        lambda_home = np.exp(
            home_advantage + attack_params[home_idx] + defense_params[away_idx]
        )
        lambda_away = np.exp(attack_params[away_idx] + defense_params[home_idx])

        home_probs = poisson.pmf(self.goals_home, lambda_home)
        away_probs = poisson.pmf(self.goals_away, lambda_away)

        # Transform to uniform scale for copula (pseudo-observations)
        home_cdf = poisson.cdf(self.goals_home, lambda_home)
        away_cdf = poisson.cdf(self.goals_away, lambda_away)

        copula_probs = frank_copula_pdf(home_cdf, away_cdf, copula_kappa)

        # Compute joint log-likelihood
        log_likelihood = np.log(home_probs * away_probs * copula_probs) * self.weights
        return -np.sum(log_likelihood)

    def fit(self):
        options = {"maxiter": 500, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 2)] + [(0.0001, 5)]

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

        home_idx = np.where(self.teams == home_team)[0][0]
        away_idx = np.where(self.teams == away_team)[0][0]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-1]

        lambda_home = np.exp(home_advantage + home_attack + away_defense)
        lambda_away = np.exp(away_attack + home_defense)

        home_goals_vector = poisson(lambda_home).pmf(np.arange(max_goals))
        away_goals_vector = poisson(lambda_away).pmf(np.arange(max_goals))

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
                + ["home_advantage", "copula_kappa"],
                self._params,
            )
        )
        return params
