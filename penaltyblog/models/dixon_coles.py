import warnings
from math import exp
from typing import Dict

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid
from .utils import numba_poisson_logpmf, numba_rho_correction, rho_correction


class DixonColesGoalModel:
    """Dixon and Coles adjusted Poisson model for predicting outcomes of football
    (soccer) matches

    Methods
    -------
    fit()
        fits a Dixon and Coles adjusted Poisson model to the data to calculate the team strengths.
        Must be called before the model can be used to predict game outcomes

    predict(home_team, away_team, max_goals=15)
        predict the outcome of a football (soccer) game between the home_team and away_team

    get_params()
        Returns the fitted parameters from the model
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
        """
        Parameters
        ----------
        goals_home : list
            A list or pd.Series of goals scored by the home_team
        goals_away : list
            A list or pd.Series of goals scored by the away_team
        teams_home : list
            A list or pd.Series of team_names for the home_team
        teams_away : list
            A list or pd.Series of team_names for the away_team
        weights : list
            A list or pd.Series of weights for the data,
            the lower the weight the less the match has on the output
        """

        self.fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
        self.fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
        self.fixtures["goals_home"] = self.fixtures["goals_home"].astype(int)
        self.fixtures["goals_away"] = self.fixtures["goals_away"].astype(int)
        self.fixtures["weights"] = weights

        self.teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))
        self.n_teams = len(self.teams)

        self._params = np.concatenate(
            (
                [1] * self.n_teams,
                [-1] * self.n_teams,
                [0.25],  # home advantage
                [-0.1],  # rho
            )
        )

        self._res = None
        self.loglikelihood = None
        self.aic = None
        self.n_params = None
        self.fitted = False

    def __repr__(self):
        lines = ["Module: Penaltyblog", "", "Model: Dixon and Coles", ""]

        if not self.fitted:
            lines.append("Status: Model not fitted")
            return "\n".join(lines)

        lines.extend(
            [
                f"Number of parameters: {self.n_params}",
                f"Log Likelihood: {round(self.loglikelihood, 3)}",
                f"AIC: {round(self.aic, 3)}",
                "",
                "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defence"),
                "-" * 60,
            ]
        )

        for idx, team in enumerate(self.teams):
            lines.append(
                "{0: <20} {1:<20} {2:<20}".format(
                    team,
                    round(self._params[idx], 3),
                    round(self._params[idx + self.n_teams], 3),
                )
            )

        lines.extend(
            [
                "-" * 60,
                f"Home Advantage: {round(self._params[-2], 3)}",
                f"Rho: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _fit(params: NDArray, fixtures: Dict, n_teams: int):
        """
        Internal method, not to called directly by the user
        """
        return _numba_neg_log_likelihood(
            params,
            n_teams,
            fixtures["home_idx"],
            fixtures["away_idx"],
            fixtures["goals_home"],
            fixtures["goals_away"],
            fixtures["weights"],
        )

    def fit(self):
        """
        Fits the model to the data and calculates the team strengths,
        home advantage and intercept. Should be called before `predict` can be used
        """
        options = {
            "maxiter": 100,
            "disp": False,
        }

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams,
            }
        ]

        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 2), (-2, 2)]

        # Pre-process fixtures to avoid redundant computation
        team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.fixtures["team_home_idx"] = (
            self.fixtures["team_home"].map(team_to_idx).astype(np.int32)
        )
        self.fixtures["team_away_idx"] = (
            self.fixtures["team_away"].map(team_to_idx).astype(np.int32)
        )
        self.fixtures["weights"] = self.fixtures["weights"].astype(np.float64)

        # Convert necessary columns to NumPy arrays for better performance
        processed_fixtures = {
            "home_idx": self.fixtures["team_home_idx"].values,
            "away_idx": self.fixtures["team_away_idx"].values,
            "goals_home": self.fixtures["goals_home"].values,
            "goals_away": self.fixtures["goals_away"].values,
            "weights": self.fixtures["weights"].values,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._fit,
                self._params,
                args=(processed_fixtures, self.n_teams),
                constraints=constraints,
                bounds=bounds,
                options=options,
            )

        self._params = self._res["x"]
        self.n_params = len(self._params)
        self.loglikelihood = self._res["fun"] * -1
        self.aic = -2 * (self.loglikelihood) + 2 * self.n_params
        self.fitted = True

    def predict(self, home_team, away_team, max_goals=15):
        """
        Predicts the probabilities of the different possible match outcomes

        Parameters
        ----------
        home_team : str
            The name of the home_team, must have been in the data the model was fitted on

        away_team : str
            The name of the away_team, must have been in the data the model was fitted on

        max_goals : int
            The maximum number of goals to calculate the probabilities over.
            Reducing this will improve performance slightly at the expensive of acuuracy

        Returns
        -------
        FootballProbabilityGrid
            A class providing access to a range of probabilites,
            such as 1x2, asian handicaps, over unders etc
        """
        if not self.fitted:
            raise ValueError(
                (
                    "Model's parameters have not been fit yet, please call the `fit()` "
                    "function before making any predictions"
                )
            )

        # check we have parameters for teams
        if home_team not in self.teams:
            raise ValueError(
                (
                    "No parameters for home team - "
                    "please ensure the team was included in the training data"
                )
            )

        if away_team not in self.teams:
            raise ValueError(
                (
                    "No parameters for away team - "
                    "please ensure the team was included in the training data"
                )
            )

        # get the relevant model parameters
        home_idx = np.where(self.teams == home_team)[0][0]
        away_idx = np.where(self.teams == away_team)[0][0]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]

        home_defence = self._params[home_idx + self.n_teams]
        away_defence = self._params[away_idx + self.n_teams]

        home_advantage = self._params[-2]
        rho = self._params[-1]

        # calculate the goal expectation
        home_goals = np.exp(home_advantage + home_attack + away_defence)
        away_goals = np.exp(away_attack + home_defence)
        home_goals_vector = poisson(home_goals).pmf(np.arange(0, max_goals))
        away_goals_vector = poisson(away_goals).pmf(np.arange(0, max_goals))

        # get the probabilities for each possible score
        m = np.outer(home_goals_vector, away_goals_vector)

        # apply Dixon and Coles adjustment
        m[0, 0] *= 1 - home_goals * away_goals * rho
        m[0, 1] *= 1 + home_goals * rho
        m[1, 0] *= 1 + away_goals * rho
        m[1, 1] *= 1 - rho

        # and return the FootballProbabilityGrid
        probability_grid = FootballProbabilityGrid(m, home_goals, away_goals)

        return probability_grid

    def get_params(self):
        """
        Provides access to the model's fitted parameters

        Returns
        -------
        dict
            A dict containing the model's parameters
        """
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet, please call the `fit()` function first"
            )

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defence_" + team for team in self.teams]
                + ["home_advantage", "rho"],
                self._res["x"],
            )
        )
        return params


def rho_correction_vec_np(goals_home, goals_away, home_exp, away_exp, rho):
    """Optimized Dixon-Coles adjustment"""
    # Create boolean masks for all conditions at once
    both_zero = (goals_home == 0) & (goals_away == 0)
    one_zero = (goals_home == 0) & (goals_away == 1)
    zero_one = (goals_home == 1) & (goals_away == 0)
    both_one = (goals_home == 1) & (goals_away == 1)

    # Use boolean indexing for faster computation
    result = np.ones_like(goals_home, dtype=float)
    result[both_zero | both_one] -= rho
    result[one_zero | zero_one] += rho

    return result


@njit
def _numba_neg_log_likelihood(
    params, n_teams, home_idx, away_idx, goals_home, goals_away, weights
):
    """Optimized negative log-likelihood function using Numba"""
    attack_params = params[:n_teams]
    defense_params = params[n_teams : 2 * n_teams]
    hfa, rho = params[-2:]

    n_matches = len(goals_home)
    log_likelihood = 0.0

    for i in range(n_matches):
        lambda_home = exp(
            hfa + attack_params[home_idx[i]] + defense_params[away_idx[i]]
        )
        lambda_away = exp(attack_params[away_idx[i]] + defense_params[home_idx[i]])

        log_likelihood += (
            numba_poisson_logpmf(goals_home[i], lambda_home)
            + numba_poisson_logpmf(goals_away[i], lambda_away)
            + np.log(
                numba_rho_correction(
                    goals_home[i], goals_away[i], lambda_home, lambda_away, rho
                )
            )
        ) * weights[i]

    return -log_likelihood
