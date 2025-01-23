import collections
import collections.abc
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid
from .utils import rho_correction_vec


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
        repr_str = ""
        repr_str += "Module: Penaltyblog"
        repr_str += "\n"
        repr_str += "\n"

        repr_str += "Model: Dixon and Coles"
        repr_str += "\n"
        repr_str += "\n"

        if not self.fitted:
            repr_str += "Status: Model not fitted"
            return repr_str

        repr_str += "Number of parameters: {0}".format(self.n_params)
        repr_str += "\n"
        repr_str += "Log Likelihood: {0}".format(round(self.loglikelihood, 3))
        repr_str += "\n"
        repr_str += "AIC: {0}".format(round(self.aic, 3))
        repr_str += "\n"
        repr_str += "\n"

        repr_str += "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defence")
        repr_str += "\n"
        repr_str += "-" * 60
        repr_str += "\n"

        for idx, team in enumerate(self.teams):
            repr_str += "{0: <20} {1:<20} {2:<20}".format(
                self.teams[idx],
                round(self._params[idx], 3),
                round(self._params[idx + self.n_teams], 3),
            )
            repr_str += "\n"

        repr_str += "-" * 60
        repr_str += "\n"

        repr_str += "Home Advantage: {0}".format(round(self._params[-2], 3))
        repr_str += "\n"
        repr_str += "Rho: {0}".format(round(self._params[-1], 3))
        repr_str += "\n"

        return repr_str

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _fit(params: NDArray, fixtures: Dict, n_teams: int):
        """
        Internal method, not to called directly by the user
        """
        """Internal method, not to called directly by the user"""
        # Extract all needed parameters and data at once
        attack_params = params[:n_teams]
        defence_params = params[n_teams : n_teams * 2]
        hfa, rho = params[-2:]

        goals_home = fixtures["goals_home"]
        goals_away = fixtures["goals_away"]
        home_idx = fixtures["home_idx"]
        away_idx = fixtures["away_idx"]

        # Compute expected goals in one step
        home_exp = np.exp(hfa + attack_params[home_idx] + defence_params[away_idx])
        away_exp = np.exp(attack_params[away_idx] + defence_params[home_idx])

        # Calculate log-likelihoods
        llk = (
            poisson.logpmf(goals_home, home_exp)
            + poisson.logpmf(goals_away, away_exp)
            + np.log(
                rho_correction_vec_np(goals_home, goals_away, home_exp, away_exp, rho)
            )
        ) * fixtures["weights"]

        return -np.sum(llk)

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
