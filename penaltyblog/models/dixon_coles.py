import warnings
from math import exp

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import poisson

from .base_model import BaseGoalsModel
from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid
from .numba_helpers import numba_poisson_logpmf, numba_rho_correction_llh


class DixonColesGoalModel(BaseGoalsModel):
    """
    Dixon and Coles adjusted Poisson model for predicting outcomes of football
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

    def __init__(
        self,
        goals_home: GoalInput,
        goals_away: GoalInput,
        teams_home: TeamInput,
        teams_away: TeamInput,
        weights: WeightInput = None,
    ):
        """
        Dixon and Coles adjusted Poisson model for predicting outcomes of football
        (soccer) matches

        Parameters
        ----------
        goals_home : array_like
            The number of goals scored by the home team in each match
        goals_away : array_like
            The number of goals scored by the away team in each match
        teams_home : array_like
            The name of the home team in each match
        teams_away : array_like
            The name of the away team in each match
        weights : array_like, optional
            The weight of each match, by default None
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        self._params = np.concatenate(
            (
                [1] * self.n_teams,
                [-1] * self.n_teams,
                [0.25],  # home advantage
                [-0.1],  # rho
            )
        )

    def __repr__(self) -> str:
        lines = ["Module: Penaltyblog", "", "Model: Dixon and Coles", ""]

        if not self.fitted:
            lines.append("Status: Model not fitted")
            return "\n".join(lines)

        assert self.aic is not None
        assert self.loglikelihood is not None
        assert self.n_params is not None

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

    def _loss_function(self, params: NDArray) -> float:
        """
        Internal method, not to called directly by the user
        """
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
        """
        Fits the model to the data and calculates the team strengths,
        home advantage and intercept. Must be called before `predict` can be used
        """
        options = {
            "maxiter": 1000,
            "disp": False,
        }

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams,
            }
        ]

        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 2), (-2, 2)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._loss_function,
                self._params,
                constraints=constraints,
                bounds=bounds,
                options=options,
            )

        if not self._res.success:
            raise ValueError(f"Optimization failed with message: {self._res.message}")

        self._params = self._res["x"]
        self.n_params = len(self._params)
        self.loglikelihood = self._res["fun"] * -1
        self.aic = -2 * (self.loglikelihood) + 2 * self.n_params
        self.fitted = True

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 15
    ) -> FootballProbabilityGrid:
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

    def get_params(self) -> ParamsOutput:
        """
        Returns the model's fitted parameters as a dictionary

        Returns
        -------
        dict
            A dict containing the model's parameters
        """
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet, please call the `fit()` function first"
            )

        assert self.n_params is not None
        assert self._res is not None

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defence_" + team for team in self.teams]
                + ["home_advantage", "rho"],
                self._res["x"],
            )
        )
        return params

    @property
    def params(self) -> dict:
        """
        Property to retrieve the fitted model parameters.
        Same as `get_params()`, but allows attribute-like access.

        Returns
        -------
        dict
            A dictionary containing attack, defense, home advantage, and correlation parameters.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        return self.get_params()


@njit
def _numba_neg_log_likelihood(
    params, n_teams, home_idx, away_idx, goals_home, goals_away, weights
) -> float:
    """
    Internal method, not to be called directly by the user

    Calculates the negative log-likelihood of the Dixon and Coles model

    Parameters
    ----------
    params : array_like
        The parameters of the model
    n_teams : int
        The number of teams in the league
    home_idx : array_like
        The indices of the home teams in the data
    away_idx : array_like
        The indices of the away teams in the data
    goals_home : array_like
        The number of goals scored by the home teams
    goals_away : array_like
        The number of goals scored by the away teams
    weights : array_like
        The weights of the matches

    Returns
    -------
    float
        The negative log-likelihood of the Dixon and Coles model
    """
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
                numba_rho_correction_llh(
                    goals_home[i], goals_away[i], lambda_home, lambda_away, rho
                )
            )
        ) * weights[i]

    return -log_likelihood
