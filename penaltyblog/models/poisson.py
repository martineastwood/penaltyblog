import warnings
from math import exp

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize

from .base_model import BaseGoalsModel
from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid
from .numba_helpers import numba_poisson_logpmf, numba_poisson_pmf


class PoissonGoalsModel(BaseGoalsModel):
    """
    Poisson model for predicting outcomes of football (soccer) matches

    Methods
    -------
    fit()
        fits a Poisson model to the data to calculate the team strengths.
        Must be called before the model can be used to predict game outcomes

    predict(home_team, away_team, max_goals=15)
        predicts the probability of each scoreline for a given home and away team

    get_params()
        provides access to the model's fitted parameters
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
        Poisson model for predicting outcomes of football (soccer) matches

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
                np.ones(self.n_teams),  # Attack params
                -np.ones(self.n_teams),  # Defense params
                [0.5],  # Home advantage
            )
        )

    def __repr__(self) -> str:
        lines = ["Module: Penaltyblog", "", "Model: Poisson", ""]

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
                f"Home Advantage: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _loss_function(self, params: NDArray) -> float:
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
        """
        Fits the Poisson model to the data using maximum likelihood estimation
        """
        options = {"maxiter": 1000, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 3)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._loss_function,
                self._params,
                constraints=constraints,
                bounds=bounds,
                options=options,
                # method="SLSQP",
            )

        if not self._res.success:
            raise ValueError(f"Optimization failed with message: {self._res.message}")

        self._params = self._res.x
        self.n_params = len(self._params)
        self.loglikelihood = -self._res.fun
        self.aic = -2 * self.loglikelihood + 2 * self.n_params
        self.fitted = True

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 15
    ) -> FootballProbabilityGrid:
        """
        Predicts the probability of each scoreline for a given home and away team

        Parameters
        ----------
        home_team : str
            The name of the home team
        away_team : str
            The name of the away team
        max_goals : int, optional
            The maximum number of goals to consider, by default 15

        Returns
        -------
        FootballProbabilityGrid
            A FootballProbabilityGrid object containing the probabilities of each scoreline
        """
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

        home_goals_vector, away_goals_vector = numba_poisson_pmf(
            lambda_home, lambda_away, max_goals
        )

        score_matrix = np.outer(home_goals_vector, away_goals_vector)

        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

    def get_params(self) -> ParamsOutput:
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Call `fit()` first."
            )

        assert self.n_params is not None

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
def _numba_neg_log_likelihood(
    params: NDArray,
    n_teams: int,
    home_idx: NDArray,
    away_idx: NDArray,
    goals_home: NDArray,
    goals_away: NDArray,
    weights: NDArray,
) -> float:
    """
    Internal method, not to be called directly by the user

    Calculates the negative log-likelihood of the Poisson model

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
        The negative log-likelihood of the Poisson model
    """
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
            numba_poisson_logpmf(goals_home[i], lambda_home)
            + numba_poisson_logpmf(goals_away[i], lambda_away)
        ) * weights[i]

    return -log_likelihood
