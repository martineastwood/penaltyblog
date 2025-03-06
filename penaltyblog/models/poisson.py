import ctypes
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from penaltyblog.golib.loss import poisson_loss_function
from penaltyblog.golib.probabilities import compute_poisson_probabilities
from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.custom_types import (
    GoalInput,
    ParamsOutput,
    TeamInput,
    WeightInput,
)
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)


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
        params = np.ascontiguousarray(params, dtype=np.float64)
        params_ctypes = params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        llk = poisson_loss_function(
            params_ctypes,
            self.n_teams,
            self.home_idx_ctypes,
            self.away_idx_ctypes,
            self.goals_home_ctypes,
            self.goals_away_ctypes,
            self.weights_ctypes,
            len(self.goals_home),
        )
        return llk

    def fit(self):
        """
        Fits the Poisson model to the data using maximum likelihood estimation
        """
        options = {"maxiter": 1000, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 3)]

        # llk = self._loss_function(self._params)
        # print(f"Initial LLK: {llk}")

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

        score_matrix, lambda_home, lambda_away = compute_poisson_probabilities(
            home_attack,
            away_attack,
            home_defense,
            away_defense,
            home_advantage,
            max_goals,
        )

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
