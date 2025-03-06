"""
Zero-Inflated Poisson Model for Football Goal Scoring

This module implements the Zero-Inflated Poisson model for predicting football match outcomes.
"""

import ctypes
import warnings

import numpy as np
from scipy.optimize import minimize

from penaltyblog.golib.loss import zero_inflated_poisson_loss_function
from penaltyblog.golib.probabilities import compute_zip_poisson_probabilities
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


class ZeroInflatedPoissonGoalsModel(BaseGoalsModel):
    """
    Zero-Inflated Poisson Model for Football Goal Scoring
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
        Initialises the ZeroInflatedPoissonGoalsModel class.

        Parameters
        ----------
        goals_home : array_like
            The number of goals scored by the home team
        goals_away : array_like
            The number of goals scored by the away team
        teams_home : array_like
            The names of the home teams
        teams_away : array_like
            The names of the away teams
        weights : array_like, optional
            The weights of the matches, by default None
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        self._params = np.concatenate(
            (
                np.ones(self.n_teams),  # Attack params
                -np.ones(self.n_teams),  # Defense params
                [0.5],  # Home advantage
                [0.1],  # Zero-inflation probability (logit scale)
            )
        )

    def __repr__(self):
        lines = ["Module: Penaltyblog", "", "Model: Zero-inflated Poisson", ""]

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
                f"Zero Inflation: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _loss_function(self, params: np.ndarray) -> float:
        """Negative Log-Likelihood optimized with Go"""
        params = np.ascontiguousarray(params, dtype=np.float64)
        params_ctypes = params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        return zero_inflated_poisson_loss_function(
            params_ctypes,
            self.n_teams,
            self.home_idx_ctypes,
            self.away_idx_ctypes,
            self.goals_home_ctypes,
            self.goals_away_ctypes,
            self.weights_ctypes,
            len(self.goals_home),
        )

    def fit(self):
        options = {"maxiter": 1000, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 3)] + [(0, 1)]

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

    def get_params(self) -> ParamsOutput:
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Call `fit()` first."
            )

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defense_" + team for team in self.teams]
                + ["home_advantage"]
                + ["zero_inflation"],
                self._params,
            )
        )
        return params

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 15
    ) -> FootballProbabilityGrid:
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
        home_advantage = self._params[-2]
        zero_inflation = self._params[-1]

        # Compute expected goals
        lambda_home = np.exp(home_advantage + home_attack + away_defense)
        lambda_away = np.exp(away_attack + home_defense)

        # Compute ZIP Poisson PMF for home and away teams
        score_matrix, lambda_home, lambda_away = compute_zip_poisson_probabilities(
            home_attack,
            away_attack,
            home_defense,
            away_defense,
            home_advantage,
            zero_inflation,
            max_goals,
        )

        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)
