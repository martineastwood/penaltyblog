"""
Zero-Inflated Poisson Model for Football Goal Scoring

This module implements the Zero-Inflated Poisson model for predicting football match outcomes.
"""

from typing import Optional

import numpy as np

from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.custom_types import (
    GoalInput,
    TeamInput,
    WeightInput,
)
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)

from .gradients import zero_inflated_poisson_gradient  # noqa
from .loss import compute_zero_inflated_poisson_loss  # noqa
from .probabilities import compute_zero_inflated_poisson_probabilities  # noqa


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

    def _get_param_names(self) -> list[str]:
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defence_{t}" for t in self.teams]
            + ["home_advantage", "zero_inflation"]
        )

    def _get_tail_param_indices(self) -> dict[str, int]:
        return {"home_advantage": -2, "zero_inflation": -1}

    def _loss_function(self, params: np.ndarray) -> float:
        # Get params
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = float(params[-2])
        zero_inflation = float(params[-1])

        return compute_zero_inflated_poisson_loss(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            attack,
            defence,
            hfa,
            zero_inflation,
        )

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = float(params[-2])
        zero_inflation = float(params[-1])

        return zero_inflated_poisson_gradient(
            attack,
            defence,
            hfa,
            zero_inflation,
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
        )

    def fit(
        self,
        minimizer_options: Optional[dict] = None,
        use_gradient: bool = True,
    ):
        """
        Fit the Zero-Inflated Poisson model using scipy.optimize.minimize.

        Parameters
        ----------
        minimizer_options : dict, optional
            Dictionary of options to pass to scipy.optimize.minimize (e.g., maxiter, ftol, disp). Default is None.

        use_gradient : bool, optional
            Whether to use the analytical gradient during optimization. Default is True.
            Setting to False will use numerical gradients instead, which may be slower but sometimes more stable.
        """
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        # Zero inflation parameter should be unbounded since it's on probability scale (not logit)
        # Based on the loss function, it appears to use direct probability, so keep bounds [0, 1]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 3)] + [(1e-6, 1 - 1e-6)]

        # Use gradient if requested and available
        jac = self._gradient if use_gradient else None

        self._fit(
            self._loss_function,
            self._params,
            constraints,
            bounds,
            minimizer_options,
            jac,
        )

    def _compute_probabilities(
        self, home_idx: int, away_idx: int, max_goals: int, normalize: bool = True
    ) -> FootballProbabilityGrid:
        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-2]
        zero_inflation = self._params[-1]

        # Preallocate the score matrix as a flattened array.
        score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)

        # Allocate one-element arrays for lambda values.
        lambda_home = np.empty(1, dtype=np.float64)
        lambda_away = np.empty(1, dtype=np.float64)

        compute_zero_inflated_poisson_probabilities(
            float(home_attack),
            float(away_attack),
            float(home_defense),
            float(away_defense),
            float(home_advantage),
            float(zero_inflation),
            int(max_goals),
            score_matrix,
            lambda_home,
            lambda_away,
        )

        score_matrix.shape = (max_goals, max_goals)

        return FootballProbabilityGrid(
            score_matrix,
            float(lambda_home[0]),
            float(lambda_away[0]),
            normalize=normalize,
        )
