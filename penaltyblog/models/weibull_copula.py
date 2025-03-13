import ctypes
import warnings
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

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

from .loss import compute_weibull_copula_loss
from .probabilities import compute_weibull_copula_probabilities


class WeibullCopulaGoalsModel(BaseGoalsModel):
    """
    Weibull Copula model for predicting outcomes of football (soccer) matches

    Methods
    -------
    fit()
        fits a Weibull Copula model to the data to calculate the team strengths.
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
        Initialises the WeibullCopulaGoalModel class.

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

        # Quick guess initialization
        rng = np.random.default_rng()
        atk_init = rng.normal(1.0, 0.1, self.n_teams)
        def_init = rng.normal(-1.0, 0.1, self.n_teams)
        home_init = np.array([0.5 + rng.normal(0, 0.1)])
        shape_init = np.array([1.2])
        kappa_init = np.array([1.5])
        self.max_goals = 15
        self.jmax = 25

        self._params = np.concatenate(
            [atk_init, def_init, home_init, shape_init, kappa_init]
        )

    def __repr__(self) -> str:
        lines = [
            "Module: Penaltyblog",
            "",
            "Model: Bivariate Weibull Count + Copula",
            "",
        ]

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
                f"Home Advantage: {round(self._params[-3], 3)}",
                f"Weibull Shape: {round(self._params[-2], 3)}",
                f"Kappa: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _neg_log_likelihood(self, params: NDArray) -> float:
        # Get params
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-3]
        shape = params[-2]
        kappa = params[-1]

        return compute_weibull_copula_loss(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            attack,
            defence,
            hfa,
            shape,
            kappa,
            self.max_goals,
        )

    def fit(self):
        """
        Fits the Weibull Copula model to the data.
        """
        # create bounds
        bnds = []
        # Attack in [-3,3]
        for _ in range(self.n_teams):
            bnds.append((-3, 3))
        # Defense in [-3,3]
        for _ in range(self.n_teams):
            bnds.append((-3, 3))
        # home advantage in [-2,2]
        bnds.append((-2, 2))
        # shape in (0.01, 2.5)
        bnds.append((1e-2, 2.5))
        # kappa in [-5,10]
        bnds.append((-5, 5))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            self._res = minimize(
                self._neg_log_likelihood,
                x0=self._params,
                bounds=bnds,
                options={"maxiter": 1000, "disp": False},
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
        Predicts the probability of each market for a given home and away team.

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
            A FootballProbabilityGrid object containing the probability of each scoreline
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
        home_advantage = self._params[-3]
        shape = self._params[-2]
        kappa = self._params[-1]

        # Preallocate the score matrix as a flattened array.
        score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)

        # Allocate one-element arrays for lambda values.
        lambda_home = np.empty(1, dtype=np.float64)
        lambda_away = np.empty(1, dtype=np.float64)

        compute_weibull_copula_probabilities(
            float(home_attack),
            float(away_attack),
            float(home_defense),
            float(away_defense),
            float(home_advantage),
            float(shape),
            float(kappa),
            int(max_goals),
            score_matrix,
            lambda_home,
            lambda_away,
        )

        score_matrix.shape = (max_goals, max_goals)

        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

    def get_params(self) -> ParamsOutput:
        """
        Return the fitted parameters in a dictionary.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        assert self.n_params is not None

        # Construct dictionary
        param_names = (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "shape", "kappa"]
        )
        vals = list(self._params)
        result = dict(zip(param_names, vals))

        return result

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
