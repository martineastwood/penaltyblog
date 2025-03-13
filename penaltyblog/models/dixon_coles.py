import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

# from .probabilities import compute_dixon_coles_probabilities
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

from .gradients import dixon_coles_gradient
from .loss import dixon_coles_loss_function
from .probabilities import compute_dixon_coles_probabilities


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

    def _gradient(self, params):
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]  # Home field advantage
        rho = params[-1]  # Dixon-Coles rho adjustment

        return dixon_coles_gradient(
            attack,
            defence,
            hfa,
            rho,
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
        )

    def _loss_function(self, params: NDArray) -> float:
        """
        Internal method, not to called directly by the user
        """
        # Get params
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]
        rho = params[-1]

        return dixon_coles_loss_function(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            attack,
            defence,
            hfa,
            rho,
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
                # jac=self._gradient,
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
        home_advantage = self._params[-2]
        rho = self._params[-1]

        # Preallocate the score matrix as a flattened array.
        score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)

        # Allocate one-element arrays for lambda values.
        lambda_home = np.empty(1, dtype=np.float64)
        lambda_away = np.empty(1, dtype=np.float64)

        compute_dixon_coles_probabilities(
            float(home_attack),
            float(away_attack),
            float(home_defense),
            float(away_defense),
            float(home_advantage),
            float(rho),
            int(max_goals),
            score_matrix,
            lambda_home,
            lambda_away,
        )

        score_matrix.shape = (max_goals, max_goals)

        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

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
