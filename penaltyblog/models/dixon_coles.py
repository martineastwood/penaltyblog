from typing import Optional

import numpy as np
from numpy.typing import NDArray

from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.custom_types import (
    GoalInput,
    TeamInput,
    WeightInput,
)
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)

from .gradients import dixon_coles_gradient  # noqa
from .loss import dixon_coles_loss_function  # noqa
from .probabilities import compute_dixon_coles_probabilities  # noqa


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

    def _get_param_names(self) -> list[str]:
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defence_{t}" for t in self.teams]
            + ["home_advantage", "rho"]
        )

    def _get_tail_param_indices(self) -> dict[str, int]:
        return {"home_advantage": -2, "rho": -1}

    def _gradient(self, params):
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]  # Home field advantage
        rho = params[-1]  # Dixon-Coles rho adjustment

        # Note: dixon_coles_gradient computes the gradient of the log-likelihood,
        # but we're minimizing the negative log-likelihood, so we negate the gradient
        return -dixon_coles_gradient(
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

        loss = dixon_coles_loss_function(
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

        if np.isnan(loss) or np.isinf(loss):
            print(params)
            return 1e10

        return loss

    def fit(self, minimizer_options: Optional[dict] = None, use_gradient: bool = True):
        """
        Fits the model to the data and calculates the team strengths,
        home advantage and intercept. Must be called before `predict` can be used

        Parameters
        ----------
        minimizer_options : dict, optional
            Dictionary of options to pass to scipy.optimize.minimize (e.g., maxiter, ftol, disp). Default is None.

        use_gradient : bool, optional
            Whether to use the analytical gradient during optimization. Default is True.
            Setting to False will use numerical gradients instead, which may be slower but sometimes more stable.
        """
        constraints = [
            {
                "type": "eq",
                "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams,
            }
        ]

        bounds = [(-2.5, 2.5)] * self.n_teams * 2 + [(0, 2), (-2.5, 2.5)]

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

        return FootballProbabilityGrid(
            score_matrix,
            float(lambda_home.item()),
            float(lambda_away.item()),
            normalize=normalize,
        )
