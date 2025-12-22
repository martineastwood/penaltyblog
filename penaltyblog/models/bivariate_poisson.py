from typing import Optional

import numpy as np
from numpy.typing import NDArray

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

from .gradients import bivariate_poisson_gradient  # noqa
from .loss import compute_bivariate_poisson_loss  # noqa
from .probabilities import compute_bivariate_poisson_probabilities  # noqa


class BivariatePoissonGoalModel(BaseGoalsModel):
    """
    Karlis & Ntzoufras Bivariate Poisson for soccer, with:
      X = W1 + W3
      Y = W2 + W3
    where W1, W2, W3 ~ independent Poisson(lambda1, lambda2, lambda3).
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
        Initialises the BivariatePoissonGoalModel class.

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
                [0.0] * self.n_teams,  # Attack
                [0.0] * self.n_teams,  # Defense
                [0.1],  # Home advantage
                [0.0],  # correlation_param => lambda3 = exp(0)=1
            )
        )

    def __repr__(self) -> str:
        lines = ["Module: Penaltyblog", "", "Model: Bivariate Poisson", ""]

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
                f"Correlation: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _loss_function(self, params: NDArray) -> float:
        # Get params
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]
        correlation = params[-1]

        return compute_bivariate_poisson_loss(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            attack,
            defence,
            hfa,
            correlation,
        )

    def _gradient(self, params: NDArray) -> NDArray:
        """
        Compute the gradient of the negative log-likelihood.
        """
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]
        correlation = params[-1]

        return bivariate_poisson_gradient(
            attack,
            defence,
            hfa,
            correlation,
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
        )

    def fit(self, minimizer_options: Optional[dict] = None, use_gradient: bool = True):
        """
        Fits the Bivariate Poisson model to the data.

        Parameters
        ----------
        minimizer_options : dict, optional
            Dictionary of options to pass to scipy.optimize.minimize (e.g., maxiter, ftol, disp). Default is None.
        use_gradient : bool, optional
            Whether to use analytical gradients for optimization. Default is True.
        """
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bnds = [(-3, 3)] * (2 * self.n_teams) + [(-2, 2), (-3, 3)]

        gradient_func = self._gradient if use_gradient else None

        self._fit(
            self._loss_function,
            self._params,
            constraints,
            bnds,
            minimizer_options,
            gradient_func,
        )

    def _compute_probabilities(
        self, home_idx: int, away_idx: int, max_goals: int, normalize: bool = True
    ) -> FootballProbabilityGrid:
        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-2]
        correlation = self._params[-1]

        # Preallocate the score matrix as a flattened array.
        score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)

        # Allocate one-element arrays for lambda values.
        lambda_home = np.empty(1, dtype=np.float64)
        lambda_away = np.empty(1, dtype=np.float64)

        compute_bivariate_poisson_probabilities(
            float(home_attack),
            float(away_attack),
            float(home_defense),
            float(away_defense),
            float(home_advantage),
            float(correlation),
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

    def _get_param_names(self) -> list[str]:
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "correlation_log"]
        )

    def _get_tail_param_indices(self) -> dict[str, int]:
        return {"home_advantage": -2, "correlation": -1}

    def get_params(self) -> ParamsOutput:
        """
        Return the fitted parameters in a dictionary.
        """
        params = super().get_params()
        params["lambda3"] = np.exp(params["correlation_log"])
        return params
