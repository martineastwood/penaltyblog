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

from .gradients import negative_binomial_gradient  # noqa
from .loss import compute_negative_binomial_loss  # noqa
from .probabilities import compute_negative_binomial_probabilities  # noqa


class NegativeBinomialGoalModel(BaseGoalsModel):
    """
    Negative Binomial model for predicting outcomes of football (soccer) matches
    handling overdispersion in goal data.

    Methods
    -------
    fit()
        fits a Negative Binomial model to the data to calculate the team strengths.
        Must be called before the model can be used to predict game outcomes

    predict(home_team, away_team, max_goals=10)
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
        Initialises the NegativeBinomialGoalModel class.

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
            ([1] * self.n_teams, [-1] * self.n_teams, [0.25], [0.1])
        )  # Home advantage and dispersion parameter

    def __repr__(self):
        lines = ["Module: Penaltyblog", "", "Model: Negative Binomial", ""]

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
                f"Dispersion: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _get_param_names(self) -> list[str]:
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defence_{t}" for t in self.teams]
            + ["home_advantage", "dispersion"]
        )

    def _get_tail_param_indices(self) -> dict[str, int]:
        return {"home_advantage": -2, "dispersion": -1}

    def _gradient(self, params):
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]  # Home field advantage
        dispersion = params[-1]  # Dispersion parameter

        # Note: negative_binomial_gradient computes the gradient of the negative log-likelihood,
        # which is correct for minimization (unlike Poisson/Dixon-Coles which compute LL gradient)
        grad = negative_binomial_gradient(
            attack,
            defence,
            hfa,
            dispersion,
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
        )

        # Remove excessive clipping that can interfere with optimization
        return grad

    def _loss_function(self, params: np.ndarray) -> float:
        """
        Calculates the negative log-likelihood of the Negative Binomial model.

        Parameters
        ----------
        params : array_like
            The parameters of the model
        data : dict
            The data used to fit the model
        n_teams : int
            The number of teams in the league

        Returns
        -------
        float
            The negative log-likelihood of the Negative Binomial model
        """
        attack = np.asarray(params[: self.n_teams], dtype=np.double, order="C")
        defence = np.asarray(
            params[self.n_teams : 2 * self.n_teams], dtype=np.double, order="C"
        )
        hfa = params[-2]
        dispersion = params[-1]

        return compute_negative_binomial_loss(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            attack,
            defence,
            hfa,
            dispersion,
        )

    def fit(
        self,
        minimizer_options: Optional[dict] = None,
        use_gradient: bool = True,
    ):
        """
        Fits the Negative Binomial model to the data.

        Parameters
        ----------
        minimizer_options : dict, optional
            Dictionary of options to pass to scipy.optimize.minimize (e.g., maxiter, ftol, disp). Default is None.

        use_gradient : bool, optional
            Whether to use the analytical gradient during optimization. Default is True.
            Setting to False will use numerical gradients instead, which may be slower but sometimes more stable.
        """
        bounds = [(-2, 2)] * self.n_teams * 2 + [(-4, 4), (1e-5, 1000)]
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]

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
        dispersion = self._params[-1]

        # Preallocate the score matrix as a flattened array.
        score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)

        # Allocate one-element arrays for lambda values.
        lambda_home = np.empty(1, dtype=np.float64)
        lambda_away = np.empty(1, dtype=np.float64)

        compute_negative_binomial_probabilities(
            float(home_attack),
            float(away_attack),
            float(home_defense),
            float(away_defense),
            float(home_advantage),
            float(dispersion),
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
