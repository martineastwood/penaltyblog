import warnings
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .base_model import BaseGoalsModel
from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid
from .numba_helpers import frank_copula_pdf


class PoissonCopulaGoalsModel(BaseGoalsModel):
    """
    Poisson Copula model for predicting outcomes of football (soccer) matches
    Methods
    -------
    fit()
        fits a Poisson Copula model to the data to calculate the team strengths.
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
        Poisson Copula model for predicting outcomes of football (soccer) matches

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
            The weight of each match (default is 1)
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        self._params = np.concatenate(
            (
                [1] * self.n_teams,  # Attack params
                [-1] * self.n_teams,  # Defense params
                [0.5],  # Home advantage
                [0.5],  # Frank copula parameter (kappa)
            )
        )

    def __repr__(self) -> str:
        lines = ["Module: Penaltyblog", "", "Model: Poisson + Copula", ""]

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
                f"Kappa: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _loss_function(self, params: np.ndarray) -> float:
        """
        Negative log-likelihood function for the Poisson Copula model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        float
            The negative log-likelihood of the model.
        """
        attack_params = params[: self.n_teams]
        defense_params = params[self.n_teams : 2 * self.n_teams]
        home_advantage = params[-2]
        copula_kappa = params[-1]

        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        home_idx = np.vectorize(team_to_idx.get)(self.teams_home)
        away_idx = np.vectorize(team_to_idx.get)(self.teams_away)

        lambda_home = np.exp(
            home_advantage + attack_params[home_idx] + defense_params[away_idx]
        )
        lambda_away = np.exp(attack_params[away_idx] + defense_params[home_idx])

        home_probs = poisson.pmf(self.goals_home, lambda_home)
        away_probs = poisson.pmf(self.goals_away, lambda_away)

        # Transform to uniform scale for copula (pseudo-observations)
        home_cdf = poisson.cdf(self.goals_home, lambda_home)
        away_cdf = poisson.cdf(self.goals_away, lambda_away)

        copula_probs = frank_copula_pdf(home_cdf, away_cdf, copula_kappa)

        # Compute joint log-likelihood
        log_likelihood = np.log(home_probs * away_probs * copula_probs) * self.weights
        return -np.sum(log_likelihood)

    def fit(self):
        """
        Fits the Poisson Copula model to the data.
        """
        options = {"maxiter": 500, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 2)] + [(0.0001, 5)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._loss_function,
                self._params,
                constraints=constraints,
                bounds=bounds,
                options=options,
                method="L-BFGS-B",
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
        Predicts the probability of each scoreline for a given home and away team.

        Parameters
        ----------
        home_team : str
            The name of the home team
        away_team : str
            The name of the away team
        max_goals : int, optional
            The maximum number of goals to consider (default is 15)

        Returns
        -------
        dict
            A dictionary containing the probability of each scoreline
        """
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Please call `fit()` first."
            )

        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Both teams must have been in the training data.")

        home_idx = np.where(self.teams == home_team)[0][0]
        away_idx = np.where(self.teams == away_team)[0][0]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-1]

        lambda_home = np.exp(home_advantage + home_attack + away_defense)
        lambda_away = np.exp(away_attack + home_defense)

        home_goals_vector = poisson(lambda_home).pmf(np.arange(max_goals))
        away_goals_vector = poisson(lambda_away).pmf(np.arange(max_goals))

        score_matrix = np.outer(home_goals_vector, away_goals_vector)

        # Return FootballProbabilityGrid
        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

    def get_params(self) -> ParamsOutput:
        """
        Returns the fitted parameters of the model.
        """
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet. Call `fit()` first."
            )

        assert self.n_params is not None

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defense_" + team for team in self.teams]
                + ["home_advantage", "copula_kappa"],
                self._params,
            )
        )
        return params
