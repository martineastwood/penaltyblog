import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from .base_model import BaseGoalsModel
from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid


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
        attack_params = params[: self.n_teams]
        defence_params = params[self.n_teams : self.n_teams * 2]
        home_adv, dispersion = params[-2:]

        lambda_home = np.exp(
            home_adv + attack_params[self.home_idx] + defence_params[self.away_idx]
        )
        lambda_away = np.exp(
            attack_params[self.away_idx] + defence_params[self.home_idx]
        )

        dispersion = np.clip(dispersion, 1e-5, None)

        home_ll = nbinom.logpmf(
            self.goals_home, dispersion, dispersion / (dispersion + lambda_home)
        )
        away_ll = nbinom.logpmf(
            self.goals_away, dispersion, dispersion / (dispersion + lambda_away)
        )

        log_likelihood = home_ll + away_ll

        if np.any(np.isnan(log_likelihood)) or np.any(np.isinf(log_likelihood)):
            return np.inf

        return -np.sum(log_likelihood * self.weights)

    def fit(self):
        """
        Fits the Negative Binomial model to the data.
        """
        options = {"maxiter": 1000, "disp": False}
        bounds = [(-2, 2)] * self.n_teams * 2 + [(-4, 4), (1e-5, 1000)]
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._loss_function,
                self._params,
                bounds=bounds,
                constraints=constraints,
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
        self, home_team: str, away_team: str, max_goals=10
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
            The maximum number of goals to consider, by default 10

        Returns
        -------
        FootballProbabilityGrid
            A FootballProbabilityGrid object containing the probability of each scoreline
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        home_idx = np.where(self.teams == home_team)[0][0]
        away_idx = np.where(self.teams == away_team)[0][0]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defence = self._params[home_idx + self.n_teams]
        away_defence = self._params[away_idx + self.n_teams]
        home_adv = self._params[-2]
        dispersion = self._params[-1]

        lambda_home = np.exp(home_adv + home_attack + away_defence)
        lambda_away = np.exp(away_attack + home_defence)

        home_goals = np.arange(max_goals)
        away_goals = np.arange(max_goals)
        home_pmf = nbinom.pmf(
            home_goals[:, None], dispersion, dispersion / (dispersion + lambda_home)
        )
        away_pmf = nbinom.pmf(
            away_goals, dispersion, dispersion / (dispersion + lambda_away)
        ).T

        m = np.outer(home_pmf, away_pmf)

        probability_grid = FootballProbabilityGrid(m, lambda_home, lambda_away)

        return probability_grid

    def get_params(self) -> ParamsOutput:
        """
        Returns the parameters of the Negative Binomial model.
        """
        if not self.fitted:
            raise ValueError(
                "Model's parameters have not been fit yet, please call the `fit()` function first"
            )

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defence_" + team for team in self.teams]
                + ["home_advantage", "dispersion"],
                self._params,
            )
        )
        return params
