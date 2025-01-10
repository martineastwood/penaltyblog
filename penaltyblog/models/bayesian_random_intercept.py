import os
from typing import Dict

import numpy as np
from scipy.stats import poisson

from .base_bayesian_model import BaseBayesianGoalModel
from .football_probability_grid import FootballProbabilityGrid


class BayesianRandomInterceptGoalModel(BaseBayesianGoalModel):
    """Bayesian Hierarchical model for predicting football match outcomes"""

    STAN_FILE = os.path.join(
        os.path.dirname(__file__),
        "stan_files",
        "bayesian_random_intercept.stan",
    )

    def _get_model_parameters(self):
        draws = self.fit_result.draws_pd()
        att_params = [x for x in draws.columns if "atts_star" in x]
        defs_params = [x for x in draws.columns if "def_star" in x]
        int_params = [x for x in draws.columns if "random_int" in x]
        return draws, att_params, defs_params, int_params

    def _format_team_parameters(self, draws, att_params, defs_params, int_params):
        attack = [None] * self.n_teams
        defence = [None] * self.n_teams
        random_int = [None] * self.n_teams
        team = self.teams["team"].tolist()

        atts = draws[att_params].mean()
        defs = draws[defs_params].mean()
        ints = draws[int_params].mean()

        for idx, _ in enumerate(team):
            attack[idx] = round(atts.iloc[idx], 3)
            defence[idx] = round(defs.iloc[idx], 3)
            random_int[idx] = round(ints.iloc[idx], 3)

        return team, attack, defence, random_int

    def get_params(self) -> Dict:
        """
        Returns the fitted parameters of the Bayesian Bivariate Goal Model.

        Returns:
            dict: A dictionary containing the fitted parameters of the model.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before getting parameters")

        draws, att_params, defs_params, int_params = self._get_model_parameters()
        team, attack, defence, ints = self._format_team_parameters(
            draws, att_params, defs_params, int_params
        )

        params = {
            "teams": team,
            "attack": dict(zip(team, attack)),
            "defence": dict(zip(team, defence)),
            "home_advantage": round(draws["home"].mean(), 3),
            "random_intercept": dict(zip(team, ints)),
        }

        return params

    def __repr__(self):
        repr_str = "Module: Penaltyblog\n\nModel: Bayesian Hierarchical Random Intercept (Stan)\n\n"

        if not self.fitted:
            return repr_str + "Status: Model not fitted"

        draws, att_params, defs_params, int_params = self._get_model_parameters()
        team, attack, defence, ints = self._format_team_parameters(
            draws, att_params, defs_params, int_params
        )

        repr_str += f"Number of parameters: {len(att_params) + len(defs_params) + 2}\n"
        repr_str += "{0: <20} {1:<20} {2:<20} {3:<20}".format(
            "Team", "Attack", "Defence", "Intercept"
        )
        repr_str += "\n" + "-" * 80 + "\n"

        for t, a, d, r in zip(team, attack, defence, ints):
            repr_str += "{0: <20} {1:<20} {2:<20} {3:<20}\n".format(t, a, d, r)

        repr_str += "-" * 60 + "\n"
        repr_str += f"Home Advantage: {round(draws['home'].mean(), 3)}\n"
        repr_str += f"Intercept: {round(draws['global_intercept'].mean(), 3)}\n"

        return repr_str

    def fit(self, draws: int = 5000, warmup: int = 2000):
        """
        Fits the Bayesian Bivariate Goal Model to the provided match data.

        Args:
            draws (int, optional): Number of posterior draws to generate, defaults to 5000.
            warmup (int, optional): Number of warmup draws, defaults to 2000.
        """
        data = {
            "N": len(self.fixtures),
            "n_teams": len(self.teams),
            "goals_home": self.fixtures["goals_home"].values,
            "goals_away": self.fixtures["goals_away"].values,
            "home_team": self.fixtures["home_index"].values,
            "away_team": self.fixtures["away_index"].values,
            "weights": self.fixtures["weights"].values,
        }

        self._compile_and_fit_stan_model(self.STAN_FILE, data, draws, warmup)

        return self

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 15, n_samples: int = 1000
    ) -> FootballProbabilityGrid:
        """
        Predicts the probability of goals scored by a home team and an away team.

        Args:
            home_team (str): The name of the home team.
            away_team (str): The name of the away team.
            max_goals (int, optional): The maximum number of goals to consider, defaults to 15.
            n_samples (int, optional): The number of samples to use for prediction, defaults to 1000.

        Returns:
                FootballProbabilityGrid: A FootballProbabilityGrid object containing
                the predicted probabilities.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")

        draws = self.fit_result.draws_pd()
        home_idx = self._get_team_index(home_team)
        away_idx = self._get_team_index(away_team)
        samples = draws.sample(n=n_samples, replace=True)

        lambda_home = np.exp(
            samples["global_intercept"]
            + samples[f"random_int[{home_idx}]"]
            + samples[f"atts[{home_idx}]"]
            + samples[f"defs[{away_idx}]"]
            + samples["home"]
        )

        lambda_away = np.exp(
            samples["global_intercept"]
            + samples[f"random_int[{away_idx}]"]
            + samples[f"atts[{away_idx}]"]
            + samples[f"defs[{home_idx}]"]
        )

        home_probs = poisson.pmf(np.arange(max_goals + 1)[:, None], lambda_home.values)
        away_probs = poisson.pmf(
            np.arange(max_goals + 1)[None, :], lambda_away.values[:, None]
        )

        score_probs = np.tensordot(home_probs, away_probs, axes=(1, 0)) / n_samples

        home_expectancy = np.sum(score_probs.sum(axis=1) * np.arange(max_goals + 1))
        away_expectancy = np.sum(score_probs.sum(axis=0) * np.arange(max_goals + 1))

        return FootballProbabilityGrid(score_probs, home_expectancy, away_expectancy)
