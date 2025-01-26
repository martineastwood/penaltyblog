import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom


class NegativeBinomialGoalModel:
    """
    Negative Binomial model for predicting soccer match outcomes
    handling overdispersion in goal data.
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
        self.fixtures = pd.DataFrame(
            {
                "goals_home": goals_home,
                "goals_away": goals_away,
                "team_home": teams_home,
                "team_away": teams_away,
                "weights": weights,
            }
        )
        self.teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))
        self.n_teams = len(self.teams)

        self._params = np.concatenate(
            ([1] * self.n_teams, [-1] * self.n_teams, [0.25], [0.1])
        )  # Home advantage and dispersion parameter

        self._res = None
        self.fitted = False
        self.aic = None

    def __repr__(self):
        repr_str = "Negative Binomial Goal Model\n"
        repr_str += "Fitted: {}\n".format(self.fitted)
        if self.fitted:
            repr_str += "AIC: {:.3f}\n".format(self.aic)
            repr_str += "Parameters:\n"
            for i, team in enumerate(self.teams):
                repr_str += f"{team}: Attack {self._params[i]:.3f}, Defence {self._params[i + self.n_teams]:.3f}\n"
            repr_str += f"Home Advantage: {self._params[-2]:.3f}\n"
            repr_str += f"Dispersion Parameter: {self._params[-1]:.3f}\n"
        else:
            repr_str += "Model has not been fitted yet.\n"
        return repr_str

    def _neg_binomial_log_likelihood(self, params, data, n_teams):
        attack_params = params[:n_teams]
        defence_params = params[n_teams : n_teams * 2]
        home_adv, dispersion = params[-2:]

        home_idx = data["home_idx"]
        away_idx = data["away_idx"]
        goals_home = data["goals_home"]
        goals_away = data["goals_away"]

        lambda_home = np.exp(
            home_adv + attack_params[home_idx] + defence_params[away_idx]
        )
        lambda_away = np.exp(attack_params[away_idx] + defence_params[home_idx])

        dispersion = np.clip(dispersion, 1e-5, None)

        home_ll = nbinom.logpmf(
            goals_home, dispersion, dispersion / (dispersion + lambda_home)
        )
        away_ll = nbinom.logpmf(
            goals_away, dispersion, dispersion / (dispersion + lambda_away)
        )

        log_likelihood = home_ll + away_ll

        if np.any(np.isnan(log_likelihood)) or np.any(np.isinf(log_likelihood)):
            return np.inf

        return -np.sum(log_likelihood * data["weights"])

    def fit(self):
        team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        processed_fixtures = {
            "home_idx": self.fixtures["team_home"].map(team_to_idx).values,
            "away_idx": self.fixtures["team_away"].map(team_to_idx).values,
            "goals_home": self.fixtures["goals_home"].values,
            "goals_away": self.fixtures["goals_away"].values,
            "weights": self.fixtures["weights"].values,
        }

        options = {"maxiter": 500, "disp": True}
        bounds = [(-2, 2)] * self.n_teams * 2 + [(0, 2)]  # Tighter bounds

        result = minimize(
            self._neg_binomial_log_likelihood,
            self._params,
            args=(processed_fixtures, self.n_teams),
            bounds=bounds,
            method="L-BFGS-B",
            options=options,
        )

        if not result.success:
            print("Optimization did not converge:", result.message)

        self._params = result.x
        self.fitted = True
        self.aic = 2 * len(self._params) + 2 * result.fun

    def predict(self, home_team, away_team, max_goals=10):
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

        score_matrix = home_pmf @ away_pmf

        return score_matrix

    def get_params(self):
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
