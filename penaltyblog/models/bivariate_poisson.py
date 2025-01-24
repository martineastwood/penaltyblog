import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class BivariatePoissonGoalModel:
    """
    Bivariate Poisson model for predicting soccer match outcomes
    using the approach by Karlis and Ntzoufras.
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
        )  # Home advantage and lambda (correlation)

        self._res = None
        self.fitted = False

    def __repr__(self):
        repr_str = "Bivariate Poisson Goal Model\n"
        repr_str += "Fitted: {}\n".format(self.fitted)
        if self.fitted:
            repr_str += "Parameters:\n"
            for i, team in enumerate(self.teams):
                repr_str += f"{team}: Attack {self._params[i]:.3f}, Defence {self._params[i + self.n_teams]:.3f}\n"
            repr_str += f"Home Advantage: {self._params[-2]:.3f}\n"
            repr_str += f"Lambda (Correlation): {self._params[-1]:.3f}\n"
        else:
            repr_str += "Model has not been fitted yet.\n"
        return repr_str

    def _log_likelihood(self, params, fixtures, n_teams):
        attack_params = params[:n_teams]
        defence_params = params[n_teams : n_teams * 2]
        home_adv, lambda_c = params[-2:]

        home_idx = fixtures["home_idx"]
        away_idx = fixtures["away_idx"]
        goals_home = fixtures["goals_home"]
        goals_away = fixtures["goals_away"]

        lambda_home = np.exp(
            home_adv + attack_params[home_idx] + defence_params[away_idx]
        )
        lambda_away = np.exp(attack_params[away_idx] + defence_params[home_idx])

        # Bivariate Poisson log-likelihood
        joint_ll = (
            np.log(
                sum(
                    poisson.pmf(goals_home - k, lambda_home)
                    * poisson.pmf(goals_away - k, lambda_away)
                    * poisson.pmf(k, lambda_c)
                    for k in range(0, min(goals_home, goals_away) + 1)
                )
            )
            * fixtures["weights"]
        )

        return -np.sum(joint_ll)

    def fit(self):
        team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.fixtures["home_idx"] = self.fixtures["team_home"].map(team_to_idx)
        self.fixtures["away_idx"] = self.fixtures["team_away"].map(team_to_idx)

        options = {"maxiter": 100, "disp": False}
        bounds = [(-3, 3)] * self.n_teams * 2 + [(0, 2), (0, 2)]

        result = minimize(
            self._log_likelihood,
            self._params,
            args=(self.fixtures, self.n_teams),
            bounds=bounds,
            options=options,
        )

        self._params = result.x
        self.fitted = True

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
        lambda_c = self._params[-1]

        lambda_home = np.exp(home_adv + home_attack + away_defence)
        lambda_away = np.exp(away_attack + home_defence)

        score_matrix = np.zeros((max_goals, max_goals))

        for i in range(max_goals):
            for j in range(max_goals):
                score_matrix[i, j] = sum(
                    poisson.pmf(i - k, lambda_home)
                    * poisson.pmf(j - k, lambda_away)
                    * poisson.pmf(k, lambda_c)
                    for k in range(0, min(i, j) + 1)
                )

        return score_matrix

    def get_params(self):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        params = dict(
            zip(
                ["attack_" + team for team in self.teams]
                + ["defence_" + team for team in self.teams]
                + ["home_advantage", "lambda_c"],
                self._params,
            )
        )
        return params
