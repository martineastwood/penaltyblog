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
        self.aic = None

    def __repr__(self):
        repr_str = "Bivariate Poisson Goal Model\n"
        repr_str += "Fitted: {}\n".format(self.fitted)
        if self.fitted:
            repr_str += "AIC: {:.3f}\n".format(self.aic)
            repr_str += "Parameters:\n"
            for i, team in enumerate(self.teams):
                repr_str += f"{team}: Attack {self._params[i]:.3f}, Defence {self._params[i + self.n_teams]:.3f}\n"
            repr_str += f"Home Advantage: {self._params[-2]:.3f}\n"
            repr_str += f"Lambda (Correlation): {self._params[-1]:.3f}\n"
        else:
            repr_str += "Model has not been fitted yet.\n"
        return repr_str

    def _log_likelihood(self, params, data, n_teams):
        attack_params = params[:n_teams]
        defence_params = params[n_teams : n_teams * 2]
        home_adv, lambda_c = params[-2:]

        home_idx = data["home_idx"]
        away_idx = data["away_idx"]
        goals_home = data["goals_home"]
        goals_away = data["goals_away"]

        lambda_home = np.exp(
            np.clip(
                home_adv + attack_params[home_idx] + defence_params[away_idx], -10, 10
            )
        )
        lambda_away = np.exp(
            np.clip(attack_params[away_idx] + defence_params[home_idx], -10, 10)
        )
        lambda_c = np.clip(lambda_c, 1e-5, None)

        # Vectorized computation
        max_goals = max(goals_home.max(), goals_away.max()) + 1
        k_range = np.arange(max_goals)

        home_pmf = poisson.pmf(goals_home[:, None] - k_range, lambda_home[:, None])
        away_pmf = poisson.pmf(goals_away[:, None] - k_range, lambda_away[:, None])
        k_pmf = poisson.pmf(k_range, lambda_c)

        likelihood_matrix = home_pmf * away_pmf * k_pmf
        likelihoods = np.sum(likelihood_matrix, axis=1)

        # Avoid zero values
        likelihoods = np.clip(likelihoods, 1e-10, None)

        if np.any(np.isnan(likelihoods)) or np.any(np.isinf(likelihoods)):
            return np.inf

        return -np.sum(np.log(likelihoods) * data["weights"])

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
        bounds = [(-2, 2)] * self.n_teams * 2 + [(0, 1), (0, 1)]  # Tighter bounds

        result = minimize(
            self._log_likelihood,
            self._params,
            args=(processed_fixtures, self.n_teams),
            bounds=bounds,
            method="L-BFGS-B",  # Better suited for bound-constrained optimization
            options=options,
        )

        if not result.success:
            print("Optimization did not converge:", result.message)

        self._params = result.x
        self.fitted = True
        self.aic = 2 * len(self._params) + 2 * result.fun
