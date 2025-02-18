import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid


class BivariatePoissonGoalModel:
    """
    Karlis & Ntzoufras Bivariate Poisson for soccer, with:
      X = W1 + W3
      Y = W2 + W3
    where W1, W2, W3 ~ independent Poisson(lambda1, lambda2, lambda3).
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
            (
                [0.0] * self.n_teams,  # Attack
                [0.0] * self.n_teams,  # Defense
                [0.1],  # Home advantage
                [0.0],  # correlation_param => lambda3 = exp(0)=1
            )
        )

        self.fitted = False
        self.aic = None
        self._res = None
        self.n_params = None
        self.loglikelihood = None

    def __repr__(self):
        lines = ["Module: Penaltyblog", "", "Model: Bivariate Poisson", ""]

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
                f"Correlation: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _log_likelihood(self, params, data):
        """
        Computes the negative log-likelihood of the Bivariate Poisson model,
        using:
        (1) Precomputation of Poisson PMFs for lambda3 (avoiding repeats),
        (2) Vectorization for the inner sum over k to reduce Python loops.
        """
        n_teams = self.n_teams

        # Unpack parameters
        attack_params = params[:n_teams]
        defense_params = params[n_teams : 2 * n_teams]
        home_adv = params[-2]
        correlation_log = params[-1]
        lambda3 = np.exp(correlation_log)

        # Unpack data arrays
        home_idx = data["home_idx"]
        away_idx = data["away_idx"]
        goals_home = data["goals_home"]
        goals_away = data["goals_away"]
        weights = data["weights"]

        # Compute lambda1, lambda2 for every match
        lambda1 = np.exp(home_adv + attack_params[home_idx] + defense_params[away_idx])
        lambda2 = np.exp(attack_params[away_idx] + defense_params[home_idx])

        # Prepare a lookup for Poisson PMF for all unique λ1/λ2 values,
        # up to max_goals in the dataset:
        max_goals = max(goals_home.max(), goals_away.max()) + 1
        unique_lambdas = np.unique(np.concatenate([lambda1, lambda2]))
        pmf_lookup = {
            lam: np.array([poisson.pmf(k, lam) for k in range(max_goals)])
            for lam in unique_lambdas
        }

        # (Part 1) Precompute the Poisson PMF for lambda3 just once
        lambda3_pmf = np.array([poisson.pmf(k, lambda3) for k in range(max_goals)])

        # Accumulate the log-likelihood in a vector
        log_likelihoods = np.zeros_like(goals_home, dtype=float)

        for i in range(len(goals_home)):
            gh = goals_home[i]
            ga = goals_away[i]

            lam1 = lambda1[i]
            lam2 = lambda2[i]

            # (Part 2) Vectorized sum over k
            kmax = min(gh, ga)
            k_range = np.arange(kmax + 1)
            like_ij = np.sum(
                pmf_lookup[lam1][gh - k_range]
                * pmf_lookup[lam2][ga - k_range]
                * lambda3_pmf[k_range]
            )

            # Numeric guard against log(0)
            like_ij = max(like_ij, 1e-10)
            log_likelihoods[i] = weights[i] * np.log(like_ij)

        return -np.sum(log_likelihoods)

    def fit(self):
        team_to_idx = {team: i for i, team in enumerate(self.teams)}
        processed_fixtures = {
            "home_idx": self.fixtures["team_home"].map(team_to_idx).values,
            "away_idx": self.fixtures["team_away"].map(team_to_idx).values,
            "goals_home": self.fixtures["goals_home"].values,
            "goals_away": self.fixtures["goals_away"].values,
            "weights": self.fixtures["weights"].values,
        }

        bnds = [(-3, 3)] * (2 * self.n_teams) + [(-2, 2), (-3, 3)]

        opt = minimize(
            fun=self._log_likelihood,
            x0=self._params,
            args=(processed_fixtures,),
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 300, "disp": False},
        )

        if not opt.success:
            print("WARNING: Optimization did not fully converge:", opt.message)

        self._params = opt.x
        self.fitted = True
        self.n_params = len(self._params)

        self.aic = 2 * len(self._params) + 2 * opt.fun
        self._res = opt
        self.loglikelihood = -self._res.fun

    def predict(self, home_team, away_team, max_goals=10):
        """
        Construct a score probability matrix P(X=x, Y=y) for x,y in [0..max_goals-1].
        Using the same bivariate Poisson formula with lam1, lam2, lam3.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        # Extract parameters
        attack_params = self._params[: self.n_teams]
        defense_params = self._params[self.n_teams : 2 * self.n_teams]
        home_adv = self._params[-2]
        correlation_log = self._params[-1]
        lam3 = np.exp(correlation_log)

        # Get the correct indices
        try:
            i_home = np.where(self.teams == home_team)[0][0]
            i_away = np.where(self.teams == away_team)[0][0]
        except IndexError:
            raise ValueError(
                f"Team not found in training set: {home_team} or {away_team}"
            )

        lam1 = np.exp(home_adv + attack_params[i_home] + defense_params[i_away])
        lam2 = np.exp(attack_params[i_away] + defense_params[i_home])

        score_matrix = np.zeros((max_goals, max_goals))

        for x in range(max_goals):
            for y in range(max_goals):
                p_xy = 0.0
                for k in range(min(x, y) + 1):
                    p_xy += (
                        poisson.pmf(x - k, lam1)
                        * poisson.pmf(y - k, lam2)
                        * poisson.pmf(k, lam3)
                    )
                score_matrix[x, y] = p_xy

        return FootballProbabilityGrid(score_matrix, lam1, lam2)

    def get_params(self):
        """
        Return the fitted parameters in a dictionary.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        # Construct dictionary
        param_names = (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_adv", "correlation_log"]
        )
        vals = list(self._params)
        result = dict(zip(param_names, vals))

        # Also show lambda3 explicitly
        result["lambda3"] = np.exp(result["correlation_log"])
        return result
