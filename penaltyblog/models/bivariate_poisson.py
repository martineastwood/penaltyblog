import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import poisson

from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid


class BivariatePoissonGoalModel:
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
        weights: WeightInput = 1,
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
            The weights of the matches, by default 1
        """
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

    def __repr__(self) -> str:
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

    def _log_likelihood(self, params, data) -> float:
        """
        Computes the negative log-likelihood of the Bivariate Poisson model,
        using:
        (1) Precomputation of Poisson PMFs for lambda3 (avoiding repeats),
        (2) Vectorization for the inner sum over k to reduce Python loops.
        """
        n_teams = self.n_teams

        # Parameter unpacking remains the same
        attack_params = params[:n_teams]
        defense_params = params[n_teams : 2 * n_teams]
        home_adv = params[-2]
        correlation_log = params[-1]
        lambda3 = np.exp(correlation_log)

        # Compute lambdas
        lambda1 = np.exp(
            home_adv
            + attack_params[data["home_idx"]]
            + defense_params[data["away_idx"]]
        )
        lambda2 = np.exp(
            attack_params[data["away_idx"]] + defense_params[data["home_idx"]]
        )

        return _compute_total_likelihood(
            data["goals_home"],
            data["goals_away"],
            lambda1,
            lambda2,
            lambda3,
            data["weights"],
            max(data["goals_home"].max(), data["goals_away"].max()) + 1,
        )

    def fit(self):
        """
        Fits the Bivariate Poisson model to the data.
        """
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

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 10
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

        score_matrix = _compute_score_matrix(lam1, lam2, lam3, max_goals)

        return FootballProbabilityGrid(score_matrix, lam1, lam2)

    def get_params(self) -> ParamsOutput:
        """
        Return the fitted parameters in a dictionary.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        # Construct dictionary
        param_names = (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "correlation_log"]
        )
        vals = list(self._params)
        result = dict(zip(param_names, vals))

        # Also show lambda3 explicitly
        result["lambda3"] = np.exp(result["correlation_log"])
        return result


@njit
def _compute_match_likelihood(
    goals_home: int,
    goals_away: int,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    weight: float,
    pmf_lookup1: np.ndarray,
    pmf_lookup2: np.ndarray,
    lambda3_pmf: np.ndarray,
) -> float:
    kmax = min(goals_home, goals_away)
    like_ij = 0.0

    for k in range(kmax + 1):
        like_ij += (
            pmf_lookup1[goals_home - k] * pmf_lookup2[goals_away - k] * lambda3_pmf[k]
        )

    like_ij = max(like_ij, 1e-10)
    return weight * np.log(like_ij)


@njit
def numba_poisson_pmf(k: int, lambda_: float, max_goals: int) -> float:
    if k < 0 or k >= max_goals or lambda_ <= 0:
        return 0.0
    return np.exp(k * np.log(lambda_) - lambda_ - np.sum(np.log(np.arange(1, k + 1))))


@njit
def _compute_total_likelihood(
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    lambda1: np.ndarray,
    lambda2: np.ndarray,
    lambda3: float,
    weights: np.ndarray,
    max_goals: int,
) -> float:
    n_matches = len(goals_home)
    log_likelihoods = np.zeros(n_matches)

    # Precompute PMF lookups
    lambda3_pmf = np.zeros(max_goals)
    for k in range(max_goals):
        lambda3_pmf[k] = numba_poisson_pmf(k, lambda3, max_goals)

    for i in range(n_matches):
        pmf1 = np.zeros(max_goals)
        pmf2 = np.zeros(max_goals)
        for k in range(max_goals):
            pmf1[k] = numba_poisson_pmf(k, lambda1[i], max_goals)
            pmf2[k] = numba_poisson_pmf(k, lambda2[i], max_goals)

        like_ij = 0.0
        kmax = min(goals_home[i], goals_away[i])

        for k in range(kmax + 1):
            like_ij += (
                pmf1[goals_home[i] - k] * pmf2[goals_away[i] - k] * lambda3_pmf[k]
            )

        like_ij = max(like_ij, 1e-10)
        log_likelihoods[i] = weights[i] * np.log(like_ij)

    return -np.sum(log_likelihoods)


@njit
def _compute_score_matrix(
    lam1: float, lam2: float, lam3: float, max_goals: int
) -> np.ndarray:
    score_matrix = np.zeros((max_goals, max_goals))

    # Precompute PMF values for each lambda
    pmf1 = np.array([numba_poisson_pmf(k, lam1, max_goals) for k in range(max_goals)])
    pmf2 = np.array([numba_poisson_pmf(k, lam2, max_goals) for k in range(max_goals)])
    pmf3 = np.array([numba_poisson_pmf(k, lam3, max_goals) for k in range(max_goals)])

    for x in range(max_goals):
        for y in range(max_goals):
            p_xy = 0.0
            for k in range(min(x, y) + 1):
                p_xy += pmf1[x - k] * pmf2[y - k] * pmf3[k]
            score_matrix[x, y] = p_xy

    return score_matrix
