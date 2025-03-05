import ctypes
import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize

from penaltyblog.golib.loss import bivariate_poisson_loss_function
from penaltyblog.golib.probabilities import (
    compute_bivariate_poisson_probabilities,
)
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


class BivariatePoissonGoalModelGo(BaseGoalsModel):
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
        params = np.ascontiguousarray(params, dtype=np.float64)
        params_ctypes = params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        return bivariate_poisson_loss_function(
            params_ctypes,
            self.n_teams,
            self.home_idx_ctypes,
            self.away_idx_ctypes,
            self.goals_home_ctypes,
            self.goals_away_ctypes,
            self.weights_ctypes,
            len(self.goals_home),
        )

    def fit(self):
        """
        Fits the Bivariate Poisson model to the data.
        """
        options = {"maxiter": 1000, "disp": False}
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}
        ]
        bnds = [(-3, 3)] * (2 * self.n_teams) + [(-2, 2), (-3, 3)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                self._loss_function,
                self._params,
                constraints=constraints,
                bounds=bnds,
                options=options,
            )

        if not self._res.success:
            raise ValueError(f"Optimization failed with message: {self._res.message}")

        self._params = self._res["x"]
        self.n_params = len(self._params)
        self.loglikelihood = self._res["fun"] * -1
        self.aic = -2 * (self.loglikelihood) + 2 * self.n_params
        self.fitted = True

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

        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Both teams must have been in the training data.")

        home_idx = self.team_to_idx[home_team]
        away_idx = self.team_to_idx[away_team]

        home_attack = self._params[home_idx]
        away_attack = self._params[away_idx]
        home_defense = self._params[home_idx + self.n_teams]
        away_defense = self._params[away_idx + self.n_teams]
        home_advantage = self._params[-2]
        correlation_log = self._params[-1]

        score_matrix, lambda_home, lambda_away = (
            compute_bivariate_poisson_probabilities(
                home_attack,
                away_attack,
                home_defense,
                away_defense,
                home_advantage,
                correlation_log,
                max_goals,
            )
        )

        return FootballProbabilityGrid(score_matrix, lambda_home, lambda_away)

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

    return float(-np.sum(log_likelihoods))


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
