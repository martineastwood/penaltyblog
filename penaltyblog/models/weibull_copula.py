import math
import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize

from .custom_types import GoalInput, ParamsOutput, TeamInput, WeightInput
from .football_probability_grid import FootballProbabilityGrid


class WeibullCopulaGoalsModel:
    """
    Weibull Copula model for predicting outcomes of football (soccer) matches

    Methods
    -------
    fit()
        fits a Weibull Copula model to the data to calculate the team strengths.
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
        Initialises the WeibullCopulaGoalModel class.

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
        self.goals_home = np.asarray(goals_home, dtype=int)
        self.goals_away = np.asarray(goals_away, dtype=int)
        self.teams_home = np.asarray(teams_home, dtype=object)
        self.teams_away = np.asarray(teams_away, dtype=object)
        if weights is None:
            weights = np.ones_like(self.goals_home, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

        self.teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))
        self.n_teams = len(self.teams)

        # Quick guess initialization
        rng = np.random.default_rng()
        atk_init = rng.normal(1.0, 0.1, self.n_teams)
        def_init = rng.normal(-1.0, 0.1, self.n_teams)
        home_init = np.array([0.5 + rng.normal(0, 0.1)])
        shape_init = np.array([1.2])
        kappa_init = np.array([1.5])

        self._params = np.concatenate(
            [atk_init, def_init, home_init, shape_init, kappa_init]
        )

        # Pre-map team -> index for faster loops
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        self.home_idx = np.vectorize(self.team_to_idx.get)(self.teams_home)
        self.away_idx = np.vectorize(self.team_to_idx.get)(self.teams_away)

        # Bookkeeping
        self._res = None
        self.loglikelihood = None
        self.aic = None
        self.n_params = len(self._params)
        self.fitted = False

        # For reusing the alpha table if shape doesn't change too much
        self._alpha_cache_shape = None
        self._alpha_cache_A = None

        # global or user-chosen
        self.max_goals = 15
        self.jmax = 25

    def __repr__(self) -> str:
        lines = [
            "Module: Penaltyblog",
            "",
            "Model: Bivariate Weibull Count + Copula",
            "",
        ]

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
                f"Home Advantage: {round(self._params[-3], 3)}",
                f"Weibull Shape: {round(self._params[-2], 3)}",
                f"Kappa: {round(self._params[-1], 3)}",
            ]
        )

        return "\n".join(lines)

    def _get_alpha_table_for_shape(self, shape) -> NDArray:
        """
        Return the precomputed alpha table for given shape,
        caching if repeated calls with same shape.
        """
        # You might do exact matches or something more sophisticated if shape changes.
        # We'll do an exact match for clarity:
        if self._alpha_cache_shape == shape:
            return self._alpha_cache_A
        # else recompute
        A = precompute_alpha_table(shape, max_goals=self.max_goals, jmax=self.jmax)
        self._alpha_cache_shape = shape
        self._alpha_cache_A = A
        return A

    def _neg_log_likelihood(self, params: NDArray) -> float:
        # unpack
        atk = params[: self.n_teams]
        dfc = params[self.n_teams : 2 * self.n_teams]
        home_adv = params[-3]
        shape = params[-2]
        kappa = params[-1]

        # If shape <= 0 => invalid, penalize
        if shape <= 0.0:
            return 1e15

        # Precompute alpha array for this shape
        A = self._get_alpha_table_for_shape(shape)
        if A is None:
            return 1e15  # shape was invalid

        return neg_log_likelihood_numba(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            atk,
            dfc,
            home_adv,
            shape,
            kappa,
            A,
            self.max_goals,
        )

    def fit(self):
        """
        Fits the Weibull Copula model to the data.
        """
        # create bounds
        bnds = []
        # Attack in [-3,3]
        for _ in range(self.n_teams):
            bnds.append((-3, 3))
        # Defense in [-3,3]
        for _ in range(self.n_teams):
            bnds.append((-3, 3))
        # home advantage in [-2,2]
        bnds.append((-2, 2))
        # shape in (0.01, 2.5)
        bnds.append((1e-2, 2.5))
        # kappa in [-5,10]
        bnds.append((-5, 5))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            res = minimize(
                self._neg_log_likelihood,
                x0=self._params,
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 250, "ftol": 1e-7, "disp": False},
            )

        self._res = res
        self._params = res.x
        self.loglikelihood = -res.fun
        self.n_params = len(res.x)
        self.aic = -2.0 * self.loglikelihood + 2.0 * self.n_params
        self.fitted = True
        return self

    def predict(
        self, home_team: str, away_team: str, max_goals: int = 15
    ) -> FootballProbabilityGrid:
        """
        Predicts the probability of each market for a given home and away team.

        Parameters
        ----------
        home_team : str
            The name of the home team
        away_team : str
            The name of the away team
        max_goals : int, optional
            The maximum number of goals to consider, by default 15

        Returns
        -------
        FootballProbabilityGrid
            A FootballProbabilityGrid object containing the probability of each scoreline
        """
        if not self.fitted:
            raise ValueError("Call fit() first.")

        atk = self._params[: self.n_teams]
        dfc = self._params[self.n_teams : 2 * self.n_teams]
        home_adv = self._params[-3]
        shape = self._params[-2]
        kappa = self._params[-1]

        # find indexes
        idx_home = np.where(self.teams == home_team)[0]
        idx_away = np.where(self.teams == away_team)[0]
        if len(idx_home) == 0 or len(idx_away) == 0:
            raise ValueError("Team not in training data.")

        iH, iA = idx_home[0], idx_away[0]
        lamH = np.exp(home_adv + atk[iH] + dfc[iA])
        lamA = np.exp(atk[iA] + dfc[iH])

        # precompute alpha table for shape
        A = self._get_alpha_table_for_shape(shape)

        # get the score matrix
        score_matrix = predict_score_matrix_numba(lamH, lamA, A, max_goals, kappa)

        return FootballProbabilityGrid(score_matrix, lamH, lamA)

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
            + ["home_adv", "shape", "kappa"]
        )
        vals = list(self._params)
        result = dict(zip(param_names, vals))

        return result


@njit
def gamma_numba(x):
    """
    Work around since numba did not recognise gamma function
    """
    return math.exp(math.lgamma(x))


@njit()
def frank_copula_cdf(u, v, kappa):
    if abs(kappa) < 1e-8:
        return u * v
    num = (np.exp(-kappa * u) - 1.0) * (np.exp(-kappa * v) - 1.0)
    denom = np.exp(-kappa) - 1.0
    inside = 1.0 + num / denom
    if inside <= 1e-14:
        return max(0.0, u * v)
    return -(1.0 / kappa) * np.log(inside)


@njit()
def precompute_alpha_table(c, max_goals=15, jmax=25):
    if c <= 0:
        return None

    A = np.zeros((max_goals + 1, jmax + 1), dtype=float)

    alpha_raw = np.zeros((max_goals + 1, jmax + 1), dtype=float)

    for j in range(jmax + 1):
        alpha_raw[0, j] = gamma_numba(c * j + 1.0) / gamma_numba(j + 1.0)

    for x in range(max_goals):
        for j in range(x + 1, jmax + 1):
            tmp_sum = 0.0
            for m in range(x, j):
                tmp_sum += (
                    alpha_raw[x, m]
                    * gamma_numba(c * j - c * m + 1.0)
                    / gamma_numba((j - m) + 1.0)
                )
            alpha_raw[x + 1, j] = tmp_sum

    for x in range(max_goals + 1):
        for j in range(jmax + 1):
            sign = (-1) ** (x + j)
            denom = gamma_numba(c * j + 1.0)
            A[x, j] = sign * (alpha_raw[x, j] / denom)

    return A


@njit()
def weibull_count_pmf(lam, A):
    if lam <= 0:
        # degenerate => all mass at 0
        pmf = np.zeros(A.shape[0], dtype=float)
        pmf[0] = 1.0
        return pmf

    max_goals = A.shape[0] - 1
    jmax = A.shape[1] - 1
    pmf = np.zeros(max_goals + 1, dtype=float)

    # Precompute lam^j up to jmax
    lam_powers = np.ones(jmax + 1, dtype=float)
    for j in range(1, jmax + 1):
        lam_powers[j] = lam_powers[j - 1] * lam

    for x in range(max_goals + 1):
        val = 0.0
        for j in range(x, jmax + 1):
            val += lam_powers[j] * A[x, j]
        pmf[x] = val if val > 0 else 0.0

    s = pmf.sum()
    if s < 1e-14:
        pmf = np.zeros_like(pmf)
        pmf[0] = 1.0
        return pmf
    return pmf / s


@njit()
def cdf_from_pmf(p):
    return np.cumsum(p)


@njit()
def compute_p_xy(x_i, y_i, cdfX, cdfY, max_goals, kappa):
    """
    Example: a stand-alone function that directly uses boundary checks
    without calling separate Python functions for FX() or FY().
    """

    def FX(k):
        if k < 0:
            return 0.0
        elif k > max_goals:
            return 1.0
        else:
            return cdfX[k]

    def FY(k):
        if k < 0:
            return 0.0
        elif k > max_goals:
            return 1.0
        else:
            return cdfY[k]

    p_xy = (
        frank_copula_cdf(FX(x_i), FY(y_i), kappa)
        - frank_copula_cdf(FX(x_i - 1), FY(y_i), kappa)
        - frank_copula_cdf(FX(x_i), FY(y_i - 1), kappa)
        + frank_copula_cdf(FX(x_i - 1), FY(y_i - 1), kappa)
    )
    return p_xy


@njit
def neg_log_likelihood_numba(
    goals_home,
    goals_away,
    weights,
    home_idx,
    away_idx,
    atk,
    dfc,
    home_adv,
    shape,
    kappa,
    A,
    max_goals,
):
    """
    The entire NLL loop in one jitted function.
    A is the precomputed alpha table for 'shape'.
    """
    n = goals_home.size
    total_ll = 0.0

    for i in range(n):
        x_i = goals_home[i]
        y_i = goals_away[i]
        w_i = weights[i]

        lam_home_i = math.exp(home_adv + atk[home_idx[i]] + dfc[away_idx[i]])
        lam_away_i = math.exp(atk[away_idx[i]] + dfc[home_idx[i]])

        pmfX = weibull_count_pmf(lam_home_i, A)
        pmfY = weibull_count_pmf(lam_away_i, A)

        cdfX = cdf_from_pmf(pmfX)
        cdfY = cdf_from_pmf(pmfY)

        p_xy = compute_p_xy(x_i, y_i, cdfX, cdfY, max_goals, kappa)
        if p_xy <= 0.0:
            total_ll += w_i * (-999999.0)
        else:
            total_ll += w_i * math.log(p_xy)

    return -total_ll


@njit
def predict_score_matrix_numba(lam_home, lam_away, A, max_goals, kappa):
    """
    Build the (max_goals+1 x max_goals+1) probability matrix for
    P(X = i, Y = j), given lam_home, lam_away, precomputed alpha-table A,
    and the Frank-copula parameter kappa.

    Returns a 2D NumPy array of shape (max_goals+1, max_goals+1).
    """
    # 1) Compute the marginal PMFs & CDFs
    pmfH = weibull_count_pmf(lam_home, A)
    pmfA = weibull_count_pmf(lam_away, A)
    cdfH = cdf_from_pmf(pmfH)
    cdfA = cdf_from_pmf(pmfA)

    # 2) Build the score_matrix
    score_matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)

    for i in range(max_goals + 1):
        # boundary-check for i and i-1 in cdfH
        if i < 0:
            Fi = 0.0
        elif i > max_goals:
            Fi = 1.0
        else:
            Fi = cdfH[i]

        i_m1 = i - 1
        if i_m1 < 0:
            Fi_m1 = 0.0
        elif i_m1 > max_goals:
            Fi_m1 = 1.0
        else:
            Fi_m1 = cdfH[i_m1]

        for j in range(max_goals + 1):
            # boundary-check for j and j-1 in cdfA
            if j < 0:
                Fj = 0.0
            elif j > max_goals:
                Fj = 1.0
            else:
                Fj = cdfA[j]

            j_m1 = j - 1
            if j_m1 < 0:
                Fj_m1 = 0.0
            elif j_m1 > max_goals:
                Fj_m1 = 1.0
            else:
                Fj_m1 = cdfA[j_m1]

            # 3) Frank copula difference
            p_ij = (
                frank_copula_cdf(Fi, Fj, kappa)
                - frank_copula_cdf(Fi_m1, Fj, kappa)
                - frank_copula_cdf(Fi, Fj_m1, kappa)
                + frank_copula_cdf(Fi_m1, Fj_m1, kappa)
            )

            if p_ij < 0.0:
                p_ij = 0.0
            score_matrix[i, j] = p_ij

    return score_matrix
