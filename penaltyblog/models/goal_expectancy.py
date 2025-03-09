import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def goal_expectancy(
    home: float, draw: float, away: float, dc_adj: bool = True, rho: float = 0.001
) -> dict:
    """
    Estimates the bookmaker's goal expectencies for the home team and away team based on the
    home, draw, and away probabilities.

    Parameters
    -----------
    home : float
        Probability of home win
    draw : float
        Probability of draw
    away : float
        Probability of away win
    dc_adj : bool
        Whether to apply the Dixon and Coles adjustment
    rho : float
        The value for rho within the Dixon and Coles adjustment if dc_adj is True

    Returns
    ----------
    Dictionary containing home team's goal expectancy, away team's goal expectancy,
    the mean squared error between actual probabilities and estimated probabilities,
    and whether the minimizer returned as successful or not
    """
    # set up the basic options for the solver so we give up
    # after 1000 attempts and don't log to screen
    options = {
        "maxiter": 1000,
        "disp": False,
    }

    res = minimize(
        fun=_mse,
        x0=[0.5, -0.5],
        args=(home, draw, away, dc_adj, rho),
        options=options,
    )

    output = {
        "home_exp": np.exp(res["x"][0]),
        "away_exp": np.exp(res["x"][1]),
        "error": res["fun"],
        "success": res["success"],
    }

    return output


def _mse(
    params: list,
    home: float,
    draw: float,
    away: float,
    dc_adj: bool = True,
    rho: float = 0.001,
):
    """
    Loss function used internally by the `goal_expectancy function` to
    calculate the mean squared error of the estimate
    """
    exp_params = np.exp(params)

    mu1 = poisson(exp_params[0]).pmf(np.arange(0, 15))
    mu2 = poisson(exp_params[1]).pmf(np.arange(0, 15))

    mat = np.outer(mu1, mu2)

    if dc_adj:
        # apply Dixon and Coles adjustment
        mat[0, 0] *= 1 - exp_params[0] * exp_params[1] * rho
        mat[0, 1] *= 1 + exp_params[0] * rho
        mat[1, 0] *= 1 + exp_params[1] * rho
        mat[1, 1] *= 1 - rho

    pred = np.array(
        [
            np.sum(np.tril(mat, -1)),  # home
            np.sum(np.diag(mat)),  # draw
            np.sum(np.triu(mat, 1)),
        ]
    )  # away

    obs = np.array([home, draw, away])

    mse = np.mean((pred - obs) ** 2)

    return mse
