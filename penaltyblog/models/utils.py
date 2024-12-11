import numpy as np
import pandas as pd
from numpy.typing import NDArray


def rho_correction_vec(df: pd.DataFrame) -> NDArray:
    dc_adj = np.select(
        [
            (df["goals_home"] == 0) & (df["goals_away"] == 0),
            (df["goals_home"] == 0) & (df["goals_away"] == 1),
            (df["goals_home"] == 1) & (df["goals_away"] == 0),
            (df["goals_home"] == 1) & (df["goals_away"] == 1),
        ],
        [
            1 - (df["home_exp"] * df["away_exp"] * df["rho"]),
            1 + (df["home_exp"] * df["rho"]),
            1 + (df["away_exp"] * df["rho"]),
            1 - df["rho"],
        ],
        default=1,
    )
    return dc_adj


def rho_correction(
    goals_home: int, goals_away: int, home_exp: float, away_exp: float, rho: float
) -> float:
    """
    Applies the dixon and coles correction
    """
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_weights(dates, xi=0.0018, base_date=None) -> NDArray:
    """
    Calculates a decay curve based on the algorithm given by
    Dixon and Coles in their paper

    Parameters
    ----------
    dates : list
        A list or pd.Series of dates to calculate weights for
    x1 : float
        Controls the steepness of the decay curve
    base_date : date
        The base date to start the decay from. If set to None
        then it uses the maximum date
    """
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights
