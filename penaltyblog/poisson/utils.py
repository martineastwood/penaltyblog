import numpy as np


def rho_correction_vec(df):
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


def rho_correction(goals_home, goals_away, home_exp, away_exp, rho):
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


def dixon_coles_weights(dates, xi=0, base_date=None):
    if base_date is None:
        base_date = max(dates)

    diffs = np.array([(base_date - x).days for x in dates])
    weights = np.exp(-xi * diffs)
    return weights
