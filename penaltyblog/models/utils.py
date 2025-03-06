import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
