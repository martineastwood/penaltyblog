"""
Kelly Criterion

Calculates the Kelly Criterion for a given set of odds and probabilities.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def criterion(
    decimal_odds: Union[float, NDArray],
    true_prob: Union[float, NDArray],
    fraction: float = 1.0,
) -> Union[float, NDArray]:
    """
    Calculate the optimal bet size using the Kelly Criterion.

    Parameters
    ----------
    decimal_odds : float or np.ndarray
        The odds in European decimal format (e.g., 1.50)
    true_prob : float or np.ndarray
        The true probability of the event (0-1)
    fraction : float, default=1.0
        Fraction of Kelly to use (e.g., 0.5 for Half Kelly)

    Returns
    -------
    float or np.ndarray
        Recommended fraction of bankroll to wager

    Examples
    --------
    >>> criterion(1.5, 0.7, 1/3)
    >>> criterion(np.array([1.5, 2.0]), np.array([0.7, 0.5]), 0.5)
    """
    crit = ((true_prob * decimal_odds) - 1) / (decimal_odds - 1)
    return np.clip(crit * fraction, 0, 1)
