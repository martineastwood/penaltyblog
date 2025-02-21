"""
Ranked Probability Score

Calculates the Ranked Probability Score (RPS) for a given set of probabilities and outcomes.
"""

from typing import List, Union

import numpy as np


def rps(probs: Union[List, np.ndarray], outcome: int) -> float:
    """
    Calculate the Ranked Probability Score (RPS) for a given set of probabilities and outcome.

    Parameters
    ----------
    probs : list or np.ndarray
        A list or array of probabilities, where each element is a probability between 0 and 1.
    outcome : int
        The index of the outcome in the list of probabilities.

    Returns
    -------
    float
        The RPS score, which is a measure of how well the probabilities match the outcome.

    Raises
    ------
    TypeError
        If probs is not a list or array, or if outcome is not an integer.
    ValueError
        If probs is not 1D, or if outcome is out of range.
    """
    if not isinstance(outcome, (int, np.integer)):
        raise TypeError("outcome must be an integer")

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probs must be a 1D array")

    n_probs = probs.shape[0]

    if outcome < 0 or outcome >= n_probs:
        raise ValueError("outcome index out of range")

    cum_probs = np.cumsum(probs)
    cum_outcomes = np.zeros_like(probs)
    cum_outcomes[outcome] = 1
    cum_outcomes = np.cumsum(cum_outcomes)

    return np.sum((cum_probs - cum_outcomes) ** 2) / (n_probs - 1)
