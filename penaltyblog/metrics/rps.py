"""
Ranked Probability Score

Calculates the Ranked Probability Score (RPS) for a given set of probabilities and outcomes.
"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from penaltyblog.utils import deprecated

from .metrics import compute_average_rps, compute_rps_array


def rps_average(probs: ArrayLike, outcomes: Union[ArrayLike, int]) -> float:
    """
    Computes the average Ranked Probability Score (RPS) for a batch of fixtures.

    Parameters:
      probs: Array-like representing probabilities.
             If a 1D array is provided, it will be reshaped to (1, n_outcomes).
      outcomes: Array-like with each element an integer outcome.
                If a single integer is provided, it will be wrapped into a 1-element array.

    Returns:
      float: The average RPS across all fixtures.
    """
    # Convert probs to a numpy array of type float64
    if not isinstance(probs, np.ndarray):
        probs = np.array(probs, dtype=np.float64)
    else:
        probs = np.ascontiguousarray(probs, dtype=np.float64)

    # If probs is 1D, reshape to (1, n_outcomes)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    # Convert outcomes to a numpy array of type int32
    if isinstance(outcomes, (int, np.integer)):
        outcomes = np.array([outcomes], dtype=np.int32)
    elif not isinstance(outcomes, np.ndarray):
        outcomes = np.array(outcomes, dtype=np.int32)
    else:
        outcomes = np.ascontiguousarray(outcomes, dtype=np.int32)

    nSets, nOutcomes = probs.shape
    avg_rps = compute_average_rps(probs, outcomes, nSets, nOutcomes)
    return float(avg_rps)


def rps_array(probs: ArrayLike, outcomes: Union[ArrayLike, int]) -> np.ndarray:
    """
    Computes individual RPS values for each fixture.

    Parameters:
      probs: 2D array-like of shape (nSets, nOutcomes)
      outcomes: Array-like of outcome indices (length nSets)

    Returns:
      np.ndarray: A numpy array of RPS values, one per fixture.
    """
    # Convert probs to a numpy array of type float64
    if not isinstance(probs, np.ndarray):
        probs = np.array(probs, dtype=np.float64)
    else:
        probs = np.ascontiguousarray(probs, dtype=np.float64)

    # If probs is 1D, reshape to (1, n_outcomes)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    # Convert outcomes to a numpy array of type int32
    if isinstance(outcomes, (int, np.integer)):
        outcomes = np.array([outcomes], dtype=np.int32)
    elif not isinstance(outcomes, np.ndarray):
        outcomes = np.array(outcomes, dtype=np.int32)
    else:
        outcomes = np.ascontiguousarray(outcomes, dtype=np.int32)

    nSets, nOutcomes = probs.shape
    out_rps = np.empty(nSets, dtype=np.float64)
    compute_rps_array(probs, outcomes, nSets, nOutcomes, out_rps)
    return out_rps


@deprecated("Use `rps_array` or `rps_average` instead.")
def rps(probs: np.ndarray, outcome: int) -> float:
    """
    Calculate the Ranked Probability Score

    Parameters
    ----------
    probs : array-like
        The predicted probabilities of each outcome occurring
    outcome : int
        Index of the observed outcome in `probs`

    Returns
    -------
    float
        The Ranked Probability Score

    Examples
    --------
    >>> rps([0.8, 0.1, 0.1], 0)
    """
    probs = np.asarray(probs)
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.zeros_like(probs)
    cum_outcomes[outcome] = 1
    cum_outcomes = np.cumsum(cum_outcomes)
    return np.sum((cum_probs - cum_outcomes) ** 2) / (len(probs) - 1)
