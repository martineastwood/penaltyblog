import numpy as np


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
