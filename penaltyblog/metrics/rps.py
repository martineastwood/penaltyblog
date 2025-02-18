import numpy as np


def rps(probs: np.ndarray, outcome: int) -> float:
    """
    Calculate the Ranked Probability Score
    """
    if not np.isscalar(outcome):
        raise TypeError("outcome must be an integer")

    probs = np.asarray(probs, dtype=np.float64)
    n_probs = probs.shape[0]

    if outcome < 0 or outcome >= n_probs:
        raise ValueError("outcome index out of range")

    cum_probs = np.cumsum(probs)
    cum_outcomes = np.zeros_like(probs)
    cum_outcomes[outcome] = 1
    cum_outcomes = np.cumsum(cum_outcomes)

    return np.sum((cum_probs - cum_outcomes) ** 2) / (n_probs - 1)
