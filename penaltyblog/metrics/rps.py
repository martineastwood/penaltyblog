import numpy as np


def rps(probs, outcome):
    """
    Calculate the Ranked Probability Score

    Parameters
    ----------
    probs : list
        A list of the predicted probabilities of each outcome occurring

    outcome : int
        An integer designating which index in `probs` was the observed outcome

    Returns
    -------
    float
        The Ranked Probability Score as floating point number

    Examples
    --------
    >>> rps([0.8, 0.1, 0.1], 0)
    """
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.zeros(len(probs))
    cum_outcomes[outcome] = 1
    cum_outcomes = np.cumsum(cum_outcomes)

    sum_rps = 0
    for i in range(len(probs)):
        sum_rps += (cum_probs[i] - cum_outcomes[i]) ** 2

    return sum_rps / (len(probs) - 1)