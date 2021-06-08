from operator import inv
import numpy as np
from scipy import optimize


def multiplicative(odds) -> dict:
    """
    The multiplicative method computes the implied probabilities by dividing the inverted odds by their sum to normalize them

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities and method used
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    normalized = inv_odds / np.sum(inv_odds)
    result = {"implied_probabilities": normalized, "method": "multiplicative"}
    return result


def additive(odds) -> dict:
    """
    The additive method removes an equal proportion from each odd to get the implied probabilities

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities and method used
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    normalized = inv_odds + 1 / len(inv_odds) * (1 - np.sum(inv_odds))
    result = {"implied_probabilities": normalized, "method": "additive"}
    return result


def power(odds) -> dict:
    """
    The power method computes the implied probabilities by solving for the power coefficient that normalizes the inverse of the odds to sum to 1.0

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities, k and method used
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds

    def _power(k, inv_odds):
        implied = inv_odds ** k
        return implied

    def _power_error(k, inv_odds):
        implied = _power(k, inv_odds)
        return 1 - np.sum(implied)

    res = optimize.ridder(_power_error, 0, 100, args=(inv_odds,))
    normalized = _power(res, inv_odds)
    result = {"implied_probabilities": normalized, "method": "power", "k": res}
    return result


def shin(odds) -> dict:
    """
    Computes the implied probabilities via the Shin (1992, 1993) method

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities, z and method used
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds

    def _shin_error(z, inv_odds):
        implied = _shin(z, inv_odds)
        return 1 - np.sum(implied)

    def _shin(z, inv_odds):
        implied = (
            (z ** 2 + 4 * (1 - z) * inv_odds ** 2 / np.sum(inv_odds)) ** 0.5 - z
        ) / (2 - 2 * z)
        return implied

    res = optimize.ridder(_shin_error, 0, 100, args=(inv_odds,))
    normalized = _shin(res, inv_odds)
    result = {"implied_probabilities": normalized, "method": "shin", "z": res}
    return result
