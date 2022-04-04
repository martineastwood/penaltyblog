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

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.multiplicative(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    normalized = inv_odds / np.sum(inv_odds)
    margin = np.sum(inv_odds) - 1
    result = {
        "implied_probabilities": normalized,
        "method": "multiplicative",
        "margin": margin,
    }
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

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.additive(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    normalized = inv_odds + 1 / len(inv_odds) * (1 - np.sum(inv_odds))
    margin = np.sum(inv_odds) - 1
    result = {
        "implied_probabilities": normalized,
        "method": "additive",
        "margin": margin,
    }
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

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.power(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1

    def _power(k, inv_odds):
        implied = inv_odds ** k
        return implied

    def _power_error(k, inv_odds):
        implied = _power(k, inv_odds)
        return 1 - np.sum(implied)

    res = optimize.ridder(_power_error, 0, 100, args=(inv_odds,))
    normalized = _power(res, inv_odds)
    result = {
        "implied_probabilities": normalized,
        "method": "power",
        "k": res,
        "margin": margin,
    }
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

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.shin(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1

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
    result = {
        "implied_probabilities": normalized,
        "method": "shin",
        "z": res,
        "margin": margin,
    }
    return result


def differential_margin_weighting(odds) -> dict:
    """
    Based on Jospeh Buchdahl's wisdom of the crowds - https://www.football-data.co.uk/The_Wisdom_of_the_Crowd.pdf

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities, z and method used

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.differential_margin_weighting(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1
    n_odds = len(odds)
    fair_odds = (n_odds * odds) / (n_odds - (margin * odds))
    result = {
        "implied_probabilities": 1 / fair_odds,
        "method": "differential_margin_weighting",
        "margin": margin,
    }
    return result


def odds_ratio(odds) -> dict:
    """
    Keith Cheung's odds ratio method, as discussed in Jospeh Buchdahl's wisdom of the crowds

    Parameters
    ----------
    odds : list
        list of odds

    Returns
    ----------
    dict
        contains implied probabilities, z and method used

    Examples
    ----------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]
    >>> pb.implied.odds_ratio(odds)
    """
    odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1

    def _or_error(c, inv_odds):
        implied = _or(c, inv_odds)
        return 1 - np.sum(implied)

    def _or(c, inv_odds):
        y = inv_odds / (c + inv_odds - (c * inv_odds))
        return y

    res = optimize.ridder(_or_error, 0, 100, args=(inv_odds,))
    normalized = _or(res, inv_odds)
    result = {
        "implied_probabilities": normalized,
        "method": "odds_ratio",
        "c": res,
        "margin": margin,
    }
    return result
