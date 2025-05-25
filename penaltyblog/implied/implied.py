"""
Implied Probabilities

Calculates the implied probabilities for a given set of odds.
"""

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from scipy import optimize


def multiplicative(odds: List[float]) -> Dict[str, Any]:
    """
    The multiplicative method computes the implied probabilities by
    dividing the inverted odds by their sum to normalize them

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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    normalized = (inv_odds / np.sum(inv_odds)).tolist()
    margin = float(np.sum(inv_odds) - 1)
    result = {
        "implied_probabilities": normalized,
        "method": "multiplicative",
        "margin": margin,
    }
    return result


def additive(odds: List[float]) -> Dict[str, Any]:
    """
    The additive method removes an equal proportion from each
    odd to get the implied probabilities

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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    normalized = (inv_odds + 1 / len(inv_odds) * (1 - np.sum(inv_odds))).tolist()
    margin = float(np.sum(inv_odds) - 1)
    result = {
        "implied_probabilities": normalized,
        "method": "additive",
        "margin": margin,
    }
    return result


def power(odds: List[float]) -> Dict[str, Any]:
    """
    The power method computes the implied probabilities by solving for the
    power coefficient that normalizes the inverse of the odds to sum to 1.0

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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)

    def _power(k: float, inv_odds: np.ndarray) -> np.ndarray:
        implied = inv_odds**k
        return implied

    def _power_error(k: float, inv_odds: np.ndarray) -> float:
        implied = _power(k, inv_odds)
        return float(1 - np.sum(implied))

    res = float(optimize.ridder(_power_error, 0, 100, args=(inv_odds,)))
    normalized = _power(res, inv_odds).tolist()
    result = {
        "implied_probabilities": normalized,
        "method": "power",
        "k": res,
        "margin": margin,
    }
    return result


def shin(odds: List[float]) -> Dict[str, Any]:
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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)

    def _shin_error(z: float, inv_odds: np.ndarray) -> float:
        implied = _shin(z, inv_odds)
        return float(1 - np.sum(implied))

    def _shin(z: float, inv_odds: np.ndarray) -> np.ndarray:
        implied = (
            (z**2 + 4 * (1 - z) * inv_odds**2 / np.sum(inv_odds)) ** 0.5 - z
        ) / (2 - 2 * z)
        return implied

    res = float(optimize.ridder(_shin_error, 0, 100, args=(inv_odds,)))
    normalized = _shin(res, inv_odds).tolist()
    result = {
        "implied_probabilities": normalized,
        "method": "shin",
        "z": res,
        "margin": margin,
    }
    return result


def differential_margin_weighting(odds: List[float]) -> Dict[str, Any]:
    """
    Based on Jospeh Buchdahl's wisdom of the crowds -
    https://www.football-data.co.uk/The_Wisdom_of_the_Crowd.pdf

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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds: npt.NDArray[np.float64] = 1.0 / odds_arr
    margin: float = float(np.sum(inv_odds) - 1)
    n_odds: int = len(odds_arr)
    fair_odds: npt.NDArray[np.float64] = (n_odds * odds_arr) / (
        n_odds - (margin * odds_arr)
    )
    implied_probs = (1 / fair_odds).tolist()
    result = {
        "implied_probabilities": implied_probs,
        "method": "differential_margin_weighting",
        "margin": margin,
    }
    return result


def odds_ratio(odds: List[float]) -> Dict[str, Any]:
    """
    Keith Cheung's odds ratio method, as discussed in
    Jospeh Buchdahl's wisdom of the crowds

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
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds: npt.NDArray[np.float64] = 1.0 / odds_arr
    margin: float = float(np.sum(inv_odds) - 1)

    def _or_error(c: float, inv_odds: npt.NDArray[np.float64]) -> float:
        implied = _or(c, inv_odds)
        return float(1 - np.sum(implied))

    def _or(c: float, inv_odds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        y = inv_odds / (c + inv_odds - (c * inv_odds))
        return y

    res = float(optimize.ridder(_or_error, 0, 100, args=(inv_odds,)))
    normalized = _or(res, inv_odds).tolist()
    result = {
        "implied_probabilities": normalized,
        "method": "odds_ratio",
        "c": res,
        "margin": margin,
    }
    return result
