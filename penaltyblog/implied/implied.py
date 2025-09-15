"""
This module provides improved functions for calculating implied odds using
type-safe dataclasses instead of dictionaries.
"""

from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize

from .models import (
    ImpliedMethod,
    ImpliedProbabilities,
    OddsFormat,
    OddsInput,
)


def calculate_implied(
    odds: Union[List[float], List[str], OddsInput],
    method: Union[str, ImpliedMethod] = ImpliedMethod.MULTIPLICATIVE,
    odds_format: Union[str, OddsFormat] = OddsFormat.DECIMAL,
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities from odds using the specified method.

    Parameters
    ----------
    odds : List[float] or List[str] or OddsInput
        The odds to convert to probabilities. Can be a list of values
        or an OddsInput object for more control.
    method : str or ImpliedMethod
        The method to use for calculating implied probabilities.
    odds_format : str or OddsFormat
        The format of the provided odds.
    market_names : List[str], optional
        Names for each market outcome.

    Returns
    -------
    ImpliedProbabilities
        Type-safe container with the calculated probabilities.

    Examples
    --------
    >>> import penaltyblog as pb
    >>> odds = [2.7, 2.3, 4.4]  # 1X2 market: Home, Draw, Away
    >>> result = pb.implied.calculate_implied(odds)
    >>> result.probabilities
    [0.35873804, 0.42112726, 0.2201347]
    >>> result.method
    ImpliedMethod.MULTIPLICATIVE
    >>> result.margin
    0.1362

    Using different odds formats:
    >>> american_odds = [+170, +130, +340]  # American odds
    >>> pb.implied.calculate_implied(
    ...     american_odds,
    ...     odds_format=OddsFormat.AMERICAN
    ... )

    >>> fractional_odds = ['7/4', '13/10', '7/2']  # Fractional odds
    >>> pb.implied.calculate_implied(
    ...     fractional_odds,
    ...     odds_format=OddsFormat.FRACTIONAL
    ... )
    """
    # Convert method to enum if string
    if isinstance(method, str):
        try:
            method = ImpliedMethod(method)
        except ValueError:
            raise ValueError(f"Unknown method: {method}")

    # Handle odds input
    if isinstance(odds, OddsInput):
        decimal_odds = odds.to_decimal()
        names = odds.market_names
    else:
        # Convert odds format to enum if string
        if isinstance(odds_format, str):
            try:
                odds_format = OddsFormat(odds_format)
            except ValueError:
                raise ValueError(f"Unknown odds format: {odds_format}")

        if isinstance(odds, list):
            if all(isinstance(x, (int, float)) for x in odds):
                odds_values: List[Union[float, str]] = [float(x) for x in odds]
            else:
                odds_values: List[Union[float, str]] = [str(x) for x in odds]
        else:
            # This should not happen due to the isinstance check above
            raise TypeError(f"Expected list, got {type(odds)}")

        # Create OddsInput and convert to decimal
        odds_input = OddsInput(
            values=odds_values, format=odds_format, market_names=market_names
        )
        decimal_odds = odds_input.to_decimal()
        names = market_names

    # Call the appropriate method
    if method == ImpliedMethod.MULTIPLICATIVE:
        return _multiplicative(decimal_odds, names)
    elif method == ImpliedMethod.ADDITIVE:
        return _additive(decimal_odds, names)
    elif method == ImpliedMethod.POWER:
        return _power(decimal_odds, names)
    elif method == ImpliedMethod.SHIN:
        return _shin(decimal_odds, names)
    elif method == ImpliedMethod.DIFFERENTIAL_MARGIN_WEIGHTING:
        return _differential_margin_weighting(decimal_odds, names)
    elif method == ImpliedMethod.ODDS_RATIO:
        return _odds_ratio(decimal_odds, names)
    elif method == ImpliedMethod.LOGARITHMIC:
        return _logarithmic(decimal_odds, names)
    else:
        raise ValueError(f"Unsupported method: {method}")


# Implementation of underlying methods with improved return types
def _multiplicative(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using the multiplicative method.

    The multiplicative method computes the implied probabilities by
    dividing the inverted odds by their sum to normalize them.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities and method metadata
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)
    normalized = (inv_odds / np.sum(inv_odds)).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.MULTIPLICATIVE,
        margin=margin,
        market_names=market_names,
    )


def _additive(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using the additive method.

    The additive method removes an equal proportion from each
    odd to get the implied probabilities.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities and method metadata
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)
    normalized = (inv_odds + 1 / len(inv_odds) * (1 - np.sum(inv_odds))).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.ADDITIVE,
        margin=margin,
        market_names=market_names,
    )


def _power(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using the power method.

    The power method computes the implied probabilities by solving for the
    power coefficient that normalizes the inverse of the odds to sum to 1.0.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities, method metadata,
        and the power coefficient 'k' in method_params
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)

    def _power_func(k: float, inv_odds: np.ndarray) -> np.ndarray:
        implied = inv_odds**k
        return implied

    def _power_error(k: float, inv_odds: np.ndarray) -> float:
        implied = _power_func(k, inv_odds)
        return float(1 - np.sum(implied))

    k = float(optimize.ridder(_power_error, 0, 100, args=(inv_odds,)))
    normalized = _power_func(k, inv_odds).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.POWER,
        margin=margin,
        market_names=market_names,
        method_params={"k": k},
    )


def _shin(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using Shin's method (1992, 1993).

    Shin's method models the bookmaker's overround as being proportional to
    the sum of the square roots of the implied probabilities.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities, method metadata,
        and the Shin 'z' parameter in method_params
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)

    def _shin_func(z: float, inv_odds: np.ndarray) -> np.ndarray:
        implied = (
            (z**2 + 4 * (1 - z) * inv_odds**2 / np.sum(inv_odds)) ** 0.5 - z
        ) / (2 - 2 * z)
        return implied

    def _shin_error(z: float, inv_odds: np.ndarray) -> float:
        implied = _shin_func(z, inv_odds)
        return float(1 - np.sum(implied))

    z = float(optimize.ridder(_shin_error, 0, 100, args=(inv_odds,)))
    normalized = _shin_func(z, inv_odds).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.SHIN,
        margin=margin,
        market_names=market_names,
        method_params={"z": z},
    )


def _differential_margin_weighting(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using differential margin weighting.

    This method is based on Joseph Buchdahl's wisdom of the crowds approach,
    which distributes the overround proportionally to the odds.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities and method metadata
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds: npt.NDArray[np.float64] = 1.0 / odds_arr
    margin: float = float(np.sum(inv_odds) - 1)
    n_odds: int = len(odds_arr)
    fair_odds: npt.NDArray[np.float64] = (n_odds * odds_arr) / (
        n_odds - (margin * odds_arr)
    )
    normalized = (1 / fair_odds).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.DIFFERENTIAL_MARGIN_WEIGHTING,
        margin=margin,
        market_names=market_names,
    )


def _odds_ratio(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Calculate implied probabilities using Keith Cheung's odds ratio method.

    This method is discussed in Joseph Buchdahl's wisdom of the crowds.
    It models the relationship between true and implied probabilities
    using an odds ratio transformation.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities, method metadata,
        and the odds ratio parameter 'c' in method_params
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds: npt.NDArray[np.float64] = 1.0 / odds_arr
    margin: float = float(np.sum(inv_odds) - 1)

    def _or_func(
        c: float, inv_odds: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        y = inv_odds / (c + inv_odds - (c * inv_odds))
        return y

    def _or_error(c: float, inv_odds: npt.NDArray[np.float64]) -> float:
        implied = _or_func(c, inv_odds)
        return float(1 - np.sum(implied))

    c = float(optimize.ridder(_or_error, 0, 100, args=(inv_odds,)))
    normalized = _or_func(c, inv_odds).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.ODDS_RATIO,
        margin=margin,
        market_names=market_names,
        method_params={"c": c},
    )


def _logarithmic(
    odds: List[float],
    market_names: Optional[List[str]] = None,
) -> ImpliedProbabilities:
    """
    Logarithmic method for overround removal using an additive shift.

    This method operates in "log-odds" (logit) space. It assumes the
    bookmaker's margin is a constant value 'c' subtracted from the
    log-odds of each outcome. The solver finds the 'c' that makes the
    final probabilities sum to 1.0. This is a common and robust method.

    Parameters
    ----------
    odds : List[float]
        List of decimal odds for each outcome
    market_names : List[str], optional
        Names for each market outcome

    Returns
    -------
    ImpliedProbabilities
        Object containing the calculated probabilities, method metadata,
        and the odds ratio parameter 'c' in method_params
    """
    odds_arr = np.array(odds, dtype=np.float64)
    inv_odds = 1.0 / odds_arr
    margin = float(np.sum(inv_odds) - 1)

    # If there is no margin, no adjustment is needed.
    if np.isclose(margin, 0.0):
        return ImpliedProbabilities(
            probabilities=inv_odds.tolist(),
            method=ImpliedMethod.LOGARITHMIC,
            margin=margin,
            market_names=market_names,
            method_params={"c": 0.0},
        )

    probs = inv_odds
    probs_safe = np.clip(probs, 1e-15, 1 - 1e-15)
    log_odds = np.log(probs_safe / (1 - probs_safe))

    # Find the adjustment constant 'c' that makes probabilities sum to 1
    def _log_odds_error(c: float, log_odds: np.ndarray) -> float:
        adjusted_log_odds = log_odds - c
        # Convert back to probabilities using sigmoid function
        adjusted_probs = 1 / (1 + np.exp(-adjusted_log_odds))
        return float(np.sum(adjusted_probs) - 1)

    # For an overround market, 'c' will be positive.
    try:
        c = float(optimize.brentq(_log_odds_error, 0, 20.0, args=(log_odds,)))
    except ValueError:
        # Fallback for extreme cases or underround markets
        c = float(optimize.brentq(_log_odds_error, -20.0, 20.0, args=(log_odds,)))

    # Apply adjustment and convert back to probabilities
    adjusted_log_odds = log_odds - c
    normalized = (1 / (1 + np.exp(-adjusted_log_odds))).tolist()

    return ImpliedProbabilities(
        probabilities=normalized,
        method=ImpliedMethod.LOGARITHMIC,
        margin=margin,
        market_names=market_names,
        method_params={"c": c},
    )
