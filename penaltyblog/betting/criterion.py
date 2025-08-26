"""
Kelly Criterion

Calculates the Kelly Criterion for a given set of odds and probabilities.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


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


def multiple_criterion(
    decimal_odds: List[float],
    true_probs: List[float],
    fraction: float = 1.0,
    max_total_stake: float = 1.0,
    method: str = "simultaneous",
) -> List[float]:
    """
    Calculate optimal bet sizes for multiple simultaneous outcomes using Kelly Criterion.

    This function handles portfolio optimization across multiple bets, ensuring that
    the total stake doesn't exceed the specified maximum and optimizes for maximum
    expected logarithmic growth.

    Parameters
    ----------
    decimal_odds : List[float]
        List of odds in European decimal format for each outcome
    true_probs : List[float]
        List of true probabilities for each outcome (should sum to ≤ 1.0)
    fraction : float, default=1.0
        Fraction of optimal Kelly to use (e.g., 0.5 for Half Kelly)
    max_total_stake : float, default=1.0
        Maximum fraction of bankroll to stake across all bets
    method : str, default="simultaneous"
        Method to use: "simultaneous" for portfolio optimization,
        "independent" for independent Kelly calculations

    Returns
    -------
    List[float]
        Recommended fraction of bankroll to wager on each outcome

    Examples
    --------
    >>> # Three-way market (e.g., match result)
    >>> multiple_criterion([2.5, 3.2, 2.8], [0.45, 0.30, 0.35])
    >>> # Two-way market with half Kelly
    >>> multiple_criterion([2.1, 1.9], [0.50, 0.48], fraction=0.5)

    Notes
    -----
    The "simultaneous" method optimizes the portfolio as a whole, while
    "independent" calculates Kelly for each bet separately and may exceed
    max_total_stake constraints.
    """
    if len(decimal_odds) != len(true_probs):
        raise ValueError("decimal_odds and true_probs must have the same length")

    if not all(p >= 0 for p in true_probs):
        raise ValueError("All probabilities must be non-negative")

    if sum(true_probs) > 1.0 + 1e-10:  # Allow small floating point errors
        raise ValueError("Sum of probabilities cannot exceed 1.0")

    if method == "independent":
        # Calculate independent Kelly for each bet
        stakes = []
        for odds, prob in zip(decimal_odds, true_probs):
            kelly = criterion(odds, prob, fraction)
            stakes.append(kelly)

        # Scale down if total exceeds max_total_stake
        total_stake = sum(stakes)
        if total_stake > max_total_stake:
            stakes = [s * (max_total_stake / total_stake) for s in stakes]

        return stakes

    elif method == "simultaneous":
        n = len(decimal_odds)

        def objective(stakes):
            """Negative expected log growth (to minimize)"""
            stakes = np.array(stakes)
            remaining_bankroll = 1.0 - np.sum(stakes)

            if remaining_bankroll <= 0:
                return 1e10  # Penalty for invalid stakes

            expected_log_growth = 0.0

            # Case where no outcome occurs (probability = 1 - sum(true_probs))
            prob_no_outcome = max(0, 1.0 - sum(true_probs))
            if prob_no_outcome > 0:
                final_bankroll = remaining_bankroll  # Lose all stakes
                if final_bankroll > 0:
                    expected_log_growth += prob_no_outcome * np.log(final_bankroll)
                else:
                    return 1e10  # Avoid log(0)

            # Cases where each outcome occurs
            for i, (odds, prob) in enumerate(zip(decimal_odds, true_probs)):
                if prob > 0:
                    winnings = stakes[i] * odds
                    final_bankroll = remaining_bankroll + winnings
                    if final_bankroll > 0:
                        expected_log_growth += prob * np.log(final_bankroll)
                    else:
                        return 1e10  # Avoid log(0)

            return -expected_log_growth  # Negative because we minimize

        # Constraints
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: max_total_stake - sum(x),
            },  # Total stake <= max
            {"type": "ineq", "fun": lambda x: 1.0 - sum(x)},  # Keep some bankroll
        ]

        # Bounds: each stake between 0 and max_total_stake
        bounds = [(0, max_total_stake) for _ in range(n)]

        # Initial guess: independent Kelly scaled down
        initial_stakes = []
        for odds, prob in zip(decimal_odds, true_probs):
            kelly = criterion(odds, prob)
            initial_stakes.append(min(kelly, max_total_stake / n))

        # Optimize
        result = minimize(
            objective,
            initial_stakes,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            stakes = result.x.tolist()
            # Apply fraction scaling
            stakes = [s * fraction for s in stakes]
            return stakes
        else:
            # Fallback to independent method if optimization fails
            return multiple_criterion(
                decimal_odds, true_probs, fraction, max_total_stake, "independent"
            )

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'simultaneous' or 'independent'"
        )
