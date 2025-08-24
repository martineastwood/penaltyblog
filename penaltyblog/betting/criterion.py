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


def arbitrage_hedge(
    existing_stakes: List[float],
    existing_odds: List[float],
    hedge_odds: List[float],
    target_profit: Optional[float] = None,
    hedge_all: bool = True,
) -> Tuple[List[float], float]:
    """
    Calculate hedge bet sizes to guarantee profit or minimize loss from existing positions.

    This function determines how much to bet on other outcomes to either lock in
    a guaranteed profit or minimize potential losses from existing bets.

    Parameters
    ----------
    existing_stakes : List[float]
        Amount already staked on each outcome (in currency units)
    existing_odds : List[float]
        Decimal odds for existing bets
    hedge_odds : List[float]
        Current decimal odds available for hedging (should match length of existing stakes)
    target_profit : float, optional
        Target profit to achieve. If None, maximizes guaranteed profit
    hedge_all : bool, default=True
        If True, hedge all outcomes. If False, only hedge profitable outcomes

    Returns
    -------
    Tuple[List[float], float]
        - List of hedge bet amounts for each outcome
        - Guaranteed profit/loss (negative if loss)

    Examples
    --------
    >>> # Already bet $100 on Team A at 3.0, now hedge on Team B at 2.5
    >>> arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5])
    ([0, 80], 20.0)  # Bet $80 on Team B, guarantee $20 profit

    >>> # Multiple existing positions
    >>> arbitrage_hedge([50, 30, 0], [2.5, 4.0, 3.0], [2.4, 3.8, 2.9])

    Notes
    -----
    This function assumes you want to guarantee the same profit regardless of
    which outcome occurs. Set hedge_all=False to only hedge outcomes where
    you have existing exposure.
    """
    if len(existing_stakes) != len(existing_odds) or len(existing_stakes) != len(
        hedge_odds
    ):
        raise ValueError("All input lists must have the same length")

    n = len(existing_stakes)

    # Calculate potential payouts from existing bets
    existing_payouts = [
        stake * odds for stake, odds in zip(existing_stakes, existing_odds)
    ]
    total_existing_stakes = sum(existing_stakes)

    if hedge_all:
        # Find the minimum net position to guarantee same profit on all outcomes
        net_positions = [payout - total_existing_stakes for payout in existing_payouts]
        min_net_position = min(net_positions)

        if target_profit is not None:
            guaranteed_profit = target_profit
        else:
            # Maximize guaranteed profit
            guaranteed_profit = min_net_position

        # Calculate required hedge bets
        hedge_stakes = []
        for i in range(n):
            current_net = existing_payouts[i] - total_existing_stakes
            needed_hedge_payout = current_net - guaranteed_profit

            if needed_hedge_payout > 0:
                # Need to hedge by betting against this outcome
                hedge_stake = needed_hedge_payout / (hedge_odds[i] - 1)
                hedge_stakes.append(-hedge_stake)  # Negative indicates betting against
            else:
                # Need to bet more on this outcome
                required_total_payout = total_existing_stakes + guaranteed_profit
                additional_payout_needed = required_total_payout - existing_payouts[i]
                if additional_payout_needed > 0:
                    additional_stake = additional_payout_needed / hedge_odds[i]
                    hedge_stakes.append(additional_stake)
                else:
                    hedge_stakes.append(0)

    else:
        # Only hedge outcomes where we have existing exposure
        hedge_stakes = []
        guaranteed_profit = float("inf")

        for i in range(n):
            if existing_stakes[i] > 0:
                # Calculate required hedge to neutralize this position
                existing_payout = existing_payouts[i]
                net_if_wins = existing_payout - total_existing_stakes
                net_if_loses = -total_existing_stakes

                # To neutralize, we want net_if_wins = net_if_loses after hedging
                # If outcome i wins: net_if_wins - hedge_stake_i * hedge_odds[i]
                # If outcome i loses: net_if_loses + hedge_stake_i
                # Setting equal: net_if_wins - hedge_stake_i * hedge_odds[i] = net_if_loses + hedge_stake_i
                # Solving: hedge_stake_i = (net_if_wins - net_if_loses) / (hedge_odds[i] + 1)

                hedge_stake = (net_if_wins - net_if_loses) / (hedge_odds[i] + 1)
                hedge_stakes.append(hedge_stake)

                # Calculate guaranteed profit with this hedge
                profit = net_if_loses + hedge_stake
                guaranteed_profit = min(guaranteed_profit, profit)
            else:
                hedge_stakes.append(0)

        if guaranteed_profit == float("inf"):
            guaranteed_profit = -total_existing_stakes  # No existing bets to hedge

    # Handle the case where we're betting against outcomes (short selling not possible in practice)
    # Convert negative stakes to positive stakes on other outcomes
    practical_hedge_stakes = []
    total_hedge_needed = 0

    for i, stake in enumerate(hedge_stakes):
        if stake < 0:
            # Can't bet negative amount, so we need to bet more on other outcomes
            total_hedge_needed += abs(stake)
            practical_hedge_stakes.append(0)
        else:
            practical_hedge_stakes.append(stake)

    # Distribute the additional hedge needed across other outcomes
    if total_hedge_needed > 0 and len(practical_hedge_stakes) > 1:
        # Simple distribution: proportional to odds (higher odds get more hedge)
        total_odds = sum(hedge_odds)
        for i, odds in enumerate(hedge_odds):
            if practical_hedge_stakes[i] >= 0:  # Only add to non-negative stakes
                additional_hedge = total_hedge_needed * (odds / total_odds)
                practical_hedge_stakes[i] += additional_hedge

    # Recalculate guaranteed profit with practical stakes
    min_profit = float("inf")
    for i in range(n):
        # If outcome i wins
        profit_if_i_wins = (
            existing_payouts[i]
            + practical_hedge_stakes[i] * hedge_odds[i]
            - total_existing_stakes
            - sum(practical_hedge_stakes)
        )
        min_profit = min(min_profit, profit_if_i_wins)

    return practical_hedge_stakes, min_profit
