from typing import List, Optional, Tuple


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
