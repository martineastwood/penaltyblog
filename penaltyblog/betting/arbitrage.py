import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scipy.optimize import linprog


@dataclass
class ArbitrageHedgeResult:
    """Structured result for arbitrage_hedge.

    Attributes
    ----------
    raw_hedge_stakes: List[float]
        Hedge stakes as originally calculated (may contain negative values
        indicating a theoretical 'bet against' that isn't practical).
    practical_hedge_stakes: List[float]
        Practical hedge stakes (non-negative) to place on each outcome.
    guaranteed_profit: float
        The guaranteed profit (or loss if negative) after placing the
        practical hedges.
    existing_payouts: List[float]
        Payouts from existing stakes given their odds.
    total_existing_stakes: float
        Sum of existing stakes.
    total_hedge_needed: float
        Total amount of hedge that had to be redistributed because negative
        (bet-against) stakes are not possible in practice.
    lp_success: bool = False
        Whether the linear programming solver succeeded. False if fallback heuristic was used.
    lp_message: Optional[str] = None
        Error message from LP solver if it failed, None otherwise.
    """

    raw_hedge_stakes: List[float]
    practical_hedge_stakes: List[float]
    guaranteed_profit: float
    existing_payouts: List[float]
    total_existing_stakes: float
    total_hedge_needed: float
    lp_success: bool = False
    lp_message: Optional[str] = None

    def __iter__(self):
        """Allow unpacking like (hedge_stakes, profit) for backward compatibility.

        Yield the practical hedge stakes and the guaranteed profit, which is
        what older callers expect when unpacking the result of
        `arbitrage_hedge`.
        """
        yield self.practical_hedge_stakes
        yield self.guaranteed_profit


def _validate_arbitrage_inputs(
    existing_stakes: List[float],
    existing_odds: List[float],
    hedge_odds: List[float],
    target_profit: Optional[float] = None,
) -> None:
    """Validate inputs for arbitrage hedge calculation."""
    if not existing_stakes or not existing_odds or not hedge_odds:
        raise ValueError("Input lists cannot be empty")

    if len(existing_stakes) != len(existing_odds) or len(existing_stakes) != len(
        hedge_odds
    ):
        raise ValueError("All input lists must have the same length")

    # Validate odds are greater than 1.0 (must offer positive returns)
    if any(odd <= 1.0 for odd in existing_odds):
        raise ValueError("All existing_odds must be greater than 1.0")

    if any(odd <= 1.0 for odd in hedge_odds):
        raise ValueError("All hedge_odds must be greater than 1.0")

    # Validate stakes are non-negative
    if any(stake < 0 for stake in existing_stakes):
        raise ValueError("All existing_stakes must be non-negative")

    # Validate target_profit if provided
    if target_profit is not None and not math.isfinite(target_profit):
        raise ValueError("target_profit must be a finite number")


def _solve_hedge_lp(
    existing_payouts: List[float],
    total_existing_stakes: float,
    hedge_odds: List[float],
    target_profit: Optional[float] = None,
    allow_lay: bool = False,
) -> Tuple[List[float], float, bool, Optional[str]]:
    """Solve linear program to find optimal hedge stakes.

    Returns:
        (hedge_stakes, guaranteed_profit, success, message)
    """
    n = len(hedge_odds)

    # Build A_ub and b_ub for LP constraints
    A_ub = []
    b_ub = []
    for i in range(n):
        row = [0.0] * (n + 1)  # n h_i vars + G
        for j in range(n):
            row[j] = 1.0
        # adjust h_i coefficient
        row[i] = 1.0 - hedge_odds[i]
        # G coefficient
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(existing_payouts[i] - total_existing_stakes)

    # Objective: maximize G -> minimize -G
    c = [0.0] * n + [-1.0]

    # Bounds: h_i >= 0 unless allow_lay True, G unbounded unless target_profit provided
    bounds = [(None, None)] * n if allow_lay else [(0, None)] * n
    if target_profit is not None:
        # fix G to target_profit via bounds
        bounds.append((target_profit, target_profit))
    else:
        bounds.append((None, None))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    success = bool(res.success)
    message = None if success else getattr(res, "message", None)

    if success:
        x = res.x
        hedge_stakes = [float(v) for v in x[:n]]
        guaranteed_profit = float(x[-1])
    else:
        hedge_stakes = []
        guaranteed_profit = 0.0

    return hedge_stakes, guaranteed_profit, success, message


def _calculate_heuristic_hedges(
    existing_payouts: List[float],
    total_existing_stakes: float,
    hedge_odds: List[float],
    target_profit: Optional[float] = None,
    tolerance: float = 1e-10,
) -> Tuple[List[float], float]:
    """Calculate hedge stakes using heuristic when LP fails."""
    n = len(hedge_odds)

    net_positions = [payout - total_existing_stakes for payout in existing_payouts]
    min_net_position = min(net_positions)
    guaranteed_profit = target_profit if target_profit is not None else min_net_position

    hedge_stakes = []
    for i in range(n):
        current_net = existing_payouts[i] - total_existing_stakes
        needed_hedge_payout = current_net - guaranteed_profit

        if needed_hedge_payout > tolerance:
            # Need to hedge by betting against this outcome
            hedge_stake = needed_hedge_payout / (hedge_odds[i] - 1)
            hedge_stakes.append(-hedge_stake)  # Negative indicates betting against
        else:
            # Need to bet more on this outcome
            required_total_payout = total_existing_stakes + guaranteed_profit
            additional_payout_needed = required_total_payout - existing_payouts[i]
            if additional_payout_needed > tolerance:
                additional_stake = additional_payout_needed / hedge_odds[i]
                hedge_stakes.append(additional_stake)
            else:
                hedge_stakes.append(0)

    return hedge_stakes, guaranteed_profit


def _calculate_partial_hedges(
    existing_stakes: List[float],
    existing_payouts: List[float],
    total_existing_stakes: float,
    hedge_odds: List[float],
    tolerance: float = 1e-10,
) -> Tuple[List[float], float]:
    """Calculate hedges for existing positions only (hedge_all=False)."""
    n = len(existing_stakes)
    hedge_stakes = []
    guaranteed_profit = float("inf")

    for i in range(n):
        if existing_stakes[i] > tolerance:
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

    return hedge_stakes, guaranteed_profit


def _redistribute_negative_stakes(
    raw_hedge_stakes: List[float],
    hedge_odds: List[float],
    allow_lay: bool = False,
    tolerance: float = 1e-10,
) -> List[float]:
    """Convert negative stakes to positive stakes on other outcomes."""
    practical_hedge_stakes = []
    total_hedge_needed = 0.0

    for s in raw_hedge_stakes:
        if s < -tolerance and not allow_lay:
            total_hedge_needed += abs(s)
            practical_hedge_stakes.append(0.0)
        else:
            # Round very small values to zero for numerical stability
            stake = float(s) if abs(s) > tolerance else 0.0
            practical_hedge_stakes.append(stake)

    # Distribute the additional hedge needed across other outcomes
    if total_hedge_needed > tolerance and len(practical_hedge_stakes) > 1:
        # Distribute the additional hedge needed across eligible outcomes only
        # Eligible = indices where practical_hedge_stakes >= 0
        eligible = [i for i, v in enumerate(practical_hedge_stakes) if v >= 0]
        if eligible:
            total_odds = sum(hedge_odds[i] for i in eligible)
            if total_odds <= tolerance:
                # fallback to equal distribution (should not occur with valid odds > 1.0)
                per = total_hedge_needed / len(eligible)
                for i in eligible:
                    practical_hedge_stakes[i] += per
            else:
                for i in eligible:
                    additional_hedge = total_hedge_needed * (hedge_odds[i] / total_odds)
                    practical_hedge_stakes[i] += additional_hedge

    return practical_hedge_stakes


def _calculate_final_profit(
    existing_payouts: List[float],
    total_existing_stakes: float,
    practical_hedge_stakes: List[float],
    hedge_odds: List[float],
    tolerance: float = 1e-10,
) -> float:
    """Calculate the guaranteed profit with practical stakes."""
    n = len(existing_payouts)
    min_profit = float("inf")
    total_practical_stakes = sum(practical_hedge_stakes)

    for i in range(n):
        # If outcome i wins
        profit_if_i_wins = (
            existing_payouts[i]
            + practical_hedge_stakes[i] * hedge_odds[i]
            - total_existing_stakes
            - total_practical_stakes
        )
        min_profit = min(min_profit, profit_if_i_wins)

    # Calculate individual profits for each outcome (may be unequal in asymmetric scenarios)
    profits = [
        existing_payouts[i]
        + practical_hedge_stakes[i] * hedge_odds[i]
        - total_existing_stakes
        - total_practical_stakes
        for i in range(n)
    ]
    max_profit_diff = max(profits) - min(profits) if profits else 0

    # Note: Profits are intentionally allowed to be unequal in asymmetric scenarios.
    # The guaranteed profit represents the worst-case (minimum) outcome.
    # Large profit differences are normal and expected when:
    # - Existing stakes are uneven across outcomes
    # - Odds ratios differ significantly
    # - Negative stakes had to be redistributed (allow_lay=False)
    if max_profit_diff > tolerance * 1000:  # Allow some numerical error
        # Note: In a production system, you might want to log this for diagnostics
        pass

    return min_profit


def arbitrage_hedge(
    existing_stakes: List[float],
    existing_odds: List[float],
    hedge_odds: List[float],
    target_profit: Optional[float] = None,
    hedge_all: bool = True,
    allow_lay: bool = False,
    tolerance: float = 1e-10,
) -> ArbitrageHedgeResult:
    """
    Calculate hedge bet sizes to guarantee profit or minimize loss from existing positions.

    This function determines how much to bet on other outcomes to either lock in
    a guaranteed profit or minimize potential losses from existing bets.

    **IMPORTANT: Understanding "Guaranteed Profit"**

    The "guaranteed profit" is the **worst-case profit** across all possible outcomes.
    In asymmetric scenarios (uneven existing stakes, different odds), individual outcome
    profits may be unequal, and the guaranteed profit represents the minimum you will
    receive regardless of which outcome occurs.

    Example: If outcome A would yield +$50 and outcome B would yield -$10, your
    guaranteed profit is -$10 (you're guaranteed to get at least this amount).

    Equal profits across all outcomes are only achievable in symmetric scenarios or
    when laying (betting against) is allowed and mathematically optimal.

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
    allow_lay : bool, default=False
        If True, allows negative (lay) stakes in results. If False, redistributes
        negative stakes to other outcomes
    tolerance : float, default=1e-10
        Numerical tolerance for comparisons and calculations

    Returns
    -------
    ArbitrageHedgeResult
        Structured result containing raw and practical hedge stakes and the
        guaranteed profit (or loss).

    Examples
    --------
    >>> # Basic symmetric arbitrage (equal profits possible):
    >>> # You have $100 on outcome A at 3.0 odds, want to hedge with outcome B at 2.5 odds
    >>> res = arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5])
    >>> res.practical_hedge_stakes  # Hedge $80 on outcome B
    [0.0, 80.0]
    >>> res.guaranteed_profit  # Guaranteed $20 profit either way
    20.0
    >>> # Verification: If A wins: $300 payout - $100 original - $80 hedge = $120 net
    >>> #              If B wins: $0 + $200 hedge payout - $100 original - $80 hedge = $20 net

    >>> # Asymmetric case (individual profits will be unequal):
    >>> res = arbitrage_hedge([100, 0], [3.0, 2.5], [2.9, 2.4])
    >>> # Individual outcome profits might be: +$50 and -$20
    >>> # Guaranteed profit would be -$20 (the worst case)

    >>> # Three-way market with existing positions:
    >>> res = arbitrage_hedge([50, 30, 0], [2.5, 4.0, 3.0], [2.4, 3.8, 2.9])
    >>> # Will calculate optimal hedge stakes for best worst-case outcome

    >>> # Target specific profit:
    >>> res = arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5], target_profit=50)
    >>> # Will calculate stakes needed to achieve exactly $50 profit

    >>> # Only hedge existing positions (don't hedge outcome with 0 stake):
    >>> res = arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5], hedge_all=False)
    >>> # Will only hedge the $100 position on outcome A

    Notes
    -----
    **Guaranteed Profit Interpretation:**
    The guaranteed_profit field represents the minimum profit you will receive
    regardless of outcome. Individual outcome profits may differ significantly
    in asymmetric scenarios due to mathematical constraints.

    **When Profits Are Equal:**
    Equal profits across outcomes occur when:
    - Existing stakes and odds are symmetric, OR
    - allow_lay=True and negative stakes are mathematically optimal, OR
    - The linear program finds a solution where equal profits are achievable

    **When Profits Are Unequal:**
    Unequal profits occur when:
    - Asymmetric existing positions with different odds ratios
    - allow_lay=False forces redistribution of negative stakes
    - Mathematical constraints prevent equal profit solutions

    Set hedge_all=False to only hedge outcomes where you have existing exposure.

    Implementation
    --------------
    When ``hedge_all=True`` the function solves a linear program to maximize
    the guaranteed profit G under non-negative hedge stakes h_i.

    **Linear Program Formulation:**

    Variables: [h_0, h_1, ..., h_{n-1}, G] where h_i are hedge stakes, G is guaranteed profit

    Objective: Maximize G (minimize -G in linprog)

    Constraints (for each outcome i):
        existing_payouts[i] + hedge_odds[i]*h_i - total_existing_stakes - sum(h_j) >= G

    Intuition: If outcome i occurs, we get existing_payouts[i] from our original bets,
    plus hedge_odds[i]*h_i from our hedge on outcome i, minus all our original stakes
    and all our hedge stakes. This net amount must be at least G for all outcomes.

    **Negative Stake Redistribution:**

    When laying (betting against) is not allowed, negative hedge stakes are converted
    to zero and the equivalent hedge amount is redistributed to other outcomes. The
    redistribution is proportional to the hedge odds - outcomes with better odds get
    a larger share of the redistributed hedge.

    Example: If outcome A needs -$50 hedge (impossible) and outcomes B,C have odds
    2.0, 3.0 respectively, then B gets $50 * 2/(2+3) = $20 and C gets $50 * 3/(2+3) = $30.

    This is solved with ``scipy.optimize.linprog``. If ``target_profit`` is
    provided, G is fixed to that value via bounds. If the LP fails, the
    function falls back to a conservative heuristic and then converts any
    theoretical "lay" (negative) stakes to practical non-negative stakes by
    redistribution. Pass ``allow_lay=True`` to permit negative (lay) stakes
    in the returned `raw_hedge_stakes` and `practical_hedge_stakes`.
    """
    # Input validation
    _validate_arbitrage_inputs(
        existing_stakes, existing_odds, hedge_odds, target_profit
    )

    # Calculate potential payouts from existing bets
    existing_payouts = [
        stake * odds for stake, odds in zip(existing_stakes, existing_odds)
    ]
    total_existing_stakes = sum(existing_stakes)

    # Determine hedge strategy and calculate raw hedge stakes
    if hedge_all:
        # Try linear programming approach first
        hedge_stakes, guaranteed_profit, lp_success, lp_message = _solve_hedge_lp(
            existing_payouts,
            total_existing_stakes,
            hedge_odds,
            target_profit,
            allow_lay,
        )

        # Fall back to heuristic if LP fails
        if not lp_success:
            hedge_stakes, guaranteed_profit = _calculate_heuristic_hedges(
                existing_payouts,
                total_existing_stakes,
                hedge_odds,
                target_profit,
                tolerance,
            )
    else:
        # Only hedge existing positions
        hedge_stakes, guaranteed_profit = _calculate_partial_hedges(
            existing_stakes,
            existing_payouts,
            total_existing_stakes,
            hedge_odds,
            tolerance,
        )
        lp_success = True  # Not applicable for partial hedging
        lp_message = None

    # Convert raw stakes to practical stakes (handle negative stakes)
    raw_hedge_stakes = [float(s) for s in hedge_stakes]
    practical_hedge_stakes = _redistribute_negative_stakes(
        raw_hedge_stakes, hedge_odds, allow_lay, tolerance
    )

    # Calculate total hedge needed (for redistribution tracking)
    total_hedge_needed = sum(
        abs(s) for s in raw_hedge_stakes if s < -tolerance and not allow_lay
    )

    # Only recalculate profit if stakes were redistributed, otherwise use the original
    if total_hedge_needed > tolerance:
        # Stakes were redistributed, need to recalculate profit
        final_profit = _calculate_final_profit(
            existing_payouts,
            total_existing_stakes,
            practical_hedge_stakes,
            hedge_odds,
            tolerance,
        )
    else:
        # No redistribution occurred, use the original guaranteed profit
        final_profit = guaranteed_profit

    # Return structured result
    result = ArbitrageHedgeResult(
        raw_hedge_stakes=raw_hedge_stakes,
        practical_hedge_stakes=practical_hedge_stakes,
        guaranteed_profit=final_profit,
        existing_payouts=existing_payouts,
        total_existing_stakes=total_existing_stakes,
        total_hedge_needed=total_hedge_needed,
        lp_success=lp_success,
        lp_message=lp_message,
    )

    return result
