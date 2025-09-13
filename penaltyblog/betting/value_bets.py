"""
Value Bet Identification

Functions for identifying value bets by comparing bookmaker odds to estimated true probabilities.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class ValueBetResult:
    """Result of value bet analysis for a single bet."""

    bookmaker_odds: float
    estimated_probability: float
    implied_probability: float
    expected_value: float
    expected_return_percentage: float
    is_value_bet: bool
    edge: float

    # Kelly recommendation based on value
    recommended_stake_kelly: float
    recommended_stake_fraction: float

    # Risk metrics
    win_probability: float
    lose_probability: float
    potential_profit: float
    potential_loss: float

    # Metadata
    margin_over_fair_odds: float
    overround_contribution: float


@dataclass
class MultipleValueBetResult:
    """Result of value bet analysis for multiple bets."""

    individual_results: List[ValueBetResult]
    bookmaker_odds: List[float]
    estimated_probabilities: List[float]

    # Portfolio metrics
    total_value_bets: int
    average_edge: float
    total_expected_value: float
    portfolio_overround: float

    # Kelly recommendations for portfolio
    kelly_stakes: List[float]
    total_kelly_stake: float
    portfolio_expected_return: float

    # Summary statistics
    best_value_bet_index: int
    best_edge: float
    worst_edge: float


@dataclass
class ArbitrageResult:
    """Result of arbitrage opportunity analysis across multiple bookmakers."""

    has_arbitrage: bool
    total_implied_probability: float
    guaranteed_return: float
    arbitrage_margin: float

    # Best odds information
    best_odds: List[float]
    best_bookmakers: List[int]

    # Stake allocation
    stake_percentages: List[float]

    # Metadata
    outcome_labels: List[str]
    num_bookmakers: int
    num_outcomes: int


def _calculate_implied_probability(decimal_odds: float) -> float:
    """Calculate implied probability from decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def _calculate_expected_value(decimal_odds: float, estimated_prob: float) -> float:
    """Calculate expected value of a bet."""
    return (estimated_prob * (decimal_odds - 1)) - (1 - estimated_prob)


def _calculate_kelly_stake(decimal_odds: float, estimated_prob: float) -> float:
    """Calculate optimal Kelly stake for a value bet."""
    edge = _calculate_expected_value(decimal_odds, estimated_prob)
    if edge <= 0:
        return 0.0
    return edge / (decimal_odds - 1)


def identify_value_bet(
    bookmaker_odds: Union[float, List[float], NDArray],
    estimated_probability: Union[float, List[float], NDArray],
    kelly_fraction: float = 1.0,
    min_edge_threshold: float = 0.0,
) -> Union[ValueBetResult, MultipleValueBetResult]:
    """
    Identify value bets by comparing bookmaker odds to estimated true probabilities.

    A value bet occurs when your estimated probability of an outcome is higher than
    the bookmaker's implied probability, creating positive expected value.

    Parameters
    ----------
    bookmaker_odds : float | list[float] | np.ndarray
        Bookmaker odds in decimal format (e.g., 2.0 for even money)
    estimated_probability : float | list[float] | np.ndarray
        Your estimated true probability of the outcome (0-1)
    kelly_fraction : float, default=1.0
        Fraction of optimal Kelly stake to recommend (e.g., 0.5 for half Kelly)
    min_edge_threshold : float, default=0.0
        Minimum edge required to consider a bet as having value

    Returns
    -------
    ValueBetResult | MultipleValueBetResult
        Comprehensive analysis of value betting opportunities

    Examples
    --------
    >>> # Single value bet analysis
    >>> result = identify_value_bet(2.5, 0.50)  # 50% chance, 2.5 odds
    >>> print(f"Expected value: {result.expected_value:.3f}")
    >>> print(f"Kelly stake: {result.recommended_stake_kelly:.2%}")

    >>> # Multiple bet analysis
    >>> odds = [2.0, 3.0, 1.8]
    >>> probs = [0.6, 0.4, 0.5]
    >>> results = identify_value_bet(odds, probs)
    >>> print(f"Found {results.total_value_bets} value bets")

    Raises
    ------
    ValueError
        If odds <= 1.0, probabilities outside [0,1], or mismatched array lengths
    """
    # Convert inputs to numpy arrays for consistent handling
    odds_array = np.asarray(bookmaker_odds)
    prob_array = np.asarray(estimated_probability)

    # Input validation
    if np.any(odds_array <= 1.0):
        raise ValueError("All bookmaker odds must be greater than 1.0")

    if np.any(prob_array < 0) or np.any(prob_array > 1):
        raise ValueError("All estimated probabilities must be between 0 and 1")

    # Handle scalar vs array inputs
    is_scalar = odds_array.ndim == 0 and prob_array.ndim == 0

    if is_scalar:
        odds_array = odds_array.flatten()
        prob_array = prob_array.flatten()

    # Check array lengths match
    if odds_array.shape != prob_array.shape:
        raise ValueError(
            "bookmaker_odds and estimated_probability must have same length"
        )

    # Flatten arrays for consistent processing
    odds_flat = odds_array.flatten()
    prob_flat = prob_array.flatten()

    individual_results = []

    for i, (odds, prob) in enumerate(zip(odds_flat, prob_flat)):
        implied_prob = _calculate_implied_probability(odds)
        expected_value = _calculate_expected_value(odds, prob)
        expected_return_pct = expected_value * 100
        edge = prob - implied_prob
        is_value = edge > min_edge_threshold

        # Kelly calculations
        kelly_stake = _calculate_kelly_stake(odds, prob)
        recommended_stake = kelly_stake * kelly_fraction

        # Risk metrics
        potential_profit = odds - 1  # Profit per unit staked if win
        potential_loss = 1.0  # Loss per unit staked if lose

        # Additional metrics
        fair_odds = 1.0 / prob if prob > 0 else float("inf")
        margin_over_fair = (
            (odds - fair_odds) / fair_odds if fair_odds != float("inf") else 0
        )

        result = ValueBetResult(
            bookmaker_odds=float(odds),
            estimated_probability=float(prob),
            implied_probability=float(implied_prob),
            expected_value=float(expected_value),
            expected_return_percentage=float(expected_return_pct),
            is_value_bet=bool(is_value),
            edge=float(edge),
            recommended_stake_kelly=float(kelly_stake),
            recommended_stake_fraction=float(recommended_stake),
            win_probability=float(prob),
            lose_probability=float(1 - prob),
            potential_profit=float(potential_profit),
            potential_loss=float(potential_loss),
            margin_over_fair_odds=float(margin_over_fair),
            overround_contribution=float(implied_prob),
        )

        individual_results.append(result)

    if is_scalar:
        return individual_results[0]

    # Portfolio-level analysis
    total_value_bets = sum(1 for r in individual_results if r.is_value_bet)

    # Calculate average edge only for value bets
    value_bet_edges = [r.edge for r in individual_results if r.is_value_bet]
    avg_edge = np.mean(value_bet_edges) if value_bet_edges else 0.0

    total_ev = sum(r.expected_value for r in individual_results)

    # Portfolio overround (sum of implied probabilities)
    portfolio_overround = sum(r.implied_probability for r in individual_results)

    # Kelly stakes for the portfolio
    kelly_stakes = [r.recommended_stake_fraction for r in individual_results]
    total_kelly_stake = sum(kelly_stakes)

    # Portfolio expected return (weighted by stakes)
    if total_kelly_stake > 0:
        portfolio_expected_return = (
            sum(
                r.recommended_stake_fraction * r.expected_return_percentage
                for r in individual_results
            )
            / total_kelly_stake
        )
    else:
        portfolio_expected_return = 0.0

    # Find best and worst edges
    all_edges = [r.edge for r in individual_results]
    best_edge = max(all_edges) if all_edges else 0.0
    worst_edge = min(all_edges) if all_edges else 0.0
    best_value_bet_index = int(np.argmax(all_edges)) if all_edges else 0

    return MultipleValueBetResult(
        individual_results=individual_results,
        bookmaker_odds=odds_flat.tolist(),
        estimated_probabilities=prob_flat.tolist(),
        total_value_bets=total_value_bets,
        average_edge=float(avg_edge),
        total_expected_value=float(total_ev),
        portfolio_overround=float(portfolio_overround),
        kelly_stakes=kelly_stakes,
        total_kelly_stake=float(total_kelly_stake),
        portfolio_expected_return=float(portfolio_expected_return),
        best_value_bet_index=best_value_bet_index,
        best_edge=float(best_edge),
        worst_edge=float(worst_edge),
    )


def calculate_bet_value(bookmaker_odds: float, estimated_probability: float) -> float:
    """
    Calculate the expected value of a bet as a simple utility function.

    Parameters
    ----------
    bookmaker_odds : float
        Decimal odds from bookmaker
    estimated_probability : float
        Your estimated probability (0-1)

    Returns
    -------
    float
        Expected value per unit staked

    Examples
    --------
    >>> value = calculate_bet_value(2.0, 0.6)  # 60% chance at 2.0 odds
    >>> print(f"Expected value: {value:.3f}")  # 0.200
    """
    if bookmaker_odds <= 1.0:
        raise ValueError("Bookmaker odds must be greater than 1.0")

    if not (0 <= estimated_probability <= 1):
        raise ValueError("Estimated probability must be between 0 and 1")

    return _calculate_expected_value(bookmaker_odds, estimated_probability)


def find_arbitrage_opportunities(
    bookmaker_odds_list: List[List[float]], outcome_labels: Optional[List[str]] = None
) -> ArbitrageResult:
    """
    Find arbitrage opportunities across multiple bookmakers for the same event.

    An arbitrage opportunity exists when you can bet on all outcomes across
    different bookmakers and guarantee a profit regardless of the result.

    Parameters
    ----------
    bookmaker_odds_list : List[List[float]]
        List of odds from each bookmaker. Each inner list contains odds for all outcomes.
        Example: [[2.1, 1.9], [2.0, 2.0]] for two bookmakers with two outcomes each.
    outcome_labels : List[str], optional
        Labels for each outcome (e.g., ["Home", "Away"])

    Returns
    -------
    ArbitrageResult
        Structured result containing arbitrage analysis:
        - has_arbitrage: bool indicating if arbitrage exists
        - total_implied_probability: float (< 1.0 indicates arbitrage)
        - best_odds: list of best odds for each outcome
        - best_bookmakers: list of bookmaker indices offering best odds
        - stake_percentages: recommended stake allocation
        - guaranteed_return: guaranteed profit percentage

    Examples
    --------
    >>> # Two bookmakers, two outcomes
    >>> odds = [[2.1, 1.85], [1.95, 2.0]]
    >>> arb = find_arbitrage_opportunities(odds, ["Home", "Away"])
    >>> if arb.has_arbitrage:
    ...     print(f"Guaranteed return: {arb.guaranteed_return:.2%}")
    """
    if not bookmaker_odds_list:
        return ArbitrageResult(
            has_arbitrage=False,
            total_implied_probability=0.0,
            guaranteed_return=0.0,
            arbitrage_margin=0.0,
            best_odds=[],
            best_bookmakers=[],
            stake_percentages=[],
            outcome_labels=outcome_labels or [],
            num_bookmakers=0,
            num_outcomes=0,
        )

    # Validate all bookmakers have same number of outcomes
    n_outcomes = len(bookmaker_odds_list[0])
    if not all(len(odds) == n_outcomes for odds in bookmaker_odds_list):
        raise ValueError(
            "All bookmakers must have odds for the same number of outcomes"
        )

    # Validate all odds are > 1.0
    for i, bookmaker_odds in enumerate(bookmaker_odds_list):
        if any(odds <= 1.0 for odds in bookmaker_odds):
            raise ValueError(f"All odds must be > 1.0 (bookmaker {i} has invalid odds)")

    # Find best odds for each outcome
    best_odds = []
    best_bookmakers = []

    for outcome_idx in range(n_outcomes):
        outcome_odds = [bookmaker[outcome_idx] for bookmaker in bookmaker_odds_list]
        best_odd = max(outcome_odds)
        best_bookmaker = outcome_odds.index(best_odd)

        best_odds.append(best_odd)
        best_bookmakers.append(best_bookmaker)

    # Calculate total implied probability using best odds
    implied_probs = [1.0 / odds for odds in best_odds]
    total_implied_prob = sum(implied_probs)

    # Check if arbitrage exists
    has_arbitrage = total_implied_prob < 1.0

    if has_arbitrage:
        # Calculate optimal stake allocation
        stake_percentages = [prob / total_implied_prob for prob in implied_probs]
        guaranteed_return = (1.0 / total_implied_prob) - 1.0
    else:
        stake_percentages = [0.0] * n_outcomes
        guaranteed_return = 0.0

    # Create outcome labels if not provided
    if outcome_labels is None:
        outcome_labels = [f"Outcome_{i+1}" for i in range(n_outcomes)]

    # Calculate arbitrage margin
    arbitrage_margin = 1.0 - total_implied_prob if has_arbitrage else 0.0

    return ArbitrageResult(
        has_arbitrage=has_arbitrage,
        total_implied_probability=total_implied_prob,
        guaranteed_return=guaranteed_return,
        arbitrage_margin=arbitrage_margin,
        best_odds=best_odds,
        best_bookmakers=best_bookmakers,
        stake_percentages=stake_percentages,
        outcome_labels=outcome_labels,
        num_bookmakers=len(bookmaker_odds_list),
        num_outcomes=n_outcomes,
    )
