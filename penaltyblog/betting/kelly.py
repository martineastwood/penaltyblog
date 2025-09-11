"""
Kelly Criterion

Calculates the Kelly Criterion for a given set of odds and probabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class RiskMetrics:
    """Comprehensive risk and return metrics for a betting strategy."""

    expected_profit: float
    expected_return: float
    kelly_growth_rate: float
    wealth_volatility: float
    log_return_volatility: float
    sharpe_ratio: float
    win_probability: float
    probability_of_ruin: float
    value_at_risk_95: float
    max_loss: float
    total_exposure: float


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation for a single bet."""

    stake: Union[float, NDArray]
    expected_growth: Union[float, NDArray]
    edge: Union[float, NDArray]
    is_favorable: Union[bool, NDArray]
    risk_of_ruin: Union[float, NDArray]
    risk_metrics: Optional[RiskMetrics]

    # Metadata
    decimal_odds: Union[float, NDArray]
    true_prob: Union[float, NDArray]
    fraction: float
    warnings: List[str]


@dataclass
class MultipleKellyResult:
    """Result of Kelly criterion calculation for multiple bets."""

    stakes: List[float]
    total_stake: float
    expected_growth: float
    expected_return: float
    portfolio_edge: float
    risk_metrics: RiskMetrics

    # Optimization details
    method: str
    optimization_success: bool
    optimization_details: Optional[Dict[str, Any]]

    # Input metadata
    decimal_odds: List[float]
    true_probs: List[float]
    fraction: float
    max_total_stake: float
    warnings: List[str]


def _validate_inputs(
    decimal_odds: Union[float, NDArray, List[float]],
    true_prob: Union[float, NDArray, List[float]],
    tolerance: float = 1e-10,
) -> List[str]:
    """
    Validate odds and probability inputs for Kelly calculations.

    Parameters
    ----------
    decimal_odds : float | np.ndarray | list[float]
        Odds in European decimal format (> 1.0). Scalar or array-like.
        If arrays are provided, they will be converted to ``np.ndarray``.
    true_prob : float | np.ndarray | list[float]
        True probability for each outcome in the range [0, 1]. Scalar or array-like.
    tolerance : float, default=1e-10
        Numerical tolerance used to guard against values that would cause
        numerical issues (e.g., odds extremely close to 1.0).

    Returns
    -------
    list[str]
        A list of non-fatal warnings describing potential data quality issues
        (e.g., very low/high probabilities, non-profitable odds). An empty list
        indicates no warnings.

    Raises
    ------
    ValueError
        If any probability is outside [0, 1], or any odds value equals 1.0
        within ``tolerance`` (division by zero risk).

    Notes
    -----
    - Inputs are converted to ``np.ndarray`` for validation. Shapes are not
      strictly enforced here; broadcasting is handled by the calling functions.
    - Odds <= 1.0 are flagged as warnings (no profit potential) but are not
      fatal unless exactly 1.0 within ``tolerance``.
    """
    warnings = []

    # Convert to numpy arrays for easier validation
    odds = np.asarray(decimal_odds)
    probs = np.asarray(true_prob)

    # Check for invalid odds
    if np.any(odds <= 1.0):
        warnings.append("Some odds are <= 1.0, which indicates no profit potential")

    # Check for invalid probabilities
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    # Check for division by zero cases
    if np.any(np.abs(odds - 1.0) < tolerance):
        raise ValueError("Odds cannot be exactly 1.0 (would cause division by zero)")

    # Check for very low probabilities that might indicate data issues
    if np.any((probs > 0) & (probs < 0.01)):
        warnings.append("Some probabilities are very low (< 1%), verify data quality")

    # Check for very high probabilities that might indicate overconfidence
    if np.any(probs > 0.95):
        warnings.append("Some probabilities are very high (> 95%), verify confidence")

    return warnings


def _calculate_risk_metrics(
    stakes: Union[float, NDArray, List[float]],
    decimal_odds: Union[float, NDArray, List[float]],
    true_prob: Union[float, NDArray, List[float]],
) -> RiskMetrics:
    """
    Calculate comprehensive risk and return metrics for a betting strategy.

    This function properly handles mutually exclusive outcomes (only one can occur)
    and calculates theoretically sound risk measures based on final wealth distribution.

    Parameters
    ----------
    stakes : float | np.ndarray | list[float]
        Fraction(s) of bankroll staked per outcome
    decimal_odds : float | np.ndarray | list[float]
        European decimal odds for each outcome
    true_prob : float | np.ndarray | list[float]
        True probabilities for each outcome

    Returns
    -------
    dict[str, float]
        Comprehensive risk metrics including:
        - expected_profit: Expected profit from the strategy
        - expected_return: Expected return on staked capital
        - kelly_growth_rate: Expected logarithmic growth (what Kelly optimizes)
        - wealth_volatility: Standard deviation of final wealth (in wealth units)
        - log_return_volatility: Standard deviation of log returns (dimensionless)
        - sharpe_ratio: Kelly growth rate divided by log-return volatility
        - win_probability: Probability of making any profit
        - probability_of_ruin: Probability of losing all stakes this round
        - value_at_risk_95: 95th percentile potential loss
        - max_loss: Maximum possible loss (worst case)
        - total_exposure: Total fraction of bankroll at risk
    """
    stakes = np.asarray(stakes, dtype=float)
    odds = np.asarray(decimal_odds, dtype=float)
    probs = np.asarray(true_prob, dtype=float)

    # Handle scalar case
    if stakes.ndim == 0:
        stakes = np.array([stakes])
        odds = np.array([odds])
        probs = np.array([probs])

    total_stake = np.sum(stakes)

    # Calculate all possible final wealth outcomes
    final_wealth_outcomes = []
    outcome_probabilities = []

    if len(stakes) > 1:
        # Multiple mutually exclusive outcomes
        # Add outcome where none of the specified events occur
        prob_no_outcome = max(0.0, 1.0 - np.sum(probs))

        # Each specified outcome occurs
        for i, (stake, odd, prob) in enumerate(zip(stakes, odds, probs)):
            if prob > 0:
                # If outcome i wins: keep remaining bankroll + winnings from bet i
                final_wealth = 1.0 - total_stake + stake * odd
                final_wealth_outcomes.append(final_wealth)
                outcome_probabilities.append(prob)

        # No specified outcome occurs (lose all stakes)
        if prob_no_outcome > 0:
            final_wealth = 1.0 - total_stake  # Lose all stakes
            final_wealth_outcomes.append(final_wealth)
            outcome_probabilities.append(prob_no_outcome)

    else:
        # Single outcome case
        prob = probs[0]
        stake = stakes[0]
        odd = odds[0]

        # Win case
        final_wealth_win = 1.0 - stake + stake * odd
        final_wealth_outcomes.append(final_wealth_win)
        outcome_probabilities.append(prob)

        # Lose case
        final_wealth_lose = 1.0 - stake
        final_wealth_outcomes.append(final_wealth_lose)
        outcome_probabilities.append(1.0 - prob)

    final_wealth_outcomes = np.array(final_wealth_outcomes)
    outcome_probabilities = np.array(outcome_probabilities)

    # Normalize probabilities (should sum to 1)
    prob_sum = np.sum(outcome_probabilities)
    if prob_sum > 0:
        outcome_probabilities = outcome_probabilities / prob_sum

    # Calculate expected final wealth and profit
    expected_wealth = np.sum(final_wealth_outcomes * outcome_probabilities)
    expected_profit = (
        expected_wealth - 1.0
    )  # Profit = final_wealth - initial_wealth(1.0)

    # Expected return on staked capital
    expected_return = expected_profit / total_stake if total_stake > 0 else 0.0

    # Calculate log wealth once (used for both Kelly growth rate and log-return volatility)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_wealth = np.log(final_wealth_outcomes)
        log_wealth = np.where(np.isfinite(log_wealth), log_wealth, -np.inf)

    # Kelly growth rate (expected log of final wealth)
    kelly_growth_rate = np.sum(outcome_probabilities * log_wealth)

    # Volatility (standard deviation of final wealth)
    wealth_variance = np.sum(
        outcome_probabilities * (final_wealth_outcomes - expected_wealth) ** 2
    )
    volatility = np.sqrt(wealth_variance)

    # Log-return volatility (standard deviation of log returns for Kelly)
    log_return_variance = np.sum(
        outcome_probabilities * (log_wealth - kelly_growth_rate) ** 2
    )
    log_return_volatility = np.sqrt(log_return_variance)

    # Sharpe ratio (log return over log return volatility - same units)
    sharpe_ratio = (
        kelly_growth_rate / log_return_volatility if log_return_volatility > 0 else 0.0
    )

    # Win probability (probability of any profit)
    win_probability = np.sum(outcome_probabilities[final_wealth_outcomes > 1.0])

    # Probability of ruin (losing all staked capital this round)
    # This is probability that final wealth <= (1 - total_stake)
    ruin_threshold = 1.0 - total_stake
    probability_of_ruin = np.sum(
        outcome_probabilities[final_wealth_outcomes <= ruin_threshold]
    )

    # Value at Risk (95th percentile loss)
    # Calculate 5th percentile of wealth distribution
    sorted_indices = np.argsort(final_wealth_outcomes)
    cumulative_probs = np.cumsum(outcome_probabilities[sorted_indices])
    var_95_idx = np.searchsorted(cumulative_probs, 0.05)
    if var_95_idx < len(final_wealth_outcomes):
        wealth_5th_percentile = final_wealth_outcomes[sorted_indices[var_95_idx]]
        value_at_risk_95 = 1.0 - wealth_5th_percentile  # Loss = 1 - final_wealth
    else:
        value_at_risk_95 = 0.0

    # Maximum loss (worst case scenario)
    min_final_wealth = np.min(final_wealth_outcomes)
    max_loss = 1.0 - min_final_wealth

    return RiskMetrics(
        expected_profit=float(expected_profit),
        expected_return=float(expected_return),
        kelly_growth_rate=float(kelly_growth_rate),
        wealth_volatility=float(volatility),
        log_return_volatility=float(log_return_volatility),
        sharpe_ratio=float(sharpe_ratio),
        win_probability=float(win_probability),
        probability_of_ruin=float(probability_of_ruin),
        value_at_risk_95=float(value_at_risk_95),
        max_loss=float(max_loss),
        total_exposure=float(total_stake),
    )


def kelly_criterion(
    decimal_odds: Union[float, NDArray, List[float]],
    true_prob: Union[float, NDArray, List[float]],
    fraction: float = 1.0,
) -> KellyResult:
    """
    Calculate optimal bet size using the Kelly Criterion with comprehensive analysis.

    This function provides a complete Kelly Criterion analysis including stake recommendations,
    expected growth rates, edge calculations, and risk metrics with robust input validation.

    Parameters
    ----------
    decimal_odds : float | np.ndarray | list[float]
        The odds in European decimal format (e.g., 1.50 for 50% implied probability).
        Scalar or array-like. Lists are accepted and will be converted to ``np.ndarray``.
    true_prob : float | np.ndarray | list[float]
        The true probability of the event (0-1). Scalar or array-like. Lists are accepted
        and will be converted to ``np.ndarray``.
    fraction : float, default=1.0
        Fraction of optimal Kelly to use (e.g., 0.5 for Half Kelly conservative betting)

    Returns
    -------
    KellyResult
        Comprehensive result object containing:
        - stake: Recommended fraction of bankroll to wager
        - expected_growth: Expected logarithmic growth rate
        - edge: Betting edge (expected value)
        - is_favorable: Whether the bet has positive expected value
        - risk_of_ruin: Simplified probability of losing stake
        - risk_metrics: Comprehensive risk analysis (for scalar inputs only)
        - warnings: List of data quality or configuration warnings

    Examples
    --------
    >>> # Single bet analysis
    >>> result = kelly_criterion(2.1, 0.55, fraction=0.5)
    >>> print(f"Stake: {result.stake:.2%}, Expected Growth: {result.expected_growth:.4%}")

    >>> # Array of bets
    >>> odds = np.array([2.0, 1.8, 3.0])
    >>> probs = np.array([0.6, 0.5, 0.4])
    >>> results = kelly_criterion(odds, probs)

    Raises
    ------
    ValueError
        If probabilities are outside [0,1] or odds are exactly 1.0
    """
    # Input validation
    input_warnings = _validate_inputs(decimal_odds, true_prob)

    # Convert to numpy arrays for consistent type handling
    odds_array = np.asarray(decimal_odds)
    prob_array = np.asarray(true_prob)

    # Calculate basic Kelly criterion
    edge = (prob_array * odds_array) - 1
    kelly_fraction = edge / (odds_array - 1)
    stake = np.clip(kelly_fraction * fraction, 0, 1)

    # Calculate additional metrics
    is_favorable = edge > 0

    # Expected log growth rate
    # E[log(1 + f * (b * p - q))] where f=stake, b=odds-1, p=true_prob, q=1-p
    stake_array = np.asarray(stake)

    # Expected growth calculation
    win_outcome = 1 + stake_array * (odds_array - 1)
    lose_outcome = 1 - stake_array

    # Handle potential log(0) cases
    with np.errstate(divide="ignore", invalid="ignore"):
        win_log = np.log(win_outcome)
        lose_log = np.log(lose_outcome)
        # Replace -inf with 0 (no growth if outcome leads to ruin)
        win_log = np.where(np.isfinite(win_log), win_log, 0)
        lose_log = np.where(np.isfinite(lose_log), lose_log, 0)

    expected_growth = prob_array * win_log + (1 - prob_array) * lose_log

    # Risk of ruin approximation (simplified)
    # This is the probability of losing the entire stake
    risk_of_ruin = 1 - prob_array if np.any(stake_array > 0) else 0.0

    # Calculate comprehensive risk metrics for single bet
    # Only calculate risk metrics for scalar inputs to maintain consistency
    risk_metrics = None
    if np.isscalar(decimal_odds) and np.isscalar(true_prob) and np.any(stake_array > 0):
        try:
            risk_metrics = _calculate_risk_metrics(stake, decimal_odds, true_prob)
        except Exception as e:
            input_warnings.append(f"Risk metrics calculation failed: {str(e)}")

    return KellyResult(
        stake=stake,
        expected_growth=expected_growth,
        edge=edge,
        is_favorable=is_favorable,
        risk_of_ruin=risk_of_ruin,
        risk_metrics=risk_metrics,
        decimal_odds=decimal_odds,
        true_prob=true_prob,
        fraction=fraction,
        warnings=input_warnings,
    )


def multiple_kelly_criterion(
    decimal_odds: Union[List[float], NDArray],
    true_probs: Union[List[float], NDArray],
    fraction: float = 1.0,
    max_total_stake: float = 1.0,
    method: Literal["simultaneous", "independent"] = "simultaneous",
    optimization_methods: List[Literal["SLSQP", "trust-constr"]] = [
        "SLSQP",
        "trust-constr",
    ],
    tolerance: float = 1e-10,
) -> MultipleKellyResult:
    """
    Calculate optimal portfolio bet sizes using Kelly Criterion with comprehensive analysis.

    This function handles portfolio optimization across multiple bets, using either simultaneous
    optimization for maximum expected log growth or independent Kelly calculations. Includes
    comprehensive risk analysis and robust optimization with multiple fallback methods.

    Parameters
    ----------
    decimal_odds : list[float] | np.ndarray
        Odds in European decimal format for each outcome. Lists or arrays are accepted
        and will be coerced to ``np.ndarray``.
    true_probs : list[float] | np.ndarray
        True probabilities for each outcome (should sum to ≤ 1.0). Lists or arrays are
        accepted and will be coerced to ``np.ndarray``.
    fraction : float, default=1.0
        Fraction of optimal Kelly to use (e.g., 0.5 for Half Kelly conservative betting)
    max_total_stake : float, default=1.0
        Maximum fraction of bankroll to stake across all bets
    method : {"simultaneous", "independent"}, default="simultaneous"
        Method to use: "simultaneous" for portfolio optimization,
        "independent" for independent Kelly calculations
    optimization_methods : list of {"SLSQP", "trust-constr"}, default=["SLSQP", "trust-constr"]
        List of optimization methods to try in order
    tolerance : float, default=1e-10
        Numerical tolerance for calculations

    Returns
    -------
    MultipleKellyResult
        Comprehensive result object containing:
        - stakes: List of recommended stakes for each outcome
        - total_stake: Total fraction of bankroll to stake
        - expected_growth: Expected logarithmic growth rate
        - expected_return: Expected return on investment
        - portfolio_edge: Weighted average betting edge
        - risk_metrics: Dict with volatility, Sharpe ratio, etc.
        - optimization details and warnings

    Examples
    --------
    >>> # Three-way football market
    >>> result = multiple_kelly_criterion([2.5, 3.2, 2.8], [0.45, 0.30, 0.25])
    >>> print(f"Stakes: {[f'{s:.2%}' for s in result.stakes]}")
    >>> print(f"Total: {result.total_stake:.2%}, Growth: {result.expected_growth:.4%}")

    >>> # Conservative Half Kelly approach
    >>> result = multiple_kelly_criterion([2.1, 1.9], [0.50, 0.48], fraction=0.5)

    Raises
    ------
    ValueError
        If array lengths don't match, probabilities are invalid, or sum exceeds 1.0
    """
    # Input validation
    if len(decimal_odds) != len(true_probs):
        raise ValueError("decimal_odds and true_probs must have the same length")

    if not all(p >= 0 for p in true_probs):
        raise ValueError("All probabilities must be non-negative")

    if sum(true_probs) > 1.0 + tolerance:
        raise ValueError("Sum of probabilities cannot exceed 1.0")

    input_warnings = []
    for i, (odds, prob) in enumerate(zip(decimal_odds, true_probs)):
        warnings_for_bet = _validate_inputs(odds, prob, tolerance)
        input_warnings.extend([f"Bet {i+1}: {w}" for w in warnings_for_bet])

    optimization_success = False
    optimization_details = {}
    stakes = []

    if method == "independent":
        # Calculate independent Kelly for each bet
        for odds, prob in zip(decimal_odds, true_probs):
            # Calculate basic Kelly criterion inline
            edge = (prob * odds) - 1
            kelly_fraction = edge / (odds - 1)
            kelly = float(np.clip(kelly_fraction, 0, 1))  # Don't apply fraction yet
            stakes.append(kelly)

        # Scale down if total exceeds max_total_stake
        total_stake = sum(stakes)
        if total_stake > max_total_stake:
            stakes = [s * (max_total_stake / total_stake) for s in stakes]

        # Apply fraction scaling
        stakes = [s * fraction for s in stakes]
        optimization_success = True
        optimization_details = {
            "method": "independent",
            "scaling_applied": total_stake > max_total_stake,
        }

    elif method == "simultaneous":
        n = len(decimal_odds)

        def objective(stakes_opt):
            """Negative expected log growth (to minimize)"""
            stakes_arr = np.array(stakes_opt)
            remaining_bankroll = 1.0 - np.sum(stakes_arr)

            if remaining_bankroll <= 1e-12:  # More conservative threshold
                return 1e8  # Large penalty for invalid stakes

            expected_log_growth = 0.0

            # Case where no outcome occurs (probability = 1 - sum(true_probs))
            prob_no_outcome = max(0, 1.0 - sum(true_probs))
            if prob_no_outcome > tolerance:
                final_bankroll = remaining_bankroll  # Lose all stakes
                if final_bankroll > 1e-12:
                    expected_log_growth += prob_no_outcome * np.log(final_bankroll)
                else:
                    return 1e8  # Avoid log(0)

            # Cases where each outcome occurs
            for i, (odds, prob) in enumerate(zip(decimal_odds, true_probs)):
                if prob > tolerance:
                    winnings = stakes_arr[i] * odds
                    final_bankroll = remaining_bankroll + winnings
                    if final_bankroll > 1e-12:
                        expected_log_growth += prob * np.log(final_bankroll)
                    else:
                        return 1e8  # Avoid log(0)

            return -expected_log_growth  # Negative because we minimize

        # Constraints
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: max_total_stake - sum(x),
            },  # Total stake <= max
            {
                "type": "ineq",
                "fun": lambda x: 0.99 - sum(x),
            },  # Keep some bankroll (more conservative)
        ]

        # Bounds: each stake between 0 and max_total_stake
        bounds = [
            (0, min(max_total_stake, 0.5)) for _ in range(n)
        ]  # Cap individual bets

        # Initial guess: independent Kelly scaled down
        initial_stakes = []
        for odds, prob in zip(decimal_odds, true_probs):
            # Calculate basic Kelly criterion inline
            edge = (prob * odds) - 1
            kelly_fraction = edge / (odds - 1)
            kelly = float(np.clip(kelly_fraction, 0, 1))
            initial_stakes.append(
                min(kelly * 0.5, max_total_stake / (2 * n))
            )  # More conservative initial guess

        # Try multiple optimization methods
        best_result = None
        for opt_method in optimization_methods:
            try:
                # Set appropriate options for each method
                if opt_method == "trust-constr":
                    options = {"xtol": tolerance, "disp": False}
                else:
                    options = {"ftol": tolerance, "disp": False}

                result = minimize(
                    objective,
                    initial_stakes,
                    method=opt_method,
                    bounds=bounds,
                    constraints=constraints,
                    options=options,
                )

                if result.success and (
                    best_result is None or result.fun < best_result.fun
                ):
                    best_result = result
                    optimization_success = True

            except Exception as e:
                input_warnings.append(
                    f"Optimization method {opt_method} failed: {str(e)}"
                )
                continue

        if best_result and best_result.success:
            stakes = best_result.x.tolist()
            # Apply fraction scaling
            stakes = [s * fraction for s in stakes]
            optimization_details = {
                "method": "simultaneous",
                "optimizer_result": {
                    "success": best_result.success,
                    "message": best_result.message,
                    "nit": getattr(best_result, "nit", None),
                    "fun": best_result.fun,
                },
            }
        else:
            # Fallback to independent method if all optimization methods fail
            input_warnings.append(
                "Simultaneous optimization failed, falling back to independent method"
            )
            return multiple_kelly_criterion(
                decimal_odds,
                true_probs,
                fraction,
                max_total_stake,
                "independent",
                optimization_methods,
                tolerance,
            )

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'simultaneous' or 'independent'"
        )

    # Calculate comprehensive metrics
    total_stake = sum(stakes)
    risk_metrics = _calculate_risk_metrics(stakes, decimal_odds, true_probs)

    # Portfolio edge (weighted average)
    edges = [(prob * odds - 1) for odds, prob in zip(decimal_odds, true_probs)]
    portfolio_edge = sum(stake * edge for stake, edge in zip(stakes, edges)) / max(
        total_stake, 1e-10
    )

    # Expected log growth for the portfolio
    remaining_bankroll = 1.0 - total_stake
    expected_growth = 0.0

    # No outcome case
    prob_no_outcome = max(0, 1.0 - sum(true_probs))
    if prob_no_outcome > tolerance and remaining_bankroll > 1e-10:
        expected_growth += prob_no_outcome * np.log(remaining_bankroll)

    # Each outcome case
    for i, (odds, prob, stake) in enumerate(zip(decimal_odds, true_probs, stakes)):
        if prob > tolerance:
            winnings = stake * odds
            final_bankroll = remaining_bankroll + winnings
            if final_bankroll > 1e-10:
                expected_growth += prob * np.log(final_bankroll)

    return MultipleKellyResult(
        stakes=stakes,
        total_stake=total_stake,
        expected_growth=expected_growth,
        expected_return=risk_metrics.expected_return,
        portfolio_edge=portfolio_edge,
        risk_metrics=risk_metrics,
        method=method,
        optimization_success=optimization_success,
        optimization_details=optimization_details,
        decimal_odds=decimal_odds,
        true_probs=true_probs,
        fraction=fraction,
        max_total_stake=max_total_stake,
        warnings=input_warnings,
    )
