===========
Bet Sizing
===========

This submodule provides powerful tools for calculating optimal bet sizes using the **Kelly Criterion**, a mathematical formula designed to maximize the long-term growth of a bankroll.

Single Bet Analysis (``kelly_criterion``)
==========================================

Use this function to perform a deep analysis of a single betting opportunity. It calculates the optimal stake and provides a wealth of metrics to help you understand the bet's risk and reward profile.

.. code-block:: python

   penaltyblog.betting.kelly_criterion(
       decimal_odds: Union[float, NDArray, List[float]],
       true_prob: Union[float, NDArray, List[float]],
       fraction: float = 1.0,
   ) -> KellyResult

Parameters
----------

- ``decimal_odds``: The decimal odds for the bet (e.g., 2.5). Can be a single number or a NumPy array for multiple independent bets.
- ``true_prob``: Your estimated "true" probability of the outcome (from 0 to 1).
- ``fraction`` (default=``1.0``): A fraction of the full Kelly stake to use. Common values are ``0.5`` for "Half Kelly" or ``0.25`` for "Quarter Kelly" to adopt a more conservative strategy.

Returns (``KellyResult`` Object)
--------------------------------

The function returns a ``KellyResult`` data object containing a complete analysis.

- ``stake`` (``float``): The recommended fraction of your bankroll to wager (e.g., 0.05 for 5%).
- ``expected_growth`` (``float``): The expected logarithmic growth rate of your bankroll if you place this bet. This is the metric that the Kelly Criterion optimizes.
- ``edge`` (``float``): Your mathematical edge on the bet. A positive edge means the bet has a positive expected value.
- ``is_favorable`` (``bool``): ``True`` if the bet has a positive edge.
- ``risk_metrics`` (``RiskMetrics``): A detailed object containing advanced risk and return metrics. See the **Understanding the Risk Metrics** section below for a full explanation.

Usage Example
-------------

.. code-block:: python

   import penaltyblog as pb

   # Analyze a single bet with a 55% chance of winning at odds of 2.1
   # We'll use a conservative Half Kelly approach (fraction=0.5)
   result = pb.betting.kelly_criterion(2.1, 0.55, fraction=0.5)

   print(f"Is the bet favorable? {result.is_favorable}")
   print(f"Recommended Stake: {result.stake:.2%} of bankroll")
   print(f"Expected Growth Rate: {result.expected_growth:.4%}")
   print("-" * 50)
   print("Advanced Risk Metrics:")
   if result.risk_metrics:
       print(f"  - Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.2f}")
       print(f"  - Volatility of Wealth: {result.risk_metrics.wealth_volatility:.4f}")
       print(f"  - 95% Value at Risk: {result.risk_metrics.value_at_risk_95:.2%} of bankroll")

.. code-block:: text

   Is the bet favorable? True
   Recommended Stake: 7.05% of bankroll
   Expected Growth Rate: 0.8177%
   --------------------------------------------------
   Advanced Risk Metrics:
     - Sharpe Ratio: 0.11
     - Volatility of Wealth: 0.0736
     - 95% Value at Risk: 7.05% of bankroll

Portfolio Betting (``multiple_kelly_criterion``)
================================================

This is a powerful function for calculating optimal stakes when you have the opportunity to bet on multiple, mutually exclusive outcomes at the same time (e.g., Home, Draw, and Away in a football match).

Instead of just calculating Kelly for each bet independently, this function uses an optimizer to treat them as a portfolio, finding the allocation that maximizes the overall growth rate of your bankroll.

.. code-block:: python

   penaltyblog.betting.multiple_kelly_criterion(
       decimal_odds: Union[List[float], NDArray],
       true_probs: Union[List[float], NDArray],
       fraction: float = 1.0,
       max_total_stake: float = 1.0,
       method: Literal["simultaneous", "independent"] = "simultaneous",
   ) -> MultipleKellyResult

Parameters
----------

- ``decimal_odds`` (``List[float]``): A list of decimal odds for each outcome.
- ``true_probs`` (``List[float]``): A list of your estimated probabilities for each outcome. The sum must be <= 1.0.
- ``fraction`` (default=``1.0``): A fraction of the optimal Kelly stakes to apply.
- ``max_total_stake`` (default=``1.0``): The maximum total fraction of your bankroll you are willing to stake across all bets combined.
- ``method`` (default=``"simultaneous"``):
    - ``simultaneous``: (Recommended) Uses a numerical optimizer to find the best possible allocation across all bets to maximize portfolio growth.
    - ``independent``: Calculates the Kelly stake for each bet individually and then scales them down if they exceed ``max_total_stake``. Less optimal but faster.

Returns (``MultipleKellyResult`` Object)
----------------------------------------

- ``stakes`` (``List[float]``): The list of recommended stakes for each outcome.
- ``total_stake`` (``float``): The total fraction of your bankroll to be staked.
- ``expected_growth`` (``float``): The expected logarithmic growth rate for the entire portfolio.
- ``risk_metrics`` (``RiskMetrics``): A detailed risk analysis for the entire portfolio.
- ``optimization_success`` (``bool``): ``True`` if the "simultaneous" optimizer found a valid solution.

Usage Example
-------------

.. code-block:: python

   import penaltyblog as pb

   # A 1X2 football market where we have a very strong edge on the Home team
   odds = [2.5, 3.2, 2.8]

   # Our probabilities show a much higher chance for a Home win (55%)
   # than the odds imply (1 / 2.5 = 40%).
   probs = [0.55, 0.25, 0.20]

   result = pb.betting.kelly.multiple_kelly_criterion(odds, probs)

   print("Optimal Portfolio Stakes:")
   print(f"- Home Win (at {odds[0]}): {result.stakes[0]:.2%} of bankroll")
   print(f"- Draw (at {odds[1]}): {result.stakes[1]:.2%} of bankroll")
   print(f"- Away Win (at {odds[2]}): {result.stakes[2]:.2%} of bankroll")
   print("-" * 20)
   print(f"Total Stake: {result.total_stake:.2%}")
   print(f"Portfolio Expected Growth: {result.expected_growth:.4%}")
   print(f"Portfolio Sharpe Ratio: {result.risk_metrics.sharpe_ratio:.2f}")

.. code-block:: text

   Optimal Portfolio Stakes:
   - Home Win (at 2.5): 27.17% of bankroll
   - Draw (at 3.2): 3.26% of bankroll
   - Away Win (at 2.8): 0.00% of bankroll
   --------------------
   Total Stake: 30.43%
   Portfolio Expected Growth: 4.6783%
   Portfolio Sharpe Ratio: 0.15

Understanding the Risk Metrics
==============================

Both functions return a ``RiskMetrics`` object that gives you a deep insight into the risk/reward profile of your strategy.

- ``expected_return``: Your expected profit as a percentage of your total stake. A 10% expected return means you expect to make £0.10 for every £1 staked.
- ``kelly_growth_rate``: The expected long-term growth rate of your bankroll, expressed as a percentage. This is the core metric Kelly optimizes. A higher number is better.
- ``wealth_volatility``: The standard deviation of your final bankroll. This measures how much your bankroll is expected to swing up and down. A lower number indicates a less risky strategy.
- ``sharpe_ratio``: A measure of risk-adjusted return (growth rate divided by its volatility). It helps you compare strategies with different risk levels. A higher Sharpe Ratio is better.
- ``probability_of_ruin``: The chance of losing your entire staked capital in this specific round of betting.
- ``value_at_risk_95``: The maximum you can expect to lose 95% of the time, expressed as a percentage of your bankroll.
