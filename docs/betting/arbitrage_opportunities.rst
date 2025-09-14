===============================
Identifying Arbitrage Opportunities
===============================

An arbitrage bet (or "arb") is a risk-free opportunity. It exists when you can bet on all outcomes of a single event across different bookmakers and guarantee a profit, because their odds are misaligned. These are rare but highly valuable.

The ``find_arbitrage_opportunities`` Function
=============================================

This function scans lists of odds from multiple bookmakers for the same event to find these risk-free opportunities.

.. code-block:: python

   penaltyblog.betting.find_arbitrage_opportunities(
       bookmaker_odds_list: List[List[float]],
       outcome_labels: List[str] = None
   ) -> ArbitrageResult

Parameters
----------

- ``bookmaker_odds_list``: A list of lists. Each inner list represents one bookmaker's odds for all outcomes of an event.
- ``outcome_labels``: Optional names for the outcomes (e.g., ["Home", "Away"]).

Returns (``ArbitrageResult``)
-----------------------------

- ``has_arbitrage`` (``bool``): ``True`` if a risk-free opportunity exists.
- ``guaranteed_return`` (``float``): The guaranteed profit as a percentage of your total stake.
- ``best_odds`` (``List[float]``): The best odds found for each outcome across all bookmakers.
- ``best_bookmakers`` (``List[int]``): The index of the bookmaker offering the best odds for each outcome.
- ``stake_percentages`` (``List[float]``): The percentage of your total stake to place on each outcome to guarantee the profit.

Usage Example
=============

.. code-block:: python

   import penaltyblog as pb

   # Odds for a soccer match (Home Win, Draw, Away Win) from three different bookmakers
   # Each inner list represents one bookmaker's odds for [Home, Draw, Away]
   odds_data = [
       [2.80, 3.50, 3.10],  # Bookmaker 1
       [3.10, 3.40, 2.90],  # Bookmaker 2
       [3.00, 3.20, 3.00],  # Bookmaker 3
   ]

   # Define the labels for the outcomes
   outcome_labels = ["Home Win", "Draw", "Away Win"]

   arb_result = pb.betting.find_arbitrage_opportunities(odds_data, outcome_labels)

   if arb_result.has_arbitrage:
       print("Arbitrage opportunity found!")
       print(f"Guaranteed Return on Investment: {arb_result.guaranteed_return:.2%}")
       print("-" * 20)

       # The function tells you exactly where to bet and how much to stake
       for i, label in enumerate(arb_result.outcome_labels):
           stake_pct = arb_result.stake_percentages[i]
           best_odd = arb_result.best_odds[i]
           # Adding 1 to the index to make it human-readable (Bookmaker 1, 2, 3)
           bookie_idx = arb_result.best_bookmakers[i] + 1

           print(f"Bet {stake_pct:.2%} on {label} at odds {best_odd} with Bookmaker {bookie_idx}")
   else:
       print("No arbitrage opportunity found.")

.. code-block:: text

   Arbitrage opportunity found!
   Guaranteed Return on Investment: 7.43%
   --------------------
   Bet 34.65% on Home Win at odds 3.1 with Bookmaker 2
   Bet 30.69% on Draw at odds 3.5 with Bookmaker 1
   Bet 34.65% on Away Win at odds 3.1 with Bookmaker 1

Simple Expected Value Calculation
=================================

If you don't need the full analysis from ``identify_value_bet`` and just want a quick Expected Value (EV) calculation, you can use this lightweight utility function.

.. code-block:: python

   penaltyblog.betting.calculate_bet_value(
       bookmaker_odds: float,
       estimated_probability: float
   ) -> float

.. code-block:: python

   import penaltyblog as pb

   # 60% chance at odds of 2.0
   ev = pb.betting.calculate_bet_value(2.0, 0.6)
   print(f"Expected Value (per £1 staked): £{ev:.2f}")

.. code-block:: text

   Expected Value (per £1 staked): £0.20
