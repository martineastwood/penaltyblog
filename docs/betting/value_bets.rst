=====================
Identifying Value Bets
=====================

A "value bet" is the cornerstone of any successful betting strategy. It's a bet where you believe the true probability of an outcome is higher than the probability implied by the bookmaker's odds. Placing value bets consistently is the key to long-term profitability.

The ``identify_value_bet`` Function
===================================

This is the core function for value analysis. It takes bookmaker odds and your own estimated probabilities, and returns a comprehensive analysis of the betting opportunity, including expected value and a recommended Kelly stake.

.. code-block:: python

   penaltyblog.betting.identify_value_bet(
       bookmaker_odds: Union[float, List[float], NDArray],
       estimated_probability: Union[float, List[float], NDArray],
       kelly_fraction: float = 1.0,
       min_edge_threshold: float = 0.0,
   ) -> Union[ValueBetResult, MultipleValueBetResult]

Parameters
----------

- ``bookmaker_odds``: A single decimal odd, or a list/array of odds.
- ``estimated_probability``: Your estimated true probability (from 0 to 1) for the corresponding outcome(s).
- ``kelly_fraction`` (default=``1.0``): The fraction of the Kelly Criterion to recommend (e.g., 0.5 for a more conservative "Half Kelly" stake).
- ``min_edge_threshold`` (default=``0.0``): The minimum required "edge" (your probability minus the implied probability) for a bet to be flagged as a value bet.

Returns
-------

The function intelligently returns one of two detailed data objects:

- ``ValueBetResult``: If you provide a single odd and probability.
- ``MultipleValueBetResult``: If you provide a list of odds and probabilities.

Understanding the Results
=========================

For Single Bets (``ValueBetResult``)
------------------------------------

When analyzing a single bet, you get a detailed breakdown:

- ``is_value_bet`` (``bool``): True if the bet has a positive edge above your threshold.
- ``expected_value`` (``float``): The amount you expect to win or lose per unit staked. A positive EV indicates a profitable bet in the long run.
- ``edge`` (``float``): The difference between your probability and the bookmaker's implied probability.
- ``recommended_stake_kelly`` (``float``): The optimal fraction of your bankroll to stake, according to the full Kelly Criterion.
- ``recommended_stake_fraction`` (``float``): The Kelly stake, adjusted by the kelly_fraction you provided.

For Multiple Bets (``MultipleValueBetResult``)
----------------------------------------------

When analyzing a list of bets (e.g., a weekend's fixtures), you get a portfolio-level summary:

- ``individual_results`` (``List[ValueBetResult]``): A list containing a detailed ``ValueBetResult`` object for each bet you provided.
- ``total_value_bets`` (``int``): The number of bets in the list that were identified as having value.
- ``average_edge`` (``float``): The average edge across all identified value bets.
- ``kelly_stakes`` (``List[float]``): A list of the recommended (fractional) Kelly stakes for the entire portfolio of bets.

Usage Examples
==============

Single Bet Analysis
-------------------

.. code-block:: python

   import penaltyblog as pb

   # We think a team has a 50% chance to win, but the odds are 2.5
   result = pb.betting.identify_value_bet(2.5, 0.50)

   if result.is_value_bet:
       print("This is a value bet!")
       print(f"Edge: {result.edge:.2%}")
       print(f"Expected Value (per £1 staked): £{result.expected_value:.2f}")
       print(f"Recommended Full Kelly Stake: {result.recommended_stake_kelly:.2%} of bankroll")
   else:
       print("This is not a value bet.")

.. code-block:: text

   This is a value bet!
   Edge: 10.00%
   Expected Value (per £1 staked): £0.25
   Recommended Full Kelly Stake: 16.67% of bankroll

Multiple Bet Analysis
---------------------

.. code-block:: python

   import penaltyblog as pb

   # Analyzing three different bets from a weekend
   odds = [2.1, 3.5, 1.8]
   my_probs = [0.5, 0.25, 0.6] # Our estimated probabilities

   results = pb.betting.identify_value_bet(odds, my_probs)

   print(f"Found {results.total_value_bets} value bets out of {len(odds)}.")
   print(f"The average edge on these value bets is {results.average_edge:.2%}")

   # You can also inspect each individual result
   for bet_result in results.individual_results:
       if bet_result.is_value_bet:
           print(f"- Bet at odds {bet_result.bookmaker_odds} has an edge of {bet_result.edge:.2%}")

.. code-block:: text

   Found 2 value bets out of 3.
   The average edge on these value bets is 3.41%
   - Bet at odds 2.1 has an edge of 2.38%
   - Bet at odds 1.8 has an edge of 4.44%
