==================
Arbitrage Hedging
==================

Arbitrage hedging is the practice of placing bets on different outcomes of an event to manage risk from an existing bet. This can be used to either lock in a guaranteed profit or to minimize a potential loss, regardless of the final result.

The primary function, ``arbitrage_hedge``, uses a **linear programming optimizer** to find the ideal hedge stakes that maximize your worst-case (guaranteed) profit.

``arbitrage_hedge()``
=====================

This function calculates the required stake(s) for hedge bets to guarantee a specific profit or minimize your loss from one or more existing positions.

.. code-block:: python

   penaltyblog.betting.arbitrage_hedge(
       existing_stakes: List[float],
       existing_odds: List[float],
       hedge_odds: List[float],
       target_profit: Optional[float] = None,
       hedge_all: bool = True,
       allow_lay: bool = False,
       tolerance: float = 1e-10,
   ) -> ArbitrageHedgeResult

.. important::
   **Understanding "Guaranteed Profit"**

   The ``guaranteed_profit`` calculated by this function is your worst-case profit across all possible outcomes.

   In many real-world scenarios (e.g., with uneven existing bets), the profit you make will be *different* for each outcome. The function returns the *minimum amount* you are guaranteed to receive. For example, if a hedge results in potential profits of ``[+£105, -£2, -£50]``, the ``guaranteed_profit`` will be **-£50**.

   Equal profits across all outcomes are only possible in perfectly symmetric situations or when "laying" (betting against an outcome) is allowed.

Parameters
----------

- ``existing_stakes`` ``(List[float])``: A list of the amounts you have already staked on each outcome.
- ``existing_odds`` ``(List[float])``: The decimal odds for your existing bets.
- ``hedge_odds`` ``(List[float])``: The current decimal odds available for placing hedge bets.
- ``target_profit`` ``(float, optional)``: A specific profit you want to lock in. If provided, the function will calculate the stakes needed to achieve this exact profit, if possible. If ``None``, it will maximize the guaranteed profit.
- ``hedge_all`` ``(bool, default=True)``: If ``True``, the function hedges against all possible outcomes. If ``False``, it only hedges the specific outcomes where you have an existing stake.
- ``allow_lay`` ``(bool, default=False)``: If ``True``, allows the function to calculate negative ("lay") stakes. Standard bookmakers typically don't allow this, so the default is ``False``, which forces the function to redistribute these amounts across other bets.
- ``tolerance`` ``(float, default=1e-10)``: A small number for handling floating-point comparisons.

Returns
-------

The function returns an ``ArbitrageHedgeResult`` object, which contains detailed information about the calculated hedge.

Understanding the Result (``ArbitrageHedgeResult``)
===================================================

The function returns a rich data object with useful attributes for analysis and execution.

- ``practical_hedge_stakes`` ``(List[float])``: A list of the actual, non-negative bet amounts you should place on each outcome. This is the primary result you'll use.
- ``guaranteed_profit`` ``(float)``: The guaranteed minimum profit (or loss, if negative) you will receive after placing the hedge bets.
- ``raw_hedge_stakes`` ``(List[float])``: The theoretical stakes calculated by the optimizer. This may contain negative values, which represent a "lay" bet.
- ``total_hedge_needed`` ``(float)``: The total value of negative stakes that had to be redistributed to other outcomes because allow_lay was ``False``.
- ``lp_success`` ``(bool)``: ``True`` if the linear programming optimizer found a solution, ``False`` if the function had to use its fallback heuristic.
- ``lp_message`` ``(str | None)``: A message from the optimizer, usually present if it failed.
- ``existing_payouts`` ``(List[float])``: The potential payout for each of your original bets.
- ``total_existing_stakes`` ``(float)``: The total amount of your original stakes.

Usage Examples
==============

Example 1: Locking in a Guaranteed Profit (A "Good" Hedge)
-----------------------------------------------------------

This is the ideal scenario. You bet on an underdog, the odds move significantly in your favour, and you can now hedge to guarantee a profit no matter what.

Let's say you bet £25 on an Away Win at high odds of 6.0. On match day, their odds have shortened to 3.0.

.. code-block:: python

   import penaltyblog as pb

   # Your existing bet: £25 on an Away Win at 6.0
   # Format: [Home, Draw, Away]
   stakes = [0, 0, 25]
   old_odds = [1.5, 4.0, 6.0]

   # On match day, the odds have shifted significantly
   new_odds = [2.5, 3.5, 3.0]

   result = pb.betting.arbitrage.arbitrage_hedge(
       existing_stakes=stakes,
       existing_odds=old_odds,
       hedge_odds=new_odds,
   )

   print(f"Hedge bets to place: [Home: £{result.raw_hedge_stakes[0]:.2f}, Draw: £{result.raw_hedge_stakes[1]:.2f}, Away: £{result.raw_hedge_stakes[2]:.2f}]")
   print(f"Guaranteed profit: £{result.guaranteed_profit:.2f}")
   print(f"Optimizer success: {result.lp_success}")

.. code-block:: text

   Hedge bets to place: [Home: £60.00, Draw: £42.86, Away: £0.00]
   Guaranteed profit: £22.14
   Optimizer success: True

**Conclusion**: The function advises betting **£47.24** on the Home Win and **£33.75** on the Draw. This eliminates the risk and locks in a **guaranteed profit of £21.87**.

Example 2: Assessing Risk (A "Bad" Hedge)
------------------------------------------

Sometimes, the function's value is in telling you **not** to hedge.

Let's use the example from before that resulted in a negative profit. You bet £50 on a Home Win at 3.5, and the odds shorten to 2.8.

.. code-block:: python

   import penaltyblog as pb

   stakes = [50, 0, 0]
   old_odds = [3.5, 3.4, 2.9]
   new_odds = [2.8, 3.8, 3.1]

   result = pb.betting.arbitrage.arbitrage_hedge(
       existing_stakes=stakes,
       existing_odds=old_odds,
       hedge_odds=new_odds,
   )

   print(f"Guaranteed profit: £{result.guaranteed_profit:.2f}")

.. code-block:: text

   Guaranteed profit: £-50.90

**Conclusion**: The function correctly calculates that there is no combination of hedge bets at the new odds that can guarantee a profit. The best possible worst-case outcome is a loss of **£50.90**, which is no better than your original risk of losing £50. The tool has successfully shown you that **hedging is not advisable here.**
