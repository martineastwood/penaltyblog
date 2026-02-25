========================================
FootballProbabilityGrid (Model Output)
========================================

All goals models in ``penaltyblog`` return a ``FootballProbabilityGrid`` when you call ``.predict(home_team, away_team)``.

This object wraps the full exact-score **probability grid** and provides **fast, vectorised access** to popular betting markets and analytics in a single, consistent interface.

Why it's useful
================

- **One object, many markets**: 1X2, BTTS, totals (with **push**), Asian handicaps (including **quarter lines**), double chance, DNB, win to nil, clean sheets, expected points, and more.
- **Internally consistent**: Every market is derived from the same score grid, so probabilities never conflict.
- **Fast**: Vectorised NumPy operations and lightweight caching for repeated calls.
- **Backwards compatible**: Older methods like ``total_goals("over", 2.5)`` and ``asian_handicap("home", -0.5)`` still work.

Quick Start
===========

.. code-block:: python

   pred = model.predict("Arsenal", "Manchester City")

   # 1X2
   pred.home_win      # P(Home win)
   pred.draw          # P(Draw)
   pred.away_win      # P(Away win)
   pred.home_draw_away  # [P(Home), P(Draw), P(Away)]

   # Expected goals (from the fitted model)
   pred.home_goal_expectation
   pred.away_goal_expectation

   # Both Teams To Score
   pred.btts_yes                 # BTTS Yes
   pred.btts_no                  # BTTS No

   # Totals (with push handling)
   pred.totals(2.0)              # -> (under, push, over)
   pred.total_goals("over", 2.5) # backward-compatible: returns P(Over 2.5)

   # Asian handicap
   pred.asian_handicap_probs("home", -0.25)  # -> {"win": ..., "push": ..., "lose": ...}
   pred.asian_handicap("home", -0.5)         # backward-compatible: win prob only

   # More handy markets
   pred.double_chance_1x
   pred.double_chance_x2
   pred.double_chance_12
   pred.draw_no_bet_home
   pred.draw_no_bet_away

   # Distributions & exact scores
   pred.exact_score(2, 1)                # P(2-1)
   pred.home_goal_distribution()         # P(H=0), P(H=1), ...
   pred.away_goal_distribution()
   pred.total_goals_distribution()       # P(T=0), P(T=1), ...

   # Team-centric analytics
   pred.win_to_nil_home()
   pred.win_to_nil_away()
   pred.expected_points_home()
   pred.expected_points_away()

API Summary
===========

+--------------------------------------------------------------+-------------------------------------------------------------------------+
| Attribute / Method                                           | Description                                                             |
+==============================================================+=========================================================================+
| ``grid``                                                     | 2D ``np.ndarray`` with exact-score probabilities ``grid[h, a] = P(H=h, A=a)`` |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``home_goal_expectation``, ``away_goal_expectation``         | Expected goals for each team                                            |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``home_win``, ``draw``, ``away_win``                         | 1X2 probabilities                                                       |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``home_draw_away``                                           | ``[P(Home), P(Draw), P(Away)]``                                         |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``btts_yes``, ``btts_no``                                    | BTTS Yes/No probabilities                                               |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``totals(line)``                                             | Returns ``(under, push, over)`` for integer/half lines (e.g., 2.0, 2.5) |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``total_goals(side, line)``                                  | Back-compat Over/Under prob (push excluded)                             |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``asian_handicap_probs(side, line)``                         | Proper **Win/Push/Lose** for integer/half/**quarter** lines             |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``asian_handicap(side, line)``                               | Back-compat: **win** probability only                                   |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``double_chance_1x``, ``double_chance_x2``, ``double_chance_12`` | Double chance markets                                                |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``draw_no_bet_home``, ``draw_no_bet_away``                   | DNB win probabilities (conditional on no draw)                          |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``exact_score(h, a)``                                        | Probability of an exact scoreline                                       |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``home_goal_distribution()``                                 | Marginal distribution over home goals                                   |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``away_goal_distribution()``                                 | Marginal distribution over away goals                                   |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``total_goals_distribution()``                               | Distribution over total goals ``T = H + A``                             |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``win_to_nil_home()``, ``win_to_nil_away()``                 | Win-to-nil probabilities                                                |
+--------------------------------------------------------------+-------------------------------------------------------------------------+
| ``expected_points_home()``, ``expected_points_away()``       | Expected points under 3/1/0                                             |
+--------------------------------------------------------------+-------------------------------------------------------------------------+

Totals: Over/Under and Pushes
==============================

Totals lines can **push** when the line is an integer (e.g., 2.0). Use ``totals(line)`` to get the full breakdown:

.. code-block:: python

   under, push, over = pred.totals(2.0)   # push > 0 possible at integer lines
   p_over_25 = pred.total_goals("over", 2.5)  # back-compat helper (no push)

- Half-lines (e.g., **2.5**) cannot push → push = 0.
- Integer lines (e.g., **2.0**) can push → non-zero push.

Asian Handicap: Integer, Half, and Quarter Lines
================================================

The grid supports correct settlement for **integer**, **half**, and **quarter** lines:

.. code-block:: python

   # Quarter lines split stake across neighbouring half-lines
   pred.asian_handicap_probs("home", -0.25)  # 50% at 0.0, 50% at -0.5 internally
   pred.asian_handicap_probs("away", +1.0)   # integer line: push possible

- ``asian_handicap_probs(side, line)`` → ``{"win": p, "push": p, "lose": p}``
- ``asian_handicap(side, line)`` → **win** probability only (backwards compatible)

Performance Notes
=================

- Operations use **NumPy masks** and **lazy caching** for frequently accessed metrics (e.g., ``home_win``, ``draw``, ``away_win``).
- The probability grid is validated on construction and (optionally) **normalised** to sum to **1**.
- You can enable normalisation via ``normalize=True`` if required.

Controlling Grid Normalisation
==============================

By default, ``FootballProbabilityGrid`` **normalises** the score grid so that all exact-score probabilities sum to 1. However, you can control this behaviour via the model's ``predict`` method:

.. code-block:: python

   # Normalised grid (default)
   pred = model.predict("Arsenal", "Manchester City", normalize_grid=True)

   # Skip normalisation (use your grid as-is)
   pred = model.predict("Arsenal", "Manchester City", normalize_grid=False)

- ``normalize_grid=True`` (default) → the returned ``FootballProbabilityGrid`` normalises its grid.
- ``normalize_grid=False`` → normalisation is skipped (useful if you already normalised externally or are auditing raw grids).

Normalising vs Not Normalising the Probability Grid
===================================================

When you call ``.predict(...)``, the model calculates probabilities for all scorelines from 0–``max_goals`` (default: 15). This means extremely high-scoring outcomes (e.g., 16–14) are excluded from the grid.

There are two approaches to handling this:

Not normalising
---------------

The probability grid keeps its *true* mass, with the missing probability sitting beyond the ``max_goals`` cut-off. This is statistically purist, but it means your derived markets (1X2, totals, Asian handicaps) will not sum exactly to 1.0. For example, you might see ``home_win + draw + away_win = 0.9999997``. This may be awkward if you need perfectly balanced pricing or hedging.

Normalising
-----------

The grid is rescaled so that all probabilities sum exactly to 1.0. The small "tail" probability beyond ``max_goals`` is implicitly reallocated proportionally across the included outcomes. This ensures all markets are internally consistent - 1X2, totals, and Asian handicap probabilities will align perfectly - and is generally safe if the missing probability mass is negligible.

By default, ``penaltyblog`` normalises the grid to avoid confusing inconsistencies in downstream markets. Advanced users can disable this with ``normalize_grid=False`` in ``.predict()`` if they want to inspect the raw, unadjusted probabilities.

Backwards Compatibility
=======================

- The following legacy-style calls still work exactly as before:
    - ``pred.total_goals("over"|"under", strike)`` - returns probability excluding pushes.
    - ``pred.asian_handicap("home"|"away", strike)`` - returns win probability only.
- Prefer the new, more explicit variants for production:
    - ``pred.totals(strike)`` to obtain (under, push, over)
    - ``pred.asian_handicap_probs(side, strike)`` for Win/Push/Lose

Reproducibility & Export (Optional Tips)
========================================

Because all markets derive from ``pred.grid``, you can export or visualise it for auditing:

.. code-block:: python

   import pandas as pd

   grid_df = pd.DataFrame(pred.grid)   # rows: home goals, cols: away goals
   grid_df.to_csv("score_grid.csv", index_label="home_goals")

This makes it easy to trace any market probability back to the underlying score distribution.

Creating Grids Directly
=======================

You can circumvent fitting models using historical match data and instead create a grid directly from your own expected goals (lambdas). This is useful if your expected goals predictions derive from external ML models (such as an XGBoost model fit on xG event data) rather than standard Poisson regression.

.. code-block:: python

   from penaltyblog.models import create_dixon_coles_grid

   # Build a grid by providing expectation parameters
   pred = create_dixon_coles_grid(home_lambda=1.5, away_lambda=1.2, rho=0.01)

   # Use the grid normally
   pred.home_win
   pred.totals(2.5)

The ``create_dixon_coles_grid`` function calculates independent probabilities and applies the low-score correlation adjustment defined by ``rho``.
