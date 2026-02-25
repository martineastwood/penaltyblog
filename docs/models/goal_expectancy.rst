===============================================
Inferring Goal Expectancies from Bookmaker Odds
===============================================

``penaltyblog`` also includes a utility that works in the **opposite** direction to the goals models: given **bookmaker 1X2 probabilities** (home/draw/away), it estimates the **implied goal expectancies** (μ_home, μ_away).

What it does
============

- Finds μ_home and μ_away such that a Poisson (optionally Dixon–Coles-adjusted) model best matches the given 1X2 probabilities.
- Uses numerical optimisation (scipy.optimize.minimize) with stable parameterisation (log μ bounded).
- Supports:
    - Time-decay adjustment for low-score events (Dixon–Coles).
    - Flexible scoring rules: Brier/MSE or cross-entropy.
    - Configurable grid size (max_goals) and normalisation after Dixon–Coles adjustment.

Parameters
==========

- **home, draw, away** - 1X2 probabilities (must be in [0,1]).
- **dc_adj** - whether to apply Dixon–Coles adjustment.
- **rho** - correlation parameter for Dixon–Coles.
- **minimizer_options** - dict of options to pass to SciPy's optimiser.
- **max_goals** - maximum goals per team in the grid (default 15 ⇒ 0–15 inclusive).
- **remove_overround** - if True, probabilities are renormalised to sum to 1 before fitting.
- **method** - optimiser method (default "L-BFGS-B").
- **bounds** - bounds on (log μ_home, log μ_away) for stability.
- **x0** - optional starting guess for (log μ_home, log μ_away).
- **renormalize_after_dc** - if True, re-normalises the probability grid after DC adjustments and clips small negatives.
- **objective** - 'brier' for mean-squared error or 'cross_entropy' for KL-style loss.
- **return_details** - if True, includes extra audit information in the result.

Returns
=======

A dict with:

- ``home_exp`` - implied home goal expectancy (μ_home)
- ``away_exp`` - implied away goal expectancy (μ_away)
- ``error`` - final mean squared error between predicted and target 1X2
- ``success`` - whether the optimiser reported success

If ``return_details=True``, also includes:

- **predicted** - model's predicted [P(home win), P(draw), P(away win)]
- **mass** - total probability in the truncated grid (≤ 1.0 if max_goals small or normalize_after_dc=False)

Quick Example
=============

.. code-block:: python

   from penaltyblog.models import goal_expectancy
   from pprint import pprint

   # Suppose your market is:
   p_home, p_draw, p_away = 0.45, 0.28, 0.29

   est = goal_expectancy(
       home=p_home,
       draw=p_draw,
       away=p_away,
       dc_adj=True,              # use Dixon–Coles low-score correction
       rho=0.001,                # typical small value
       minimizer_options={"maxiter": 5000},
       remove_overround=True
   )

   pprint(est)

.. code-block:: text

   {'away_exp': 1.018968393752446,
    'error': 1.4642670964617868e-12,
    'home_exp': 1.3415375327000219,
    'mass': 0.9999999999999999,
    'predicted': array([0.44117672, 0.27450821, 0.28431507]),
    'success': True}

You can then use these expectancies directly in your own Poisson simulator or as a prior/anchor when comparing to model-based expectancies.

Notes & Best Practices
======================

- **Probabilities vs odds:** If you start from odds, convert to probabilities and (optionally) remove overround before passing to this function.
- **Truncation:** Only scores up to ``max_goals`` are considered; very small tail mass may be lost if ``normalize_after_dc=False``.
- **DC adjustment:** Can help fit when draw prices are high; rho is typically small (0.001–0.01).
- **Stability:** The optimiser works on bounded log-μ space, preventing non-physical negative goal expectancies.
- **Diagnostics:** Use ``return_details=True`` to check mass and predicted 1X2 to understand residual errors.

Behind the Scenes: How the Optimiser Works
==========================================

This function reverse-engineers μ_home and μ_away via a small **non-linear optimisation** problem.

1. Parameterisation
-------------------

- The optimiser works in **log μ** space (``log_mu_home``, ``log_mu_away``), ensuring μ > 0 at all times.
- Bounds on log μ (default ``[-3, 3]``) correspond to μ ∈ [0.05, 20] – covering realistic football scoring ranges but preventing runaway values that could destabilise the fit.

2. Objective Function
---------------------

- Default: **Brier score** (mean squared error) between the model's predicted 1X2 probabilities and the input values.
- Alternative: **Cross-entropy** loss (KL divergence direction) if ``objective='cross_entropy'``.

3. Probability Grid
-------------------

- A Poisson model generates a probability matrix over scores ``(0..max_goals) × (0..max_goals)``.
- The Dixon–Coles adjustment optionally tweaks four low-score cells to better match real-world correlation in low-scoring games.
- If ``renormalize_after_dc=True``, the grid is re-scaled to sum exactly to 1.0 after the adjustment (and any small negatives are clipped).

4. From Grid to 1X2
--------------------

- ``P(home win)`` = sum of all cells below the main diagonal.
- ``P(draw)`` = sum of diagonal cells.
- ``P(away win)`` = sum of cells above the diagonal.

5. Optimisation
---------------

- The optimiser (``scipy.optimize.minimize``) searches log μ space to minimise the chosen loss.
- By default, the **L-BFGS-B** method is used, as it handles bounds well and converges quickly for small parameter spaces.
- The starting guess defaults to a mild home advantage (``log(1.3)``, ``log(1.1)``), but you can override with ``x0``.

6. Diagnostics & Tail Mass
---------------------------

- The returned ``mass`` is the sum of all probabilities in the truncated grid.
  If ``max_goals`` is too low, ``mass`` < 1.0 means you've cut off a non-negligible tail - increasing ``max_goals`` will reduce this.
- With ``normalize_after_dc=False``, residuals may include both truncation error and DC-induced mass shifts.


Extended Goal Expectancy Inference
==================================

If you have both the 1X2 market probabilities and the Over/Under 2.5 probabilities available, you can use ``goal_expectancy_extended`` to simultaneously reverse-engineer the implied expected goals (μ_home, μ_away) and a custom Dixon-Coles adjustment parameter (``rho``) that matches all constraints.

.. code-block:: python

   from penaltyblog.models import goal_expectancy_extended

   # Assuming you derived the following market probabilities:
   p_home, p_draw, p_away = 0.45, 0.28, 0.29
   p_over25, p_under25 = 0.48, 0.52

   est_ext = goal_expectancy_extended(
       home=p_home,
       draw=p_draw,
       away=p_away,
       over25=p_over25,
       under25=p_under25,
       remove_overround=True
   )

   print(est_ext["home_exp"])     # Implied home lambda
   print(est_ext["away_exp"])     # Implied away lambda
   print(est_ext["implied_rho"])  # Implied Dixon-Coles rho

Generating Full Match Probabilities
===================================

Once you have inferred the implied goal expectancies (and optionally the Dixon-Coles ``rho`` parameter) using either ``goal_expectancy`` or ``goal_expectancy_extended``, you can feed these parameters directly into ``create_dixon_coles_grid``. This allows you to generate a complete probability grid and extract odds for any other market (e.g., Asian Handicaps, Both Teams to Score, exact correct scores) from just basic 1X2 and Over/Under prices.

.. code-block:: python

   from penaltyblog.models import create_dixon_coles_grid

   # Using the results from goal_expectancy_extended above
   home_lambda = est_ext["home_exp"]
   away_lambda = est_ext["away_exp"]
   rho = est_ext["implied_rho"]

   # Create the full probability grid
   pred = create_dixon_coles_grid(home_lambda, away_lambda, rho)

   # Now you can query any market
   print(pred.btts_yes)                           # Both Teams to Score (Yes)
   print(pred.asian_handicap_probs("home", -0.5)) # Asian Handicap
   print(pred.exact_score(2, 1))                  # Probability of a 2-1 exact score
