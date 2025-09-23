==============================
Overview of the Different Models
==============================

The ``penaltyblog`` package provides a suite of **robust, ready-to-use statistical models** for predicting football (soccer) match scores. All models are **highly optimised with Cython** for speed, making them fast, even for large-scale forecasting, betting analysis, and live in-play applications. Whether you want a quick baseline or an advanced model capturing complex goal-scoring patterns, ``penaltyblog``'s models provide a consistent, user-friendly API.

1. Poisson Goals Model
======================

The simplest and most widely used approach.

- **Idea**: Goals follow a Poisson distribution, with rates determined by attack strength, defense strength, and home advantage.
- **Strengths**: Easy to understand, quick to fit, good baseline accuracy.
- **Weaknesses**: Overpredicts high scores, struggles with low-score biases.
- **Best for**: General forecasting, fast model training.

2. Dixon and Coles Goals Model
==============================

A refinement of the Poisson model that corrects for the higher-than-expected frequency of low-score draws (e.g., 0-0, 1-0, 1-1).

- **Strengths**: More realistic score predictions in low-scoring leagues, improved match outcome accuracy.
- **Weaknesses**: Adds parameter tuning, assumes the same low-score adjustment for all matches.
- **Best for**: Leagues with many draws or defensive play styles.

3. Bivariate Poisson Goals Model
================================

Extends the Poisson model by introducing correlation between teams' goal counts.

- **Strengths**: Captures match dynamics affecting both teams, better for high-scoring matches.
- **Weaknesses**: More complex and harder to interpret, slower to fit.
- **Best for**: Leagues where team performances are strongly linked (e.g., end-to-end attacking games).

4. Zero-Inflated Poisson Goals Model
====================================

Adds an explicit mechanism for handling excess goalless matches.

- **Strengths**: Improves accuracy in ultra-defensive contexts.
- **Weaknesses**: Only adjusts for excess 0-0 games; doesn't fix other Poisson issues.
- **Best for**: Competitions or teams with frequent goalless draws.

5. Negative Binomial Goals Model
================================

Handles **overdispersion** - when goal count variance is greater than the mean.

- **Strengths**: More realistic for high-scoring or unpredictable leagues, better at extreme results.
- **Weaknesses**: Still assumes independence between team scores.
- **Best for**: High-variance, goal-heavy competitions.

6. Weibull Count + Copula Goals Model
=====================================

A more flexible alternative to Poisson-based models, using a Weibull distribution for goals and a copula for score correlation.

- **Strengths**: Handles complex goal patterns and diverse score dependencies.
- **Weaknesses**: Statistically and computationally intensive, not always worth the added complexity.
- **Best for**: Advanced modelling in leagues with unusual scoring patterns.

Model Comparison
================

+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| Model                      | Strengths                                         | Weaknesses                                      | Best Used For |
+============================+===================================================+=================================================+===============+
| **Poisson**                | Simple, efficient, widely used                   | Overpredicts high scores, ignores low-score bias| General forecasting |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| **Dixon & Coles**          | Corrects low-score bias, better match accuracy   | Fixed adjustment across matches, extra tuning  | Low-scoring leagues |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| **Bivariate Poisson**      | Models score correlation, useful for high-scoring| Complex, harder to interpret                    | High-scoring leagues |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| **Zero-Inflated Poisson**  | Better at goalless matches                       | Only fixes 0-0 bias                            | Defensive teams |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| **Negative Binomial**      | Handles overdispersion, realistic extreme scores | Still independent goal counts                   | High-scoring, volatile leagues |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+
| **Weibull + Copula**       | Flexible distribution & correlation modelling    | Highly complex, slow to fit                     | Complex goal patterns |
+----------------------------+---------------------------------------------------+-------------------------------------------------+---------------+

Consistent API Across Models
=============================

All goal models in ``penaltyblog`` share the **same interface**, making it simple to switch between them, run comparisons, or fine-tune parameters without rewriting your code.

This design means you can:

- Swap out a Poisson model for a Dixon & Coles model in one line.
- Benchmark multiple models on the same dataset with minimal changes.
- Apply optimisations (like lookback windows or time weighting) consistently across all models.

Common Methods
--------------

Every model implements the following core methods:

- ``fit(minimizer_options)``: Train the model using your dataset.
- ``predict(home_team, away_team, max_goals, normalize)``: Predict scoreline probabilities for a given fixture.
- ``get_params()``: Retrieve the model's fitted parameters.
- ``save(filepath)``: Save the model to disk as a pickled file.
- ``load(filepath)``: Load the saved model.

Example
-------

Switching from a Poisson model to a Dixon and Coles model is as simple as:

.. code-block:: python

   from penaltyblog.models import PoissonGoalsModel, DixonColesGoalsModel

   # Train a Poisson model
   model = PoissonGoalsModel(
       train["goals_home"],
       train["goals_away"],
       train["team_home"],
       train["team_away"],
   )
   model.fit()

   # Swap to Dixon & Coles
   model = DixonColesGoalsModel(
       train["goals_home"],
       train["goals_away"],
       train["team_home"],
       train["team_away"],
   )
   model.fit()

   # Predict probabilities for a fixture
   prediction = model.predict("Arsenal", "Manchester City")
   print(prediction.home_draw_away)

Because the API is consistent, you can automate model testing and tuning. For example, by looping through a list of model classes, fitting each one, and comparing metrics like Ranked Probability Score (RPS) without special-case code.

Time Weighting to Prioritise Recent Matches
============================================

Football is dynamic - teams change managers, players, and tactics over time. Using too much historical data can let outdated results dilute your predictions.

To address this, all ``penaltyblog`` models support **time weighting**, allowing you to give recent fixtures more influence than older ones.

The most common approach is the **Dixon and Coles exponential decay weighting**, where a decay factor ``ξ`` controls how quickly older matches lose importance:

- ``ξ = 0`` → all matches are weighted equally.
- Small ``ξ`` (e.g., 0.001) → older matches still contribute, but recent ones matter more.
- Large ``ξ`` (e.g., 0.03) → the model focuses heavily on the most recent results.

Example
-------

.. code-block:: python

   from penaltyblog.models import PoissonGoalsModel, dixon_coles_weights

   # Generate weights with a decay factor of 0.001
   weights = dixon_coles_weights(train["date"], xi=0.001)

   # Fit a Poisson model using time weighting
   model = PoissonGoalsModel(
       train["goals_home"],
       train["goals_away"],
       train["team_home"],
       train["team_away"],
       weights=weights
   )
   model.fit()

Rich Probability Outputs for Betting and Analytics
==================================================

All goal models in ``penaltyblog`` return their predictions as a ``FootballProbabilityGrid`` object.

This class automatically gives you access to a wide range of **pre-calculated betting markets and metrics**, with no extra coding required.

When you call ``.predict(home_team, away_team)``, you receive:

- The **full probability grid** for every possible scoreline (e.g., 0–0, 1–0, 2–3, …)
- Expected goals for each team (``home_goal_expectation``, ``away_goal_expectation``)
- Ready-to-use probabilities for popular markets:
  - **Match result** (``home_win``, ``draw``, ``away_win``, ``home_draw_away``)
  - **Both Teams to Score** (``both_teams_to_score``)
  - **Over/Under Total Goals** (``total_goals("over", strike)``)
  - **Asian Handicap** (``asian_handicap("home", strike)`` / ``asian_handicap("away", strike)``)

Example
-------

.. code-block:: python

   prediction = model.predict("Arsenal", "Manchester City")

   # Expected goals
   print(prediction.home_goal_expectation)  # e.g. 1.45
   print(prediction.away_goal_expectation)  # e.g. 1.12

   # Match odds (1X2)
   print(prediction.home_draw_away)  # [P(Home), P(Draw), P(Away)]

   # Both teams to score
   print(prediction.both_teams_to_score)  # Probability both teams score

   # Over/Under 2.5 goals
   print(prediction.total_goals("over", 2.5))

   # Asian handicap (home -0.5)
   print(prediction.asian_handicap("home", -0.5))

Because the grid is generated directly from the underlying scoreline probabilities, **all these markets are perfectly internally consistent** - a crucial advantage for betting analytics and trading models. No more recalculating market probabilities from scratch; the ``FootballProbabilityGrid`` makes it instant.

Faster Fitting with Gradients (Optional)
=========================================

All goals models now support **analytical gradients** during fitting to speed up convergence. Gradients are **on by default** but can be **turned off** for backward compatibility or if they don't suit your data.

- **Why use gradients?** Faster, more stable optimisation and fewer iterations.
- **When to turn them off?** If you're experimenting, debugging, or working with unusual data where numerical optimisation behaves better.

Example
-------

.. code-block:: python

   # Gradients enabled (default)
   model.fit()

   # Turn gradients off (backward compatible behaviour)
   model.fit(use_gradient=False)

.. note::
   Under the hood, when ``use_gradient=True``, the model supplies a ``jac`` function to ``scipy.optimize.minimize``. When ``use_gradient=False``, it omits ``jac``, falling back to numerical approximations.

Passing Options to the Optimiser
=================================

You can pass keyword options straight through to SciPy's optimiser via the ``minimizer_options`` argument. Typical knobs include ``maxiter``, ``ftol``, ``gtol``, etc. (the optimisation method is chosen per-model to suit its bounds/constraints).

Example
-------

.. code-block:: python

   # Increase iterations and tighten tolerances
   model.fit(
       minimizer_options={
           "maxiter": 5000,
           "gtol": 1e-8,
           "ftol": 1e-9,
           "disp": False,  # silence SciPy output
       }
   )

   # Combine with gradient toggle
   model.fit(
       use_gradient=True,
       minimizer_options={"maxiter": 3000, "gtol": 1e-8}
   )

.. note::
   Each model chooses an appropriate optimisation method internally based on bounds/constraints. The ``minimizer_options`` you provide are forwarded to ``scipy.optimize.minimize(options=...)``.

Inspecting Fit Results
======================

After fitting, models expose common diagnostics:

- ``model.fitted`` — boolean flag
- ``model.loglikelihood`` — maximised log-likelihood
- ``model.n_params`` — number of fitted parameters
- ``model.aic`` — Akaike Information Criterion
- ``model.params`` / ``model.get_params()`` — dict of named parameters

Example
-------

.. code-block:: python

   model.fit()
   print(model.fitted, model.loglikelihood, model.aic)
   print(model.params)

Saving and Loading Models
=========================

Use built-in persistence helpers to save a fitted model to disk and load it later without retraining

.. code-block:: python

   model.fit()
   model.save("models/eredivisie_dc.pkl")

.. code-block:: python

   from penaltyblog.models import DixonColesGoalModel  # or the relevant class

   loaded = DixonColesGoalModel.load("models/eredivisie_dc.pkl")
   prediction = loaded.predict("Ajax", "PSV")
   print(prediction.home_draw_away)

.. note::
   Models are serialised with ``pickle``. Ensure you import the same model class before loading.

Minimal End-to-end Example
==========================

.. code-block:: python

   import penaltyblog as pb

   # Prepare your training arrays (goals & teams) and optional weights
   gh, ga = train["goals_home"], train["goals_away"]
   th, ta = train["team_home"], train["team_away"]
   w = pb.models.dixon_coles_weights(train["date"], xi=0.001)  # optional

   # Choose a model (swap freely thanks to the shared API)
   model = pb.models.DixonColesGoalsModel(gh, ga, th, ta, weights=w)

   # Fit fast with gradients and optional custom optimiser options
   model.fit(
       use_gradient=True,
       minimizer_options={"maxiter": 3000, "gtol": 1e-8}
   )

   # Predict and access rich markets
   pred = model.predict("Ajax", "PSV")
   print(pred.home_draw_away)               # [P(Home), P(Draw), P(Away)]
   print(pred.totals(2.5))                  # (under, push, over)
   print(model.aic, model.loglikelihood)    # diagnostics

   # Save for later reuse
   model.save("models/eredivisie_dc.pkl")
