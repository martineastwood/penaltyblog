Changelog
===========

Version Numbering
###################

``penaltyblog`` follows the SemVer versioning guidelines. For more information,
see `semver.org <http://semver.org/>`_


v1.5.0 (2025-08-15)
^^^^^^^^^^^^^^^^^^^^

Package Updates
---------------

- ``Pitch``

  - Initial release of interactive ``Pitch`` plotting library

- ``MatchFlow``

  - ``Flow`` now has it's own query language, with support for boolean expressions and field comparisons via ``.query``

- ``Goals Models``

  - All Goals Model's ``.fit`` functions now take an optional dictionary of arguments to pass to scipy's optimiser
  - All GoalsModels now fit using an optional gradient (defaults to True), which improves the fit time by approx 5-10x

- ``FootballProbabilityGrid``

  - Updated class to include more betting markets
  - Now supports fractional Asian handicaps and totals
  - Optionally normalizes probabilities to sum to 1 (default: True)
  - Calculations now use vectorized numpy operations for improved performance
  - Caching of results for repeated queries to improve efficiency

- ``Goal Expectancy``

  - Added support for removing overrounding from input probabilities
  - Improved handling of edge cases in probability distributions
  - Altered to using probabilities rather than odds
  - Added more diagnostic output for debugging
  - Optionally normalizes probabilities to sum to 1

Documentation Improvements
----------------------------

- Added Pitch documentation
- Updated Flow documentation with ``.query`` examples
- Completely rewritten documentation for Goals Models and goal expectancy
- Removed obsolete examples

v1.4.1 (2025-06-24)
^^^^^^^^^^^^^^^^^^^^

Package Updates
---------------

- Fixed bug in `Flow.cache` executor logic


v1.4.0 (2025-06-19)
^^^^^^^^^^^^^^^^^^^^

Package Updates
---------------

- Introduced optional ``FlowOptimizer`` for smart plan rewrites
  - New ``optimize=True`` flag on all flows (off by default)
  - Safe, conservative rewrites: pushdown, fusion, and simplification
  - Enhanced ``.explain(compare=True)`` for before/after plan introspection
  - Optimizer is backwards-compatible and fully opt-in
- Added ``.plot_plan()`` on ``Flow`` and ``FlowGroup`` to visualize pipeline structure
- ``.explain()`` now works on ``FlowGroup``, and supports ``compare=True``
- New ``.with_schema({...})`` method to cast and validate fields
  - Example: ``Flow.with_schema({"x": int, "ts": parse_datetime})``
- Added ``.rolling_summary()`` to ``FlowGroup`` for windowed group summaries
  (e.g. rolling 5-minute aggregates per player or team)
- Added ``.time_bucket()`` to ``FlowGroup`` for time-based binning summaries
- Added ``.show()`` method to pretty-print results using tabulate
- ``Flow.collect()`` now supports optional progress bars during execution

Documentation Improvements
--------------------------

- Refreshed documentation to include:
  - ``FlowOptimizer`` and ``.optimize=True``
  - ``.with_schema``, ``.rolling_summary``, ``.show()``
  - Plan introspection via ``.explain(compare=True)`` and ``.plot_plan()``
  - Enhanced type hints throughout the package for improved compatibility with `mypy`.

v1.3.0 (2025-05-20)
^^^^^^^^^^^^^^^^^^^^

Package Updates
-----------------

- Initial release of MatchFlow

Documentation Improvements
----------------------------

- Added MatchFlow documentation
- Added MatchFlow recipes documentation
- Added API references for all of ``penaltyblog``
- Added stub file for metric Cython code
- Added stub file for model Cython code


v1.2.0 (2025-04-10)
^^^^^^^^^^^^^^^^^^^^

Package Updates
-----------------

- Updated Elo Ratings model to be more football-specific so that it now includes home field advantage and can predict draw probabilities
- Added new Cythonised Ignorance Score metric
- Added new Cythonised Multiclass Briar Score metric
- RPS functions now raise a ValueError exception if outcome is out of bounds

Documentation Improvements
----------------------------

- Updated Elo documentation
- Added Pi Ratings documentation
- Added examples for ignorance score
- Added examples for multiclass briar score
- Updated examples for RPS

---

v1.1.0 (2025-03-15)
^^^^^^^^^^^^^^^^^^^^

Performance Enhancements
------------------------

- Rewrote Dixon-Coles model using Cython, achieving approximately 250x speed improvement.
- Rewrote Poisson model using Cython, achieving approximately 250x speed improvement.
- Implemented Negative Binomial Goals Model in Cython for enhanced performance.
- Added high-performance Cython implementation of the Bivariate Poisson Goals Model based on Karlis & Ntzoufras.
- Introduced Cython implementation of the Bivariate Weibull Count Copula Goals Model (`Boshnakov et al. paper <https://blogs.salford.ac.uk/business-school/wp-content/uploads/sites/7/2016/09/paper.pdf>`_).
- Added Pi Ratings System (`Constantinou paper <http://www.constantinou.info/downloads/papers/pi-ratings.pdf>`_).
- Migrated ranked probability score functions to Cython for improved speed.

Package Updates
---------------

- Temporarily removed Stan-based models due to dependency management challenges. Investigating improved packaging strategies for future reintegration.
- Temporarily removed Rue and Salvesen model pending revision to accurately reflect its intended methodology (previously implemented as a hybrid Dixon-Coles variant).

Documentation Improvements
--------------------------

- Updated and expanded model examples in the documentation.
- Enhanced type hints throughout the package for improved compatibility with `mypy`.
- Updated documentation to `pydata` Sphinx theme.

CI/CD and Testing
-----------------

- Expanded GitHub Actions workflows to perform unit tests across all supported Python versions.
- Extended GitHub Actions workflows to perform unit tests on Windows, macOS, and Linux.
- Configured GitHub Actions to automatically build wheels for all supported Python versions across Windows, macOS, and Linux.

---

v1.0.4 (2025-01-10)
^^^^^^^^^^^^^^^^^^^^

Package Updates
---------------

- Moved Stan code to separate files to prevent access denied issues on Windows.

---

v1.0.3 (2024-12-19)
^^^^^^^^^^^^^^^^^^^^

Bug Fixes
---------

- Fixed bug in how the Bayesian models indexed teams in the `predict` function.
- Goals models now only predict individual team names rather than iterables of team names, fixing compatibility issues between different sequence objects.

---

v1.0.2 (2024-12-18)
^^^^^^^^^^^^^^^^^^^^

Bug Fixes
---------

- Updated how the Bayesian models handle the Stan files to prevent access denied issues on Windows.

---

v1.0.1 (2024-12-13)
^^^^^^^^^^^^^^^^^^^^

Improvements
------------

- Updated `install_stan` to install the C++ toolchain on Windows if required.

---

v1.0.0 (2024-12-12)
^^^^^^^^^^^^^^^^^^^^

Performance Enhancements
------------------------

- Removed `pymc` as a dependency.
- Optimized `RPS` calculation.
- Optimized `ELO` code.
- Optimized `Kelly Criterion` code.
- Updated `FootballProbabilityGrid` to store its internal matrix as a NumPy array.

Model Updates
-------------

- Rewrote `BayesianHierarchicalGoalModel` in Stan instead of `pymc`, updating the prediction method to integrate over the posterior rather than sampling the mid-point.
- Rewrote `BayesianRandomInterceptGoalModel` in Stan, improved the random intercept, and updated the prediction method.
- Rewrote `BayesianBivariateGoalModel` in Stan for better convergence and updated the prediction method.
- Added `BayesianSkellamGoalModel` for predicting football match outcomes using the Skellam distribution.

Package Updates
---------------

- Added support for Python 3.13.
- Removed obsolete **SoFifa** and **ESPN** scrapers.
- Updated all example notebooks.
- Increased unit test coverage.
- Added CI/CD workflows.
- Removed `Poetry` from the build step.
- Updated documentation.
- Added type hinting to `Colley` and `Massey` classes.

---

v0.8.1 (2023-09-31)
^^^^^^^^^^^^^^^^^^^^

Bug Fixes
---------

- Changed FBRef `born` column to `Int64` dtype to allow `NULL` values.

---

v0.8.0 (2023-08-31)
^^^^^^^^^^^^^^^^^^^^

New Features
------------

- Added initial **Backtest framework** for backtesting betting strategies.
- Added function to calculate the **Kelly Criterion**.
- Added class for calculating **Elo ratings**.

Bug Fixes
---------

- Fixed bug in FBRef scraper for player age and year of birth.
- All goal models can now accept iterables as team inputs.
- Fixed mapping of Belgium leagues in the **FootballData** scraper.

---

v0.7.0 (2023-03-13)
^^^^^^^^^^^^^^^^^^^^

New Features
------------

- Added **FBRef scraper**.

Package Updates
---------------

- Minimum Python version supported is now **Python 3.8**.

---

v0.6.1 (2023-01-06)
^^^^^^^^^^^^^^^^^^^^

Bug Fixes
---------

- Tweaked **Understat scraper** to avoid bot detection.

---

v0.6.0 (2022-12-02)
^^^^^^^^^^^^^^^^^^^^

New Features
------------

- Added `goal_expectancy` function.
- Added **Bayesian Random Intercept Model**.

Performance Enhancements
------------------------

- Tweaked `pymc` settings for Bayesian goal models to improve speed.

Bug Fixes
---------

- Fixed bug in **Bayesian Bivariate Goals Model**.
- Fixed bug in **FootballData scraper** where a null value was breaking the index column.

---

v0.5.1 (2022-11-03)
^^^^^^^^^^^^^^^^^^^^

Bug Fixes
---------

- Fixed bug in goal models when printing an instance before fitting it.
- Fixed bug in Bayesian goal models' weighted decay.
- Fixed default value of `xi` in `dixon_coles_weights` to `0.0018`.

---

v0.5.0 (2022-10-11)
^^^^^^^^^^^^^^^^^^^^

New Features
------------

- Added `get_player_season` and `get_player_shots` to **Understat scraper**.
- Added **Bayesian Hierarchical Goal Model**.
- Added **Bayesian Bivariate Poisson Goal Model**.
- Added **Bayesian Random Intercept Poisson Goal Model**.

Bug Fixes
---------

- `get_fixtures` in **Understat scraper** now only returns completed fixtures (consistent with FootballData scraper).
- Fixed bug in **FootballData scraper** for older seasons missing the `Time` column.

Package Updates
---------------

- Added **SoFifa scraper**.
- Added compatibility for **Python 3.7**.

---

v0.4.0 (2022-08-08)
^^^^^^^^^^^^^^^^^^^^

General Improvements
--------------------

- General bug fixes.
- Reorganized internal package structure.
- Added unit tests.
- Added documentation and uploaded to **ReadTheDocs**.

New Features
------------

- Added **FPL scraper**.
- Added **FPL optimizer**.
- Added **ESPN scraper**.
- Added **Understat scraper**.
- Added **pre-commit checks** to repository.
- Added both-teams-to-score probability to football goals models.
- Refactored **FootballData scraper** for consistency with other scrapers.
- Refactored **Club Elo scraper** for consistency with other scrapers.

Performance Enhancements
------------------------

- Refactored **Colley ratings** and **Massey ratings** for consistency.
- Updated example notebooks and included them in documentation.
