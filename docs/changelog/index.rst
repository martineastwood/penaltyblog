Changelog
===========

Version Numbering
###################

`penaltyblog` follows the SemVer versioning guidelines. For more information,
see `semver.org <http://semver.org/>`_

v1.1.0 ()
^^^^^^^^^^^^^^^^^^^^
- Refactored Dixon and Coles goals model to fit ~20x faster
- Refactored Poisson goals model to fit ~20x faster
- Added Negative Binomial Goals Model
- Added Bivariate Poisson Goals Model based on Karlis & Ntzoufras approach
- Added Poisson Copula Goals Model
- Added Bivariate Weibull Count Copula Goals Model based on https://blogs.salford.ac.uk/business-school/wp-content/uploads/sites/7/2016/09/paper.pdf
- Added Pi Ratings System based on http://www.constantinou.info/downloads/papers/pi-ratings.pdf
- Temporarily removed Stan models from package due to ongoing issues on Windows. Hopefully, I'll find a nicer way to package these up so I can include them again.
- Temporarily removed Rue and Salvesen model while I revist its implementation (It wasn't the proper model, more of a hydrid Dixon and Coles model With some of Rue and Salvesen's ideas thrown in)

v1.0.4 (2025-01-10)
^^^^^^^^^^^^^^^^^^^^
- Moved stan code to separate files to prevent access denied issues on Windows

v1.0.3 (2024-12-19)
^^^^^^^^^^^^^^^^^^^^
- Fixed bug in how the Bayesian models indexed teams in the predict function
- Goals model now only predict individual team names rather than iterables of team names as was causing compatibility issues between different sequence objects.

v1.0.2 (2024-12-18)
^^^^^^^^^^^^^^^^^^^^
- Updated how the Bayesian models handle the Stan files to prevent access denied issues on Windows

v1.0.1 (2024-12-13)
^^^^^^^^^^^^^^^^^^^^
- Updated `install_stan` to install the C++ toolchain on Windows if required

v1.0.0 (2024-12-12)
^^^^^^^^^^^^^^^^^^^^
- Removed pymc as a dependency
- Updated all other dependency versions
- Added support for Python 3.13
- Rewrote `BayesianHierarchicalGoalModel` model into Stan instead of pymc and updated prediction method to integrate over the posterior rather than just sampling the mid-point
- Rewrote `BayesianRandomInterceptGoalModel` into Stan instead of pymc, updated model to use a more accurate random intercept, and updated prediction method to integrate over the posterior rather than just sampling the mid-point
- Rewrote `BayesianBivariateGoalModel` into Stan instead of pymc, improved model so converges better, and updated prediction method to integrate over the posterior rather than just sampling the mid-point
- Added `BayesianSkellamGoalModel` model for predicting outcomes of football (soccer) matches based on the Skellam distribution
- Removed obsolete sofifa and espn scrapers
- Optimised `RPS` calculation
- Optimised `ELO` code
- Optimised `Kelly Criterion` code
- Updated `FootballProbabilityGrid` to store its internal matrix as a numpy array
- Updated all example notebooks
- Increased unit test coverage
- Added CI/CD
- Removed Poetry from build step
- Updated documentation
- Added type hinting to `Colley` class
- Added type hinting to `Massey` class

v0.8.1 (2023-09-31)
^^^^^^^^^^^^^^^^^^^^
- Changed fbref `born` column to Int64 dtype to allow NULL values

v0.8.0 (2023-08-31)
^^^^^^^^^^^^^^^^^^^^
- Added initial Backtest framework for backtesting betting stategies
- Added function to calculate the Kelly Criterion
- Added class for calculating Elo ratings
- Fixed bug in FBRef scraper for player age and year of birth
- All goal models can now take iterables as team inputs
- Fixed mapping of Belgium leagues in football-data scraper

v0.7.0 (2023-03-13)
^^^^^^^^^^^^^^^^^^^^^^
- Added FBRef scraper
- Minimum python version supported is now py3.8

v0.6.1 (2023-01-06)
^^^^^^^^^^^^^^^^^^^^^^
- Tweaked Understat scraper to avoid their bot detection

v0.6.0 (2022-12-02)
^^^^^^^^^^^^^^^^^^^^^^

- Added `goal expectancy` function
- Fixed bug in Bayesian Bivariate Goals model
- Added Bayesian Random Intercept model
- Tweaked pymc settings for Bayesian goal models so should now run faster
- Fixed bug in Footballdata scraper where a null value was breaking the index column

v0.5.1 (2022-11-03)
^^^^^^^^^^^^^^^^^^^^^

- Fixed bug in goals models when printing out instance before fitting it
- Changed the default value for `xi` in `dixon_coles_weights` weights function to `0.0018`
- Improved how the weighted decay was applied in the Bayesian goal models


v0.5.0 (2022-10-11)
^^^^^^^^^^^^^^^^^^^^^

- Added `get_player_season` to understat scraper
- Added `get_player_shots` to understat scraper
- Understat scraper `get_fixtures` only returns fixtures that have been completed to make consistent with FootballData scraper
- Fixed bug in FootballData scraper for older seasons lacking the `Time` column
- Added initial SoFifa scraper
- Added Bayesian Hierarchical Goal Model for predicting outcomes of football (soccer) matches
- Added Bayesian Bivariate Poisson Goal Model for predicting outcomes of football (soccer) matches
- Added Bayesian Random Intercept Poisson Goal Model for predicting outcomes of football (soccer) matches
- Added compatibility for Python 3.7 (was previously Python >=3.8)


v0.4.0 (2022-08-08)
^^^^^^^^^^^^^^^^^^^^^

- General bug fixes
- Reorganized internal structure of package
- Added unit tests
- Added documention and uploaded to readthedocs
- Added FPL scraper
- Added FPL optimizer
- Added ESPN scraper
- Added Understat scraper
- Refactored FootballData scraper to make consistent with other scrapers
- Refactored Club Elo scraper to make consistent with other scrapers
- Added both teams to score probability to football goals models
- Added pre-commit checks to repo
- Updated examples notebooks and added to docs
- Refactored Colley ratings to make consistent
- Refactored Massey ratings to make consistent
