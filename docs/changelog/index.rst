Changelog
===========

Version Numbering
#################

`penaltyblog` follows the SemVer versioning guidelines. For more information,
see `semver.org <http://semver.org/>`_

v0.8.0 (xxx)
^^^^^^^^^^^^^^
- Added Backtest framework for backtesting betting stategies
- Added function to calculate the Kelly Criterion
- Added class for calculating Elo ratings
- Fixed bug in FBRef scraper for player age and year of birth
- All goal models can now take iterables as team inputs

v0.7.0 (2023-03-13)
^^^^^^^^^^^^^^
- Added FBRef scraper
- Minimum python version supported is now py3.8

v0.6.1 (2023-01-06)
^^^^^^^^^^^^^^
- Tweaked Understat scraper to avoid their bot detection

v0.6.0 (2022-12-02)
^^^^^^^^^^^^^^

- Added `goal expectancy` function
- Fixed bug in Bayesian Bivariate Goals model
- Added Bayesian Random Intercept model
- Tweaked pymc settings for Bayesian goal models so should now run faster
- Fixed bug in Footballdata scraper where a null value was breaking the index column

v0.5.1 (2022-11-03)
^^^^^^^^^^^^^^^^^^^

- Fixed bug in goals models when printing out instance before fitting it
- Changed the default value for `xi` in `dixon_coles_weights` weights function to `0.0018`
- Improved how the weighted decay was applied in the Bayesian goal models


v0.5.0 (2022-10-11)
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

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
