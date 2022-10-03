Changelog
===========

Version Numbering
#################

penaltyblog follows the SemVer versioning guidelines. For more information,
see `semver.org <http://semver.org/>`_

v0.5.0 (xxxxxx)
^^^^^^^^^^^^^^^^^^^

- Added `get_player_season` to understat scraper
- Added `get_player_shots` to understat scraper
- Understat scraper `get_fixtures` only returns fixtures that have been completed to make consistent with FootballData scraper
- Fixed bug in FootballData scraper for older seasons lacking the `Time` column
- Added initial SoFifa scraper
- Added Bayesian Hierarchical Goal Model for predicting outcomes of football (soccer) matches
- Added Bayesian Bivariate Poisson Goal Model for predicting outcomes of football (soccer) matches
- Added Bayesian Random Intercept Poisson Goal Model for predicting outcomes of football (soccer) matches


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
