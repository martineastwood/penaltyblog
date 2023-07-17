Welcome to penaltyblog!
=======================================

The **penaltyblog** Python package contains lots of useful code from http://pena.lt/y/blog for working with football (soccer) data.

**penaltyblog** includes functions for:

- Scraping football data from sources such as football-data.co.uk, ESPN, Club Elo, Understat and Fantasy Premier League
- Modelling of football matches using Poisson-based models, such as Dixon and Coles
- Predicting probabilities for many betting markets, e.g. Asian handicaps, over/under, total goals etc
- Modelling football team's abilities using Massey ratings, Colley ratings and Elo ratings
- Estimating the implied odds from bookmaker's odds by removing the overround using multiple different methods
- Backtesting betting strategies
- Mathematically optimising your fantasy football team by formation, budget etc


Installation
#################

.. code:: bash

   pip install penaltyblog


.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   backtest/index
   fpl/index
   implied/index
   metrics/index
   models/index
   ratings/index
   scrapers/index
   changelog/index


Useful Links
####################

- `Walkthrough of using the web scrapers <http://www.pena.lt/y/2022/08/05/scraping-football-data-using-penaltyblog-python-package/>`_
- `Poisson goal model example <http://www.pena.lt/y/2021/06/18/predicting-football-results-using-the-poisson-distribution/>`_
- `Dixon and Coles goal model example <http://www.pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/>`_
- `Bayesian Hierarchical goal model example <http://www.pena.lt/y/2021/08/25/predicting-football-results-using-bayesian-statistics-with-python-and-pymc3/>`_
- `How to calculate goal expectancy <https://pena.lt/y/2022/12/02/goal-expectancy-from-bookmakers-odds-using-python/>`_
