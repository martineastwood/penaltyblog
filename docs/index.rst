Welcome to penaltyblog!
=======================================

The **penaltyblog** Python package contains lots of useful code from http://pena.lt/y/blog for working with football (soccer) data.

**penaltyblog** includes functions for:

- Scraping football data from sources such as football-data.co.uk, ESPN, Club Elo, Understat and Fantasy Premier League
- Modelling of football matches using Poisson-based models, such as Dixon and Coles
- Predicting probabilities for many betting markets, e.g. Asian handicaps, over/under, total goals etc
- Modelling football team's abilities using Massey ratings and Colley ratings
- Estimating the implied odds from bookmaker's odds by removing the overround using multiple different methods
- Mathematically optimising your fantasy football team by formation, budget etc


Installation
#################

.. code:: bash

   pip install penaltyblog


.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   fpl/index
   implied/index
   metrics/index
   models/index
   ratings/index
   scrapers/index
