.. penaltyblog documentation master file, created by
   sphinx-quickstart on Tue Aug 31 11:27:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to penaltyblog!
=======================================

The **penaltyblog** Python package contains lots of useful code from http://pena.lt/y/blog for working with football (soccer) data.

**penaltyblog** includes functions for:

- Scraping football data from sources such as football-data.co.uk, ESPN, Club Elo
- Modelling the outcomes of football matches using Poisson-based models, such as Dixon and Coles
- Modelling football team's abilities using Massey ratings and Colley ratings
- Estimating the implied odds from bookmaker's odds by removing the overround
- Mathematically optimising your fantasy football team


Installation
#################

.. code:: bash

   pip install penaltyblog


.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   implied/index
   models/index
   ratings/index
   scrapers/index
   metrics/index
   reference/index
