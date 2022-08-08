# Penalty Blog

<div align="center">

  <a href="">[![Python Version](https://img.shields.io/pypi/pyversions/penaltyblog)](https://pypi.org/project/penaltyblog/)</a>
  <a href="">[![Coverage Status](https://coveralls.io/repos/github/martineastwood/penaltyblog/badge.svg?branch=master&service=github)](https://coveralls.io/repos/github/martineastwood/penaltyblog/badge.svg?branch=master&service=github)</a>
  <a href="">[![PyPI](https://img.shields.io/pypi/v/penaltyblog.svg)](https://pypi.org/project/penaltyblog/)</a>
  <a href="">[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)</a>
  <a href="">[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)</a>
  <a href="">[![Code style: pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)</a>

</div>


The **penaltyblog** Python package contains lots of useful code from [pena.lt/y/blog](http://pena.lt/y/blog.html) for working with football (soccer) data.

**penaltyblog** includes functions for:

- Scraping football data from sources such as football-data.co.uk, ESPN, Club Elo, Understat and Fantasy Premier League
- Modelling of football matches using Poisson-based models, such as Dixon and Coles
- Predicting probabilities for many betting markets, e.g. Asian handicaps, over/under, total goals etc
- Modelling football team's abilities using Massey ratings and Colley ratings
- Estimating the implied odds from bookmaker's odds by removing the overround using multiple different methods
- Mathematically optimising your fantasy football team

## Installation

`pip install penaltyblog`


## Documentation

To learn how to use penaltyblog, you can read the [documentation](https://penaltyblog.readthedocs.io/en/latest/) and look at the
examples for:

- [Scraping football data](https://penaltyblog.readthedocs.io/en/latest/scrapers/index.html)
- [Predicting football matches and betting markets](https://penaltyblog.readthedocs.io/en/latest/models/index.html)
- [Estimating the implied odds from bookmakers odds](https://penaltyblog.readthedocs.io/en/latest/implied/index.html)
- [Calculate Massey and Colley ratings](https://penaltyblog.readthedocs.io/en/latest/ratings/index.html)

## References

- Mark J. Dixon and Stuart G. Coles (1997) Modelling Association Football Scores and Inefficiencies in the Football Betting Market.
- Håvard Rue and Øyvind Salvesen (1999) Prediction and Retrospective Analysis of Soccer Matches in a League.
- Anthony C. Constantinou and Norman E. Fenton (2012) Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models
- Hyun Song Shin (1992) Prices of State Contingent Claims with Insider Traders, and the Favourite-Longshot Bias
- Hyun Song Shin (1993) Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims
- Joseph Buchdahl (2015) The Wisdom of the Crowd
