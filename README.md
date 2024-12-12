<img src="https://raw.githubusercontent.com/martineastwood/penaltyblog/refs/heads/master/logo.png" width="0" height="0" style="display:none;"/>

<meta property="og:image" content="https://raw.githubusercontent.com/martineastwood/penaltyblog/refs/heads/master/logo.png" />
<meta property="og:image:alt" content="penaltyblog python package for soccer modeling" />
<meta name="twitter:image" content="https://raw.githubusercontent.com/martineastwood/penaltyblog/refs/heads/master/logo.png">
<meta name="twitter:card" content="summary_large_image">

# Penalty Blog

<div align="center">

  <a href="">[![Python Version](https://img.shields.io/pypi/pyversions/penaltyblog)](https://pypi.org/project/penaltyblog/)</a>
<a href="https://codecov.io/github/martineastwood/penaltyblog" >
<img src="https://codecov.io/github/martineastwood/penaltyblog/branch/master/graph/badge.svg?token=P0WDHRGIG2"/>
</a>
  <a href="">[![PyPI](https://img.shields.io/pypi/v/penaltyblog.svg)](https://pypi.org/project/penaltyblog/)</a>
  <a href="">[![Downloads](https://static.pepy.tech/badge/penaltyblog)](https://pepy.tech/project/penaltyblog)</a>
  <a href="">[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)</a>
  <a href="">[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)</a>
  <a href="">[![Code style: pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)</a>

</div>


<div align="center">
  <img src="logo.png" alt="Penalty Blog Logo" width="200">
</div>



The **penaltyblog** Python package contains lots of useful code from [pena.lt/y/blog](http://pena.lt/y/blog.html) for working with football (soccer) data.

**penaltyblog** includes functions for:

- Scraping football data from sources such as football-data.co.uk, FBRef, Club Elo, Understat and Fantasy Premier League
- Modelling of football matches using Poisson-based models, such as Dixon and Coles, and Bayesian models
- Predicting probabilities for many betting markets, e.g. Asian handicaps, over/under, total goals etc
- Modelling football team's abilities using Massey ratings, Colley ratings and Elo ratings
- Estimating the implied odds from bookmaker's odds by removing the overround using multiple different methods
- Estimating goal expectancy from bookmaker's odds
- Mathematically optimising your fantasy football team

## Installation

`pip install penaltyblog`


## Stan

The Bayesian models in `penaltyblog` require the [Stan probabilistic programming language](https://mc-stan.org/) to function. You can use the following command to install Stan directly from the penaltyblog package:

```python
import penaltyblog as pb

pb.install_stan()
```

## Documentation

To learn how to use `penaltyblog`, you can read the [documentation](https://penaltyblog.readthedocs.io/en/stable/) and look at the
examples for:

- [Scraping football data](https://penaltyblog.readthedocs.io/en/stable/scrapers/index.html)
- [Predicting football matches and betting markets](https://penaltyblog.readthedocs.io/en/stable/models/index.html)
- [Estimating the implied odds from bookmakers odds](https://penaltyblog.readthedocs.io/en/stable/implied/index.html)
- [Calculate Massey, Colley and Elo ratings](https://penaltyblog.readthedocs.io/en/stable/ratings/index.html)

## References

- Mark J. Dixon and Stuart G. Coles (1997) Modelling Association Football Scores and Inefficiencies in the Football Betting Market
- Håvard Rue and Øyvind Salvesen (1999) Prediction and Retrospective Analysis of Soccer Matches in a League
- Anthony C. Constantinou and Norman E. Fenton (2012) Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models
- Hyun Song Shin (1992) Prices of State Contingent Claims with Insider Traders, and the Favourite-Longshot Bias
- Hyun Song Shin (1993) Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims
- Joseph Buchdahl (2015) The Wisdom of the Crowd
- Gianluca Baio and Marta A. Blangiardo (2010) Bayesian Hierarchical Model for the Prediction of Football Results
