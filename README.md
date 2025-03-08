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


# penaltyblog: Football Data & Modelling Made Easy

**penaltyblog** is a production-ready Python package designed for football (soccer) analytics, providing powerful tools from [pena.lt/y/blog](https://pena.lt/y/blog) for data analysis, outcome modelling, and betting insights. Otimized with Cython, **penaltyblog** delivers high-performance modelling to power faster, efficient predictions.

## Features
- ⚽ **Scrape Data** – Collect match statistics from sources like FBRef, Understat, Club Elo, and Fantasy Premier League.
- 📊 **Model Matches Efficiently** – High-performance implementations of Poisson, Bivariate Poisson, Dixon-Coles, and other advanced statistical models, optimized with Cython for rapid analysis.
- 💰 **Bet Smarter** – Precisely estimate probabilities for Asian handicaps, over/under totals, match outcomes, and more.
- 🏆 **Rank Teams** – Evaluate team strengths with sophisticated methods including Elo, Massey, Colley, and Pi ratings.
- 📈 **Decode Bookmaker Odds** – Accurately extract implied probabilities by removing bookmaker margins (overrounds).
- 🎯 **Fantasy Football Optimisation** – Mathematically optimize your fantasy football squad to maximize performance.

Take your football analytics and betting strategy to the next level with **penaltyblog** 🚀

## Installation

```bash
pip install penaltyblog
```

## Documentation

Learn more about how to utilize `penaltyblog` by exploring the [official documentation](https://penaltyblog.readthedocs.io/en/stable/) and detailed examples:

- [Scraping football data](https://penaltyblog.readthedocs.io/en/stable/scrapers/index.html)
- [Predicting football matches and betting markets](https://penaltyblog.readthedocs.io/en/stable/models/index.html)
- [Estimating implied odds from bookmaker prices](https://penaltyblog.readthedocs.io/en/stable/implied/index.html)
- [Calculating Massey, Colley, and Elo ratings](https://penaltyblog.readthedocs.io/en/stable/ratings/index.html)

## References

- Baio, Gianluca, and Marta A. Blangiardo (2010). *Bayesian Hierarchical Model for the Prediction of Football Results*.
- *A bivariate Weibull count model for forecasting association football scores*
- Boshnakov, Georgi, Tarak Kharrat, and Ian G. McHale (2017). A bivariate Weibull count model for forecasting association football scores.
- Buchdahl, Joseph (2015). *The Wisdom of the Crowd*.
- Constantinou, Anthony C., and Norman E. Fenton (2012). *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models*.
- Constantinou, Anthony C., and Norman E. Fenton (2013). *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries*.
- Dixon, Mark J., and Stuart G. Coles (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*.
- Karlis, Dimitris, and Ioannis Ntzoufras (2003). *Analysis of Sports Data by Using Bivariate Poisson Models*.
- Rue, Håvard, and Øyvind Salvesen (1999). *Prediction and Retrospective Analysis of Soccer Matches in a League*.
- Shin, Hyun Song (1992). *Prices of State Contingent Claims with Insider Traders, and the Favourite-Longshot Bias*.
- Shin, Hyun Song (1993). *Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims*.
