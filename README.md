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

**penaltyblog** is a production-ready Python package designed for football (soccer) analytics, providing powerful tools from [pena.lt/y/blog](https://pena.lt/y/blog) for data analysis, outcome modelling, and betting insights. Optimized with Cython, **penaltyblog** delivers high-performance modelling to power faster, efficient predictions.

## Features

- 🔄 **Streamline JSON Workflows with MatchFlow:** Process nested football data using a lazy, streaming pipeline built for JSON. Filter, select, flatten, join, group, and summarize large datasets without loading everything into memory.
- 🔌 **Connect to Professional APIs:** Seamlessly stream and filter data directly from industry leaders, like StatsBomb and Opta. Query matches, events, and stats using lazy loading without handling massive JSON dumps.
- 📊 **Model Matches Efficiently:** High-performance implementations of Poisson, Bivariate Poisson, Dixon-Coles, and other advanced statistical models, optimized with Cython for rapid analysis.
- ⚽ **Scrape Data:** Collect match statistics from sources like Understat, Club Elo, and Fantasy Premier League.
- 💰 **Bet Smarter:** Precisely estimate probabilities for Asian handicaps, over/under totals, match outcomes, and more.
- 🏆 **Rank Teams:** Evaluate team strengths with sophisticated methods including Elo, Massey, Colley, and Pi ratings.
- 📈 **Decode Bookmaker Odds:** Accurately extract implied probabilities by removing bookmaker margins (overrounds).
- 🎯 **Fantasy Football Optimisation:** Mathematically optimize your fantasy football squad to maximize performance.
- 🎨 **Visualize with Style:** Create publication-ready pitch visualizations and data flow diagrams with customizable themes, supporting multiple data providers and flexible layouts.

Take your football analytics and betting strategy to the next level with **penaltyblog** 🚀

## Installation

```bash
pip install penaltyblog
```

## 🚀 Quick Start - Try it Now!

Run these examples directly in your browser (no installation required):

| Example | Description | Colab |
|---------|-------------|-------|
| **Predict Soccer Match Results** | Build a match prediction model from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GjrDG_iq_9_lxEQK_aBmr-jCCCnFt0v7?usp=sharing) |
| **Process Soccer Data the Easy Way** | Lazy processing of football data using Matchflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rRJV8mNOTLTXmn5cOGT4faxIwIP44pC-?usp=sharing) |
| **Calculate Massey Ratings** | Calculate teams' attack and defense strengths | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d_WPJwQgrogeSI9oIO9fY8s18CPPZ8nL?usp=sharing) |
| **Use Pi Ratings** | Like Elo ratings, but better | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12qEDCNYG-FFHOJ_kURe0cm80sScandyh?usp=sharing) |
| **Create Interactive Charts** | Create your own interactive football vizualisations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xFfIdvmbFcjHlS_2eHEu3NxD-xLNrbpY?usp=sharing) |
| **Work Directly With Statsbomb Data** | Connect directly to Statsbomb's API, including their free open data sets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xFfIdvmbFcjHlS_2eHEu3NxD-xLNrbpY?usp=sharing) |
| **Calculate Implied Probabilities** | Calculate implied probabilities from bookmaker's odds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o-tOetyWmSY_1WczN8WhWsl62Uz5T65F?usp=sharing) |


## Documentation

Learn more about how to utilize `penaltyblog` by exploring the [official documentation](https://penaltyblog.readthedocs.io/en/latest/) and detailed examples:

- [Processing football event data with MatchFlow](https://penaltyblog.readthedocs.io/en/latest/matchflow/index.html)
- [Scraping football data](https://penaltyblog.readthedocs.io/en/latest/scrapers/index.html)
- [Predicting football matches and betting markets](https://penaltyblog.readthedocs.io/en/latest/models/index.html)
- [Estimating implied odds from bookmaker prices](https://penaltyblog.readthedocs.io/en/latest/implied/index.html)
- [Calculating Massey, Colley, Pi, and Elo ratings](https://penaltyblog.readthedocs.io/en/latest/ratings/index.html)
- [Calculating metrics such as Ranked Probability Scores](https://penaltyblog.readthedocs.io/en/latest/metrics/index.html)

## Why Penaltyblog?

Unlike many football analytics resources that are academic, one-off, or hard to scale, `penaltyblog` is designed from the ground up to be **production-ready**, **performance-optimized**, and **practically useful**.

It combines advanced statistical models, efficient implementations (via **Cython**), and real-world workflows, from scraping public data to modelling outcomes and optimising fantasy teams.

Built by [Martin Eastwood](https://pena.lt/y/about), who has worked with football clubs, governing bodies, player agencies and betting syndicates worldwide, `penaltyblog` is the foundation for football analytics.

Whether you're a club analyst, independent researcher, or just data-curious, `penaltyblog` gives you the tools to go from data to insight - fast, flexibly, and with confidence.

## Support & Collaboration

If you're working at a football club, agency, or organisation and want help applying these tools to your own data or workflows, I'm open to:

- 📂 Integration support
- 🔧 Custom model development
- 🧠 Technical collaboration on football analytics projects

➡️ Get in touch [here](https://pena.lt/y/contact)
