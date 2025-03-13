.. _api_ref:

API Reference
=============

.. currentmodule:: penaltyblog

Backtest
----------------------------

.. currentmodule:: penaltyblog.backtest

.. autosummary::
    :toctree: generated/
    :recursive:
    :template: autosummary/class_with_base.rst

    Account
    Backtest
    Context


Fantasy Premier League
----------------------------

.. currentmodule:: penaltyblog.fpl

.. autosummary::
    :toctree: generated/

    get_current_gameweek
    get_entry_picks_by_gameweek
    get_entry_transfers
    get_gameweek_info
    get_player_id_mappings
    get_player_history
    get_rankings
    optimise_team


Kelly Criterion
----------------------------

.. currentmodule:: penaltyblog.kelly

.. autosummary::
    :toctree: generated/

    criterion


Implied Probability
----------------------------

.. currentmodule:: penaltyblog.implied

.. autosummary::
    :toctree: generated/

    multiplicative
    additive
    power
    shin
    differential_margin_weighting
    odds_ratio


Metrics
----------------------------

.. currentmodule:: penaltyblog.metrics

.. autosummary::
    :toctree: generated/

    rps_average
    rps_array


Predictive Models
----------------------------

.. currentmodule:: penaltyblog.models

.. autosummary::
    :toctree: generated/
    :recursive:
    :template: autosummary/class_with_base.rst

    PoissonGoalsModel
    DixonColesGoalModel
    BivariatePoissonGoalModel
    ZeroInflatedPoissonGoalsModel
    NegativeBinomialGoalModel
    WeibullCopulaGoalsModel
    FootballProbabilityGrid
    goal_expectancy
    dixon_coles_weights


Ratings and Rankings
----------------------------

.. currentmodule:: penaltyblog.ratings

.. autosummary::
    :toctree: generated/
    :recursive:
    :template: autosummary/class_with_base.rst

    Colley
    Massey
    Elo
    PiRatingSystem


Scrapers
----------------------------

.. currentmodule:: penaltyblog.scrapers

.. autosummary::
    :toctree: generated/
    :recursive:
    :template: autosummary/class_with_base.rst

    ClubElo
    FBRef
    FootballData
    Understat
