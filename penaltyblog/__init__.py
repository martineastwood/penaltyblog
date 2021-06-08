from .version import __version__
from .data import get_example_data
from .rps import rps
from .poisson import (
    PoissonGoalsModel,
    DixonColesGoalModel,
    RueSalvesenGoalModel,
    dixon_coles_weights,
)
from .footballdata import fetch_data, list_countries
from .clubelo import fetch_rankings_by_date, fetch_rankings_by_team, list_all_teams
from .implied import (
    additive,
    multiplicative,
    power,
    shin,
    differential_margin_weighting,
    odds_ratio,
)
