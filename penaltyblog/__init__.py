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
