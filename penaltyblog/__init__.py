import platform

from . import backtest, fpl, implied, kelly, metrics, models, ratings, scrapers
from .version import __version__


def install_stan():
    """
    Install Stan runtime for compiling the Bayesian goal models
    """
    import cmdstanpy

    cmdstanpy.install_cmdstan(compiler=True)
