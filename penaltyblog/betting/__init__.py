"""
Betting Utilities

Functions for bet sizing, Kelly Criterion, arbitrage, and other betting strategies.
"""

from .arbitrage import arbitrage_hedge  # noqa
from .kelly import kelly_criterion, multiple_kelly_criterion  # noqa

__all__ = [
    "kelly_criterion",
    "multiple_kelly_criterion",
    "arbitrage_hedge",
]
