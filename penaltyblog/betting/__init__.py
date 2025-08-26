"""
Betting Utilities

Functions for bet sizing, Kelly Criterion, arbitrage, and other betting strategies.
"""

from .arbitrage import arbitrage_hedge  # noqa
from .criterion import criterion, multiple_criterion  # noqa

__all__ = [
    "criterion",
    "multiple_criterion",
    "arbitrage_hedge",
]
