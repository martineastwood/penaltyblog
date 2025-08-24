"""
Betting Utilities

Functions for bet sizing, Kelly Criterion, arbitrage, and other betting strategies.
"""

from .criterion import arbitrage_hedge, criterion, multiple_criterion  # noqa

__all__ = [
    "criterion",
    "multiple_criterion",
    "arbitrage_hedge",
]
