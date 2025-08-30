"""
Betting Utilities

Functions for bet sizing, Kelly Criterion, arbitrage, value betting, and other betting strategies.
"""

from .arbitrage import arbitrage_hedge  # noqa
from .kelly import kelly_criterion, multiple_kelly_criterion  # noqa
from .odds import convert_odds  # noqa
from .value_bets import (  # noqa
    ArbitrageResult,
    MultipleValueBetResult,
    ValueBetResult,
    calculate_bet_value,
    find_arbitrage_opportunities,
    identify_value_bet,
)

__all__ = [
    "kelly_criterion",
    "multiple_kelly_criterion",
    "arbitrage_hedge",
    "identify_value_bet",
    "calculate_bet_value",
    "find_arbitrage_opportunities",
    "ValueBetResult",
    "MultipleValueBetResult",
    "ArbitrageResult",
    "convert_odds",
]
