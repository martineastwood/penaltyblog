"""
Opta API integration modules.

This package contains the refactored Opta API functionality,
separated into focused modules for better maintainability.
"""

from .client import OptaClient
from .exceptions import OptaAPIError, OptaAuthenticationError, OptaRequestError
from .parsers import (
    extract_match_events,
    extract_player_stats,
    extract_season_player_stats,
    extract_season_team_stats,
    extract_team_stats,
    flatten_stats,
    parse_match_basic,
    parse_match_stats_basic,
    parse_match_stats_player,
    parse_match_stats_team,
    parse_tournament_schedule,
)

__all__ = [
    "OptaClient",
    "OptaAPIError",
    "OptaAuthenticationError",
    "OptaRequestError",
    "extract_match_events",
    "extract_player_stats",
    "extract_season_player_stats",
    "extract_season_team_stats",
    "extract_team_stats",
    "flatten_stats",
    "parse_match_basic",
    "parse_match_stats_basic",
    "parse_match_stats_player",
    "parse_match_stats_team",
    "parse_tournament_schedule",
]
