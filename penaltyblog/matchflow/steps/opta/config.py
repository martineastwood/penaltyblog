"""
Configuration and constants for Opta API endpoints.
"""

from typing import Dict, List, Set

# Sources that are not paginated
NON_PAGINATED_SOURCES: Set[str] = {
    "tournament_schedule",  # MA0
    "match_basic",  # MA1 (Single)
    "match_stats_basic",  # MA2 (Single or Multi via fx)
    "match_events",  # MA3
    "player_season_stats",  # TM4
    "team_season_stats",  # TM4
    "contestant_participation",  # TM16
    "xg_shots",  # MA12
    "xg_player_summary",  # MA12
    "xg_team_summary",  # MA12
}

# Default pagination settings
DEFAULT_PAGE_SIZE = 1000
DEFAULT_PAGE_NUM = 1

# Endpoint configurations
ENDPOINT_CONFIGS: Dict[str, Dict] = {
    "tournament_calendars": {
        "path_template": "/tournamentcalendar/{auth_key}",
        "status_variants": {
            "active": "/active",
            "authorized": "/authorized",
            "active_authorized": "/active/authorized",
            "all": "",
        },
    },
    "tournament_schedule": {
        "path_template": "/tournamentschedule/{auth_key}/{tournament_calendar_uuid}",
    },
    "matches_basic": {
        "path_template": "/match/{auth_key}",
    },
    "match_basic": {
        "path_template": "/match/{auth_key}/{fixture_uuid}",
    },
    "match_stats_basic": {
        "path_template": "/matchstats/{auth_key}",
        "supports_multi": True,  # Can handle multiple fixture_uuids via fx param
    },
    "match_events": {
        "path_template": "/matchevent/{auth_key}/{fixture_uuid}",
    },
    "teams": {
        "path_template": "/team/{auth_key}",
    },
    "squads": {
        "path_template": "/squads/{auth_key}",
    },
    "player_season_stats": {
        "path_template": "/seasonstats/{auth_key}",
    },
    "team_season_stats": {
        "path_template": "/seasonstats/{auth_key}",
    },
    "contestant_participation": {
        "path_template": "/contestantparticipation/{auth_key}",
    },
}

# Parameter mappings for different endpoints
PARAMETER_MAPPINGS: Dict[str, Dict[str, str]] = {
    "matches_basic": {
        "fixture_uuids": "fx",
        "tournament_calendar_uuid": "tmcl",
        "competition_uuids": "comp",
        "contestant_uuid": "ctst",
        "opponent_uuid": "ctst2",
        "contestant_position": "ctstpos",
        "date_range": "mt.mDt",
        "delta_timestamp": "_dlt",
        "use_opta_names": "_lcl",
        "tmcl": "tmcl",
        "ctst": "ctst",
        "ctst2": "ctst2",
        "ctstpos": "ctstpos",
        "comp": "comp",
        "mt_mDt": "mt.mDt",
        "_dlt": "_dlt",
        "_lcl": "_lcl",
    },
    "match_basic": {
        "use_opta_names": "_lcl",
    },
    "match_stats_basic": {
        "use_opta_names": "_lcl",
    },
    "match_events": {
        "contestant_uuid": "ctst",
        "person_uuid": "prsn",
        "event_types": "type",
        "use_opta_names": "_lcl",
    },
    "teams": {
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
        "country_uuid": "ctry",
        "stage_uuid": "stg",
        "series_uuid": "srs",
    },
    "squads": {
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
        "use_opta_names": "_lcl",
    },
    "player_season_stats": {
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
    },
    "team_season_stats": {
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
    },
    "contestant_participation": {
        "contestant_uuid": "ctst",
        "active": "active",
    },
}

# Response parsing configurations
RESPONSE_PARSERS: Dict[str, str] = {
    "tournament_schedule": "parse_tournament_schedule",
    "match_basic": "parse_match_basic",
    "match_stats_basic": "parse_match_stats_basic",
    "match_events": "parse_match_events",
    "player_season_stats": "parse_season_player_stats",
    "team_season_stats": "parse_season_team_stats",
}

# Pagination response key mappings
PAGINATION_RESPONSE_KEYS: Dict[str, List[str]] = {
    "tournament_calendars": ["competition"],
    "matches_basic": ["match", "matches.match"],
    "teams": ["contestants.contestant", "contestant"],
    "squads": ["squad", "teamSquads.squad"],
    "contestant_participation": ["contestant.competition", "competition"],
}
