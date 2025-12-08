"""
Configuration and constants for Opta API endpoints.
"""

from typing import Dict, List, Set

# Sources that are not paginated
NON_PAGINATED_SOURCES: Set[str] = {
    "referees_person",
    "rankings",  # PE4
    "injuries_person_path",
    "player_career_person",
    "area_specific",
    "tournament_schedule",  # MA0
    "match_basic",  # MA1 (Single)
    "match_stats_player",  # MA2 (Player stats)
    "match_stats_team",  # MA2 (Team stats)
    "match_events",  # MA3
    "pass_matrix",  # MA4
    "possession",  # MA5
    "player_season_stats",  # TM4
    "team_season_stats",  # TM4
    "contestant_participation",  # TM16
    "team_standings",  # TM2
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
    "venues": {
        "path_template": "/venues/{auth_key}",
    },
    "tournament_schedule": {
        "path_template": "/tournamentschedule/{auth_key}/{tournament_calendar_uuid}",
    },
    "areas_all": {
        "path_template": "/areas/{auth_key}",
    },
    "area_specific": {
        "path_template": "/areas/{auth_key}/{area_uuid}",
    },
    "matches_basic": {
        "path_template": "/match/{auth_key}",
    },
    "match_basic": {
        "path_template": "/match/{auth_key}/{fixture_uuid}",
    },
    "match_stats_player": {
        "path_template": "/matchstats/{auth_key}",
        "supports_multi": True,  # Can handle multiple fixture_uuids via fx param
    },
    "match_stats_team": {
        "path_template": "/matchstats/{auth_key}",
        "supports_multi": True,  # Can handle multiple fixture_uuids via fx param
    },
    "match_events": {
        "path_template": "/matchevent/{auth_key}/{fixture_uuid}",
    },
    "pass_matrix": {
        "path_template": "/passmatrix/{auth_key}/{fixture_uuid}",
    },
    "possession": {
        "path_template": "/possession/{auth_key}/{fixture_uuid}",
    },
    "player_career_person": {
        "path_template": "/playercareer/{auth_key}/{person_uuid}",
    },
    "player_career_contestant": {
        "path_template": "/playercareer/{auth_key}",
    },
    "injuries_person_path": {
        "path_template": "/injuries/{auth_key}/{person_uuid}",
    },
    "injuries_query": {
        "path_template": "/injuries/{auth_key}",
    },
    "referees": {
        "path_template": "/referees/{auth_key}",
    },
    "referees_person": {
        "path_template": "/referees/{auth_key}",
    },
    "rankings": {
        "path_template": "/rankings/{auth_key}",
        "live_path_template": "/rankings/{auth_key}",
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
    "transfers": {
        "path_template": "/transfers/{auth_key}",
    },
    "team_standings": {
        "path_template": "/standings/{auth_key}",
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
        "live": "live",
        "lineups": "lineups",
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
    "match_stats_player": {
        "fixture_uuids": "fx",
        "use_opta_names": "_lcl",
    },
    "match_stats_team": {
        "fixture_uuids": "fx",
        "use_opta_names": "_lcl",
    },
    "match_events": {
        "contestant_uuid": "ctst",
        "person_uuid": "prsn",
        "event_types": "type",
        "use_opta_names": "_lcl",
    },
    "pass_matrix": {
        "use_opta_names": "_lcl",
    },
    "possession": {
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
    "areas_all": {
        "use_opta_names": "_lcl",
    },
    "area_specific": {
        "use_opta_names": "_lcl",
    },
    "venues": {
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
        "venue_uuid": "venue",
        "use_opta_names": "_lcl",
    },
    "player_career_person": {
        "person_uuid": "prsn",
        "use_opta_names": "_lcl",
    },
    "player_career_contestant": {
        "contestant_uuid": "ctst",
        "active": "active",
        "use_opta_names": "_lcl",
    },
    "injuries_person_path": {
        "person_uuid": "prsn",
        "use_opta_names": "_lcl",
    },
    "injuries_query": {
        "person_uuid": "prsn",
        "tournament_calendar_uuid": "tmcl",
        "contestant_uuid": "ctst",
        "use_opta_names": "_lcl",
    },
    "referees": {
        "person_uuid": "prsn",
        "tournament_calendar_uuid": "tmcl",
        "stage_uuid": "stg",
        "use_opta_names": "_lcl",
    },
    "referees_person": {
        "person_uuid": "prsn",
        "tournament_calendar_uuid": "tmcl",
        "stage_uuid": "stg",
        "use_opta_names": "_lcl",
    },
    "rankings": {
        "tournament_calendar_uuid": "tmcl",
        "live": "live",
        "use_opta_names": "_lcl",
    },
    "transfers": {
        "person_uuid": "prsn",
        "contestant_uuid": "ctst",
        "competition_uuid": "comp",
        "tournament_calendar_uuid": "tmcl",
        "start_date": "strtDt",
        "end_date": "endDt",
        "use_opta_names": "_lcl",
    },
    "team_standings": {
        "tournament_calendar_uuid": "tmcl",
        "stage_uuid": "stg",
        "live": "live",
        "type": "type",
        "use_opta_names": "_lcl",
    },
}

# Response parsing configurations
RESPONSE_PARSERS: Dict[str, str] = {
    "transfers": "parse_transfers",
    "injuries_person_path": "parse_injuries_person",
    "injuries_query": "parse_injuries_query",
    "player_career_person": "parse_player_career_person",
    "area_specific": "parse_area_specific",
    "tournament_schedule": "parse_tournament_schedule",
    "match_basic": "parse_match_basic",
    "match_stats_player": "parse_match_stats_player",
    "match_stats_team": "parse_match_stats_team",
    "match_events": "parse_match_events",
    "pass_matrix": "parse_pass_matrix",
    "possession": "parse_possession",
    "player_season_stats": "parse_season_player_stats",
    "team_season_stats": "parse_season_team_stats",
}

# Pagination response key mappings
PAGINATION_RESPONSE_KEYS: Dict[str, List[str]] = {
    "transfers": ["person"],
    "referees": ["referee"],
    "injuries_query": ["person"],
    "player_career_contestant": ["person"],
    "areas_all": ["area"],
    "venues": ["venue"],
    "tournament_calendars": ["competition"],
    "matches_basic": ["match", "matches.match"],
    "teams": ["contestants.contestant", "contestant"],
    "squads": ["squad", "teamSquads.squad"],
    "contestant_participation": ["contestant.competition", "competition"],
}
