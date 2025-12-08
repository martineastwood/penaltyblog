"""
Data parsing functions for Opta API responses.
"""

from typing import Any, Dict, Iterator, List


def flatten_stats(
    stats_list: List[Dict[str, Any]], key_name: str = "type"
) -> Dict[str, Any]:
    """
    Converts a list of {'key_name': 'name', 'value': 'val'} into a flat dict.
    key_name can be 'type' (for MA2) or 'name' (for TM4).
    """
    stats_dict = {}
    for stat in stats_list:
        stat_key = stat.get(key_name)
        if stat_key:
            value = stat.get("value")
            try:
                if isinstance(value, str) and "." in value:
                    value = float(value)
            except ValueError:
                pass
            stats_dict[stat_key] = value
    return stats_dict


def extract_player_stats(match: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests player stats from a MA2 match record."""
    match_info = match.get("matchInfo", {})
    live_data = match.get("liveData", {})
    lineups = live_data.get("lineUp", [])

    for team in lineups:
        contestant_id = team.get("contestantId")
        players = team.get("player", [])
        for player in players:
            player_stats = flatten_stats(player.get("stat", []), key_name="type")
            if not player_stats:
                continue
            player_record = {
                **player,
                **player_stats,
                "_match_uuid": match_info.get("id"),
                "_contestant_id": contestant_id,
                "_match_info": match_info,
            }
            player_record.pop("stat", None)
            yield player_record


def extract_team_stats(match: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests team stats from a MA2 match record."""
    match_info = match.get("matchInfo", {})
    live_data = match.get("liveData", {})
    lineups = live_data.get("lineUp", [])

    for team_lineup in lineups:
        team_stats_data = team_lineup.get("stat")
        if not team_stats_data:
            continue

        team_stats = flatten_stats(team_stats_data, key_name="type")
        if not team_stats:
            continue

        team_record = {
            "contestantId": team_lineup.get("contestantId"),
            **team_stats,
            "_match_uuid": match_info.get("id"),
            "_match_info": match_info,
        }
        yield team_record


def extract_match_events(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests match events from a MA3 feed response."""
    match_info = data.get("matchInfo", {})
    live_data = data.get("liveData", {})
    match_details = live_data.get("matchDetails", {})

    # Handle both possible structures: liveData.event or liveData.events.event
    event_list = live_data.get("event", [])
    if not event_list:
        # Try the nested structure
        events = live_data.get("events", {})
        event_list = events.get("event", [])

    for event in event_list:
        event["_match_info"] = match_info
        event["_match_details"] = match_details
        yield event


def extract_season_player_stats(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Un-nests player seasonal stats from a TM4 response.
    Yields one record per player from the top-level 'player' list.
    """
    competition_info = data.get("competition", {})
    tournament_info = data.get("tournamentCalendar", {})

    players = data.get("player", [])  # Get top-level player list
    if not isinstance(players, list):
        return  # No player data

    for player in players:
        # TM4 uses "name" as the key
        player_stats = flatten_stats(player.get("stat", []), key_name="name")

        player_record = {
            **player,
            **player_stats,
            "_competition": competition_info,
            "_tournamentCalendar": tournament_info,
        }
        player_record.pop("stat", None)
        yield player_record


def extract_season_team_stats(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Un-nests team seasonal stats from a TM4 response.
    Yields the single team record from the 'contestant' dict.
    """
    competition_info = data.get("competition", {})
    tournament_info = data.get("tournamentCalendar", {})

    team_data = data.get("contestant")  # Get top-level contestant dict
    if not isinstance(team_data, dict):
        return  # No team data

    team_stats = flatten_stats(team_data.get("stat", []), key_name="name")

    team_record = {
        **team_data,
        **team_stats,
        "_competition": competition_info,
        "_tournamentCalendar": tournament_info,
    }
    team_record.pop("stat", None)
    team_record.pop("player", None)  # Remove nested player list if present
    yield team_record


def parse_tournament_schedule(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse tournament schedule (MA0) response."""
    competition_info = data.get("competition", {})
    tournament_info = data.get("tournamentCalendar", {})
    match_dates = data.get("matchDate", [])
    for day in match_dates:
        for match in day.get("match", []):
            match["_competition"] = competition_info
            match["_tournamentCalendar"] = tournament_info
            match["_matchDate"] = day.get("date")
            yield match


def parse_match_basic(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse basic match (MA1) response."""
    yield data


def extract_contestant_participation(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests contestant participation from a TM16 response."""
    contestant_info = data.get("contestant", {})
    yield from contestant_info


def parse_match_stats_basic(
    data: Dict[str, Any], include_players: bool = True
) -> Iterator[Dict[str, Any]]:
    """Parse match stats (MA2) response."""
    matches = []  # Default to empty list

    # Check 1: Is it a single match object? (like MA2 single-match sample)
    if isinstance(data.get("matchInfo"), dict):
        matches = [data]

    # Check 2: Is it a list under "matchStats" key? (like MA1 multi-match sample)
    elif isinstance(data.get("matchStats"), list):
        matches = data.get("matchStats")

    for match in matches:
        if include_players:
            yield from extract_player_stats(match)
        else:
            yield from extract_team_stats(match)


def parse_match_stats_player(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse match player stats (MA2) response - only player stats."""
    matches = []  # Default to empty list

    # Check 1: Is it a single match object?
    if isinstance(data.get("matchInfo"), dict):
        matches = [data]

    # Check 2: Is it a list under "matchStats" key?
    elif isinstance(data.get("matchStats"), list):
        matches = data.get("matchStats")

    for match in matches:
        yield from extract_player_stats(match)


def parse_match_stats_team(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse match team stats (MA2) response - only team stats."""
    matches = []  # Default to empty list

    # Check 1: Is it a single match object?
    if isinstance(data.get("matchInfo"), dict):
        matches = [data]

    # Check 2: Is it a list under "matchStats" key?
    elif isinstance(data.get("matchStats"), list):
        matches = data.get("matchStats")

    for match in matches:
        yield from extract_team_stats(match)


def parse_area_specific(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse specific area (OT4) response."""
    # The response for a specific area is just the area object itself
    yield data


def parse_player_career_person(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse specific player career (PE2) response."""
    # Extract person data from the nested structure
    if "person" in data:
        if isinstance(data["person"], list):
            yield from data["person"]
        elif isinstance(data["person"], dict):
            yield data["person"]
    else:
        # Fallback: yield the entire data if no person key found
        yield data


def parse_injuries_person(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse specific player injuries (PE7) response from path parameter."""
    # The response for a specific person might be the person object itself,
    # or a list under the "person" key.
    if "person" in data and isinstance(data["person"], list):
        yield from data["person"]
    else:
        yield data


def parse_injuries_query(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Parse player injuries (PE7) response from query parameters.
    This handles both non-paginated (tmcl+prsn) and paginated (tmcl) results.
    """
    # If 'person' is a list, it's a paginated (tmcl) or (tmcl+ctst) result.
    if "person" in data and isinstance(data["person"], list):
        yield from data["person"]
    # If 'person' is a dict, it's a non-paginated (tmcl+prsn) result.
    elif "person" in data and isinstance(data["person"], dict):
        yield data["person"]
    # Fallback for unexpected structures, though 'person' should be the key.
    elif "injuries" in data:
        yield data


def parse_transfers(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Parse player transfers (TM7) response.
    This handles the non-paginated (prsn) case, which returns a single
    person object.
    """
    # The response for a specific person is just the person object itself
    if "person" in data and isinstance(data["person"], list):
        yield from data["person"]
    else:
        yield data


def parse_rankings(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Parse rankings (PE4) response.

    Yields records for matches, teams, and players found in the rankings.

    The rankings response contains three types of data:
    1. Match rankings (matchData) - Match-level statistics and rankings
    2. Team rankings (team) - Team-specific statistics and rankings
    3. Player rankings (team.player) - Player-specific statistics and rankings

    Match rankings contain combined statistics for the entire match (e.g., total goals
    scored by both teams), while team and player rankings contain individual statistics.
    """
    # The actual API response structure is direct, not wrapped in seasonRankings
    competition = data.get("competition", {})
    tournament_calendar = data.get("tournamentCalendar", {})

    context = {
        "_competition": competition,
        "_tournament_calendar": tournament_calendar,
    }

    # Yield match rankings - matchData is at the top level, not nested under tournamentCalendar
    # These are MATCH-LEVEL statistics (combined for both teams)
    match_data = data.get("matchData", [])
    if isinstance(match_data, dict):  # Handle case where there's only one match
        match_data = [match_data]

    for match in match_data:
        stats = flatten_stats(match.get("stat", []), key_name="type")
        record = {**match, **stats, **context, "_record_type": "match"}
        record.pop("stat", None)

        # Extract team information for clarity
        team_data = match.get("teamData", [])
        if team_data:
            home_team = next((t for t in team_data if t.get("side") == "Home"), None)
            away_team = next((t for t in team_data if t.get("side") == "Away"), None)

            if home_team:
                record["_home_team_id"] = home_team.get("id")
            if away_team:
                record["_away_team_id"] = away_team.get("id")

        yield record

    # Yield team and player rankings - teams are at the top level, not nested under tournamentCalendar
    # These are TEAM-LEVEL statistics (individual team stats)
    team_data = data.get("team", [])
    if isinstance(team_data, dict):  # Handle case where there's only one team
        team_data = [team_data]

    for team in team_data:
        # Yield player records first
        player_data = team.get("player", [])
        if isinstance(player_data, dict):  # Handle case where there's only one player
            player_data = [player_data]

        for player in player_data:
            stats = flatten_stats(player.get("stat", []), key_name="type")
            player_record = {
                **player,
                **stats,
                **context,
                "_record_type": "player",
                "_team_id": team.get("id"),
            }
            player_record.pop("stat", None)
            yield player_record

        # Yield team record
        stats = flatten_stats(team.get("stat", []), key_name="type")
        team_record = {
            **team,
            **stats,
            **context,
            "_record_type": "team",
        }
        team_record.pop("stat", None)
        team_record.pop("player", None)  # Remove nested players
        yield team_record


def parse_pass_matrix(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Parse pass matrix and average formation (MA4) response.

    The MA4 feed provides pass matrix data and average pitch positions for players.
    This parser extracts the pass matrix data and yields it as a single record
    along with match information.
    """
    # Extract match information
    match_info = data.get("matchInfo", {})
    live_data = data.get("liveData", {})

    # Extract pass matrix data
    pass_matrix = data.get("passMatrix", {})

    # Create a comprehensive record with all the data
    record = {
        "_match_info": match_info,
        "_live_data": live_data,
        "_pass_matrix": pass_matrix,
    }

    # Add top-level pass matrix fields to the record for easier access
    if isinstance(pass_matrix, dict):
        record.update(pass_matrix)

    # Add match info fields to the record
    if isinstance(match_info, dict):
        record.update(match_info)

    # Add live data fields to the record
    if isinstance(live_data, dict):
        record.update(live_data)

    yield record


def parse_possession(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Parse possession and territorial advantage (MA5) response.

    The MA5 feed provides possession breakdown including overall % possession,
    territorial advantage, and time-based splits (last 5, 10, 15, 20, 25, 30 minutes).
    """
    # Extract match information
    match_info = data.get("matchInfo", {})
    live_data = data.get("liveData", {})

    # Extract possession data
    possession = data.get("possession", {})

    # Extract possession territory data
    possession_territory = data.get("possessionTerritory", {})

    # Create a comprehensive record with all data
    record = {
        "_match_info": match_info,
        "_live_data": live_data,
        "_possession": possession,
        "_possession_territory": possession_territory,
    }

    # Add top-level possession fields to record for easier access
    if isinstance(possession, dict):
        record.update(possession)

    # Add possession territory fields to record
    if isinstance(possession_territory, dict):
        record.update(possession_territory)

    # Add match info fields to record
    if isinstance(match_info, dict):
        record.update(match_info)

    # Add live data fields to record
    if isinstance(live_data, dict):
        record.update(live_data)

    yield record


def parse_referees(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Parse referees (PE3) response."""
    if "referee" in data and isinstance(data["referee"], list):
        yield from data["referee"]
    elif "referee" in data and isinstance(data["referee"], dict):
        yield data["referee"]
    else:
        yield data
