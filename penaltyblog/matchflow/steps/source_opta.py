from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Union, cast

import requests

if TYPE_CHECKING:
    from ..flow import Flow

# --- HELPER FUNCTIONS (RE-INTRODUCED) ---


def _flatten_stats(
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


def _extract_player_stats(match: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests player stats from a MA2 match record."""
    match_info = match.get("matchInfo", {})
    live_data = match.get("liveData", {})
    lineups = live_data.get("lineUp", [])

    for team in lineups:
        contestant_id = team.get("contestantId")
        players = team.get("player", [])
        for player in players:
            player_stats = _flatten_stats(player.get("stat", []), key_name="type")
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


def _extract_team_stats(match: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests team stats from a MA2 match record."""
    match_info = match.get("matchInfo", {})
    live_data = match.get("liveData", {})
    team_stats_list = live_data.get("teamStats", [])

    for team in team_stats_list:
        team_stats = _flatten_stats(team.get("stat", []), key_name="type")
        if not team_stats:
            continue
        team_record = {
            **team,
            **team_stats,
            "_match_uuid": match_info.get("id"),
            "_match_info": match_info,
        }
        team_record.pop("stat", None)
        yield team_record


def _extract_match_events(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Un-nests match events from a MA3 feed response."""
    match_info = data.get("matchInfo", {})
    live_data = data.get("liveData", {})
    match_details = live_data.get("matchDetails", {})
    event_list = live_data.get("event", [])
    for event in event_list:
        event["_match_info"] = match_info
        event["_match_details"] = match_details
        yield event


# --- HELPERS FOR TM4 (Based on new understanding) ---


def _extract_season_player_stats(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
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
        player_stats = _flatten_stats(player.get("stat", []), key_name="name")

        player_record = {
            **player,
            **player_stats,
            "_competition": competition_info,
            "_tournamentCalendar": tournament_info,
        }
        player_record.pop("stat", None)
        yield player_record


def _extract_season_team_stats(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Un-nests team seasonal stats from a TM4 response.
    Yields the single team record from the 'contestant' dict.
    """
    competition_info = data.get("competition", {})
    tournament_info = data.get("tournamentCalendar", {})

    team_data = data.get("contestant")  # Get top-level contestant dict
    if not isinstance(team_data, dict):
        return  # No team data

    team_stats = _flatten_stats(team_data.get("stat", []), key_name="name")

    team_record = {
        **team_data,
        **team_stats,
        "_competition": competition_info,
        "_tournamentCalendar": tournament_info,
    }
    team_record.pop("stat", None)
    team_record.pop("player", None)  # Remove nested player list if present
    yield team_record


# --- OPTA HTTP LOGIC ---


def _build_opta_request_details(step: dict) -> tuple[str, dict, dict]:
    """Helper to prepare URL, params, and headers based on creds."""
    args = step.get("args", {})
    creds = args.get("creds", {})
    params = {}
    headers = {}

    # 1. Handle Auth & Base URL
    if creds.get("auth_key") and creds.get("rt_mode"):
        base_url = f"{step['base_url']}/{step['asset_type']}"
        params["_rt"] = creds["rt_mode"]
    else:
        raise ValueError(...)  # Unchanged

    params["_fmt"] = "json"

    # 3. Map 'source' to the actual endpoint path
    source = step.get("source")
    endpoint_path = ""
    auth_key = creds.get("auth_key")

    # OT2
    if source == "tournament_calendars":
        status = args.get("status", "all")
        base_ot2_path = f"/tournamentcalendar/{auth_key}"
        if status == "active":
            endpoint_path = f"{base_ot2_path}/active"
        elif status == "authorized":
            endpoint_path = f"{base_ot2_path}/authorized"
        elif status == "active_authorized":
            endpoint_path = f"{base_ot2_path}/active/authorized"
        else:
            endpoint_path = base_ot2_path
        params["comp"] = args.get("comp")
        params["ctst"] = args.get("ctst")
        params["stages"] = args.get("stages")
        params["coverage"] = args.get("coverage")
    # MA0
    elif source == "tournament_schedule":
        tournament_calendar_uuid = args.get("tournament_calendar_uuid")
        if not tournament_calendar_uuid:
            raise ValueError(...)
        endpoint_path = f"/tournamentschedule/{auth_key}/{tournament_calendar_uuid}"
        params["cvlv"] = args.get("cvlv")
        params["_lcl"] = args.get("_lcl")
    # MA1 Basic (Paginated List)
    elif source == "matches_basic":
        endpoint_path = f"/match/{auth_key}"
        params["fx"] = args.get("fx")
        params["tmcl"] = args.get("tmcl")
        params["comp"] = args.get("comp")
        params["ctst"] = args.get("ctst")
        params["ctst2"] = args.get("ctst2")
        params["ctstpos"] = args.get("ctstpos")
        params["mt.mDt"] = args.get("mt_mDt")
        params["_dlt"] = args.get("_dlt")
        params["live"] = args.get("live")
        params["lineups"] = args.get("lineups")
        params["_lcl"] = args.get("_lcl")
    # MA1 Basic (Single Match)
    elif source == "match_basic":
        fixture_uuid = args.get("fixture_uuid")
        if not fixture_uuid:
            raise ValueError(...)
        endpoint_path = f"/match/{auth_key}/{fixture_uuid}"
        params["live"] = args.get("live")
        params["lineups"] = args.get("lineups")
        params["_lcl"] = args.get("_lcl")
    # MA2 Basic (Stats)
    elif source == "match_stats_basic":
        fixture_uuids = args.get("fixture_uuids")
        if not fixture_uuids:
            raise ValueError(...)
        if isinstance(fixture_uuids, str):
            endpoint_path = f"/matchstats/{auth_key}/{fixture_uuids}"
        elif isinstance(fixture_uuids, list):
            endpoint_path = f"/matchstats/{auth_key}"
            params["fx"] = ",".join(fixture_uuids)
        params["people"] = args.get("people")
        params["_lcl"] = args.get("_lcl")
    # MA3 (Events)
    elif source == "match_events":
        fixture_uuid = args.get("fixture_uuid")
        if not fixture_uuid:
            raise ValueError(...)
        endpoint_path = f"/matchevent/{auth_key}/{fixture_uuid}"
        params["ctst"] = args.get("ctst")
        params["prsn"] = args.get("prsn")
        params["type"] = args.get("type")
        params["_lcl"] = args.get("_lcl")
    # TM1 (Teams)
    elif source == "teams":
        endpoint_path = f"/team/{auth_key}"
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        params["ctry"] = args.get("ctry")
        params["stg"] = args.get("stg")
        params["srs"] = args.get("srs")
    # TM3 (Squads)
    elif source == "squads":
        endpoint_path = f"/squads/{auth_key}"
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        params["_lcl"] = args.get("_lcl")

    # --- ADDED BLOCK FOR TM4 ---
    elif source in ("player_season_stats", "team_season_stats"):
        endpoint_path = f"/seasonstats/{auth_key}"
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        params["detailed"] = args.get("detailed")
    # --- END ADDED BLOCK ---

    else:
        raise ValueError(f"Unknown Opta source type in plan: {source}")

    final_url = base_url + endpoint_path
    final_params = {k: v for k, v in params.items() if v is not None}

    return final_url, final_params, headers


# Define which sources are not paginated
NON_PAGINATED_SOURCES = {
    "tournament_schedule",  # MA0
    "match_basic",  # MA1 (Single)
    "match_stats_basic",  # MA2 (Single or Multi via fx)
    "match_events",  # MA3
    "player_season_stats",  # TM4
    "team_season_stats",  # TM4
}


def from_opta(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from an Opta API endpoint, handling pagination.
    Yields a single stream of individual records (e.g., matches, calendars).
    """
    source = step.get("source")
    args = step.get("args", {})

    proxies = args.get("proxies")

    with requests.Session() as session:
        if proxies:
            session.proxies = proxies

        base_url, base_params, headers = _build_opta_request_details(step)

        # --- Handle Non-Paginated Endpoints ---
        if source in NON_PAGINATED_SOURCES:
            try:
                response = session.get(base_url, params=base_params, headers=headers)
                if response.status_code == 404:
                    raise IOError(
                        f"Opta API request returned 404 Not Found for URL: {response.url}"
                    )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise IOError(f"Opta API request failed: {e}") from e
            except requests.exceptions.JSONDecodeError:
                raise IOError(f"Opta API returned non-JSON response: {response.text}")
            if "errorCode" in data:
                raise RuntimeError(f"Opta API Error: {data['errorCode']}")

            # --- PARSING LOGIC FOR NON-PAGINATED ---

            if source == "tournament_schedule":
                competition_info = data.get("competition", {})
                tournament_info = data.get("tournamentCalendar", {})
                match_dates = data.get("matchDate", [])
                for day in match_dates:
                    for match in day.get("match", []):
                        match["_competition"] = competition_info
                        match["_tournamentCalendar"] = tournament_info
                        match["_matchDate"] = day.get("date")
                        yield match
                return

            elif source == "match_basic":
                yield data
                return

            elif source == "match_stats_basic":
                matches = []  # Default to empty list

                # Check 1: Is it a single match object? (like MA2 single-match sample)
                if isinstance(data.get("matchInfo"), dict):
                    matches = [data]

                # Check 2: Is it a list under the "match" key? (like MA1 multi-match sample)
                elif isinstance(data.get("matchStats"), list):
                    matches = data.get("matchStats")

                include_players = args.get("people", "yes") == "yes"
                for match in matches:
                    if include_players:
                        yield from _extract_player_stats(match)
                    else:
                        yield from _extract_team_stats(match)
                return

            elif source == "match_events":
                yield from _extract_match_events(data)
                return

            # --- ADDED BLOCK FOR TM4 ---
            elif source == "player_season_stats":
                yield from _extract_season_player_stats(data)
                return

            elif source == "team_season_stats":
                yield from _extract_season_team_stats(data)
                return
            # --- END ADDED BLOCK ---

            # Fallback for any other non-paginated source
            yield data
            return

        # --- Handle Paginated Endpoints ---
        # In scope: OT2, MA1-basic (list), TM1, TM3
        page_num = 1
        page_size = 1000

        while True:
            params = base_params.copy()
            params["_pgNm"] = page_num
            params["_pgSz"] = page_size
            try:
                response = session.get(base_url, params=params, headers=headers)
                if response.status_code == 404:
                    break
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.HTTPError as e:
                raise IOError(f"Opta API request failed: {e}") from e
            except requests.exceptions.RequestException as e:
                raise IOError(f"Opta API request (non-HTTP) failed: {e}") from e
            except requests.exceptions.JSONDecodeError:
                raise IOError(f"Opta API returned non-JSON response: {response.text}")
            if "errorCode" in data:
                raise RuntimeError(f"Opta API Error: {data['errorCode']}")

            # --- Extract list and yield from it ---
            records_list = []
            try:
                if source == "tournament_calendars":  # OT2
                    records_list = data.get("competition", [])
                elif source == "matches_basic":  # MA1 Basic list
                    records_list = data.get("match", [])
                    if not records_list:
                        records_list = data.get("matches", {}).get("match", [])
                elif source == "teams":  # TM1
                    records_list = data.get("contestants", {}).get("contestant", [])
                    if not records_list:
                        records_list = data.get("contestant", [])
                elif source == "squads":  # TM3
                    records_list = data.get("squad", [])
                    if not records_list:
                        records_list = data.get("teamSquads", {}).get("squad", [])

                # TM4 is non-paginated, so it's not handled here

            except Exception as e:
                print(
                    f"Warning: Could not parse items from page {page_num} for {source}. Stopping. Error: {e}"
                )
                break

            if not records_list or not isinstance(records_list, list):
                break

            yield from records_list

            if len(records_list) < page_size:
                break

            page_num += 1
