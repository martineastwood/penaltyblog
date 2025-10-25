# penaltyblog/matchflow/steps/source_opta.py

from typing import TYPE_CHECKING, Any, Dict, Iterator

import requests

if TYPE_CHECKING:
    from ..flow import Flow


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
        # Add filter params
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        # detailed param omitted
        params["_lcl"] = args.get("_lcl")

    else:
        # Removed placeholder logic, raise error for unknown
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
}


def from_opta(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from an Opta API endpoint, handling pagination.
    Yields RAW JSON data directly from the API.
    """
    source = step.get("source")

    base_url, base_params, headers = _build_opta_request_details(step)

    # --- Handle Non-Paginated Endpoints ---
    if source in NON_PAGINATED_SOURCES:
        try:
            response = requests.get(base_url, params=base_params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise IOError(f"Opta API request failed: {e}") from e
        except requests.exceptions.JSONDecodeError:
            raise IOError(f"Opta API returned non-JSON response: {response.text}")
        if "errorCode" in data:
            raise RuntimeError(f"Opta API Error: {data['errorCode']}")

        # Yield the raw data directly
        yield data
        return  # Stop generator

    # --- Handle Paginated Endpoints ---
    page_num = 1
    page_size = (
        1000  # Use 1000 page size, adjust if specific endpoints have lower limits
    )

    while True:
        params = base_params.copy()
        params["_pgNm"] = page_num
        params["_pgSz"] = page_size
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise IOError(f"Opta API request failed: {e}") from e
        except requests.exceptions.JSONDecodeError:
            raise IOError(f"Opta API returned non-JSON response: {response.text}")
        if "errorCode" in data:
            raise RuntimeError(f"Opta API Error: {data['errorCode']}")

        # Yield raw JSON for the current page
        yield data

        # --- Determine if last page ---
        items_on_page = 0
        try:
            if source == "tournament_calendars":
                items = data.get("tournamentCalendars", {}).get(
                    "tournamentCalendar", []
                )
                if not items:
                    items = data.get("tournamentCalendar", [])
                items_on_page = len(items)
            elif source == "matches_basic":  # MA1 Basic list
                items = data.get("matches", {}).get("match", [])
                if not items:
                    items = data.get("match", [])
                items_on_page = len(items)
            elif source == "teams":  # TM1 Check
                items = data.get("contestants", {}).get("contestant", [])
                if not items:
                    items = data.get("contestant", [])
                items_on_page = len(items)
            elif source == "squads":  # TM3 Check
                items = data.get("teamSquads", {}).get("squad", [])
                if not items:
                    items = data.get("squad", [])
                items_on_page = len(items)
            else:
                items_on_page = page_size  # Pessimistic guess

        except Exception:
            items_on_page = page_size

        if items_on_page < page_size:
            break  # Last page

        page_num += 1
