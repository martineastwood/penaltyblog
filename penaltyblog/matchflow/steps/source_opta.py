# penaltyblog/matchflow/steps/source_opta.py

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Union, cast

import requests

if TYPE_CHECKING:
    from ..flow import Flow

# --- HELPER FUNCTIONS REMOVED ---
# No parsing helpers needed for raw output


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
        raise ValueError(
            "Invalid Opta credentials. Provide 'auth_key' and 'rt_mode' "
            "in DEFAULT_CREDS or via the 'creds' parameter."
        )

    params["_fmt"] = "json"

    # 3. Map 'source' to the actual endpoint path
    source = step.get("source")
    endpoint_path = ""
    auth_key = creds.get("auth_key")

    # OT2 - Tournament Calendars (Paginated)
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
    # MA0 - Tournament Schedule (Non-Paginated)
    elif source == "tournament_schedule":
        tournament_calendar_uuid = args.get("tournament_calendar_uuid")
        if not tournament_calendar_uuid:
            raise ValueError(
                "tournament_schedule source requires 'tournament_calendar_uuid'"
            )
        endpoint_path = f"/tournamentschedule/{auth_key}/{tournament_calendar_uuid}"
        params["cvlv"] = args.get("cvlv")
        params["_lcl"] = args.get("_lcl")
    # MA1 Basic - Fixtures & Results (Paginated List)
    elif source == "matches_basic":
        endpoint_path = f"/match/{auth_key}"  # Use /match endpoint
        params["fx"] = args.get("fx")
        params["tmcl"] = args.get("tmcl")
        params["comp"] = args.get("comp")
        params["ctst"] = args.get("ctst")
        params["mt.mDt"] = args.get("mt_mDt")
        params["_dlt"] = args.get("_dlt")
        params["live"] = args.get("live")
        params["lineups"] = args.get("lineups")
        params["_lcl"] = args.get("_lcl")
    # MA1 Basic - Fixtures & Results (Single Match, Non-Paginated)
    elif source == "match_basic":
        fixture_uuid = args.get("fixture_uuid")
        if not fixture_uuid:
            raise ValueError("match_basic source requires 'fixture_uuid'")
        endpoint_path = f"/match/{auth_key}/{fixture_uuid}"  # Use /match endpoint
        params["live"] = args.get("live")
        params["lineups"] = args.get("lineups")
        params["_lcl"] = args.get("_lcl")
    # MA2 Basic - Match Stats (Non-Paginated, handles single/multi fx)
    elif source == "match_stats_basic":
        fixture_uuids = args.get("fixture_uuids")
        if not fixture_uuids:
            raise ValueError(f"{source} source requires 'fixture_uuids'")
        if isinstance(fixture_uuids, str):
            endpoint_path = (
                f"/matchstats/{auth_key}/{fixture_uuids}"  # Use /matchstats endpoint
            )
        elif isinstance(fixture_uuids, list):
            endpoint_path = f"/matchstats/{auth_key}"  # Use /matchstats endpoint
            params["fx"] = ",".join(fixture_uuids)
        # Removed 'detailed' param
        params["people"] = args.get("people")
        params["_lcl"] = args.get("_lcl")
    # MA3 - Match Events (Non-Paginated)
    elif source == "match_events":
        fixture_uuid = args.get("fixture_uuid")
        if not fixture_uuid:
            raise ValueError("match_events source requires 'fixture_uuid'")
        endpoint_path = f"/matchevent/{auth_key}/{fixture_uuid}"
        params["ctst"] = args.get("ctst")
        params["prsn"] = args.get("prsn")
        params["type"] = args.get("type")
        params["_lcl"] = args.get("_lcl")
    # TM1 - Teams (Paginated)
    elif source == "teams":
        endpoint_path = f"/team/{auth_key}"
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        params["ctry"] = args.get("ctry")
        params["stg"] = args.get("stg")
        params["srs"] = args.get("srs")
    # TM3 - Squads (Paginated)
    elif source == "squads":
        endpoint_path = f"/squads/{auth_key}"
        params["tmcl"] = args.get("tmcl")
        params["ctst"] = args.get("ctst")
        params["_lcl"] = args.get("_lcl")
    else:
        # Catch any source names not explicitly handled
        raise ValueError(f"Unknown Opta source type in plan: {source}")

    final_url = base_url + endpoint_path
    final_params = {k: v for k, v in params.items() if v is not None}

    return final_url, final_params, headers


# Define which sources are not paginated based on our revised scope
NON_PAGINATED_SOURCES = {
    "tournament_schedule",  # MA0
    "match_basic",  # MA1 (Single)
    "match_stats_basic",  # MA2 (Single or Multi via fx - API returns one JSON)
    "match_events",  # MA3
}


def from_opta(step) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from an Opta API endpoint, handling pagination.
    Yields RAW JSON data directly from the API (one dict per page for paginated feeds).
    """
    source = step.get("source")
    args = step.get("args", {})
    # raw_output flag removed - raw is the only behavior

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

        # Yield the raw data directly for non-paginated sources
        yield data
        return  # Stop generator

    # --- Handle Paginated Endpoints ---
    # In scope: OT2, MA1-basic (list), TM1, TM3
    page_num = 1
    # Use 1000 page size, adjust if specific endpoints have lower limits found during testing
    page_size = 1000

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

        # --- Determine if last page by checking item count ---
        items_on_page = 0
        try:
            if source == "tournament_calendars":  # OT2
                items = data.get("tournamentCalendars", {}).get(
                    "tournamentCalendar", []
                )
                if not items:
                    items = data.get("tournamentCalendar", [])  # Fallback
                items_on_page = len(items)
            elif source == "matches_basic":  # MA1 Basic list
                items = data.get("match", [])  # Primary structure from sample
                if not items:
                    items = data.get("matches", {}).get("match", [])  # Fallback
                items_on_page = len(items)
            elif source == "teams":  # TM1
                items = data.get("contestants", {}).get("contestant", [])
                if not items:
                    items = data.get("contestant", [])  # Fallback
                items_on_page = len(items)
            elif source == "squads":  # TM3
                items = data.get("squad", [])  # Structure from sample
                items_on_page = len(items)
            else:
                # Should not happen if _build_opta_request_details is correct,
                # but act pessimistically if it does.
                items_on_page = page_size

        except Exception:
            # If inspecting the structure fails, assume not last page
            items_on_page = page_size

        if items_on_page < page_size:
            break  # Assume last page

        page_num += 1
