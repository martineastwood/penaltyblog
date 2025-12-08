"""
Refactored Opta API source integration.

This module provides a clean interface to Opta API data sources,
with separated concerns for HTTP handling, endpoint building,
data parsing, and pagination.
"""

from typing import Any, Dict, Iterator

from .opta.client import OptaClient
from .opta.endpoints import OptaEndpointBuilder
from .opta.paginator import OptaPaginator
from .opta.parsers import (
    extract_contestant_participation,
    extract_match_events,
    extract_season_player_stats,
    extract_season_team_stats,
    parse_match_basic,
    parse_match_stats_basic,
    parse_match_stats_player,
    parse_match_stats_team,
    parse_player_career_person,
    parse_rankings,
    parse_referees,
    parse_tournament_schedule,
    parse_transfers,
)


def from_opta(step: Dict[str, Any]) -> Iterator[Dict[Any, Any]]:
    """
    Create a Flow from an Opta API endpoint, handling pagination.
    Yields a single stream of individual records (e.g., matches, calendars).

    Args:
        step: Configuration dictionary containing:
            - source: The endpoint source name
            - base_url: Base URL for Opta API
            - asset_type: Asset type (e.g., "soccerdata")
            - args: Arguments for the endpoint including credentials

    Yields:
        Individual records from the API response

    Raises:
        OptaAuthenticationError: If credentials are invalid
        OptaRequestError: If API request fails
        OptaAPIError: If API returns an error
    """
    source = step.get("source")
    args = step.get("args", {})

    # Extract credentials and validate
    creds = args.get("creds", {})
    proxies = args.get("proxies")

    # Initialize components
    with OptaClient(proxies=proxies) as client:
        client.validate_credentials(creds)

        # Build endpoint details
        endpoint_builder = OptaEndpointBuilder(
            base_url=step["base_url"],
            asset_type=step["asset_type"],
            auth_key=creds["auth_key"],
        )

        url, params = endpoint_builder.build_request_details(source, args)
        headers = {}

        # Add rt_mode parameter for authentication
        params["_rt"] = creds["rt_mode"]

        # Handle paginated vs non-paginated endpoints
        if OptaPaginator.is_paginated(source, args):
            yield from _handle_paginated_endpoint(client, source, url, params, headers)
        else:
            yield from _handle_non_paginated_endpoint(
                client, source, url, params, headers, args
            )


def _handle_paginated_endpoint(
    client: OptaClient,
    source: str,
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
) -> Iterator[Dict[str, Any]]:
    """
    Handle paginated endpoints using the paginator.

    Args:
        client: OptaClient instance
        source: Endpoint source name
        url: Request URL
        params: Request parameters
        headers: Request headers

    Yields:
        Individual records from paginated responses
    """
    paginator = OptaPaginator(client)
    yield from paginator.fetch_paginated_data(source, url, params, headers)


def _handle_non_paginated_endpoint(
    client: OptaClient,
    source: str,
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
    args: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    """
    Handle non-paginated endpoints with appropriate parsing.

    Args:
        client: OptaClient instance
        source: Endpoint source name
        url: Request URL
        params: Request parameters
        headers: Request headers
        args: Original arguments for parsing configuration

    Yields:
        Parsed records from the response

    Raises:
        OptaRequestError: If request fails
        OptaAPIError: If API returns an error
    """
    # Make the request
    data = client.make_request(url, params, headers)

    # Route to appropriate parser based on source
    if source == "tournament_schedule":
        yield from parse_tournament_schedule(data)

    elif source == "match_basic":
        yield from parse_match_basic(data)

    elif source == "match_stats_basic":
        include_players = args.get("include_players", True)
        yield from parse_match_stats_basic(data, include_players)

    elif source == "match_stats_player":
        yield from parse_match_stats_player(data)

    elif source == "match_stats_team":
        yield from parse_match_stats_team(data)

    elif source == "match_events":
        yield from extract_match_events(data)

    elif source == "player_season_stats":
        yield from extract_season_player_stats(data)

    elif source == "team_season_stats":
        yield from extract_season_team_stats(data)

    elif source == "contestant_participation":
        yield from extract_contestant_participation(data)

    elif source == "transfers":
        yield from parse_transfers(data)

    elif source == "player_career_person":
        yield from parse_player_career_person(data)

    elif source == "referees_person":
        yield from parse_referees(data)

    elif source == "rankings":
        yield from parse_rankings(data)

    else:
        # Fallback for any other non-paginated source
        yield data
