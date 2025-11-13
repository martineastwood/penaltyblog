"""
Endpoint builders for Opta API requests.
"""

from typing import Any, Dict, Tuple

from .config import ENDPOINT_CONFIGS, PARAMETER_MAPPINGS
from .exceptions import OptaConfigurationError


class OptaEndpointBuilder:
    """
    Builds URLs and parameters for different Opta API endpoints.
    """

    def __init__(self, base_url: str, asset_type: str, auth_key: str):
        """
        Initialize endpoint builder.

        Args:
            base_url: Base URL for Opta API
            asset_type: Asset type (e.g., "soccerdata")
            auth_key: Authentication key
        """
        self.base_url = f"{base_url}/{asset_type}"
        self.auth_key = auth_key

    def build_request_details(
        self, source: str, args: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build URL and parameters for a specific endpoint.

        Args:
            source: The endpoint source name
            args: Arguments for the endpoint

        Returns:
            Tuple of (url, parameters)

        Raises:
            OptaConfigurationError: If source is unknown or required args are missing
        """
        if source not in ENDPOINT_CONFIGS:
            raise OptaConfigurationError(f"Unknown Opta source type: {source}")

        config = ENDPOINT_CONFIGS[source]
        endpoint_path = self._build_endpoint_path(source, config, args)
        params = self._build_parameters(source, args)

        url = self.base_url + endpoint_path
        final_params = {k: v for k, v in params.items() if v is not None}

        return url, final_params

    def _build_endpoint_path(
        self, source: str, config: Dict[str, Any], args: Dict[str, Any]
    ) -> str:
        """Build the endpoint path from configuration and arguments."""
        path_template = config["path_template"]

        # Handle tournament_calendars with status variants
        if source == "tournament_calendars":
            status = args.get("status", "all")
            status_variants = config.get("status_variants", {})
            status_suffix = status_variants.get(status, "")
            base_path = path_template.format(auth_key=self.auth_key)
            return base_path + status_suffix

        # Handle endpoints that support multiple fixture UUIDs
        elif source in ["match_stats_player", "match_stats_team"] and isinstance(
            args.get("fixture_uuids"), list
        ):
            # For multiple fixtures, don't include UUIDs in path
            return path_template.format(auth_key=self.auth_key)

        # Standard path building with required parameters
        else:
            try:
                return path_template.format(
                    auth_key=self.auth_key, **self._extract_path_params(source, args)
                )
            except KeyError as e:
                raise OptaConfigurationError(
                    f"Missing required path parameter for {source}: {e}"
                ) from e

    def _extract_path_params(self, source: str, args: Dict[str, Any]) -> Dict[str, str]:
        """Extract path parameters from arguments for a specific source."""
        path_params = {}

        if source == "tournament_schedule":
            tournament_calendar_uuid = args.get("tournament_calendar_uuid")
            if not tournament_calendar_uuid:
                raise OptaConfigurationError(
                    "tournament_schedule source requires 'tournament_calendar_uuid'"
                )
            path_params["tournament_calendar_uuid"] = tournament_calendar_uuid

        elif source == "match_basic":
            fixture_uuid = args.get("fixture_uuid")
            if not fixture_uuid:
                raise OptaConfigurationError(
                    "match_basic source requires 'fixture_uuid'"
                )
            path_params["fixture_uuid"] = fixture_uuid

        elif source == "match_events":
            fixture_uuid = args.get("fixture_uuid")
            if not fixture_uuid:
                raise OptaConfigurationError(
                    "match_events source requires 'fixture_uuid'"
                )
            path_params["fixture_uuid"] = fixture_uuid

        elif source == "pass_matrix":
            fixture_uuid = args.get("fixture_uuid")
            if not fixture_uuid:
                raise OptaConfigurationError(
                    "pass_matrix source requires 'fixture_uuid'"
                )
            path_params["fixture_uuid"] = fixture_uuid

        elif source == "possession":
            fixture_uuid = args.get("fixture_uuid")
            if not fixture_uuid:
                raise OptaConfigurationError(
                    "possession source requires 'fixture_uuid'"
                )
            path_params["fixture_uuid"] = fixture_uuid

        elif source == "match_stats_basic" and isinstance(
            args.get("fixture_uuids"), str
        ):
            path_params["fixture_uuids"] = args["fixture_uuids"]

        elif source == "area_specific":
            area_uuid = args.get("area_uuid")
            if not area_uuid:
                raise OptaConfigurationError(
                    "area_specific source requires 'area_uuid'"
                )
            path_params["area_uuid"] = area_uuid

        elif source == "player_career_person":
            person_uuid = args.get("person_uuid")
            if not person_uuid:
                raise OptaConfigurationError(
                    "player_career_person source requires 'person_uuid'"
                )
            path_params["person_uuid"] = person_uuid

        elif source == "injuries_person_path":
            person_uuid = args.get("person_uuid")
            if not person_uuid:
                raise OptaConfigurationError(
                    "injuries_person_path source requires 'person_uuid'"
                )
            path_params["person_uuid"] = person_uuid

        elif source == "rankings":
            tournament_calendar_uuid = args.get("tournament_calendar_uuid")
            if not tournament_calendar_uuid:
                raise OptaConfigurationError(
                    "rankings source requires 'tournament_calendar_uuid'"
                )
            path_params["tournament_calendar_uuid"] = tournament_calendar_uuid

        return path_params

    def _build_parameters(self, source: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Build query parameters from arguments."""
        params = {"_fmt": "json"}  # Always request JSON format

        # Get parameter mapping for this source
        param_mapping = PARAMETER_MAPPINGS.get(source, {})

        # Handle special cases
        if source in [
            "match_stats_player",
            "match_stats_team",
            "match_stats_basic",
            "matches_basic",
        ] and isinstance(args.get("fixture_uuids"), list):
            fx_uuids = args.get("fixture_uuids")
            if isinstance(fx_uuids, list):
                params["fx"] = ",".join(fx_uuids)
            elif isinstance(fx_uuids, str):
                params["fx"] = fx_uuids

        # Map arguments to parameter names
        for arg_name, param_name in param_mapping.items():
            value = args.get(arg_name)
            if value is not None:
                # Handle special parameter transformations
                if arg_name == "fixture_uuids" and isinstance(value, list):
                    continue  # Already handled above
                elif arg_name in [
                    "live",
                    "lineups",
                    "include_stages",
                    "include_coverage",
                    "detailed",
                    "active",
                    "include_players",
                ] and isinstance(value, bool):
                    params[param_name] = (
                        "yes" if value else ("no" if value is False else None)
                    )  # Handle False explicitly
                elif arg_name == "competition_uuids" and isinstance(value, list):
                    params[param_name] = ",".join(value)
                elif arg_name == "contestant_uuid" and isinstance(value, list):
                    params[param_name] = ",".join(value)
                elif arg_name == "event_types" and isinstance(value, list):
                    params[param_name] = ",".join(map(str, value))
                elif arg_name == "coverage_level" and isinstance(value, list):
                    params["cvlv"] = ",".join(map(str, value))
                elif arg_name == "coverage_level" and isinstance(value, int):
                    params["cvlv"] = str(value)
                else:
                    params[param_name] = value

        # Add source-specific parameters that aren't in the mapping
        if source == "tournament_calendars":
            for param in ["comp", "ctst", "stages", "coverage"]:
                if param in args and args[param] is not None:
                    params[param] = args[param]

        elif source in ["match_basic", "match_stats_basic", "match_events"]:
            if "live" in args and args["live"] is not None:
                params["live"] = args["live"]
            if "lineups" in args and args["lineups"] is not None:
                params["lineups"] = args["lineups"]

        elif source == "player_season_stats" or source == "team_season_stats":
            if "detailed" in args and args["detailed"] is not None:
                params["detailed"] = "yes" if args["detailed"] else "no"

        elif source == "team_standings":
            if "live" in args and args["live"] is not None:
                params["live"] = "yes" if args["live"] else "no"

        return params
