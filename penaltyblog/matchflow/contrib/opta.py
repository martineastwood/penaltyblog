# penaltyblog/matchflow/contrib/opta.py
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from typing_extensions import Literal  # Keep Literal for status

if TYPE_CHECKING:
    from ..flow import Flow


# --- Helper for formatting dates ---
def _format_opta_datetime(dt: Union[str, datetime]) -> str:
    """Formats a datetime object or string into Opta's required Z-format."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt


class Opta:

    DEFAULT_CREDS = {
        "auth_key": os.environ.get("OPTA_AUTH_KEY"),
        "rt_mode": os.environ.get("OPTA_RT_MODE", "b"),
    }
    BASE_URL = os.environ.get("OPTA_BASE_URL", "http://api.performfeeds.com")
    ASSET_TYPE = "soccerdata"

    def _step(self, source: str, optimize: bool = False, **args) -> "Flow":
        """Internal helper to build a lazy Flow plan."""
        from ..flow import Flow

        # Ensure 'raw' is always True in args passed to the plan
        # We handle this internally now, not exposed to user
        args_with_raw = args.copy()
        args_with_raw["raw"] = True  # Force raw internally

        return Flow(
            plan=[
                {
                    "op": "from_opta",
                    "source": source,
                    "base_url": self.BASE_URL,
                    "asset_type": self.ASSET_TYPE,
                    # Pass the modified args dict
                    "args": args_with_raw,
                }
            ],
            optimize=optimize,
        )

    # --- IN-SCOPE METHODS ---

    def tournament_calendars(
        self,
        status: Literal["all", "active", "authorized", "active_authorized"] = "all",
        competition_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        include_stages: bool = False,
        include_coverage: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw tournament calendar data (Feed OT2).
        This feed is paginated. Yields one raw JSON object per page.

        Parameters
        ----------
        status : Literal["all", "active", "authorized", "active_authorized"], optional
            Filter calendars by status (default: "all").
        competition_uuid : str, optional
            Filter by competition UUID.
        contestant_uuid : str, optional
            Filter by contestant UUID.
        include_stages : bool, optional
            Request stage details (default: False).
        include_coverage : bool, optional
            Request coverage details (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding raw JSON data per page.
        """
        return self._step(
            "tournament_calendars",
            status=status,
            comp=competition_uuid,
            ctst=contestant_uuid,
            stages="yes" if include_stages else None,
            coverage="yes" if include_coverage else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def tournament_schedule(
        self,
        tournament_calendar_uuid: str,
        coverage_level: Optional[Union[int, List[int]]] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw tournament schedule data (Feed MA0).
        Yields one raw JSON object for the entire schedule.

        Parameters
        ----------
        tournament_calendar_uuid : str
            The UUID for the specific tournament calendar.
        coverage_level : int or List[int], optional
            Filter matches by coverage level(s).
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding one raw JSON object.
        """
        cvlv_str = None
        if isinstance(coverage_level, list):
            cvlv_str = ",".join(map(str, coverage_level))
        elif isinstance(coverage_level, int):
            cvlv_str = str(coverage_level)

        return self._step(
            "tournament_schedule",  # Uses MA0 endpoint
            tournament_calendar_uuid=tournament_calendar_uuid,
            cvlv=cvlv_str,
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def matches(
        self,
        fixture_uuids: Optional[List[str]] = None,
        tournament_calendar_uuid: Optional[str] = None,
        competition_uuids: Optional[List[str]] = None,
        contestant_uuids: Optional[List[str]] = None,
        date_from: Optional[Union[str, datetime]] = None,
        date_to: Optional[Union[str, datetime]] = None,
        delta_timestamp: Optional[Union[str, datetime]] = None,
        live: bool = False,
        lineups: bool = False,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match data (Feed MA1 - Basic).
        This feed is paginated. Yields one raw JSON object per page.

        Parameters
        ----------
        fixture_uuids : List[str], optional
            Get specific matches by UUID.
        tournament_calendar_uuid : str, optional
            Filter by tournament calendar.
        competition_uuids : List[str], optional
            Filter by competition(s).
        contestant_uuids : List[str], optional
            Filter by contestant(s).
        date_from : str or datetime, optional
            Start of date range.
        date_to : str or datetime, optional
            End of date range.
        delta_timestamp : str or datetime, optional
            Get updates since this time.
        live : bool, optional
            Request live data (default: False).
        lineups : bool, optional
            Request lineup data (requires live=True) (default: False).
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding raw JSON data per page.
        """
        date_range_str = None
        if date_from and date_to:
            date_range_str = f"[{_format_opta_datetime(date_from)} TO {_format_opta_datetime(date_to)}]"
        elif date_from or date_to:
            raise ValueError("Both 'date_from' and 'date_to' must be provided")

        return self._step(
            "matches_basic",  # Changed source to indicate basic MA1
            fx=",".join(fixture_uuids) if fixture_uuids else None,
            tmcl=tournament_calendar_uuid,
            comp=",".join(competition_uuids) if competition_uuids else None,
            ctst=",".join(contestant_uuids) if contestant_uuids else None,
            mt_mDt=date_range_str,
            _dlt=_format_opta_datetime(delta_timestamp) if delta_timestamp else None,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def match(
        self,
        fixture_uuid: str,
        live: bool = False,
        lineups: bool = False,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw data for a single match (Feed MA1 - Basic).
        Yields one raw JSON object.

        Parameters
        ----------
        fixture_uuid : str
            The UUID for the specific match.
        live : bool, optional
            Request live data (default: False).
        lineups : bool, optional
            Request lineup data (requires live=True) (default: False).
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding one raw JSON object.
        """
        return self._step(
            "match_basic",  # Changed source to indicate basic MA1
            fixture_uuid=fixture_uuid,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def match_stats(
        self,
        fixture_uuids: Union[str, List[str]],
        # detailed parameter removed
        include_players: bool = True,  # Renamed 'people' for clarity
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match stats data (Feed MA2 - Basic).
        Yields one raw JSON object (contains both team and player stats if requested).

        Parameters
        ----------
        fixture_uuids : str or List[str]
            A single match UUID or a list of match UUIDs.
        include_players : bool, optional
            Include player stats alongside team stats (default: True). If False, only
            team stats are returned (maps to 'people=no').
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding one raw JSON object per match requested.
            Note: If multiple UUIDs are requested, the API returns a structure
            containing multiple matches, yielded here as a single JSON object.
            Use matchflow operations downstream to process further.
        """
        return self._step(
            "match_stats_basic",  # Changed source to indicate basic MA2
            fixture_uuids=fixture_uuids,
            # detailed parameter removed, defaults to basic
            people="yes" if include_players else "no",  # Use renamed param
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def events(
        self,
        fixture_uuid: str,
        contestant_uuid: Optional[str] = None,
        person_uuid: Optional[str] = None,
        event_types: Optional[Union[int, List[int]]] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match events data (Feed MA3).
        Yields one raw JSON object containing all events.

        Parameters
        ----------
        fixture_uuid : str
            The UUID for the specific match.
        contestant_uuid : str, optional
            Filter events by contestant.
        person_uuid : str, optional
            Filter events by person.
        event_types : int or List[int], optional
            Filter by event type ID(s).
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding one raw JSON object.
        """
        type_str = None
        if isinstance(event_types, list):
            type_str = ",".join(map(str, event_types))
        elif isinstance(event_types, int):
            type_str = str(event_types)

        return self._step(
            "match_events",  # Uses MA3 endpoint
            fixture_uuid=fixture_uuid,
            ctst=contestant_uuid,
            prsn=person_uuid,
            type=type_str,
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    # --- PLACEHOLDERS for TM1 and TM3 ---

    def teams(
        self,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        country_uuid: Optional[str] = None,
        stage_uuid: Optional[str] = None,
        series_uuid: Optional[str] = None,
        # detailed parameter not used as we always get raw
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw team data (Feed TM1).
        Requires either tournament_calendar_uuid or contestant_uuid.
        This feed is paginated. Yields one raw JSON object per page.

        Parameters
        ----------
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar (season).
        contestant_uuid : str, optional
            Filter by a specific contestant (team).
        country_uuid : str, optional
            Filter by country UUID.
        stage_uuid : str, optional
            Filter by stage UUID.
        series_uuid : str, optional
            Filter by series UUID.
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding raw JSON data per page.

        Raises
        ------
        ValueError
            If neither tournament_calendar_uuid nor contestant_uuid is provided.
        """
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the teams feed."
            )

        return self._step(
            "teams",  # Use final source name
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
            ctry=country_uuid,
            stg=stage_uuid,
            srs=series_uuid,
            # detailed=yes parameter removed from sample calls, omitting
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def squads(
        self,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw squad data (Feed TM3).
        Requires either tournament_calendar_uuid or contestant_uuid.
        This feed is paginated. Yields one raw JSON object per page.

        Parameters
        ----------
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar (season).
        contestant_uuid : str, optional
            Filter by a specific contestant (team).
        use_opta_names : bool, optional
            Request 'en-op' locale (default: False).
        creds : dict, optional
            Credentials for Opta API.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding raw JSON data per page.

        Raises
        ------
        ValueError
            If neither tournament_calendar_uuid nor contestant_uuid is provided.
        """
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the squads feed."
            )

        return self._step(
            "squads",  # Use final source name
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
            # detailed=yes parameter removed
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )


# Bind singleton instance
opta = Opta()
