# penaltyblog/matchflow/contrib/opta.py
import os
from datetime import date, datetime
from typing import TYPE_CHECKING, List, Optional, Union

from typing_extensions import Literal

if TYPE_CHECKING:
    from ..flow import Flow


# --- Helper for formatting dates ---
def _format_opta_datetime(dt: Union[str, datetime]) -> str:
    """
    Format a datetime object or string into Opta's required Z-format.

    Parameters
    ----------
    dt : str or datetime
            The datetime object or string to format.

    Returns
    -------
    str
            The formatted datetime string in ISO 8601 format with 'Z' suffix.
    """
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt


def _format_opta_date(dt: Union[str, datetime, date]) -> str:
    """
    Format a datetime/date object or string into Opta's required YYYY-MM-DD format.
    """
    if isinstance(dt, (datetime, date)):
        return dt.strftime("%Y-%m-%d")
    return dt


class Opta:
    """
    A lazy Flow builder for the Stats Perform (Opta) Soccer API.

    This class provides methods that correspond to specific Opta feeds.
    Calling a method returns a Flow object that, when executed,
    will make the necessary API requests to fetch the data.

    Supported Feeds (in order):
    - OT2 (Tournament Calendars):  .tournament_calendars()
    - OT3 (Venues):                .venues()
    - OT4 (Areas):                 .areas()
    - MA0 (Tournament Schedule):   .tournament_schedule()
    - MA1 (Match - Basic):         .matches() / .match()
    - MA2 (Match Stats - Basic):   .match_stats()
    - MA3 (Match Events):          .events()
    - PE2 (Player Career):         .player_career()
    - PE3 (Referees):              .referees()
    - PE7 (Injuries):              .injuries()
    - TM1 (Teams):                 .teams()
    - TM3 (Squads):                .squads()
    - TM4 (Season Stats):          .player_season_stats() / .team_season_stats()
    - TM7 (Transfers):             .transfers()
    - TM16 (Contestant Part.):     .contestant_participation()
    """

    DEFAULT_CREDS = {
        "auth_key": os.environ.get("OPTA_AUTH_KEY"),
        "rt_mode": os.environ.get("OPTA_RT_MODE", "b"),
    }

    BASE_URL = os.environ.get("OPTA_BASE_URL", "http://api.performfeeds.com")
    ASSET_TYPE = "soccerdata"

    def _step(self, source: str, optimize: bool = False, **args) -> "Flow":
        """
        Internal helper to build a lazy Flow plan for Opta API requests.

        Parameters
        ----------
        source : str
            The name of the Opta data source/feed to request.
        optimize : bool, optional
            Whether to optimize the execution plan (default: False).
        **args : dict
            Additional arguments to pass to the Opta API request.

        Returns
        -------
        Flow
            A Flow object representing the lazy execution plan for the Opta request.
        """
        from ..flow import Flow

        args = args.copy()
        return Flow(
            plan=[
                {
                    "op": "from_opta",
                    "source": source,
                    "base_url": self.BASE_URL,
                    "asset_type": self.ASSET_TYPE,
                    "args": args,
                }
            ],
            optimize=optimize,
        )

    # --- IN-SCOPE METHODS (Ordered by Feed ID) ---

    def tournament_calendars(
        self,
        status: Literal["all", "active", "authorized", "active_authorized"] = "all",
        competition_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        include_stages: bool = False,
        include_coverage: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw tournament calendar data (Feed OT2).

        Parameters
        ----------
        status : Literal["all", "active", "authorized", "active_authorized"], optional
            Filter tournaments by status (default: "all").
        competition_uuid : str, optional
            Filter by a specific competition UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        include_stages : bool, optional
            Include tournament stage information (default: False).
        include_coverage : bool, optional
            Include coverage information (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw tournament calendar data.
        """
        return self._step(
            "tournament_calendars",
            status=status,
            competition_uuid=competition_uuid,
            contestant_uuid=contestant_uuid,
            include_stages="yes" if include_stages else None,
            include_coverage="yes" if include_coverage else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def venues(
        self,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        venue_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw venue data (Feed OT3).
        This feed is paginated.

        Note: You must specify at least one of 'tournament_calendar_uuid',
        'contestant_uuid', or 'venue_uuid'.

        Parameters
        ----------
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        venue_uuid : str, optional
            Filter by a specific venue UUID.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw venue data.

        Raises
        ------
        ValueError
            If no filter (tmcl, ctst, or venue) is provided.
        """
        if not tournament_calendar_uuid and not contestant_uuid and not venue_uuid:
            raise ValueError(
                "At least one of 'tournament_calendar_uuid', 'contestant_uuid', or 'venue_uuid' must be provided."
            )

        return self._step(
            "venues",
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            venue_uuid=venue_uuid,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def areas(
        self,
        area_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw area data (Feed OT4).

        If 'area_uuid' is provided, fetches data for a specific area.
        If 'area_uuid' is None, fetches a paginated list of all areas.

        Parameters
        ----------
        area_uuid : str, optional
            The UUID for a specific area.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw area data.
        """
        if area_uuid:
            # Fetch a specific area (non-paginated)
            source = "area_specific"
        else:
            # Fetch all areas (paginated)
            source = "areas_all"

        return self._step(
            source,
            area_uuid=area_uuid,
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def tournament_schedule(
        self,
        tournament_calendar_uuid: str,
        coverage_level: Optional[Union[int, List[int]]] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw tournament schedule data (Feed MA0).

        Parameters
        ----------
        tournament_calendar_uuid : str
            The UUID for the specific tournament calendar (season).
        coverage_level : int or List[int], optional
            Filter by coverage level(s). Can be a single integer or list of integers.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw tournament schedule data.
        """
        cvlv_str = None
        if isinstance(coverage_level, list):
            cvlv_str = ",".join(map(str, coverage_level))
        elif isinstance(coverage_level, int):
            cvlv_str = str(coverage_level)
        return self._step(
            "tournament_schedule",
            tournament_calendar_uuid=tournament_calendar_uuid,
            coverage_level=cvlv_str,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def matches(
        self,
        fixture_uuids: Optional[List[str]] = None,
        tournament_calendar_uuid: Optional[str] = None,
        competition_uuids: Optional[List[str]] = None,
        contestant_uuid: Optional[str] = None,
        opponent_uuid: Optional[str] = None,
        contestant_position: Optional[Literal["home", "away"]] = None,
        date_from: Optional[Union[str, datetime]] = None,
        date_to: Optional[Union[str, datetime]] = None,
        delta_timestamp: Optional[Union[str, datetime]] = None,
        live: bool = False,
        lineups: bool = False,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match data (Feed MA1 - Basic).
        This feed is paginated. Yields one raw JSON object per page.

        Parameters
        ----------
        fixture_uuids : List[str], optional
            Get specific matches by UUID (comma-separated).
        tournament_calendar_uuid : str, optional
            Filter by tournament calendar.
        competition_uuids : List[str], optional
            Filter by competition(s) (comma-separated).
        contestant_uuid : str, optional
            Filter by a specific contestant (team).
        opponent_uuid : str, optional
            Filter for matches where contestant_uuid played opponent_uuid (maps to ctst2).
        contestant_position : Literal["home", "away"], optional
            Filter for matches where contestant_uuid played home or away.
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
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'})
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
            "matches_basic",
            fixture_uuids=fixture_uuids,
            tournament_calendar_uuid=tournament_calendar_uuid,
            competition_uuids=competition_uuids,
            contestant_uuid=contestant_uuid,
            opponent_uuid=opponent_uuid,
            contestant_position=contestant_position,
            date_range=date_range_str,
            delta_timestamp=delta_timestamp,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def match(
        self,
        fixture_uuid: str,
        live: bool = False,
        lineups: bool = False,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw data for a single match (Feed MA1 - Basic).

        Parameters
        ----------
        fixture_uuid : str
            The UUID for the specific match/fixture.
        live : bool, optional
            Request live data (default: False).
        lineups : bool, optional
            Request lineup data (requires live=True) (default: False).
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw match data for the specified fixture.
        """
        return self._step(
            "match_basic",
            fixture_uuid=fixture_uuid,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def match_stats(
        self,
        fixture_uuids: Union[str, List[str]],
        include_players: bool = True,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match stats data (Feed MA2 - Basic).

        Parameters
        ----------
        fixture_uuids : str or List[str]
            The UUID(s) for the specific match/fixture(es). Can be a single UUID or list of UUIDs.
        include_players : bool, optional
            Include player-level statistics (default: True).
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw match statistics data.
        """
        if isinstance(fixture_uuids, str):
            fixture_uuids = [fixture_uuids]
        return self._step(
            "match_stats_basic",
            fixture_uuids=fixture_uuids,
            include_players="yes" if include_players else "no",
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
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
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw match events data (Feed MA3).

        Parameters
        ----------
        fixture_uuid : str
            The UUID for the specific match/fixture.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        person_uuid : str, optional
            Filter by a specific person (player) UUID.
        event_types : int or List[int], optional
            Filter by event type(s). Can be a single integer or list of integers.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw match events data.
        """
        return self._step(
            "match_events",
            fixture_uuid=fixture_uuid,
            contestant_uuid=contestant_uuid,
            person_uuid=person_uuid,
            event_types=event_types,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def player_career(
        self,
        person_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        active: bool = True,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw player career data (Feed PE2).

        Fetches by person (non-paginated) or contestant (paginated).
        You must specify exactly one of 'person_uuid' or 'contestant_uuid'.

        Parameters
        ----------
        person_uuid : str, optional
            Filter by a specific person UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        active : bool, optional
            When using 'contestant_uuid', filter for active players
            (default: True). Ignored for 'person_uuid'.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw player career data.

        Raises
        ------
        ValueError
            If neither or both 'person_uuid' and 'contestant_uuid' are provided.
        """
        if not person_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'person_uuid' or 'contestant_uuid' must be provided."
            )
        if person_uuid and contestant_uuid:
            raise ValueError("Cannot provide both 'person_uuid' and 'contestant_uuid'.")

        if person_uuid:
            source = "player_career_person"
            active_param = None  # 'active' only applies to ctst
        else:
            source = "player_career_contestant"
            active_param = active

        return self._step(
            source,
            person_uuid=person_uuid,
            contestant_uuid=contestant_uuid,
            active=active_param,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def referees(
        self,
        person_uuid: Optional[str] = None,
        tournament_calendar_uuid: Optional[str] = None,
        stage_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw referee data (Feed PE3).
        This feed is paginated.

        You must specify exactly one of 'person_uuid', 'tournament_calendar_uuid',
        or 'stage_uuid'.

        Parameters
        ----------
        person_uuid : str, optional
            Filter by a specific person (referee) UUID.
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar UUID.
        stage_uuid : str, optional
            Filter by a specific stage UUID.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw referee data.

        Raises
        ------
        ValueError
            If the parameter combination is invalid (e.g., none or multiple
            filter UUIDs are provided).
        """
        # Check that exactly one filter is provided
        filters_provided = sum(
            1
            for f in [person_uuid, tournament_calendar_uuid, stage_uuid]
            if f is not None
        )
        if filters_provided == 0:
            raise ValueError(
                "One of 'person_uuid', 'tournament_calendar_uuid', or 'stage_uuid' must be provided."
            )
        if filters_provided > 1:
            raise ValueError(
                "Only one of 'person_uuid', 'tournament_calendar_uuid', or 'stage_uuid' can be provided at a time."
            )

        return self._step(
            "referees",
            person_uuid=person_uuid,
            tournament_calendar_uuid=tournament_calendar_uuid,
            stage_uuid=stage_uuid,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def injuries(
        self,
        person_uuid: Optional[str] = None,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw player injury data (Feed PE7).

        Fetches by person (non-paginated) or by tournament calendar (paginated).
        You must specify 'person_uuid' OR 'tournament_calendar_uuid'.
        'contestant_uuid' can only be used in combination with 'tournament_calendar_uuid'.

        Parameters
        ----------
        person_uuid : str, optional
            Filter by a specific person UUID. Can be used as the primary
            filter (path parameter) or with 'tournament_calendar_uuid'
            (query parameter).
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
            Must be used with 'tournament_calendar_uuid'.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw injury data.

        Raises
        ------
        ValueError
            If parameter combinations are invalid (e.g., no args,
            'ctst' without 'tmcl', or 'person_uuid' and 'contestant_uuid' together).
        """
        if not person_uuid and not tournament_calendar_uuid:
            raise ValueError(
                "Either 'person_uuid' or 'tournament_calendar_uuid' must be provided."
            )
        if contestant_uuid and not tournament_calendar_uuid:
            raise ValueError(
                "'contestant_uuid' can only be used in combination with 'tournament_calendar_uuid'."
            )
        if contestant_uuid and person_uuid:
            raise ValueError(
                "Cannot use 'contestant_uuid' and 'person_uuid' in the same request."
            )

        # Use specific path-based source if ONLY person_uuid is given
        if person_uuid and not tournament_calendar_uuid:
            source = "injuries_person_path"
        else:
            # All other combinations use the query endpoint
            source = "injuries_query"

        return self._step(
            source,
            person_uuid=person_uuid,
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def transfers(
        self,
        person_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        competition_uuid: Optional[str] = None,
        tournament_calendar_uuid: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw player transfer data (Feed TM7).

        Fetches by person (non-paginated) or by other criteria (paginated).
        You must specify at least one of 'person_uuid', 'contestant_uuid',
        'competition_uuid', or 'tournament_calendar_uuid'.

        Parameters
        ----------
        person_uuid : str, optional
            Filter by a specific person UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        competition_uuid : str, optional
            Filter by a specific competition UUID. (Recommended to use with dates).
        tournament_calendar_uuid : str, optional
            Filter by a specific tournament calendar UUID.
        start_date : str, datetime, or date, optional
            The start date for filtering (YYYY-MM-DD). Requires 'end_date'.
        end_date : str, datetime, or date, optional
            The end date for filtering (YYYY-MM-DD). Requires 'start_date'.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw transfer data (grouped by person).

        Raises
        ------
        ValueError
            If parameter combinations are invalid (e.g., no filters,
            or partial date range).
        """
        if not any(
            [
                person_uuid,
                contestant_uuid,
                competition_uuid,
                tournament_calendar_uuid,
            ]
        ):
            raise ValueError(
                "At least one of 'person_uuid', 'contestant_uuid', "
                "'competition_uuid', or 'tournament_calendar_uuid' must be provided."
            )

        start_date_str = None
        end_date_str = None
        if start_date and end_date:
            start_date_str = _format_opta_date(start_date)
            end_date_str = _format_opta_date(end_date)
        elif start_date or end_date:
            raise ValueError(
                "Both 'start_date' and 'end_date' must be provided together."
            )

        return self._step(
            "transfers",
            person_uuid=person_uuid,
            contestant_uuid=contestant_uuid,
            competition_uuid=competition_uuid,
            tournament_calendar_uuid=tournament_calendar_uuid,
            start_date=start_date_str,
            end_date=end_date_str,
            use_opta_names="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def teams(
        self,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        country_uuid: Optional[str] = None,
        stage_uuid: Optional[str] = None,
        series_uuid: Optional[str] = None,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw team data (Feed TM1).

        Parameters
        ----------
        tournament_calendar_uuid : str, optional
            Filter by tournament calendar UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        country_uuid : str, optional
            Filter by country UUID.
        stage_uuid : str, optional
            Filter by stage UUID.
        series_uuid : str, optional
            Filter by series UUID.
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw team data.

        Raises
        ------
        ValueError
            If neither 'tournament_calendar_uuid' nor 'contestant_uuid' is provided.
        """
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the teams feed."
            )
        return self._step(
            "teams",
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            country_uuid=country_uuid,
            stage_uuid=stage_uuid,
            series_uuid=series_uuid,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def squads(
        self,
        tournament_calendar_uuid: Optional[str] = None,
        contestant_uuid: Optional[str] = None,
        use_opta_names: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of raw squad data (Feed TM3).

        Parameters
        ----------
        tournament_calendar_uuid : str, optional
            Filter by tournament calendar UUID.
        contestant_uuid : str, optional
            Filter by a specific contestant (team) UUID.
        use_opta_names : bool, optional
            Request 'en-op' locale for Opta-specific names (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw squad data.

        Raises
        ------
        ValueError
            If neither 'tournament_calendar_uuid' nor 'contestant_uuid' is provided.
        """
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the squads feed."
            )
        return self._step(
            "squads",
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            _lcl="en-op" if use_opta_names else None,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def player_season_stats(
        self,
        tournament_calendar_uuid: str,
        contestant_uuid: str,
        detailed: bool = True,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of cumulative player stats for a season (Feed TM4).
        This feed is paginated.

        Parameters
        ----------
        tournament_calendar_uuid : str
            The UUID for the specific tournament calendar (season).
        contestant_uuid : str, optional
            Filter by a specific contestant (team).
        detailed : bool, optional
            Request detailed stats (default: True).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding a stream of player stat records.
        """
        return self._step(
            "player_season_stats",  # New source name
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            detailed="yes" if detailed else "no",
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def team_season_stats(
        self,
        tournament_calendar_uuid: str,
        contestant_uuid: str = None,
        detailed: bool = True,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of cumulative team stats for a season (Feed TM4).
        This feed is paginated.

        Parameters
        ----------
        tournament_calendar_uuid : str
            The UUID for the specific tournament calendar (season).
        contestant_uuid : str
            Filter by a specific contestant (team).
        detailed : bool, optional
            Request detailed stats (default: True).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests.
        optimize : bool, optional
            Whether to optimize the plan.

        Returns
        -------
        Flow
            A Flow yielding a stream of team stat records.
        """
        return self._step(
            "team_season_stats",  # New source name
            tournament_calendar_uuid=tournament_calendar_uuid,
            contestant_uuid=contestant_uuid,
            detailed="yes" if detailed else "no",
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )

    def contestant_participation(
        self,
        contestant_uuid: Union[str, List[str]],
        active: bool = False,
        creds: Optional[dict] = None,
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow of contestant participation data (Feed TM16).
        Provides a list of competitions and tournament calendars for a team.
        This feed is paginated.

        Parameters
        ----------
        contestant_uuid : str or List[str]
            The UUID(s) for the specific contestant(s) (team).
            Can be a single UUID, list of UUIDs, or an Opta ID.
        active : bool, optional
            Filter for active tournament calendars only (default: False).
        creds : dict, optional
            Credentials for Opta API.
        proxies : dict, optional
            Proxies dictionary for requests (e.g., {'http': 'socks5h://...'}).
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            A Flow yielding raw competition/tournament calendar data.

        Raises
        ------
        ValueError
            If 'contestant_uuid' is not provided.
        """
        if not contestant_uuid:
            raise ValueError(
                "'contestant_uuid' must be provided for the contestant_participation feed."
            )
        return self._step(
            "contestant_participation",
            contestant_uuid=contestant_uuid,
            active=active,
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )


# Bind singleton instance
opta = Opta()
