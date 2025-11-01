# penaltyblog/matchflow/contrib/opta.py
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

from typing_extensions import Literal

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

        args_with_raw = args.copy()
        args_with_raw["raw"] = True
        return Flow(
            plan=[
                {
                    "op": "from_opta",
                    "source": source,
                    "base_url": self.BASE_URL,
                    "asset_type": self.ASSET_TYPE,
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
        proxies: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """Return a Flow of raw tournament calendar data (Feed OT2)."""
        return self._step(
            "tournament_calendars",
            status=status,
            comp=competition_uuid,
            ctst=contestant_uuid,
            stages="yes" if include_stages else None,
            coverage="yes" if include_coverage else None,
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
        """Return a Flow of raw tournament schedule data (Feed MA0)."""
        cvlv_str = None
        if isinstance(coverage_level, list):
            cvlv_str = ",".join(map(str, coverage_level))
        elif isinstance(coverage_level, int):
            cvlv_str = str(coverage_level)
        return self._step(
            "tournament_schedule",
            tournament_calendar_uuid=tournament_calendar_uuid,
            cvlv=cvlv_str,
            _lcl="en-op" if use_opta_names else None,
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
            fx=",".join(fixture_uuids) if fixture_uuids else None,
            tmcl=tournament_calendar_uuid,
            comp=",".join(competition_uuids) if competition_uuids else None,
            ctst=contestant_uuid,
            ctst2=opponent_uuid,
            ctstpos=contestant_position,
            mt_mDt=date_range_str,
            _dlt=_format_opta_datetime(delta_timestamp) if delta_timestamp else None,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            _lcl="en-op" if use_opta_names else None,
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
        """Return a Flow of raw data for a single match (Feed MA1 - Basic)."""
        return self._step(
            "match_basic",
            fixture_uuid=fixture_uuid,
            live="yes" if live or lineups else "no",
            lineups="yes" if lineups else "no",
            _lcl="en-op" if use_opta_names else None,
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
        """Return a Flow of raw match stats data (Feed MA2 - Basic)."""
        return self._step(
            "match_stats_basic",
            fixture_uuids=fixture_uuids,
            people="yes" if include_players else "no",
            _lcl="en-op" if use_opta_names else None,
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
        """Return a Flow of raw match events data (Feed MA3)."""
        type_str = None
        if isinstance(event_types, list):
            type_str = ",".join(map(str, event_types))
        elif isinstance(event_types, int):
            type_str = str(event_types)
        return self._step(
            "match_events",
            fixture_uuid=fixture_uuid,
            ctst=contestant_uuid,
            prsn=person_uuid,
            type=type_str,
            _lcl="en-op" if use_opta_names else None,
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
        """Return a Flow of raw team data (Feed TM1)."""
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the teams feed."
            )
        return self._step(
            "teams",
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
            ctry=country_uuid,
            stg=stage_uuid,
            srs=series_uuid,
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
        """Return a Flow of raw squad data (Feed TM3)."""
        if not tournament_calendar_uuid and not contestant_uuid:
            raise ValueError(
                "Either 'tournament_calendar_uuid' or 'contestant_uuid' must be provided for the squads feed."
            )
        return self._step(
            "squads",
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
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
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
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
            tmcl=tournament_calendar_uuid,
            ctst=contestant_uuid,
            detailed="yes" if detailed else "no",
            creds=creds or self.DEFAULT_CREDS,
            proxies=proxies,
            optimize=optimize,
        )


# Bind singleton instance
opta = Opta()
