"""
StatsBomb integration for Flow.
Provides lazy-loading access to StatsBomb API via plan-based execution.
"""

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..flow import Flow


class StatsBomb:
    @property
    def DEFAULT_CREDS(self) -> dict:
        """Get default credentials from environment variables."""
        return {
            "user": os.environ.get("SB_USERNAME"),
            "passwd": os.environ.get("SB_PASSWORD"),
        }

    def _step(self, source: str, optimize: bool = False, **args) -> "Flow":
        """
        Build a plan that represents a single StatsBomb source operation and
        return a Flow wrapping that plan.

        This is an internal helper used by the public methods. It does not
        execute the plan; it only constructs the Flow object which can be
        executed later by the caller.

        Parameters
        ----------
        source : str
            The StatsBomb resource name (for example, 'competitions',
            'matches', 'events', etc.).
        optimize : bool, optional
            If True, the returned Flow will be created with optimization
            enabled, by default False.
        **args
            Additional keyword arguments are forwarded into the plan's
            "args" field (for example, competition_id, season_id, match_id,
            creds, etc.).

        Returns
        -------
        Flow
            A Flow instance that contains a single-plan operation which will
            load the requested StatsBomb resource when executed.
        """
        from ..flow import Flow

        return Flow(
            plan=[
                {
                    "op": "from_statsbomb",
                    "source": source,
                    "args": args,
                }
            ],
            optimize=optimize,
        )

    def competitions(self, optimize: bool = False) -> "Flow":
        """
        Return a Flow for competitions from StatsBomb.

        Parameters
        ----------
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            Flow object for competitions resource.
        """
        return self._step("competitions", optimize=optimize)

    def matches(
        self,
        competition_id: int,
        season_id: int,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow for matches from StatsBomb.

        Parameters
        ----------
        competition_id : int
            Competition ID.
        season_id : int
            Season ID.
        creds : dict, optional
            Credentials for StatsBomb API. Defaults to StatsBomb.DEFAULT_CREDS.
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            Flow object for matches resource.
        """
        return self._step(
            "matches",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def events(
        self,
        match_id: int,
        include_360_metrics=False,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow for events from StatsBomb.

        Parameters
        ----------
        match_id : int
            Match ID.
        include_360_metrics : bool, optional
            Whether to include 360 metrics (default: False).
        creds : dict, optional
            Credentials for StatsBomb API. Defaults to StatsBomb.DEFAULT_CREDS.
        optimize : bool, optional
            Whether to optimize the plan (default: False).

        Returns
        -------
        Flow
            Flow object for events resource.
        """
        return self._step(
            "events",
            match_id=match_id,
            include_360_metrics=include_360_metrics,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def lineups(
        self, match_id: int, creds: Optional[dict] = None, optimize: bool = False
    ) -> "Flow":
        """
        Return a Flow for player lineups for a given match.

        Parameters
        ----------
        match_id : int
            The StatsBomb match identifier for which to fetch lineups.
        creds : dict, optional
            Optional credentials to use for the StatsBomb API. If omitted,
            `StatsBomb.DEFAULT_CREDS` will be used.
        optimize : bool, optional
            If True, the returned Flow will have optimization enabled.

        Returns
        -------
        Flow
            A Flow that will fetch the lineups when executed.
        """
        return self._step(
            "lineups",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def player_match_stats(
        self, match_id: int, creds: Optional[dict] = None, optimize: bool = False
    ) -> "Flow":
        """
        Return a Flow for per-player statistics for a single match.

        Parameters
        ----------
        match_id : int
            The StatsBomb match identifier.
        creds : dict, optional
            Credentials to authenticate with the StatsBomb API. Defaults to
            `StatsBomb.DEFAULT_CREDS` when not provided.
        optimize : bool, optional
            If True, the returned Flow will be optimized.

        Returns
        -------
        Flow
            A Flow that will fetch player match statistics when executed.
        """
        return self._step(
            "player_match_stats",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def player_season_stats(
        self,
        competition_id: int,
        season_id: int,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow for per-player aggregated statistics for a season.

        Parameters
        ----------
        competition_id : int
            StatsBomb competition identifier.
        season_id : int
            StatsBomb season identifier.
        creds : dict, optional
            Optional credentials for the StatsBomb API. Defaults to
            `StatsBomb.DEFAULT_CREDS` when omitted.
        optimize : bool, optional
            If True, create the Flow with optimization enabled.

        Returns
        -------
        Flow
            A Flow that will fetch player season statistics when executed.
        """
        return self._step(
            "player_season_stats",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def team_match_stats(
        self, match_id: int, creds: Optional[dict] = None, optimize: bool = False
    ) -> "Flow":
        """
        Return a Flow for team-level statistics for a single match.

        Parameters
        ----------
        match_id : int
            The StatsBomb match identifier.
        creds : dict, optional
            Credentials for StatsBomb; falls back to
            `StatsBomb.DEFAULT_CREDS` if not provided.
        optimize : bool, optional
            Whether to optimize the constructed Flow.

        Returns
        -------
        Flow
            A Flow that will fetch team match statistics when executed.
        """
        return self._step(
            "team_match_stats",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def team_season_stats(
        self,
        competition_id: int,
        season_id: int,
        creds: Optional[dict] = None,
        optimize: bool = False,
    ) -> "Flow":
        """
        Return a Flow for team-level aggregated statistics for a season.

        Parameters
        ----------
        competition_id : int
            StatsBomb competition identifier.
        season_id : int
            StatsBomb season identifier.
        creds : dict, optional
            Optional credentials to use with the StatsBomb API; defaults to
            `StatsBomb.DEFAULT_CREDS` when omitted.
        optimize : bool, optional
            Whether to enable optimizations for the created Flow.

        Returns
        -------
        Flow
            A Flow that will fetch team season statistics when executed.
        """
        return self._step(
            "team_season_stats",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )


# Bind singleton instance
statsbomb = StatsBomb()
