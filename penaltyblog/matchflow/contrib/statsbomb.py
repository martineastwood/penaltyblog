"""
StatsBomb integration for Flow.
Provides lazy-loading access to StatsBomb API via plan-based execution.
"""

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..flow import Flow


class StatsBomb:
    DEFAULT_CREDS = {
        "user": os.environ.get("SB_USERNAME"),
        "passwd": os.environ.get("SB_PASSWORD"),
    }

    def _step(self, source: str, optimize: bool = False, **args) -> "Flow":
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

        Args:
            optimize (bool): Whether to optimize the plan (default: False)
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

        Args:
            competition_id (int): Competition ID
            season_id (int): Season ID
            creds (dict, optional): Credentials
            optimize (bool): Whether to optimize the plan (default: False)
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

        Args:
            match_id (int): Match ID
            include_360_metrics (bool): Whether to include 360 metrics
            creds (dict, optional): Credentials
            optimize (bool): Whether to optimize the plan (default: False)
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
        return self._step(
            "lineups",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )

    def player_match_stats(
        self, match_id: int, creds: Optional[dict] = None, optimize: bool = False
    ) -> "Flow":
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
        return self._step(
            "team_season_stats",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
            optimize=optimize,
        )


# Bind singleton instance
statsbomb = StatsBomb()
