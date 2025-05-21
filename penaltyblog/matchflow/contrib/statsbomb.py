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

    def _step(self, source: str, **args) -> "Flow":
        from ..flow import Flow

        return Flow(
            plan=[
                {
                    "op": "from_statsbomb",
                    "source": source,
                    "args": args,
                }
            ]
        )

    def competitions(self) -> "Flow":
        return self._step("competitions")

    def matches(
        self, competition_id: int, season_id: int, creds: Optional[dict] = None
    ) -> "Flow":
        return self._step(
            "matches",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
        )

    def events(
        self, match_id: int, include_360_metrics=False, creds: Optional[dict] = None
    ) -> "Flow":
        return self._step(
            "events",
            match_id=match_id,
            include_360_metrics=include_360_metrics,
            creds=creds or self.DEFAULT_CREDS,
        )

    def lineups(self, match_id: int, creds: Optional[dict] = None) -> "Flow":
        return self._step(
            "lineups",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
        )

    def player_match_stats(self, match_id: int, creds: Optional[dict] = None) -> "Flow":
        return self._step(
            "player_match_stats",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
        )

    def player_season_stats(
        self, competition_id: int, season_id: int, creds: Optional[dict] = None
    ) -> "Flow":
        return self._step(
            "player_season_stats",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
        )

    def team_match_stats(self, match_id: int, creds: Optional[dict] = None) -> "Flow":
        return self._step(
            "team_match_stats",
            match_id=match_id,
            creds=creds or self.DEFAULT_CREDS,
        )

    def team_season_stats(
        self, competition_id: int, season_id: int, creds: Optional[dict] = None
    ) -> "Flow":
        return self._step(
            "team_season_stats",
            competition_id=competition_id,
            season_id=season_id,
            creds=creds or self.DEFAULT_CREDS,
        )


# Bind singleton instance
statsbomb = StatsBomb()
