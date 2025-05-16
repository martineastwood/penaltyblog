import os
from typing import Optional

import requests

try:
    import statsbombpy
except ImportError:
    statsbombpy = None


class statsbomb:

    DEFAULT_CREDS = {
        "user": os.environ.get("SB_USERNAME"),
        "passwd": os.environ.get("SB_PASSWORD"),
    }

    @staticmethod
    def _require_statsbombpy():
        if statsbombpy is None:
            raise ImportError(
                "statsbombpy is required. Install with `pip install statsbombpy`"
            )

    @classmethod
    def competitions(cls) -> "Flow":
        """
        Get Flow of all available competitions.

        Returns:
            Flow: A Flow of competition records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(sb.competitions(fmt="dict").values())
        return Flow(data)

    @classmethod
    def matches(
        cls, competition_id: int, season_id: int, creds: Optional[dict] = None
    ) -> "Flow":
        """
        Get Flow of matches for a given competition + season.

        Args:
            competition_id (int): The competition ID.
            season_id (int): The season ID.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.matches(
                competition_id=competition_id,
                season_id=season_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
            ).values()
        )
        return Flow(data)

    @classmethod
    def lineups(cls, match_id: int, creds: Optional[dict] = None) -> "Flow":
        """
        Get Flow of lineups for a given match.

        Args:
            match_id (int): The match ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.lineups(
                match_id=match_id, fmt="dict", creds=creds or cls.DEFAULT_CREDS
            ).values()
        )
        return Flow(data)

    @classmethod
    def events(
        cls,
        match_id: int,
        include_360_metrics: bool = False,
        creds: Optional[dict] = None,
    ) -> "Flow":
        """
        Get Flow of events for a given match.

        Args:
            match_id (int): The match ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.events(
                match_id=match_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
                include_360_metrics=include_360_metrics,
            ).values()
        )
        return Flow(data)

    @classmethod
    def player_match_stats(
        cls,
        match_id: int,
        creds: Optional[dict] = None,
    ) -> "Flow":
        """
        Get Flow of a player's match stats for a given match_id

        Args:
            match_id (int): The match ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.player_match_stats(
                match_id=match_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
            ).values()
        )
        return Flow(data)

    @classmethod
    def player_season_stats(
        cls,
        competition_id: int,
        season_id: int,
        creds: Optional[dict] = None,
    ) -> "Flow":
        """
        Get Flow of a player's season stats for a given competition_id and season_id

        Args:
            competition_id (int): The competition ID.
            season_id (int): The season ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.player_season_stats(
                competition_id=competition_id,
                season_id=season_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
            ).values()
        )
        return Flow(data)

    @classmethod
    def team_match_stats(
        cls,
        match_id: int,
        creds: Optional[dict] = None,
    ) -> "Flow":
        """
        Get Flow of a team's match stats for a given match_id

        Args:
            match_id (int): The match ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.team_match_stats(
                match_id=match_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
            ).values()
        )
        return Flow(data)

    @classmethod
    def team_season_stats(
        cls,
        competition_id: int,
        season_id: int,
        creds: Optional[dict] = None,
    ) -> "Flow":
        """
        Get Flow of a team's season stats for a given competition_id and season_id

        Args:
            competition_id (int): The competition ID.
            season_id (int): The season ID.
            creds (dict, optional): StatsBomb credentials. Defaults to DEFAULT_CREDS.

        Returns:
            Flow: A Flow of match records.
        """
        cls._require_statsbombpy()
        from statsbombpy import sb

        from .flow import Flow

        data = list(
            sb.team_season_stats(
                competition_id=competition_id,
                season_id=season_id,
                fmt="dict",
                creds=creds or cls.DEFAULT_CREDS,
            ).values()
        )
        return Flow(data)

    @staticmethod
    def from_github_file(file_id: int, type: str = "events") -> "Flow":
        """
        Load a StatsBomb event data file from GitHub.

        Consumes the HTTP response; the resulting Flow is a stream of records.

        Args:
            file_id (int): The StatsBomb file ID.
            type (str, optional): The type of data to load. Defaults to "events". Can be one of:
                - "events"
                - "lineups"
                - "three-sixty"
                - "matches"

        Returns:
            Flow: A Flow object.
        """
        from .flow import Flow

        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/{type}/{file_id}.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return Flow(r for r in data)
        else:
            return Flow(iter([data]))
