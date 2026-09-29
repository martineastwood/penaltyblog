import os
import re
from typing import Any

import pandas as pd
import requests

from .base_scrapers import RequestsScraper
from .common import COMPETITION_MAPPINGS, create_game_id, move_column_inplace


class FootballCharts(RequestsScraper):
    """
    Loads football results from football-charts.com as pandas dataframes.

    Parameters
    ----------
    competition : str
        Canonical penaltyblog competition name. See
        ``FootballCharts.list_competitions()`` for available choices.

    season : str
        Season of interest. European seasons use formats such as
        ``2025-2026`` and calendar-year leagues use formats such as ``2026``.

    team_mappings : dict or None
        Dict (or None) of team name mappings in format
        ``{"Canonical Team": ["Alternative Name"]}``.

    api_key : str or None
        Optional Football Charts API key. If omitted, ``FC_API_KEY`` is used
        when it is present in the environment.

    Notes
    -----
    Football Charts requires attribution when its data is used. The returned
    dataframe stores ``"Data by football-charts.com"`` in its ``attribution``
    attribute.
    """

    source = "footballcharts"
    api_base_url = "https://footballcharts-backend.onrender.com/api/v1"

    def __init__(self, competition, season, team_mappings=None, api_key=None):
        self._check_competition(competition)

        self.competition = competition
        self.season = season
        self.league = COMPETITION_MAPPINGS[competition][self.source]["slug"]
        self.api_key = api_key

        super().__init__(team_mappings=team_mappings)

    @classmethod
    def from_league(
        cls,
        league,
        season,
        team_mappings=None,
        api_key=None,
    ):
        """
        Create a scraper for a Football Charts league code.

        This provides access to leagues that do not have a canonical
        penaltyblog competition mapping, for example ``"germany3"``.

        Parameters
        ----------
        league : str
            Football Charts league code.

        season : str
            Season of interest.

        team_mappings : dict or None
            Optional team name mappings.

        api_key : str or None
            Optional Football Charts API key.
        """
        if not isinstance(league, str) or not league.strip():
            raise ValueError("league must be a non-empty Football Charts code")

        scraper = cls.__new__(cls)
        scraper.competition = league
        scraper.season = season
        scraper.league = league
        scraper.api_key = api_key
        RequestsScraper.__init__(scraper, team_mappings=team_mappings)
        return scraper

    @classmethod
    def list_leagues(cls, api_key=None) -> pd.DataFrame:
        """Return the Football Charts league catalogue and available seasons."""
        data = cls._get_json("/leagues/", api_key=api_key)
        columns = ["league", "name", "country", "seasons"]
        return pd.DataFrame(data.get("leagues", []), columns=columns)

    @staticmethod
    def _parse_score(score: Any):
        if score is None:
            return None, None

        match = re.fullmatch(r"\s*(\d+)\s*[:\-–]\s*(\d+)\s*", str(score))
        if match is None:
            return None, None

        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _get_json(cls, path, params=None, api_key=None):
        key = api_key or os.environ.get("FC_API_KEY")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/102.0.0.0 Safari/537.36"
            )
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"

        response = requests.get(
            f"{cls.api_base_url}{path}",
            params=params,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 403:
            raise PermissionError(
                f"Football Charts denied {path} with params {params}: "
                "this season may require a paid API key "
                "(the free tier includes the current and previous season)."
            )

        response.raise_for_status()
        return response.json()

    def get_fixtures(self) -> pd.DataFrame:
        """Get completed fixtures for the selected competition and season."""
        data = self._get_json(
            f"/leagues/{self.league}/results/",
            params={"season": self.season},
            api_key=self.api_key,
        )

        rows = []
        for match in data.get("matches", []):
            goals_home, goals_away = self._parse_score(match.get("score"))
            if goals_home is None:
                # Results can contain abandoned or not-yet-played matches.
                continue

            date = pd.to_datetime(match.get("date"), errors="coerce")
            if pd.isna(date):
                continue

            match_time = match.get("time")
            if match_time:
                datetime = pd.to_datetime(
                    f"{match.get('date')} {match_time}", errors="coerce"
                )
            else:
                datetime = date

            hthg, htag = self._parse_score(match.get("ht_result"))
            rows.append(
                {
                    "date": date,
                    "datetime": datetime if not pd.isna(datetime) else date,
                    "season": self.season,
                    "competition": self.competition,
                    "team_home": match.get("homeTeam"),
                    "team_away": match.get("awayTeam"),
                    "goals_home": goals_home,
                    "goals_away": goals_away,
                    "fthg": goals_home,
                    "ftag": goals_away,
                    "hthg": hthg,
                    "htag": htag,
                    "first_goal_minute": match.get("first_goal_time"),
                    "footballcharts_id": match.get("id"),
                }
            )

        columns = [
            "date",
            "datetime",
            "season",
            "competition",
            "team_home",
            "team_away",
            "goals_home",
            "goals_away",
            "fthg",
            "ftag",
            "hthg",
            "htag",
            "first_goal_minute",
            "footballcharts_id",
        ]
        df = pd.DataFrame(rows, columns=columns)

        if df.empty:
            df.index = pd.Index([], name="id")
            df.attrs["attribution"] = "Data by football-charts.com"
            return df

        df = (
            df.pipe(self._map_teams, columns=["team_home", "team_away"])
            .pipe(create_game_id)
            .set_index("id")
            .sort_index()
        )

        move_column_inplace(df, "competition", 0)
        move_column_inplace(df, "season", 1)
        move_column_inplace(df, "datetime", 2)

        df.attrs["attribution"] = "Data by football-charts.com"
        return df
