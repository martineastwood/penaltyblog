import json
import re
from typing import Any, Dict

import pandas as pd
import requests
from lxml import html

from .base_scrapers import RequestsScraper
from .common import (
    COMPETITION_MAPPINGS,
    create_game_id,
    move_column_inplace,
    sanitize_columns,
)


class Understat(RequestsScraper):
    """
    Scrapes data from understat and returns as a pandas dataframes

    Parameters
    ----------
    league : str
        Name of the league of interest,
        see the `Understat.list_competitions()` output for available choices

    season : str
        Name of the season of interest in format 2020-2021

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    source = "understat"

    def __init__(self, competition, season, team_mappings=None):

        self._check_competition(competition)

        self.base_url = "https://understat.com/"
        self.competition = competition
        self.season = season
        self.mapped_season = self._map_season(self.season)
        self.mapped_competition = COMPETITION_MAPPINGS[self.competition]["understat"][
            "slug"
        ]

        super().__init__(team_mappings=team_mappings)

        self.cookies = {"beget": "begetok"}

    def _map_season(self, season) -> str:
        """
        Internal function to map the season name

        Parameters
        ----------
        season : str
            Name of the season
        """
        years = season.split("-")
        part1 = years[0]
        return part1

    def _convert_date(self, df):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        return df

    def get_fixtures(self) -> pd.DataFrame:
        """
        Gets the fixtures / results for the selected competition / season
        """
        # Use the API endpoint which requires specific headers
        url = (
            self.base_url
            + "getLeagueData/"
            + self.mapped_competition
            + "/"
            + self.mapped_season
        )

        # The API requires X-Requested-With and Referer headers
        headers = self.headers.copy()
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Referer"] = (
            f"{self.base_url}league/{self.mapped_competition}/{self.mapped_season}"
        )

        response = requests.get(url, headers=headers, cookies=self.cookies)
        response.raise_for_status()

        # The API returns gzip-compressed JSON
        try:
            data = response.json()
            # The response is a dict with 'dates', 'teams', 'players' keys
            events = data.get("dates", [])
            if not events:
                raise ValueError("No dates data found in response")
        except json.JSONDecodeError:
            raise ValueError("Error: no data found or invalid JSON response")

        fixtures = list()
        for e in events:
            if not e["isResult"]:
                continue

            tmp: dict[str, Any] = dict()
            tmp["understat_id"] = str(e["id"])
            tmp["datetime"] = e["datetime"]
            tmp["team_home"] = e["h"]["title"]
            tmp["team_away"] = e["a"]["title"]
            tmp["goals_home"] = int(e["goals"]["h"])
            tmp["goals_away"] = int(e["goals"]["a"])
            tmp["xg_home"] = float(e["xG"]["h"])
            tmp["xg_away"] = float(e["xG"]["a"])
            tmp["forecast_w"] = float(e["forecast"]["w"])
            tmp["forecast_d"] = float(e["forecast"]["d"])
            tmp["forecast_l"] = float(e["forecast"]["l"])
            fixtures.append(tmp)

        df = (
            pd.DataFrame(fixtures)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._convert_date)
            .pipe(self._map_teams, columns=["team_home", "team_away"])
            .pipe(create_game_id)
            .set_index("id")
            .sort_index()
        )

        return df

    def get_shots(self, understat_id: str) -> pd.DataFrame:
        """
        Gets the shots for the selected understat_id

        Parameters
        ----------
        understat_id : str
            Id for the match of interest,
            Ids can be found using the `get_fixtures()` function
        """
        url = f"{self.base_url}getMatchData/{understat_id}"

        # The API requires X-Requested-With and Referer headers
        headers = self.headers.copy()
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Referer"] = f"{self.base_url}match/{understat_id}"

        response = requests.get(url, headers=headers, cookies=self.cookies)
        response.raise_for_status()

        try:
            data = response.json()
            # The response has 'shots' key containing home and away shots
            events = data.get("shots", {})
            if not events:
                raise ValueError("No shots data found in response")
        except json.JSONDecodeError:
            raise ValueError("Error: no data found or invalid JSON response")

        shots = list()
        shots.extend(events["h"])
        shots.extend(events["a"])

        col_renames = {
            "h_team": "team_home",
            "a_team": "team_away",
            "h_goals": "goals_home",
            "a_goals": "goals_away",
        }

        df = (
            pd.DataFrame(shots)
            .pipe(sanitize_columns)
            .rename(columns=col_renames)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .assign(datetime=lambda x: pd.to_datetime(x.date))
            .assign(date=lambda x: x.datetime.dt.date)
            .pipe(create_game_id)
            .set_index("id")
            .sort_index()
        )

        move_column_inplace(df, "competition", 0)
        move_column_inplace(df, "season", 1)
        move_column_inplace(df, "datetime", 2)

        return df

    def get_fixture_info(self, understat_id: str) -> pd.DataFrame:
        """
        Gets the match info for the selected understat_id

        Parameters
        ----------
        understat_id : str
            Id for the match of interest,
            Ids can be found using the `get_fixtures()` function
        """
        url = "https://understat.com/match/{}".format(understat_id)
        content = self.get(url)
        tree = html.fromstring(content)
        events = None

        for s in tree.cssselect("script"):
            if s.text and "match_info" in s.text:
                script_text = s.text
                script_text = " ".join(script_text.split())
                script_text = str(script_text.encode(), "unicode-escape")
                match_info_parts = script_text.split(" match_info")
                if len(match_info_parts) > 1:
                    match_info = "match_info" + match_info_parts[1]
                    match = re.match(
                        r"match_info = JSON\.parse\('(?P<json>.*?)'\)", match_info
                    )
                    if match is not None:
                        json_str = match.group("json")
                        events = json.loads(json_str)
                        break

        if events is None:
            raise ValueError("Error: no data found")

        col_renames = {
            "team_h": "team_home",
            "team_a": "team_away",
            "h_goals": "goals_home",
            "a_goals": "goals_away",
            "h_xg": "xg_home",
            "a_xg": "xg_away",
            "h_shot": "shots_home",
            "a_shot": "shots_away",
            "h_shotOnTarget": "shots_on_target_home",
            "a_shotOnTarget": "shots_on_target_away",
            "h_deep": "deep_home",
            "a_deep": "deep_away",
            "h_ppda": "ppda_home",
            "a_ppda": "ppda_away",
            "match_id": "understat_fixture_id",
        }

        df = (
            pd.DataFrame([events])
            .rename(columns=col_renames)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .assign(datetime=lambda x: pd.to_datetime(x.date))
            .assign(date=lambda x: x.datetime.dt.date)
            .pipe(create_game_id)
            .set_index("id")
            .sort_index()
        )

        move_column_inplace(df, "competition", 0)
        move_column_inplace(df, "season", 1)
        move_column_inplace(df, "datetime", 2)

        return df

    def get_player_season(self, player_id: str) -> pd.DataFrame:
        """
        Gets the season info for the selected player_id

        Parameters
        ----------
        player_id : str
            Id for the player of interest,
        """
        import requests

        url = f"{self.base_url}getPlayerData/{player_id}"

        # The API requires X-Requested-With and Referer headers
        headers = self.headers.copy()
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Referer"] = f"{self.base_url}player/{player_id}"

        response = requests.get(url, headers=headers, cookies=self.cookies)
        response.raise_for_status()

        try:
            data = response.json()
            # The response has 'groups' key containing season data
            events = data.get("groups", {})
            if not events:
                raise ValueError("No groups data found in response")
        except json.JSONDecodeError:
            raise ValueError("Error: no data found or invalid JSON response")

        df = (
            pd.DataFrame(events["season"])
            .pipe(sanitize_columns)
            .set_index("season")
            .sort_index()
        )

        move_column_inplace(df, "team", 0)

        return df

    def get_player_shots(self, player_id: str) -> pd.DataFrame:
        """
        Gets the shot data for the selected player_id

        Parameters
        ----------
        player_id : str
            Id for the player of interest,
        """
        import requests

        url = f"{self.base_url}getPlayerData/{player_id}"

        # The API requires X-Requested-With and Referer headers
        headers = self.headers.copy()
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Referer"] = f"{self.base_url}player/{player_id}"

        response = requests.get(url, headers=headers, cookies=self.cookies)
        response.raise_for_status()

        try:
            data = response.json()
            # The response has 'shots' key containing shot data
            events = data.get("shots", [])
            if not events:
                raise ValueError("No shots data found in response")
        except json.JSONDecodeError:
            raise ValueError("Error: no data found or invalid JSON response")

        col_renames = {
            "h_team": "team_home",
            "a_team": "team_away",
            "match_id": "understat_fixture_id",
            "id": "understat_shot_id",
            "player_id": "understat_player_id",
            "h_goals": "goals_home",
            "a_goals": "goals_away",
        }

        df = (
            pd.DataFrame(events)
            .rename(columns=col_renames)
            .pipe(sanitize_columns)
            .assign(datetime=lambda x: pd.to_datetime(x.date))
            .assign(date=lambda x: x.datetime.dt.date)
            .pipe(create_game_id)
            .set_index("understat_shot_id")
            .sort_index()
        )

        return df
