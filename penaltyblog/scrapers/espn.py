import json
from datetime import datetime

import pandas as pd

from .base_scrapers import RequestsScraper
from .common import COMPETITION_MAPPINGS, create_game_id, sanitize_columns


class ESPN(RequestsScraper):
    """
    Scrapes data from fbref.com as pandas dataframes

    Parameters
    ----------
    league : str
        Name of the league of interest,
        see the `ESPN.list_competitions()` output for available choices

    season : str
        Name of the season of interest in format 2020-2021

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    source = "espn"

    def __init__(self, competition, season, team_mappings=None):

        self._check_competition(competition)

        self.base_url = (
            "https://site.api.espn.com/apis/site/v2/sports/soccer/"
            "{competition}/scoreboard?dates={date}"
        )
        self.competition = competition
        self.season = season
        self.mapped_season = self._map_season(self.season)
        self.mapped_competition = COMPETITION_MAPPINGS[self.competition]["espn"]["slug"]
        self.start_date = COMPETITION_MAPPINGS[self.competition]["espn"]["start_date"]

        super().__init__(team_mappings=team_mappings)

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

    def _map_fixture_column_types(self, df) -> pd.DataFrame:
        """
        Internal function to set column dtypes

        Parameters
        ----------
        df : pd.DataFrame
            the scraped fixtures
        """
        mappings = {
            "total_shots_home": int,
            "total_shots_away": int,
            "attendance": int,
            "goals_home": int,
            "goals_away": int,
            "fouls_committed_home": int,
            "fouls_committed_away": int,
            "won_corners_home": int,
            "won_corners_away": int,
            "possession_pct_home": float,
            "possession_pct_away": float,
            "goal_assists_home": int,
            "goal_assists_away": int,
            "shots_on_target_home": int,
            "shots_on_target_away": int,
            "total_goals_home": int,
            "total_goals_away": int,
            "shot_assists_home": int,
            "shot_assists_away": int,
        }
        for k, v in mappings.items():
            df[k] = df[k].astype(v)

        return df

    def _convert_date(self, df):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        return df

    def get_fixtures(self) -> pd.DataFrame:
        """
        Gets the fixtures / results for the selected competition / season
        """
        url = self.base_url.format(
            date=self.mapped_season + self.start_date,
            competition=self.mapped_competition,
        )

        content = self.get(url)
        content = json.loads(content)
        fixture_dates = content["leagues"][0]["calendar"]

        fixtures = list()
        for date in fixture_dates:

            url = self.base_url.format(
                date=datetime.strptime(date, "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d"),
                competition=self.mapped_competition,
            )
            content = self.get(url)
            content = json.loads(content)
            events = self._scrape_fixture_events(content)
            fixtures.extend(events)

        df = pd.DataFrame(fixtures)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = (
            df.pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._convert_date)
            .pipe(self._map_teams, columns=["team_home", "team_away"])
            .pipe(create_game_id)
            .pipe(self._map_fixture_column_types)
            .set_index("id")
            .sort_index()
        )

        return df

    def _scrape_fixture_events(self, content) -> list:
        """
        Internal method to extract the relevant info about the fixtures from the JSON
        """
        events = list()

        for e in content.get("events", []):
            tmp = dict()
            tmp["espn_id"] = e["competitions"][0].get("id")
            tmp["datetime"] = e["competitions"][0].get("date")
            tmp["attendance"] = e["competitions"][0].get("attendance")
            tmp["team_home"] = e["competitions"][0]["competitors"][0]["team"].get(
                "name"
            )
            tmp["team_away"] = e["competitions"][0]["competitors"][1]["team"].get(
                "name"
            )
            tmp["goals_home"] = e["competitions"][0]["competitors"][0].get("score")
            tmp["goals_away"] = e["competitions"][0]["competitors"][1].get("score")

            for stat in e["competitions"][0]["competitors"][0]["statistics"]:
                tmp[stat["name"] + "_home"] = stat["displayValue"]

            for stat in e["competitions"][0]["competitors"][1]["statistics"]:
                tmp[stat["name"] + "_away"] = stat["displayValue"]

            events.append(tmp)

        return events
