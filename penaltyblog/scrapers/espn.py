import calendar
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .base_scrapers import RequestsScraper
from .common import COMPETITION_MAPPINGS, create_game_id, sanitize_columns


class ESPN(RequestsScraper):
    """
    Scrapes data from espn as pandas dataframes

    Parameters
    ----------

    league : str
        Name of the league of interest, see
        the `ESPN.list_competitions()` output
        for available choices

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
        Gets the fixtures / results for the selected competition / season and returns as a
        pandas data frame
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

    def get_player_stats(self, espn_id) -> pd.DataFrame:
        """
        Get the player stats for a given fixture

        Parameters
        ----------
        espn_id : str
            ID for the fixture of interest,
            see the `ESPN.get_fixtures()` output for available IDs
        """
        url = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/summary?event={}".format(
            self.mapped_competition, espn_id
        )

        content = self.get(url)
        content = json.loads(content)

        output = list()
        for roster in content["rosters"]:
            for player in roster["roster"]:
                tmp = dict()
                tmp["espn_id"] = espn_id
                tmp["datetime"] = content["header"]["competitions"][0]["date"]
                tmp["espn_player_id"] = player["athlete"]["id"]
                tmp["home"] = 1 if roster["homeAway"] == "home" else 0
                tmp["team"] = roster["team"]["displayName"]
                tmp["player_name"] = player["athlete"]["fullName"]
                tmp["position"] = player["position"]["displayName"]
                tmp["formation_place"] = player["formationPlace"]

                subs = [
                    x
                    for x in player.get("plays", [])
                    if "substitution" in x and x["substitution"]
                ]

                tmp["starter"] = 1 if player["starter"] else 0
                if player["subbedOut"]:
                    tmp["subbed_out"] = subs[-1]["clock"]["displayValue"]
                    tmp["subbed_out"] = re.findall(r"\d+", tmp["subbed_out"])[0]
                    tmp["subbed_out"] = int(tmp["subbed_out"])

                if player["subbedIn"]:
                    tmp["subbed_in"] = subs[0]["clock"]["displayValue"]
                    tmp["subbed_in"] = re.findall(r"\d+", tmp["subbed_in"])[0]
                    tmp["subbed_in"] = int(tmp["subbed_in"])

                for stat in player["stats"]:
                    tmp[stat["name"]] = stat["value"]

                output.append(tmp)

        df = (
            pd.DataFrame(output)
            .replace({np.NaN: None})
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._convert_date)
            .pipe(self._map_teams, columns=["team"])
        )

        id = "{}---{}---{}".format(
            int(calendar.timegm(df["date"].iloc[0].timetuple())),
            df.query("home == 1")["team"].iloc[0],
            df.query("home == 0")["team"].iloc[0],
        )

        df = df.assign(id=id.replace(" ", "_").lower()).set_index("id").sort_index()

        return df

    def get_team_stats(self, espn_id) -> pd.DataFrame:
        """
        Get the team stats for a given fixture

        Parameters
        ----------
        espn_id : str
            ID for the fixture of interest,
            see the `ESPN.get_fixtures()` output for available IDs
        """
        url = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/summary?event={}".format(
            self.mapped_competition, espn_id
        )

        content = self.get(url)
        content = json.loads(content)

        output = list()
        for team in content["boxscore"]["teams"]:
            tmp = dict()
            tmp["espn_id"] = espn_id
            tmp["espn_team_id"] = team["team"]["id"]
            tmp["datetime"] = content["header"]["competitions"][0]["date"]
            tmp["team"] = team["team"]["displayName"]

            for stat in team["statistics"]:
                tmp[stat["name"]] = stat["displayValue"]

            for roster in content["rosters"]:
                if roster["team"]["displayName"] == tmp["team"]:
                    tmp["home"] = 1 if roster["homeAway"] == "home" else 0

            output.append(tmp)

        df = (
            pd.DataFrame(output)
            .replace({np.NaN: None})
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._convert_date)
            .pipe(self._map_teams, columns=["team"])
        )

        id = "{}---{}---{}".format(
            int(calendar.timegm(df["date"].iloc[0].timetuple())),
            df.query("home == 1")["team"].iloc[0],
            df.query("home == 0")["team"].iloc[0],
        )

        df = df.assign(id=id.replace(" ", "_").lower()).set_index("id").sort_index()

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
