import io
from datetime import datetime

import pandas as pd

from .base_scrapers import RequestsScraper
from .common import sanitize_columns


class ClubElo(RequestsScraper):
    """
    Collects data from clubelo.com.com as pandas dataframes

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    source = "clubelo"

    def __init__(self, team_mappings=None):
        self.base_url = "http://api.clubelo.com/"

        super().__init__(team_mappings=team_mappings)

    def _season_mapping(self, season):
        years = season.split("-")
        part1 = years[0][-2:]
        part2 = years[1][-2:]
        mapped = part1 + part2
        return mapped

    def _column_name_mapping(self, df) -> pd.DataFrame:
        """
        Internal function to rename columns to make consistent with other data sources
        """
        cols = {"Club": "team"}
        df = df.rename(columns=cols)
        return df

    def _convert_date(self, df):
        df["From"] = pd.to_datetime(df["From"])
        df["To"] = pd.to_datetime(df["To"])
        return df

    def get_elo_by_date(self, date=None) -> pd.DataFrame:
        """
        Get team's elo ratings on a specified date

        Parameters
        ----------
        date : str
            date of interest in format 2020-08-30, defaults to current date
        """
        if date is None:
            date = datetime.now().date()
        else:
            date = datetime.strptime(date, "%Y-%m-%d")

        url = (
            self.base_url + str(date.year) + "-" + str(date.month) + "-" + str(date.day)
        )

        content = self.get(url)
        df = (
            pd.read_csv(io.StringIO(content))
            .pipe(self._convert_date)
            .pipe(self._column_name_mapping)
            .pipe(sanitize_columns)
            .pipe(self._map_teams, columns=["team"])
            .sort_values("elo", ascending=False)
            .set_index("team")
        )

        return df

    def get_elo_by_team(self, team) -> pd.DataFrame:
        """
        Get team's historical elo ratings. Acceptable team names can be found using the ..

        Parameters
        ----------
        team : str
            team of interest
        """
        url = self.base_url + team

        content = self.get(url)
        df = (
            pd.read_csv(io.StringIO(content))
            .pipe(self._convert_date)
            .pipe(self._column_name_mapping)
            .pipe(sanitize_columns)
            .pipe(self._map_teams, columns=["team"])
            .sort_values("from", ascending=False)
            .set_index("from")
        )

        return df

    def get_team_names(self) -> pd.DataFrame:
        """
        Gets the names of all available teams through Club Elo
        """
        date = datetime.now().date()

        url = (
            self.base_url + str(date.year) + "-" + str(date.month) + "-" + str(date.day)
        )

        content = self.get(url)
        df = pd.read_csv(io.StringIO(content))[["Club"]].rename(
            columns={"Club": "team"}
        )

        return df
