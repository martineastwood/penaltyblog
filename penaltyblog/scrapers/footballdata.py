import pandas as pd
import io
from .common import (
    COMPETITION_MAPPINGS,
    sanitize_columns,
    create_game_id,
)
from .team_mappings import santize_team_names
from .base_scrapers import BaseScraperRequests


class FootballData(BaseScraperRequests):
    """
    Scrapes data from fbref.com as pandas dataframes

    Parameters
    ----------
    league : str
        Name of the league of interest (optional)

    season : str
        Name of the season of interest (optional) in format 2020-2021
    """

    source = "footballdata"

    def __init__(self, competition, season):
        self.base_url = (
            "https://www.football-data.co.uk/mmz4281/{season}/{competition}.csv"
        )
        self.competition = competition
        self.season = season
        self.mapped_season = self._season_mapping(self.season)
        self.mapped_competition = COMPETITION_MAPPINGS[self.competition][
            "footballdata"
        ]["slug"]

        super().__init__()

    def _season_mapping(self, season):
        """
        Internal function to map season to football-data's format
        """
        years = season.split("-")
        part1 = years[0][-2:]
        part2 = years[1][-2:]
        mapped = part1 + part2
        return mapped

    def _column_name_mapping(self, df) -> pd.DataFrame:
        """
        Internal function to rename columns to make consistent with other data sources
        """
        cols = {"HomeTeam": "team_home", "AwayTeam": "team_away"}
        df = df.rename(columns=cols)
        return df

    def _convert_date(self, df):
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    def get_fixtures(self) -> pd.DataFrame:
        """
        Downloads the fixtures and returns them as a pandas data frame
        """
        url = self.base_url.format(
            season=self.mapped_season, competition=self.mapped_competition
        )

        content = self.get(url)
        df = (
            pd.read_csv(io.StringIO(content))
            .pipe(self._convert_date)
            .pipe(self._column_name_mapping)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(santize_team_names)
            .pipe(create_game_id)
            .drop(["date", "time"], axis=1)
            .set_index(["competition", "season", "id"])
            .sort_index()
        )

        return df
