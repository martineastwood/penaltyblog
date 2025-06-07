import io

import pandas as pd

from .base_scrapers import RequestsScraper
from .common import (
    COMPETITION_MAPPINGS,
    create_game_id,
    move_column_inplace,
    sanitize_columns,
)


class FootballData(RequestsScraper):
    """
    Scrapes data from football-data.co.uk as pandas dataframes

    Parameters
    ----------
    competition : str
        Name of the league of interest. See the
        `FootballData.list_competitions()` function
        for available competitions

    season : str
        Name of the season of interest in format 2020-2021

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`

    """

    source = "footballdata"

    def __init__(self, competition, season, team_mappings=None):

        self._check_competition(competition)

        self.base_url = (
            "https://www.football-data.co.uk/mmz4281/{season}/{competition}.csv"
        )
        self.competition = competition
        self.season = season
        self.mapped_season = self._season_mapping(self.season)
        self.mapped_competition = COMPETITION_MAPPINGS[self.competition][
            "footballdata"
        ]["slug"]

        super().__init__(team_mappings=team_mappings)

    def _season_mapping(self, season):
        """
        Internal function to map season to football-data's format
        """
        years = season.split("-")
        part1 = years[0][-2:]
        part2 = years[1][-2:]
        mapped = part1 + part2
        return mapped

    def _convert_date(self, df):
        # Check for date format - could be either %d/%m/%Y or %d/%m/%y
        # Sample the first date to determine format
        if df.empty:
            return df

        sample_date = df["Date"].iloc[0] if not df["Date"].empty else ""
        date_format = "%d/%m/%y" if len(sample_date) <= 8 else "%d/%m/%Y"

        # Convert datetime column if Time exists
        if "Time" in df.columns:
            time_format = date_format + " %H:%M"
            df["datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], format=time_format, errors="coerce"
            )

        # Convert Date column
        df["Date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")

        return df

    def get_fixtures(self) -> pd.DataFrame:
        """
        Downloads the fixtures and returns them as a pandas data frame
        """
        url = self.base_url.format(
            season=self.mapped_season, competition=self.mapped_competition
        )

        col_renames = {
            "HomeTeam": "team_home",
            "AwayTeam": "team_away",
        }

        content = self.get(url)
        df = (
            pd.read_csv(io.StringIO(content))
            .pipe(self._convert_date)
            .rename(columns=col_renames)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .assign(goals_home=lambda x: x["fthg"])
            .assign(goals_away=lambda x: x["ftag"])
            .pipe(self._map_teams, columns=["team_home", "team_away"])
            .dropna(subset=["date"])
            .pipe(create_game_id)
            .set_index(["id"])
            .sort_index()
        )

        cols = ["competition", "season", "datetime", "date"]
        i = 0
        for c in cols:
            if c in df.columns:
                move_column_inplace(df, c, i)
                i += 0

        return df
