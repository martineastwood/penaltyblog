import io

import pandas as pd

from .base_scrapers import RequestsScraper
from .common import (
    COMPETITION_MAPPINGS,
    create_game_id,
    move_column_inplace,
    sanitize_columns,
)


class FBRef(RequestsScraper):
    """
    Scrapes data from FBRef and returns as a pandas dataframes

    Parameters
    ----------
    league : str
        Name of the league of interest,
        see the `FBRef.list_competitions()` output for available choices

    season : str
        Name of the season of interest in format 2020-2021

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    source = "fbref"

    def __init__(self, competition, season, team_mappings=None):
        self._check_competition(competition)

        self.base_url = "https://fbref.com/en/comps/"
        self.competition = competition
        self.season = season
        self.mapped_season = self._map_season(self.season)
        self.mapped_competition = COMPETITION_MAPPINGS[self.competition]["fbref"][
            "slug"
        ]

        super().__init__(team_mappings=team_mappings)

    def _map_season(self, season) -> str:
        """
        Internal function to map the season name

        Parameters
        ----------
        season : str
            Name of the season
        """
        return season

    def _convert_date(self, df):
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _rename_fixture_columns(self, df) -> pd.DataFrame:
        """
        Internal function to rename columns to make consistent with other data sources
        """
        cols = {
            "Wk": "week",
            "Home": "team_home",
            "Away": "team_away",
            "xG": "xg_home",
            "xG.1": "xg_away",
        }
        df = df.rename(columns=cols)
        return df

    def _drop_fixture_spacer_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to drop the spacer rows from the fixtures df
        """
        return df[~df["week"].isna()]

    def _drop_unplayed_fixtures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to drop the spacer rows from the fixtures df
        """
        return df[~df["xg_home"].isna()]

    def _split_score(self, df) -> pd.DataFrame:
        """
        Internal function to split the score column into goals_home and goals_away
        """
        df["goals_home"] = df["score"].str.split("–", expand=True)[0]
        df["goals_away"] = df["score"].str.split("–", expand=True)[1]

        df["goals_home"] = df["goals_home"].astype(float)
        df["goals_away"] = df["goals_away"].astype(float)

        return df.drop("score", axis=1)

    def _flatten_stats_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to flatten multi-level column names
        """
        columns = list()
        for col in df.columns:
            p1 = "" if "Unnamed:" in col[0] else col[0].replace(" ", "_").lower() + "_"
            p2 = col[1].replace(" ", "_").replace("# ", "").lower()
            columns.append(p1 + p2)
        df.columns = pd.Index(columns)
        return df

    def _set_stat_col_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to set the data types for the stat columns
        """
        for col in df.columns:
            if "90" in col:
                df[col] = df[col].astype(float)
            elif "playing_time" in col:
                df[col] = df[col].astype(float)
            elif "expected" in col:
                df[col] = df[col].astype(float)
        return df

    def _player_ages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to format the player ages
        """
        if "born" in df.columns:
            df["born"] = df["born"].astype("Int64")

        if "age" in df.columns:
            try:
                df["age_years"] = df["age"].str.split("-", expand=True)[0].astype(float)
                df["age_days"] = df["age"].str.split("-", expand=True)[1].astype(float)
            except Exception:
                df["age_years"] = df["age"].astype(float)

            df = df.drop("age", axis=1)

        return df

    def get_fixtures(self) -> pd.DataFrame:
        """
        Gets the fixtures / results for the selected competition / season
        """

        url = (
            self.base_url
            + self.mapped_competition
            + "/"
            + self.mapped_season
            + "/schedule/"
        )

        content = self.get(url)

        df = (
            pd.read_html(io.StringIO(content))[0]
            .drop(["Match Report", "Notes"], axis=1)
            .pipe(self._rename_fixture_columns)
            .pipe(sanitize_columns)
            .pipe(self._drop_fixture_spacer_rows)
            .pipe(self._drop_unplayed_fixtures)
            .pipe(self._split_score)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._convert_date)
            .pipe(self._map_teams, columns=["team_home", "team_away"])
            .pipe(create_game_id)
            .set_index("id")
            .sort_index()
        )

        move_column_inplace(df, "competition", 0)
        move_column_inplace(df, "season", 1)
        move_column_inplace(df, "datetime", 2)

        return df

    def list_stat_types(self) -> list:
        return COMPETITION_MAPPINGS[self.competition]["fbref"]["stats"]

    def get_stats(self, stat_type: str = "standard") -> dict:
        """
        Gets squad / player stats for the selected stat type

        Parameters
        ----------
        stat_type : str
            see self.list_stat_types for the available stats


        Returns
        ----------
        Returns a dict of dataframes, with a keys for
        `squad_for`, `squad_against` and `players` stats
        """
        if stat_type not in self.list_stat_types():
            raise ValueError("Stat type not available for this competition")

        if stat_type == "standard":
            page = "stats"
        elif stat_type == "goalkeeping":
            page = "keepers"
        elif stat_type == "advanced_goalkeeping":
            page = "keepersadv"
        elif stat_type == "goal_shot_creation":
            page = "gca"
        elif stat_type == "defensive_actions":
            page = "defense"
        elif stat_type == "playing_time":
            page = "playingtime"
        else:
            page = stat_type

        url = (
            self.base_url
            + self.mapped_competition
            + "/"
            + self.mapped_season
            + "/"
            + page
            + "/"
        )

        content = self.get(url)
        content = content.replace("<!--", "").replace("-->", "")

        output = dict()

        dfs = pd.read_html(io.StringIO(content))

        output["squad_for"] = (
            dfs[0]
            .pipe(self._flatten_stats_col_names)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._set_stat_col_types)
            .set_index("squad")
            .sort_index()
        )

        output["squad_against"] = (
            dfs[1]
            .pipe(self._flatten_stats_col_names)
            .pipe(sanitize_columns)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._set_stat_col_types)
            .set_index("squad")
            .sort_index()
        )

        output["players"] = (
            dfs[2]
            .pipe(self._flatten_stats_col_names)
            .pipe(sanitize_columns)
            .query("rk != 'Rk'")
            .drop(["rk", "matches"], axis=1)
            .assign(season=self.season)
            .assign(competition=self.competition)
            .pipe(self._set_stat_col_types)
            .pipe(self._player_ages)
            .set_index("player")
            .sort_index()
        )

        return output
