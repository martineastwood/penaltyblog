import calendar
import re

import pandas as pd

COMPETITION_MAPPINGS = {
    "DEU Bundesliga 1": {"fbref": "", "footballdata": {"slug": "D1"}},
    "DEU Bundesliga 2": {"fbref": "", "footballdata": {"slug": "D2"}},
    "ENG Premier League": {
        "fbref": "https://fbref.com/en/comps/9/history/Premier-League-Seasons",
        "footballdata": {"slug": "E0"},
        "espn": {"slug": "eng.1", "start_date": "0801"},
    },
    "ENG Championship": {
        "fbref": "",
        "footballdata": {"slug": "E1"},
        "espn": {"slug": "eng.2"},
    },
    "ENG League 1": {"fbref": "", "footballdata": {"slug": "E2"}},
    "ENG League 2": {"fbref": "", "footballdata": {"slug": "E3"}},
    "ENG Conference": {"fbref": "", "footballdata": {"slug": "EC"}},
    "FRA Ligue 1": {"fbref": "", "footballdata": {"slug": "F1"}},
    "FRA Ligue 2": {"fbref": "", "footballdata": {"slug": "F2"}},
    "GRC Super League": {"fbref": "", "footballdata": {"slug": "G1"}},
    "ITA Serie A": {"fbref": "", "footballdata": {"slug": "I1"}},
    "ITA Serie B": {"fbref": "", "footballdata": {"slug": "I2"}},
    "NLD Eredivisie": {"fbref": "", "footballdata": {"slug": "N1"}},
    "PRT Liga 1": {"fbref": "", "footballdata": {"slug": "P1"}},
    "SCO Premier League": {"fbref": "", "footballdata": {"slug": "SC0"}},
    "SCO Division 1": {"fbref": "", "footballdata": {"slug": "SC1"}},
    "SCO Division 2": {"fbref": "", "footballdata": {"slug": "SC2"}},
    "SCO Division 3": {"fbref": "", "footballdata": {"slug": "SC3"}},
    "TUR Super Lig": {"fbref": "", "footballdata": {"slug": "T1"}},
}


def move_column_inplace(df, col, pos):
    """
    Reorder specific columns, taken from
    https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
    """
    col = df.pop(col)
    df = df.insert(pos, col.name, col)
    return df


def sanitize_columns(df, rename_mappings=None):
    """
    Make the columns names consistent, e.g.
    lowercase, snakecase, consistent names for team columns
    """
    if rename_mappings:
        df = df.rename(columns=rename_mappings)

    df.columns = [to_snake_case(x) for x in df.columns]
    return df


def to_snake_case(name: str):
    """
    Taken from
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("__([A-Z])", r"_\1", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def create_game_id(df: pd.DataFrame):
    """
    Creates a unique id for each fixture based on datetime and team names
    """

    def _create_game_id(row: pd.Series):
        tmpl = "{}---{}---{}".format(
            int(calendar.timegm(row["date"].timetuple())),
            row["team_home"],
            row["team_away"],
        )
        tmpl = tmpl.replace(" ", "_").lower()
        return tmpl

    df["id"] = df.apply(_create_game_id, axis=1)
    return df
