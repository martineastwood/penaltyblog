import calendar
import re
from typing import Any, Dict

import pandas as pd

COMPETITION_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "BEL First Division A": {
        "fbref": {
            "slug": "37",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "B1"},
        "espn": {"slug": "ger.2"},
    },
    "BEL First Division B": {
        "fbref": {
            "slug": "69",
            "stats": [
                "standard",
                "goalkeeping",
                "playing_time",
                "misc",
            ],
        },
        "espn": {"slug": "ger.2"},
    },
    "DEU Bundesliga 1": {
        "fbref": {
            "slug": "20",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "D1"},
        "understat": {"slug": "Bundesliga"},
        "espn": {"slug": "ger.1"},
    },
    "DEU Bundesliga 2": {
        "fbref": {
            "slug": "33",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "D2"},
        "espn": {"slug": "ger.2"},
    },
    "ENG Premier League": {
        "fbref": {
            "slug": "9",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "E0"},
        "espn": {"slug": "eng.1", "start_date": "0801"},
        "understat": {"slug": "EPL"},
    },
    "ENG Championship": {
        "fbref": {
            "slug": "10",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "E1"},
        "espn": {"slug": "eng.2"},
    },
    "ENG League 1": {
        "fbref": {
            "slug": "15",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "E2"},
        "espn": {"slug": "eng.3"},
    },
    "ENG League 2": {
        "fbref": {
            "slug": "15",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "E3"},
        "espn": {"slug": "eng.4"},
    },
    "ENG Conference": {
        "fbref": {
            "slug": "34",
            "stats": [
                "standard",
                "goalkeeping",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "EC"},
    },
    "ESP La Liga": {
        "fbref": {
            "slug": "12",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "SP1"},
        "understat": {"slug": "La_Liga"},
        "espn": {"slug": "esp.1"},
    },
    "ESP La Liga Segunda": {
        "fbref": {
            "slug": "17",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "SP2"},
        "espn": {"slug": "esp.2"},
    },
    "FRA Ligue 1": {
        "fbref": {
            "slug": "13",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "F1"},
        "understat": {"slug": "Ligue_1"},
        "espn": {"slug": "fra.1"},
    },
    "FRA Ligue 2": {
        "fbref": {
            "slug": "60",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "F2"},
        "espn": {"slug": "fra.2"},
    },
    "GRC Super League": {
        "fbref": {
            "slug": "27",
            "stats": [
                "standard",
                "goalkeeping",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "G1"},
        "espn": {"slug": "gre.1"},
    },
    "ITA Serie A": {
        "fbref": {
            "slug": "11",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "I1"},
        "understat": {"slug": "Serie_A"},
        "espn": {"slug": "ita.1"},
    },
    "ITA Serie B": {
        "fbref": {
            "slug": "18",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "I2"},
        "espn": {"slug": "ita.2"},
    },
    "NLD Eredivisie": {
        "fbref": {
            "slug": "23",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "N1"},
        "espn": {"slug": "ned.2"},
    },
    "PRT Liga 1": {
        "fbref": {
            "slug": "32",
            "stats": [
                "standard",
                "goalkeeping",
                "advanced_goalkeeping",
                "shooting",
                "passing",
                "passing_types",
                "goal_shot_creation",
                "defensive_actions",
                "possession",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "P1"},
        "espn": {"slug": "por.1"},
    },
    "RUS Premier League": {
        "fbref": {
            "slug": "30",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "understat": {"slug": "RFPL"},
        "espn": {"slug": "rus.1"},
    },
    "SCO Premier League": {
        "fbref": {
            "slug": "40",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "SC0"},
        "espn": {"slug": "sco.1"},
    },
    "SCO Division 1": {
        "fbref": {
            "slug": "72",
            "stats": [
                "standard",
                "goalkeeping",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "SC1"},
        "espn": {"slug": "sco.2"},
    },
    "SCO Division 2": {
        "footballdata": {"slug": "SC2"},
        "espn": {"slug": "sco.3"},
    },
    "SCO Division 3": {
        "footballdata": {"slug": "SC3"},
        "espn": {"slug": "sco.4"},
    },
    "TUR Super Lig": {
        "fbref": {
            "slug": "26",
            "stats": [
                "standard",
                "goalkeeping",
                "shooting",
                "playing_time",
                "misc",
            ],
        },
        "footballdata": {"slug": "T1"},
    },
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
