from typing import Iterable

import pandas as pd
import requests

from .common import COMPETITION_MAPPINGS


class BaseScraper:
    """
    Base scraper that all scrapers inherit from

    Parameters
    ----------
    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    src: str = ""

    def __init__(self, team_mappings=None):
        if team_mappings is not None:
            self.team_mappings = dict()
            for team, options in team_mappings.items():
                for option in options:
                    self.team_mappings[option] = team
        else:
            self.team_mappings = None

    def _check_competition(self, competition):
        available = self.list_competitions()
        if competition not in available:
            raise ValueError(
                "{} not available for this data source".format(competition)
            )

    @classmethod
    def list_competitions(cls) -> list:
        if not hasattr(cls, "source"):
            raise AttributeError(f"{cls.__name__} has no attribute 'source'")
        competitions = list()
        for k, v in COMPETITION_MAPPINGS.items():
            if cls.source in v.keys():
                competitions.append(k)
        return competitions

    def _map_teams(self, df: pd.DataFrame, columns: Iterable) -> pd.DataFrame:
        """
        Internal function to apply team mappings if they've been provided

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of scraped data

        columns : Iterable
            iterable of columns to map
        """
        if self.team_mappings is not None:
            for c in columns:
                df[c] = df[c].replace(self.team_mappings)
        return df


class RequestsScraper(BaseScraper):
    """
    Base scraper that all request-based scrapers inherit from
    """

    def __init__(self, team_mappings=None):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/102.0.0.0 Safari/537.36"
            )
        }

        self.cookies = None

        super().__init__(team_mappings=team_mappings)

    def get(self, url: str) -> str:
        if self.cookies is not None:
            return requests.get(url, headers=self.headers, cookies=self.cookies).text
        else:
            return requests.get(url, headers=self.headers).text
