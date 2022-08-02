from typing import Iterable

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager

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


class SeleniumScraper(BaseScraper):
    """
    Base Scraper class that all selenium-based scrapers inherit from
    """

    def __init__(self):
        self.options = FirefoxOptions()
        self.options.add_argument("--headless")
        self.options.add_argument("--blink-settings=imagesEnabled=false")
        self.options.set_preference("dom.max_script_run_time", 15)

        self.driver = webdriver.Firefox(
            executable_path=GeckoDriverManager().install(), options=self.options
        )
        self.driver.delete_all_cookies()

        super().__init__()

    def close_browser(self):
        """
        Quit the browsers and frees its resources
        """
        self.driver.quit()

    def get(self, url: str):
        """
        Loads in the url into selenium

        Parameters
        ----------
        url : str
            The URL of interest
        """
        self.driver.get(url)


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

        super().__init__(team_mappings=team_mappings)

    def get(self, url: str):
        return requests.get(url, headers=self.headers).text
