from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager
import requests
from .common import COMPETITION_MAPPINGS


class BaseScraperSelenium:
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

    @classmethod
    def list_competitions(cls):
        competitions = list()
        for k, v in COMPETITION_MAPPINGS.items():
            if cls.source in v.keys():
                competitions.append(k)
        return competitions


class BaseScraperRequests:
    """
    Base scraper that all request-based scrapers inherit from
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        }

    def get(self, url: str):
        return requests.get(url, headers=self.headers).text

    @classmethod
    def list_competitions(cls):
        competitions = list()
        for k, v in COMPETITION_MAPPINGS.items():
            if cls.source in v.keys():
                competitions.append(k)
        return competitions
