import io
from datetime import datetime

import pandas as pd
from lxml import html

from .base_scrapers import RequestsScraper
from .common import sanitize_columns


class SoFifa(RequestsScraper):
    """
    Collects data from sofifa.com as pandas dataframes

    team_mappings : dict or None
        dict (or None) of team name mappings in format
        `{
            "Manchester United: ["Man Utd", "Man United],
        }`
    """

    source = "sofifa"

    def __init__(self, team_mappings=None):
        self.base_url = "https://sofifa.com"

        self.sort_by_mappings = {
            "age": "ae",
            "overall": "oa",
            "potential": "pt",
            "team": "tm",
            "value": "vl",
            "wage": "wg",
            "total": "tt",
        }

        super().__init__(team_mappings=team_mappings)

    def get_player_info(
        self, sort_by: str = None, sort_direction: str = "desc", max_pages: int = 10
    ) -> pd.DataFrame:
        """
        Gets a dataframe of top-level player info, including IDs we can use to
        scrape individual players with.

        Parameters
        ----------
        sort_by : str
            column to sort players by, must be one of
            {None, 'age', 'overall', 'potential', 'team', 'value', 'wage', 'total'}

        sort_direction: str
            direction to sort by, must be one of {None, 'asc', 'desc'}

        max_pages: int
            maximum number of pages of data to scrape.
            Each page contains up to 60 players.
        """

        base_url = "https://sofifa.com/players?offset={offset}"

        if sort_by is not None:

            if sort_by not in self.sort_by_mappings:
                raise ValueError("sort_by not recognised")

            if sort_direction not in ["asc", "desc"]:
                raise ValueError("sort_direction not recognised")

            col = "&col={col}&sort={sort}".format(
                col=self.sort_by_mappings[sort_by], sort=sort_direction
            )
        else:
            col = ""

        for i in range(max_pages):
            offset = i * 60
            url = base_url.format(offset=offset) + col
            content = self.get(url)
            tree = html.fromstring(content)
            trs = tree.cssselect("table tbody tr")

            players = list()
            for tr in trs:
                tmp = dict()
                tmp["name"] = tr.xpath(
                    "td[contains(@class, 'col-name')]/a[@role='tooltip']"
                )[0].get("aria-label")

                tmp["url"] = tr.xpath(
                    "td[contains(@class, 'col-name')]/a[@role='tooltip']"
                )[0].get("href")

                tmp["id"] = tmp["url"].split("player/")[1].split("/")[0]

                tmp["nationality"] = tr.xpath("td[contains(@class, 'col-name')]/img")[
                    0
                ].get("title")

                tmp["team"] = tr.xpath("td[contains(@class, 'col-name')]/div/a")[0].text

                tmp["age"] = int(tr.xpath("td[contains(@class, 'col-ae')]")[0].text)

                tmp["overall_rating"] = int(
                    tr.xpath("td[contains(@class, 'col-oa')]/span")[0].text
                )

                tmp["potential"] = int(
                    tr.xpath("td[contains(@class, 'col-pt')]/span")[0].text
                )

                tmp["value"] = tr.xpath("td[contains(@class, 'col-vl')]")[0].text

                tmp["wage"] = tr.xpath("td[contains(@class, 'col-wg')]")[0].text

                tmp["total"] = int(
                    tr.xpath("td[contains(@class, 'col-tt')]/span")[0].text
                )

                pos = tr.xpath(
                    "td[contains(@class, 'col-name')]/a[contains(@rel, 'nofollow')]/span"
                )
                tmp["positions"] = [x.text for x in pos]

                players.append(tmp)

        df = (
            pd.DataFrame(players)
            .pipe(self._map_teams, columns=["team"])
            .set_index("id")
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
