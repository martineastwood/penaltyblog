import re

import pandas as pd
from lxml import html

from .base_scrapers import RequestsScraper


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

    def get_players(
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
                raise ValueError("sort_by not recognised - {}".format(sort_by))

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
            parser = html.HTMLParser(encoding="utf-8")
            tree = html.document_fromstring(content, parser=parser)
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

    def get_player(self, sofifa_id) -> pd.DataFrame:  # noqa: C901
        """
        Get a player's detailed stats

        Parameters
        ----------
        sofifa_id : str
            player_id of interest
        """
        url = "https://sofifa.com/player/{id}".format(id=sofifa_id)
        content = self.get(url)
        parser = html.HTMLParser(encoding="utf-8")
        tree = html.document_fromstring(content, parser=parser)

        tmp = dict()
        tmp["name"] = tree.xpath("//div[contains(@class, 'info')]/h1")[0].text

        info = tree.xpath("//div[contains(@class, 'info')]/div")[0].text_content()
        tmp["weight"] = int(re.search("[0-9]*kg", info).group(0).split("kg")[0])
        tmp["height"] = int(re.search("[0-9]*cm", info).group(0).split("cm")[0])
        tmp["age"] = int(re.search(r"[0-9]*y\.o", info).group(0).split("y.o")[0])
        tmp["position"] = [
            x.text for x in tree.xpath("//div[contains(@class, 'info')]/div/span")
        ]

        stats = tree.xpath(
            "//div[contains(@class, 'player')]//div[contains(@class, 'block-quarter')]"
        )
        for stat in stats[:2]:
            v = int(stat.xpath("div/span")[0].text)
            k = stat.xpath("div/div")[0].text
            k = k.lower().replace(" ", "_")
            tmp[k] = v

        for stat in stats[2:]:
            v = stat.xpath("div")[0].text
            k = stat.xpath("div/div/text()")[0]
            k = k.lower().replace(" ", "_")
            tmp[k] = v

        lis = tree.xpath(
            "//div[contains(@class, 'block-quarter')]//ul[contains(@class, 'pl')]/li"
        )
        for li in lis:
            label = li.find("label")
            if label is None:
                continue
            k = li.xpath("label/text()")[0]
            k = k.lower().replace(" ", "_")

            span = li.find("span")
            if span is not None:
                span = li.xpath("span/text()")[0]
            else:
                span = li.xpath("text()")[0].strip()
            tmp[k] = span

        cards = tree.xpath(
            "//div[contains(@class, 'block-quarter')]/div[contains(@class, 'card')]"
        )

        for card in cards:
            title = card.xpath("h5")[0].text_content() + "_"

            if card.xpath("h5/a"):
                title = ""

            elif title == "Player Specialities":
                tmp["player_specialities"] = card.xpath("ul/li/span/text()")
                continue

            elif title == "Traits_":
                tmp["traits"] = card.xpath("ul/li/span/text()")
                continue

            for li in card.xpath("ul/li"):
                k = title
                if li.xpath("label"):
                    k += li.xpath("label/text()")[0]
                elif li.xpath("span[@role = 'tooltip']"):
                    k += li.xpath("span[@role = 'tooltip']/text()")[0]
                else:
                    continue

                if li.xpath("span"):
                    v = li.xpath("span/text()")[0].strip()
                if li.xpath("span"):
                    v = li.xpath("span/text()")[0].strip()
                else:
                    v = li.xpath("text()")[0].strip()

                k = k.lower().replace(" ", "_")

                tmp[k] = v

                for k, v in tmp.items():
                    try:
                        tmp[k] = int(v)
                    except (ValueError, TypeError):
                        pass

        return pd.DataFrame([tmp])
