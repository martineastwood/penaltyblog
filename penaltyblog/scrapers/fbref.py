# from typing import Optional

# import pandas as pd
# from lxml import etree, html

# from .common import COMPETITION_MAPPINGS, BaseScraper


# class FbRef(BaseScraper):
#     """
#     Scrapes data from fbref.com as pandas dataframes

#     Parameters
#     ----------
#     league : str
#         Name of the league of interest (optional)

#     season : str
#         Name of the season of interest (optional)
#     """

#     def __init__(self, competition, season):
#         self.base_url = "https://fbref.com/en"
#         self.competition = competition
#         self.season = season

#         super().__init__()

#         self._get_season_link()

#     def _get_season_link(self):
#         url = COMPETITION_MAPPINGS[self.competition]["fbref"]
#         self.get(url)

#         tree = html.fromstring(self.driver.page_source)
#         tbl = tree.xpath("//table[@id='seasons']")[0]
#         tmp = pd.read_html(etree.tostring(tbl, method="html"))[0]

#         print(tmp)

#         # tmp = tmp.query("Season == @self.season")
#         # self.season_link = tmp[""]

#     def competitions(self):
#         """
#         Returns names of all competitions available
#         """
#         url = self.base_url + "/comps"

#         print("Getting", url)
#         self.get(url)

#         tree = html.fromstring(self.driver.page_source)

#         df = list()
#         for tbl in tree.xpath("//table[contains(@id, 'comps')]"):
#             tmp = pd.read_html(etree.tostring(tbl, method="html"))[0]
#             df.append(tmp)
#         df = pd.concat(df, axis=0)

#         print(df)

#         # import pdb

#         # pdb.set_trace()

#         self.close_browser()

#         # //*[@id="comps_intl_club_cup"]/tbody/tr[1]/th
