import penaltyblog as pb
import pandas as pd


def test_footballdata_get_fixtures():
    fb = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
    df = fb.get_fixtures()
    assert type(df) == pd.DataFrame


# def test_footballdata_list_competitions():
#     fb = pb.scrapers.FootballData.list_competitions()
#     df = fb.get_fixtures()
#     assert type(df) == pd.DataFrame
