import penaltyblog as pb
import pandas as pd


def test_footballdata():
    fb = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
    df = fb.get_fixtures()

    assert type(df) == pd.DataFrame
