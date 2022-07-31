import penaltyblog as pb
import pandas as pd


def test_clubelo():
    ce = pb.scrapers.ClubElo()
    df = ce.get_elo_by_date()

    assert type(df) == pd.DataFrame
