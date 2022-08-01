import pandas as pd

import penaltyblog as pb


def test_espn_get_fixtures():
    espn = pb.scrapers.ESPN("ENG Premier League", "2021-2022")
    df = espn.get_fixtures()
    assert type(df) == pd.DataFrame
    assert "1628812800---brentford---arsenal" in [x[2] for x in df.index.values]


def test_espn_list_competitions():
    df = pb.scrapers.ESPN.list_competitions()
    assert type(df) == list
