import pandas as pd

import penaltyblog as pb


def test_sofifa_get_player_info():
    sofifa = pb.scrapers.SoFifa()
    df = sofifa.get_players(max_pages=1, sort_by="potential")
    assert type(df) == pd.DataFrame
    assert df.shape[0] > 0


def test_sofifa_get_player():
    sofifa = pb.scrapers.SoFifa()
    df = sofifa.get_player("237692")
    assert df["name"].iloc[0] == "Philip Foden"
    assert type(df) == pd.DataFrame
    assert df.shape[0] > 0
