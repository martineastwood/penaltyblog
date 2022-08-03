import pandas as pd
import pytest

import penaltyblog as pb


def test_footballdata_wrong_league():
    with pytest.raises(ValueError):
        _ = pb.scrapers.ESPN("FRA Premier League", "2021-2022")


def test_espn_get_fixtures():
    espn = pb.scrapers.ESPN("ENG Premier League", "2021-2022")
    df = espn.get_fixtures()
    assert type(df) == pd.DataFrame
    assert "1628812800---brentford---arsenal" in df.index


def test_espn_list_competitions():
    df = pb.scrapers.ESPN.list_competitions()
    assert type(df) == list


def test_espn_get_player_stats():
    espn = pb.scrapers.ESPN("ENG Premier League", "2021-2022")
    df = espn.get_player_stats("606029")
    assert type(df) == pd.DataFrame
    assert "1628812800---brentford---arsenal" in df.index.unique()


def test_espn_get_team_stats():
    espn = pb.scrapers.ESPN("ENG Premier League", "2021-2022")
    df = espn.get_team_stats("606029")
    assert type(df) == pd.DataFrame
    assert "1628812800---brentford---arsenal" in df.index.unique()
