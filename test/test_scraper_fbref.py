import pandas as pd
import pytest

import penaltyblog as pb


def test_fbref_wrong_league():
    with pytest.raises(ValueError):
        _ = pb.scrapers.FBRef("FRA Premier League", "2020-2021")


def test_fbref_get_fixtures():
    fb = pb.scrapers.FBRef("ENG Premier League", "2021-2022")
    df = fb.get_fixtures()
    assert type(df) == pd.DataFrame
    assert "1628812800---brentford---arsenal" in df.index


def test_fbref_list_competitions():
    df = pb.scrapers.FBRef.list_competitions()
    assert type(df) == list


def test_fbref_team_mappings():
    team_mappings = pb.scrapers.get_example_team_name_mappings()
    fb = pb.scrapers.FBRef("ENG Premier League", "2021-2022", team_mappings)
    df = fb.get_fixtures()
    assert "Wolverhampton Wanderers" in df["team_home"].unique()


def test_fbref_wrong_stat_type():
    with pytest.raises(ValueError):
        fb = pb.scrapers.FBRef("ENG Premier League", "2021-2022")
        fb.get_stats("wrong_stat_type")
