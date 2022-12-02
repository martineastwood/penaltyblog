import pandas as pd
import pytest

import penaltyblog as pb


def test_footballdata_wrong_league():
    with pytest.raises(ValueError):
        _ = pb.scrapers.FootballData("FRA Premier League", "2020-2021")


def test_footballdata_get_fixtures():
    fb = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
    df = fb.get_fixtures()
    assert type(df) == pd.DataFrame


def test_footballdata_id():
    fb = pb.scrapers.FootballData("ENG Premier League", "2021-2022")
    df = fb.get_fixtures()
    assert "1628812800---brentford---arsenal" in df.index


def test_footballdata_list_competitions():
    df = pb.scrapers.FootballData.list_competitions()
    assert type(df) == list


def test_footballdata_team_mappings():
    team_mappings = pb.scrapers.get_example_team_name_mappings()
    fb = pb.scrapers.FootballData("ENG Premier League", "2021-2022", team_mappings)
    df = fb.get_fixtures()
    assert "Wolverhampton Wanderers" in df["team_home"].unique()


def test_footballdata_nat_error():
    """
    pandas was reading an extra blank row at end of csv that
    was causing a NaT error to be thrown from having a null in index
    """
    mappings = pb.scrapers.get_example_team_name_mappings()
    fb = pb.scrapers.FootballData("ENG Premier League", "2014-2015", mappings)
    df = fb.get_fixtures()
    assert df.shape[0] == 380
