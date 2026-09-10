from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

import penaltyblog as pb

FOOTBALL_DATA_CSV = """Date,HomeTeam,AwayTeam,FTHG,FTAG
01/08/20,Arsenal,Fulham,3,0
"""


@pytest.mark.local
def test_footballdata_wrong_league():
    with pytest.raises(ValueError):
        _ = pb.scrapers.FootballData("FRA Premier League", "2020-2021")


def test_footballdata_tries_fallback_domain():
    unavailable = Mock()
    unavailable.raise_for_status.side_effect = requests.HTTPError("503")

    available = Mock()
    available.text = FOOTBALL_DATA_CSV

    with patch(
        "penaltyblog.scrapers.base_scrapers.requests.get",
        side_effect=[unavailable, available],
    ) as mock_get:
        df = pb.scrapers.FootballData("ENG Premier League", "2020-2021").get_fixtures()

    assert len(df) == 1
    assert [call.args[0] for call in mock_get.call_args_list] == [
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://football-data.co.uk/mmz4281/2021/E0.csv",
    ]


@pytest.mark.local
def test_footballdata_get_fixtures():
    fb = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
    df = fb.get_fixtures()
    assert type(df) == pd.DataFrame


@pytest.mark.local
def test_footballdata_id():
    fb = pb.scrapers.FootballData("ENG Premier League", "2021-2022")
    df = fb.get_fixtures()
    assert "1628812800---brentford---arsenal" in df.index


@pytest.mark.local
def test_footballdata_list_competitions():
    df = pb.scrapers.FootballData.list_competitions()
    assert type(df) == list


@pytest.mark.local
def test_footballdata_team_mappings():
    team_mappings = pb.scrapers.get_example_team_name_mappings()
    fb = pb.scrapers.FootballData("ENG Premier League", "2021-2022", team_mappings)
    df = fb.get_fixtures()
    assert "Wolverhampton Wanderers" in df["team_home"].unique()


@pytest.mark.local
def test_footballdata_nat_error():
    """
    pandas was reading an extra blank row at end of csv that
    was causing a NaT error to be thrown from having a null in index
    """
    mappings = pb.scrapers.get_example_team_name_mappings()
    fb = pb.scrapers.FootballData("ENG Premier League", "2014-2015", mappings)
    df = fb.get_fixtures()
    assert df.shape[0] == 380
