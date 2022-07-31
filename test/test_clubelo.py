import penaltyblog as pb
import pandas as pd


def test_clubelo_by_date():
    ce = pb.scrapers.ClubElo()
    df = ce.get_elo_by_date()
    assert type(df) == pd.DataFrame


def test_clubelo_by_team():
    ce = pb.scrapers.ClubElo()
    df = ce.get_elo_by_team("Barcelona")
    assert type(df) == pd.DataFrame


def test_clubelo_team_names():
    ce = pb.scrapers.ClubElo()
    df = ce.get_team_names()
    assert type(df) == pd.DataFrame
