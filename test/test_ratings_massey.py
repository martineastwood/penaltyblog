import pandas as pd

import penaltyblog as pb


def test_massey():
    fb = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
    df = fb.get_fixtures()

    massey = pb.ratings.Massey(df["fthg"], df["ftag"], df["team_home"], df["team_away"])
    ratings = massey.get_ratings()
    assert type(ratings) == pd.DataFrame
