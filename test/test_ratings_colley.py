import pandas as pd

import penaltyblog as pb


def test_colley(fixtures):
    df = fixtures
    massey = pb.ratings.Colley(df["fthg"], df["ftag"], df["team_home"], df["team_away"])
    ratings = massey.get_ratings()
    assert type(ratings) == pd.DataFrame
