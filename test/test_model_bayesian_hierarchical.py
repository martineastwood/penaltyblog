import pandas as pd
import pytest

import penaltyblog as pb


def test_model():
    df = pd.concat(
        [
            pb.scrapers.FootballData("ENG Premier League", "2018-2019").get_fixtures(),
            pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures(),
        ]
    )
    df["weights"] = pb.models.dixon_coles_weights(df["date"])

    clf = pb.models.BayesianHierarchicalGoalModel(
        df["goals_home"],
        df["goals_away"],
        df["team_home"],
        df["team_away"],
        df["weights"],
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 0.5

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.6 < probs.total_goals("over", 1.5) < 0.8
    assert 0.3 < probs.asian_handicap("home", 1.5) < 0.5
    assert 0.4 < probs.both_teams_to_score < 0.7


def test_unfitted_raises_error():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()
    clf = pb.models.BayesianHierarchicalGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


def test_unfitted_repr():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()
    clf = pb.models.BayesianHierarchicalGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr
