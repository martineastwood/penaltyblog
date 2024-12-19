import pytest

import penaltyblog as pb


def test_model():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()

    clf = pb.models.BayesianSkellamGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.0 < probs.total_goals("over", 1.5) < 1.0
    assert 0.0 < probs.asian_handicap("home", 1.5) < 1.0
    assert 0.0 < probs.both_teams_to_score < 1.0
    assert clf._get_team_index("Arsenal") == 1

    params = clf.get_params()
    assert 0.1 < params["home_advantage"] < 0.3


def test_unfitted_raises_error():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()
    clf = pb.models.BayesianBivariateGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


def test_unfitted_repr():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()
    clf = pb.models.BayesianSkellamGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr
