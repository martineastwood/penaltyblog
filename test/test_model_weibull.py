import pytest

import penaltyblog as pb


def test_poisson_model(fixtures):
    df = fixtures

    clf = pb.models.WeibullCopulaGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert 0.65 < params["attack_Man City"] < 2.0
    assert 0.1 < params["home_advantage"] < 0.4

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.3 < probs.total_goals("over", 1.5) < 0.9
    assert 0.2 < probs.asian_handicap("home", 1.5) < 0.5


def test_unfitted_raises_error(fixtures):
    df = fixtures
    clf = pb.models.WeibullCopulaGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


def test_unfitted_repr(fixtures):
    df = fixtures
    clf = pb.models.WeibullCopulaGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr
