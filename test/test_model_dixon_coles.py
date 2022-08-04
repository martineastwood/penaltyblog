import penaltyblog as pb


def test_dc_model():
    fb = pb.scrapers.FootballData("ENG Premier League", "2019-2020")
    df = fb.get_fixtures()

    clf = pb.models.DixonColesGoalModel(
        df["home_goals"], df["away_goals"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert 0.2 < params["home_advantage"] < 0.3

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.6 < probs.total_goals("over", 1.5) < 0.8
    assert 0.3 < probs.asian_handicap("home", 1.5) < 0.4
    assert 0.4 < probs.both_teams_to_score < 0.7
