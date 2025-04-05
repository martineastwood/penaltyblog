import pytest

from penaltyblog.ratings.elo import Elo


def test_initial_team_rating():
    elo = Elo()
    assert elo.get_team_rating("Team A") == 1500.0
    assert elo.get_team_rating("Team B") == 1500.0


def test_home_win_probability():
    elo = Elo()
    prob = elo.home_win_probability("Team A", "Team B")
    assert 0.0 <= prob <= 1.0


def test_calculate_match_probabilities():
    elo = Elo()
    probs = elo.calculate_match_probabilities("Team A", "Team B")
    assert 0.0 <= probs["home_win"] <= 1.0
    assert 0.0 <= probs["draw"] <= 1.0
    assert 0.0 <= probs["away_win"] <= 1.0
    assert (
        pytest.approx(probs["home_win"] + probs["draw"] + probs["away_win"], 0.01)
        == 1.0
    )


def test_update_ratings_home_win():
    elo = Elo()
    elo.update_ratings("Team A", "Team B", 0)
    assert elo.get_team_rating("Team A") > 1500.0
    assert elo.get_team_rating("Team B") < 1500.0


def test_update_ratings_draw():
    elo = Elo()
    initial_rating_home = elo.get_team_rating("Team A")
    initial_rating_away = elo.get_team_rating("Team B")

    elo.update_ratings("Team A", "Team B", 1)

    # Calculate expected ratings after a draw
    expected_home = elo.home_win_probability("Team A", "Team B")
    expected_away = 1 - expected_home

    expected_rating_home = initial_rating_home + elo.k * (0.5 - expected_home)
    expected_rating_away = initial_rating_away + elo.k * (0.5 - expected_away)

    assert pytest.approx(elo.get_team_rating("Team A"), 0.01) == expected_rating_home
    assert pytest.approx(elo.get_team_rating("Team B"), 0.01) == expected_rating_away


def test_update_ratings_away_win():
    elo = Elo()
    elo.update_ratings("Team A", "Team B", 2)
    assert elo.get_team_rating("Team A") < 1500.0
    assert elo.get_team_rating("Team B") > 1500.0


def test_invalid_result():
    elo = Elo()
    with pytest.raises(ValueError, match="Invalid result: must be 0, 1, or 2"):
        elo.update_ratings("Team A", "Team B", 3)


def test_custom_k_factor():
    elo = Elo(k=40.0)
    elo.update_ratings("Team A", "Team B", 0)
    assert elo.get_team_rating("Team A") > 1500.0


def test_custom_home_field_advantage():
    elo = Elo(home_field_advantage=200.0)
    prob = elo.home_win_probability("Team A", "Team B")
    assert prob > 0.5
