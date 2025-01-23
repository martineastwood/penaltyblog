import pytest
from scipy.stats import norm

import penaltyblog as pb


@pytest.fixture
def pi_rating():
    """Fixture to initialize the PiRatingSystem instance for tests."""
    return pb.ratings.PiRatingSystem()


def test_initialize_team(pi_rating):
    """Test if team ratings are correctly initialized."""
    pi_rating.initialize_team("Team A")
    assert "Team A" in pi_rating.team_ratings
    assert pi_rating.team_ratings["Team A"]["home"] == 0.0
    assert pi_rating.team_ratings["Team A"]["away"] == 0.0


def test_expected_goal_difference(pi_rating):
    """Test expected goal difference calculation."""
    pi_rating.team_ratings["Team A"] = {"home": 1.2, "away": 0.8}
    pi_rating.team_ratings["Team B"] = {"home": 1.0, "away": 0.5}
    expected_diff = pi_rating.expected_goal_difference("Team A", "Team B")
    assert expected_diff == pytest.approx(1.2 - 0.5, rel=1e-2)


def test_diminishing_error(pi_rating):
    """Test diminishing error function."""
    assert pi_rating.diminishing_error(2) == pytest.approx(2 / (1 + 0.75 * 2), rel=1e-2)
    assert pi_rating.diminishing_error(-3) == pytest.approx(
        -3 / (1 + 0.75 * 3), rel=1e-2
    )


def test_update_ratings(pi_rating):
    """Test if team ratings update correctly after a match."""
    pi_rating.update_ratings("Team A", "Team B", 2)
    assert pi_rating.team_ratings["Team A"]["home"] > 0
    assert pi_rating.team_ratings["Team B"]["away"] < 0


def test_calculate_probabilities(pi_rating):
    """Test probability calculations for a match."""
    pi_rating.team_ratings["Team A"] = {"home": 1.0, "away": 0.8}
    pi_rating.team_ratings["Team B"] = {"home": 0.5, "away": 0.3}

    probabilities = pi_rating.calculate_match_probabilities("Team A", "Team B")

    # Expected goal difference
    expected_diff = 1.0 - 0.3

    # Using normal distribution to validate the results
    draw_threshold = 0.5
    expected_draw = norm.cdf(draw_threshold, loc=expected_diff, scale=1.0) - norm.cdf(
        -draw_threshold, loc=expected_diff, scale=1.0
    )
    expected_home_win = 1 - norm.cdf(draw_threshold, loc=expected_diff, scale=1.0)
    expected_away_win = norm.cdf(-draw_threshold, loc=expected_diff, scale=1.0)

    assert probabilities["home_win"] == pytest.approx(expected_home_win, rel=1e-2)
    assert probabilities["draw"] == pytest.approx(expected_draw, rel=1e-2)
    assert probabilities["away_win"] == pytest.approx(expected_away_win, rel=1e-2)


def test_get_team_rating(pi_rating):
    """Test retrieving the average team rating."""
    pi_rating.team_ratings["Team A"] = {"home": 1.5, "away": 0.5}
    avg_rating = pi_rating.get_team_rating("Team A")
    assert avg_rating == pytest.approx((1.5 + 0.5) / 2, rel=1e-2)


def test_display_ratings(capsys, pi_rating):
    """Test that the display function prints correct output."""
    pi_rating.team_ratings["Team A"] = {"home": 1.0, "away": 0.5}
    pi_rating.display_ratings()
    captured = capsys.readouterr()
    assert "Team A: Home = 1.00, Away = 0.50, Average = 0.75" in captured.out
