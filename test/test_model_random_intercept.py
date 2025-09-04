import pandas as pd
import pytest

import penaltyblog as pb


@pytest.fixture
def small_fixture():
    """A small, self-contained fixture for testing."""
    data = {
        "team_home": ["Team A", "Team C", "Team A", "Team D", "Team B", "Team C"],
        "team_away": ["Team B", "Team D", "Team C", "Team A", "Team D", "Team B"],
        "goals_home": [1, 0, 2, 1, 3, 0],
        "goals_away": [1, 0, 2, 1, 1, 0],
    }
    return pd.DataFrame(data)


def test_model_initialization(small_fixture):
    """Test that the BayesianRandomInterceptModel initializes correctly."""
    df = small_fixture
    model = pb.models.BayesianRandomInterceptModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    assert model.n_teams == 4
    assert model.n_matches == 6
    assert not model.fitted


def test_predict_unfitted_raises_error(small_fixture):
    """Test that calling predict on an unfitted model raises a ValueError."""
    df = small_fixture
    model = pb.models.BayesianRandomInterceptModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.predict("Team A", "Team B")


@pytest.mark.local
def test_model_fit_and_predict(small_fixture):
    """
    Test that the model fits without errors and can make predictions.
    This is a smoke test to ensure the MCMC process runs, not a test of
    statistical correctness.
    """
    df = small_fixture
    model = pb.models.BayesianRandomInterceptModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Calculate ndim to set n_walkers
    # ndim = 2 * n_teams + n_matches + 7
    ndim = 2 * 4 + 6 + 7  # 8 + 6 + 7 = 21
    n_walkers = 2 * ndim  # 42

    # Use MCMC parameters that will produce a chain
    model.fit(n_walkers=n_walkers, n_steps=30, n_burn=10)

    assert model.fitted
    assert model.chain is not None
    assert len(model.chain) > 0

    # Test prediction
    probs = model.predict("Team A", "Team B")
    assert isinstance(probs, pb.models.FootballProbabilityGrid)

    # Check that probabilities are sensible
    assert len(probs.home_draw_away) == 3
    assert 0 < probs.home_draw_away[0] < 1
    assert 0 < probs.home_draw_away[1] < 1
    assert 0 < probs.home_draw_away[2] < 1
    assert pytest.approx(sum(probs.home_draw_away), 1e-6) == 1.0
