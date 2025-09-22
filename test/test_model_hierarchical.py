import numpy as np
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_hierarchical_model(fixtures):
    """Test basic hierarchical Bayesian Dixon-Coles model fitting and prediction."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Fit with MCMC parameters suitable for testing
    # Hierarchical model has more parameters: 2*n_teams + 6
    n_params = (
        2 * clf.n_teams + 6
    )  # attack_offsets + defense_offsets + 6 hyperparameters
    n_walkers = max(50, 2 * n_params)
    clf.fit(n_walkers=n_walkers, n_steps=100, n_burn=50)

    assert clf.fitted
    assert clf.chain is not None
    assert clf.sampler is not None

    # Check that we have posterior samples
    assert clf.chain.shape[0] > 0  # Should have samples after burn-in and thinning
    assert clf.chain.shape[1] == n_params  # Should match expected parameter count

    # Test prediction
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3

    # Test that probabilities sum to approximately 1
    total_prob = sum(probs.home_draw_away)
    assert 0.99 < total_prob < 1.01


@pytest.mark.local
def test_hierarchical_unfitted_raises_error(fixtures):
    """Test that unfitted hierarchical model raises appropriate errors."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Should raise error when trying to predict without fitting
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        clf.predict("Liverpool", "Wolves")


@pytest.mark.local
def test_hierarchical_custom_mcmc_params(fixtures):
    """Test hierarchical model with custom MCMC parameters."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test with custom parameters
    n_params = 2 * clf.n_teams + 6
    n_walkers = max(50, 2 * n_params)
    clf.fit(n_walkers=n_walkers, n_steps=150, n_burn=75)

    assert clf.fitted
    assert clf.chain.shape[0] > 0

    # Test prediction works
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_hierarchical_predict_match(fixtures):
    """Test match prediction functionality for hierarchical model."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    n_params = 2 * clf.n_teams + 6
    n_walkers = max(50, 2 * n_params)
    clf.fit(n_walkers=n_walkers, n_steps=100, n_burn=50)

    # Test prediction for a specific match
    probs = clf.predict("Liverpool", "Wolves")

    # Check that we get reasonable probability values
    assert all(0 <= p <= 1 for p in probs.home_draw_away)

    # Test that probabilities are properly normalized
    total_prob = sum(probs.home_draw_away)
    assert abs(total_prob - 1.0) < 0.01

    # Test some specific probability methods
    assert 0 <= probs.total_goals("over", 1.5) <= 1
    assert 0 <= probs.btts_yes <= 1


@pytest.mark.local
def test_hierarchical_parameter_structure(fixtures):
    """Test that hierarchical model parameters have correct structure."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Check parameter names
    param_names = clf._get_param_names()
    expected_length = (
        2 * clf.n_teams + 6
    )  # attack_offsets + defense_offsets + 6 hyperparameters
    assert len(param_names) == expected_length

    # Check that attack and defense offset parameters are present for all teams
    for team in clf.teams:
        assert f"attack_offset_{team}" in param_names
        assert f"defense_offset_{team}" in param_names

    # Check hyperparameter names
    assert "mu_attack" in param_names
    assert "log_sigma_attack" in param_names
    assert "mu_defense" in param_names
    assert "log_sigma_defense" in param_names
    assert "home_advantage" in param_names
    assert "rho" in param_names

    # Test parameter unpacking
    test_params = np.random.randn(expected_length)
    unpacked = clf._unpack_params(test_params)

    assert "attack" in unpacked
    assert "defense" in unpacked
    assert "hfa" in unpacked
    assert "rho" in unpacked

    assert len(unpacked["attack"]) == clf.n_teams
    assert len(unpacked["defense"]) == clf.n_teams
    assert isinstance(unpacked["hfa"], (int, float))
    assert isinstance(unpacked["rho"], (int, float))


@pytest.mark.local
def test_hierarchical_unfitted_repr(fixtures):
    """Test string representation of unfitted hierarchical model."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr_str = str(clf)
    assert "Bayesian Hierarchical Dixon-Coles" in repr_str


@pytest.mark.local
def test_hierarchical_with_weights(fixtures):
    """Test hierarchical model with match weights."""
    df = fixtures

    # Create some weights (e.g., recency weighting)
    n_matches = len(df)
    weights = np.exp(
        -0.01 * np.arange(n_matches)[::-1]
    )  # Recent matches weighted higher

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"],
        df["goals_away"],
        df["team_home"],
        df["team_away"],
        weights=weights,
    )

    n_params = 2 * clf.n_teams + 6
    n_walkers = max(50, 2 * n_params)
    clf.fit(n_walkers=n_walkers, n_steps=100, n_burn=50)

    assert clf.fitted
    assert clf.chain is not None

    # Test that predictions still work with weights
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_hierarchical_initial_params(fixtures):
    """Test hierarchical model with custom initial parameters."""
    df = fixtures

    clf = pb.models.BayesianHierarchicalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Create custom initial parameters
    n_params = 2 * clf.n_teams + 6
    custom_init = np.random.normal(0, 0.1, n_params)

    # Fit with custom initialization
    n_walkers = max(50, 2 * n_params)
    clf.fit(n_walkers=n_walkers, n_steps=100, n_burn=50, initial_params=custom_init)

    assert clf.fitted
    assert clf.chain is not None

    # Test prediction
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_hierarchical_log_probability_function(fixtures):
    """Test the hierarchical log probability function directly."""
    df = fixtures

    # Get some test data
    home_idx = np.array([0, 1])
    away_idx = np.array([1, 0])
    goals_home = np.array([2, 1])
    goals_away = np.array([1, 1])
    weights = np.array([1.0, 1.0])
    n_teams = 2

    # Create test parameters
    params = np.array(
        [
            0.1,
            -0.1,  # attack_offsets
            0.2,
            -0.2,  # defense_offsets
            0.0,
            0.0,  # mu_attack, log_sigma_attack
            0.0,
            0.0,  # mu_defense, log_sigma_defense
            0.25,
            -0.1,
        ]
    )  # hfa, rho

    # Import the function directly
    from penaltyblog.models.hierarchical import hierarchical_log_probability

    # Test that function runs without error
    log_prob = hierarchical_log_probability(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )

    # Should return a finite value
    assert np.isfinite(log_prob)

    # Test with invalid rho (should return -inf)
    invalid_params = params.copy()
    invalid_params[-1] = 2.0  # rho > 1
    log_prob_invalid = hierarchical_log_probability(
        invalid_params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )
    assert log_prob_invalid == -np.inf
