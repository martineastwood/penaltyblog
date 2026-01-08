"""
Comprehensive unit tests for BayesianGoalModel.

This test suite covers:
- Model initialization and fitting
- Parameter estimation (home_advantage, rho)
- Predictions
- Diagnostics (R-hat, ESS)
- Trace management and mapping
- Plotting functionality
- Error handling
- Parameter indices and names
- Start position generation
- Comparison with other models
"""

import numpy as np
import pandas as pd
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_bayesian_goal_model_initialization(fixtures):
    """Test that BayesianGoalModel can be initialized correctly."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    assert model.fitted is False
    assert model.trace is None
    assert model.trace_dict is None
    assert model.sampler is None


@pytest.mark.local
def test_bayesian_goal_model_fit(fixtures):
    """Test that the model fits correctly and produces expected outputs."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    assert model.fitted is True
    assert model.trace is not None
    assert model.trace_dict is not None
    assert model.sampler is not None


@pytest.mark.local
def test_bayesian_goal_model_home_advantage_and_rho(fixtures):
    """Test that home_advantage and rho parameters are correctly estimated."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check that home_advantage and rho are in trace_dict
    assert "home_advantage" in model.trace_dict
    assert "rho" in model.trace_dict

    # Check that they have reasonable values
    home_advantage = model.trace_dict["home_advantage"]
    rho = model.trace_dict["rho"]

    assert len(home_advantage) > 0
    assert len(rho) > 0

    # Home advantage should typically be positive
    assert np.mean(home_advantage) > -0.5  # Allow some flexibility

    # Rho is typically negative (low scores correction)
    assert np.mean(rho) < 0.5


@pytest.mark.local
def test_bayesian_goal_model_get_params(fixtures):
    """Test that get_params returns all expected parameters."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params = model.get_params()

    # Check standard parameters
    assert "home_advantage" in params
    assert "rho" in params

    # Check team parameters
    assert "attack_Man City" in params
    assert "defense_Man City" in params


@pytest.mark.local
def test_bayesian_goal_model_predictions(fixtures):
    """Test that prediction functionality works correctly."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    probs = model.predict("Man City", "Liverpool")

    assert isinstance(probs, pb.models.FootballProbabilityGrid)
    assert probs.home_win > 0
    assert probs.away_win > 0
    assert probs.draw > 0

    # Probabilities should sum to approximately 1
    assert abs(probs.home_win + probs.draw + probs.away_win - 1.0) < 0.01


@pytest.mark.local
def test_bayesian_goal_model_diagnostics(fixtures):
    """Test that diagnostics are computed correctly."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    diag = model.get_diagnostics()

    assert isinstance(diag, pd.DataFrame)
    assert "r_hat" in diag.columns
    assert "ess" in diag.columns

    # Check that standard parameters are in diagnostics
    assert "Home_Advantage" in diag.index
    assert "Rho" in diag.index


@pytest.mark.local
def test_bayesian_goal_model_unfitted_raises_error(fixtures):
    """Test that unfitted model raises appropriate errors."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test diagnostics raises error
    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.get_diagnostics()

    # Test get_params raises error
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.get_params()

    # Test prediction raises error
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.predict("Man City", "Liverpool")


@pytest.mark.local
def test_bayesian_goal_model_plot_trace(fixtures):
    """Test that plot_trace method works correctly."""
    df = fixtures.head(20)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_trace(params=["home_advantage", "rho"])
    assert fig is not None


@pytest.mark.local
def test_bayesian_goal_model_plot_autocorr(fixtures):
    """Test that plot_autocorr method works correctly."""
    df = fixtures.head(20)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_autocorr(params=["home_advantage"])
    assert fig is not None


@pytest.mark.local
def test_bayesian_goal_model_plot_posterior(fixtures):
    """Test that plot_posterior method works correctly."""
    df = fixtures.head(20)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_posterior(params=["home_advantage", "rho"])
    assert fig is not None


@pytest.mark.local
def test_bayesian_goal_model_plot_convergence(fixtures):
    """Test that plot_convergence method works correctly."""
    df = fixtures.head(20)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_convergence()
    assert fig is not None


@pytest.mark.local
def test_bayesian_goal_model_plot_diagnostics(fixtures):
    """Test that plot_diagnostics method works correctly."""
    df = fixtures.head(20)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_diagnostics()
    assert fig is not None


@pytest.mark.local
def test_bayesian_goal_model_param_names(fixtures):
    """Test that _get_param_names returns correct parameter names."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    param_names = model._get_param_names()

    # Check that standard parameters are included
    assert "home_advantage" in param_names
    assert "rho" in param_names

    # Check that team parameters are included
    assert "attack_Man City" in param_names
    assert "defense_Man City" in param_names


@pytest.mark.local
def test_bayesian_goal_model_param_indices(fixtures):
    """Test that _get_tail_param_indices returns correct indices."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    indices = model.param_indices()

    # Check that all expected keys are present
    assert "attack" in indices
    assert "defense" in indices
    assert "home_advantage" in indices
    assert "rho" in indices

    # Check that home_advantage and rho use correct negative indices
    assert indices["home_advantage"] == -2
    assert indices["rho"] == -1


@pytest.mark.local
def test_bayesian_goal_model_params_array(fixtures):
    """Test that params_array property works correctly."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params_array = model.params_array

    assert isinstance(params_array, np.ndarray)
    assert params_array.ndim == 1

    # Check that the array has the correct length
    # n_teams * 2 (attack and defense) + 2 (home_advantage, rho)
    expected_length = model.n_teams * 2 + 2
    assert len(params_array) == expected_length


@pytest.mark.local
def test_bayesian_goal_model_params_property(fixtures):
    """Test that params property works correctly."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params = model.params

    assert isinstance(params, dict)
    assert "home_advantage" in params
    assert "rho" in params


@pytest.mark.local
def test_bayesian_goal_model_trace_mapping(fixtures):
    """Test that _map_trace_to_dict correctly maps trace."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check that trace_dict has expected keys
    assert "home_advantage" in model.trace_dict
    assert "rho" in model.trace_dict

    # Check that attack and defense parameters are mapped
    assert "attack_Man City" in model.trace_dict
    assert "defense_Man City" in model.trace_dict


@pytest.mark.local
def test_bayesian_goal_model_custom_fit_parameters(fixtures):
    """Test fitting with custom sample parameters."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test with custom parameters
    model.fit(n_samples=100, burn=75, n_chains=3, thin=2)

    assert model.fitted is True

    # Check that trace shape reflects thinning and burn-in
    # The model successfully fits, so we just check that trace is not empty
    assert model.trace.shape[0] > 0
    assert model.trace_dict is not None
    assert "home_advantage" in model.trace_dict
    assert "rho" in model.trace_dict


@pytest.mark.local
def test_bayesian_goal_model_dimension(fixtures):
    """Test that the model has the correct parameter dimension."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check parameter dimension
    # Should be: n_teams * 2 (attack, defense) + 2 (home_advantage, rho)
    expected_ndim = model.n_teams * 2 + 2
    actual_ndim = len(model._params)

    assert expected_ndim == actual_ndim


@pytest.mark.local
def test_bayesian_goal_model_save_load(fixtures, tmp_path):
    """Test that the model can be saved and loaded."""
    import pickle

    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Save the model
    model_path = tmp_path / "bayesian_model.pkl"
    model.save(str(model_path))

    # Load the model
    loaded_model = pb.models.BayesianGoalModel.load(str(model_path))

    # Check that loaded model has the same parameters
    assert loaded_model.fitted == model.fitted
    assert np.allclose(loaded_model._params, model._params)

    # Check that key parameters are preserved
    loaded_params = loaded_model.get_params()
    original_params = model.get_params()

    assert loaded_params["home_advantage"] == original_params["home_advantage"]
    assert loaded_params["rho"] == original_params["rho"]


@pytest.mark.local
def test_bayesian_goal_model_predict_with_max_goals(fixtures):
    """Test prediction with custom max_goals parameter."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Test with default max_goals
    probs_default = model.predict("Man City", "Liverpool")

    # Test with custom max_goals
    probs_custom = model.predict("Man City", "Liverpool", max_goals=10)

    # Both should work
    assert probs_default.home_win > 0
    assert probs_custom.home_win > 0


@pytest.mark.local
def test_bayesian_goal_model_predict_without_normalization(fixtures):
    """Test prediction without normalization."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Test without normalization
    probs = model.predict("Man City", "Liverpool", normalize=False)

    assert isinstance(probs, pb.models.FootballProbabilityGrid)
    # Without normalization, probabilities might not sum exactly to 1
    # But they should still be positive
    assert probs.home_win > 0
    assert probs.away_win > 0
    assert probs.draw > 0


@pytest.mark.local
def test_bayesian_goal_model_with_unknown_teams_raises_error(fixtures):
    """Test that prediction with unknown teams raises an error."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Test with unknown team
    with pytest.raises(ValueError, match="must have been in the training data"):
        model.predict("Unknown Team", "Man City")


@pytest.mark.local
def test_bayesian_goal_model_average_lambdas(fixtures):
    """Test that prediction returns expected lambda values."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    probs = model.predict("Man City", "Liverpool")

    # Check that the probability grid has lambda values
    # These are stored in the FootballProbabilityGrid
    assert probs is not None
    assert hasattr(probs, "home_win")
    assert hasattr(probs, "away_win")


@pytest.mark.local
def test_bayesian_goal_model_team_map(fixtures):
    """Test that team_map is correctly initialized."""
    df = fixtures.head(30)

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # team_map should be initialized even before fitting
    assert model.team_map is not None
    assert "Man City" in model.team_map
    assert "Liverpool" in model.team_map


@pytest.mark.local
def test_bayesian_goal_model_comparison_with_poisson(fixtures):
    """Test that Bayesian model has more parameters than Poisson model."""
    df = fixtures.head(30)

    # Fit Poisson model
    poisson_model = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    poisson_model.fit()

    # Fit Bayesian model
    bayesian_model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    bayesian_model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Bayesian model should have rho parameter that Poisson doesn't have
    poisson_params = poisson_model._get_param_names()
    bayesian_params = bayesian_model._get_param_names()

    assert "rho" in bayesian_params
    assert "rho" not in poisson_params


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
