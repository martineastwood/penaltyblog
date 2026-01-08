"""
Comprehensive unit tests for HierarchicalBayesianGoalModel.

This test suite covers:
- Model initialization and fitting
- Parameter estimation (including hierarchical sigma parameters)
- Predictions
- Diagnostics (R-hat, ESS)
- Trace management and mapping
- Plotting functionality
- Error handling
- Parameter indices and names
"""

import numpy as np
import pandas as pd
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_hierarchical_bayesian_model_initialization(fixtures):
    """Test that HierarchicalBayesianGoalModel can be initialized correctly."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    assert model.fitted is False
    assert model.trace is None
    assert model.trace_dict is None
    assert model.sampler is None


@pytest.mark.local
def test_hierarchical_bayesian_model_fit(fixtures):
    """Test that the model fits correctly and produces expected outputs."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    assert model.fitted is True
    assert model.trace is not None
    assert model.trace_dict is not None
    assert model.sampler is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_sigma_parameters(fixtures):
    """Test that hierarchical sigma parameters are correctly estimated."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check that sigma parameters are in trace_dict
    assert "sigma_attack" in model.trace_dict
    assert "sigma_defense" in model.trace_dict

    # Check that sigma parameters have reasonable values
    sigma_attack = model.trace_dict["sigma_attack"]
    sigma_defense = model.trace_dict["sigma_defense"]

    assert len(sigma_attack) > 0
    assert len(sigma_defense) > 0

    # Sigmas should be positive
    assert np.mean(sigma_attack) > 0
    assert np.mean(sigma_defense) > 0


@pytest.mark.local
def test_hierarchical_bayesian_model_get_params(fixtures):
    """Test that get_params returns all expected parameters."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params = model.get_params()

    # Check standard parameters
    assert "home_advantage" in params
    assert "rho" in params

    # Check hierarchical sigma parameters
    assert "sigma_attack" in params
    assert "sigma_defense" in params

    # Check team parameters
    assert "attack_Man City" in params
    assert "defense_Man City" in params

    # Check sigma values are positive
    assert params["sigma_attack"] > 0
    assert params["sigma_defense"] > 0


@pytest.mark.local
def test_hierarchical_bayesian_model_predictions(fixtures):
    """Test that prediction functionality works correctly."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
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
def test_hierarchical_bayesian_model_diagnostics(fixtures):
    """Test that diagnostics are computed correctly including sigma parameters."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    diag = model.get_diagnostics()

    assert isinstance(diag, pd.DataFrame)
    assert "r_hat" in diag.columns
    assert "ess" in diag.columns

    # Check that sigma parameters are in diagnostics
    assert "Sigma_Attack" in diag.index
    assert "Sigma_Defense" in diag.index

    # Check that standard parameters are in diagnostics
    assert "Home_Advantage" in diag.index
    assert "Rho" in diag.index


@pytest.mark.local
def test_hierarchical_bayesian_model_unfitted_raises_error(fixtures):
    """Test that unfitted model raises appropriate errors."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test prediction raises error
    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.get_diagnostics()

    # Test get_params raises error
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.get_params()

    # Test prediction raises error
    with pytest.raises(ValueError, match="Model is not yet fitted"):
        model.predict("Man City", "Liverpool")


@pytest.mark.local
def test_hierarchical_bayesian_model_plot_trace(fixtures):
    """Test that plot_trace method works correctly."""
    df = fixtures.head(20)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_trace(params=["sigma_attack", "sigma_defense"])
    assert fig is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_plot_autocorr(fixtures):
    """Test that plot_autocorr method works correctly."""
    df = fixtures.head(20)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_autocorr(params=["sigma_attack"])
    assert fig is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_plot_posterior(fixtures):
    """Test that plot_posterior method works correctly."""
    df = fixtures.head(20)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_posterior(params=["sigma_attack", "sigma_defense"])
    assert fig is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_plot_convergence(fixtures):
    """Test that plot_convergence method works correctly."""
    df = fixtures.head(20)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_convergence()
    assert fig is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_plot_diagnostics(fixtures):
    """Test that plot_diagnostics method works correctly."""
    df = fixtures.head(20)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    fig = model.plot_diagnostics()
    assert fig is not None


@pytest.mark.local
def test_hierarchical_bayesian_model_param_names(fixtures):
    """Test that _get_param_names returns correct parameter names."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    param_names = model._get_param_names()

    # Check that hierarchical sigma parameters are included
    assert "sigma_attack" in param_names
    assert "sigma_defense" in param_names

    # Check that standard parameters are included
    assert "home_advantage" in param_names
    assert "rho" in param_names


@pytest.mark.local
def test_hierarchical_bayesian_model_param_indices(fixtures):
    """Test that _get_tail_param_indices returns correct indices."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    indices = model.param_indices()

    # Check that all expected keys are present
    assert "attack" in indices
    assert "defense" in indices
    assert "home_advantage" in indices
    assert "rho" in indices
    assert "sigma_attack" in indices
    assert "sigma_defense" in indices

    # Check that hierarchical parameters use correct negative indices
    assert indices["sigma_attack"] == -2
    assert indices["sigma_defense"] == -1

    # Check that home_advantage and rho are adjusted for hierarchical model
    assert indices["home_advantage"] == -4
    assert indices["rho"] == -3


@pytest.mark.local
def test_hierarchical_bayesian_model_params_array(fixtures):
    """Test that params_array property works correctly."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params_array = model.params_array

    assert isinstance(params_array, np.ndarray)
    assert params_array.ndim == 1

    # Check that the array has the correct length
    # n_teams * 2 (attack and defense) + 4 (home_advantage, rho, sigma_attack, sigma_defense)
    expected_length = model.n_teams * 2 + 4
    assert len(params_array) == expected_length


@pytest.mark.local
def test_hierarchical_bayesian_model_params_property(fixtures):
    """Test that params property works correctly."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    params = model.params

    assert isinstance(params, dict)
    assert "sigma_attack" in params
    assert "sigma_defense" in params


@pytest.mark.local
def test_hierarchical_bayesian_model_trace_mapping(fixtures):
    """Test that _map_trace_to_dict correctly maps trace including sigma parameters."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check that trace_dict has expected keys
    assert "sigma_attack" in model.trace_dict
    assert "sigma_defense" in model.trace_dict

    # Check that attack and defense parameters are mapped
    assert "attack_Man City" in model.trace_dict
    assert "defense_Man City" in model.trace_dict


@pytest.mark.local
def test_hierarchical_bayesian_model_custom_fit_parameters(fixtures):
    """Test fitting with custom sample parameters."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test with custom parameters
    model.fit(n_samples=100, burn=75, n_chains=3, thin=2)

    assert model.fitted is True

    # Check that trace shape reflects thinning and burn-in
    # The model successfully fits, so we just check that trace is not empty
    assert model.trace.shape[0] > 0
    assert model.trace_dict is not None
    assert "sigma_attack" in model.trace_dict
    assert "sigma_defense" in model.trace_dict


@pytest.mark.local
def test_hierarchical_bayesian_model_comparison_with_standard_bayesian(fixtures):
    """Test that hierarchical model has additional parameters compared to standard Bayesian model."""
    df = fixtures.head(30)

    # Fit standard Bayesian model
    standard_model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    standard_model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Fit hierarchical Bayesian model
    hierarchical_model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    hierarchical_model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Hierarchical model should have more parameters
    standard_params = standard_model._get_param_names()
    hierarchical_params = hierarchical_model._get_param_names()

    assert len(hierarchical_params) == len(standard_params) + 2

    # Check that hierarchical model has sigma parameters
    assert "sigma_attack" in hierarchical_params
    assert "sigma_defense" in hierarchical_params
    assert "sigma_attack" not in standard_params
    assert "sigma_defense" not in standard_params


@pytest.mark.local
def test_hierarchical_bayesian_model_dimension(fixtures):
    """Test that the model has the correct parameter dimension."""
    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Check parameter dimension
    # Should be: n_teams * 2 (attack, defense) + 4 (home_advantage, rho, sigma_attack, sigma_defense)
    expected_ndim = model.n_teams * 2 + 4
    actual_ndim = len(model._params)

    assert expected_ndim == actual_ndim


@pytest.mark.local
def test_hierarchical_bayesian_model_save_load(fixtures, tmp_path):
    """Test that the model can be saved and loaded."""
    import pickle

    df = fixtures.head(30)

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    # Save the model
    model_path = tmp_path / "hierarchical_model.pkl"
    model.save(str(model_path))

    # Load the model
    loaded_model = pb.models.HierarchicalBayesianGoalModel.load(str(model_path))

    # Check that loaded model has the same parameters
    assert loaded_model.fitted == model.fitted
    assert np.allclose(loaded_model._params, model._params)

    # Check that sigma parameters are preserved
    loaded_params = loaded_model.get_params()
    original_params = model.get_params()

    assert loaded_params["sigma_attack"] == original_params["sigma_attack"]
    assert loaded_params["sigma_defense"] == original_params["sigma_defense"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
