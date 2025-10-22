import numpy as np
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_poisson_gradient_with_weights(fixtures):
    """Test that the Poisson gradient correctly uses weights."""
    df = fixtures.copy()

    # Create weights that give higher importance to recent matches
    # Let's say we have 100 matches, give weight 2.0 to last 20 matches, 1.0 to others
    n_matches = len(df)
    weights = np.ones(n_matches)
    weights[-20:] = 2.0  # Double weight for last 20 matches

    # Create model with weights
    clf_weighted = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )

    # Create model without weights for comparison
    clf_unweighted = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Fit both models
    clf_weighted.fit(use_gradient=True)
    clf_unweighted.fit(use_gradient=True)

    # The weighted model should produce different parameters than unweighted
    params_weighted = clf_weighted.get_params()
    params_unweighted = clf_unweighted.get_params()

    # Parameters should be different (at least for some teams)
    differences = []
    for team in clf_weighted.teams:
        attack_diff = abs(
            params_weighted[f"attack_{team}"] - params_unweighted[f"attack_{team}"]
        )
        defense_diff = abs(
            params_weighted[f"defense_{team}"] - params_unweighted[f"defense_{team}"]
        )
        differences.extend([attack_diff, defense_diff])

    # At least some parameters should be different
    assert any(
        diff > 0.01 for diff in differences
    ), "Weighted and unweighted models should produce different parameters"


@pytest.mark.local
def test_poisson_gradient_weighted_vs_unweighted_consistency(fixtures):
    """Test that gradient with uniform weights matches unweighted gradient."""
    df = fixtures

    # Create uniform weights (all 1.0)
    n_matches = len(df)
    uniform_weights = np.ones(n_matches)

    # Model with uniform weights
    clf_uniform = pb.models.PoissonGoalsModel(
        df["goals_home"],
        df["goals_away"],
        df["team_home"],
        df["team_away"],
        uniform_weights,
    )

    # Model without weights
    clf_none = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Fit both models
    clf_uniform.fit(use_gradient=True)
    clf_none.fit(use_gradient=True)

    # Parameters should be very similar
    params_uniform = clf_uniform.get_params()
    params_none = clf_none.get_params()

    for team in clf_uniform.teams:
        attack_key = f"attack_{team}"
        defense_key = f"defense_{team}"

        # Allow for small numerical differences
        assert (
            abs(params_uniform[attack_key] - params_none[attack_key]) < 1e-6
        ), f"Attack parameter for {team} should be nearly identical with uniform weights vs no weights"
        assert (
            abs(params_uniform[defense_key] - params_none[defense_key]) < 1e-6
        ), f"Defense parameter for {team} should be nearly identical with uniform weights vs no weights"

    # Home advantage should also be nearly identical
    assert (
        abs(params_uniform["home_advantage"] - params_none["home_advantage"]) < 1e-6
    ), "Home advantage should be nearly identical with uniform weights vs no weights"


@pytest.mark.local
def test_poisson_gradient_numerical_check_with_weights(fixtures):
    """Test that analytical gradient matches numerical gradient when using weights."""
    from scipy.optimize import check_grad

    df = fixtures

    # Create non-uniform weights
    n_matches = len(df)
    weights = np.random.uniform(
        0.5, 2.0, n_matches
    )  # Random weights between 0.5 and 2.0

    clf = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )

    # Test gradient at initial parameters
    initial_params = clf._params.copy()

    # Use scipy's check_grad function to compare analytical vs numerical gradients
    gradient_error = check_grad(
        clf._loss_function,  # Function to differentiate
        clf._gradient,  # Analytical gradient function
        initial_params,  # Point at which to check
        epsilon=1e-7,  # Step size for numerical differentiation
    )

    # check_grad returns the 2-norm of the difference between gradients
    # For Poisson models with weights, gradients should still be accurate
    assert (
        gradient_error < 1e-4
    ), f"Gradient error {gradient_error:.2e} is too large with weights"


@pytest.mark.local
def test_poisson_gradient_zero_weights(fixtures):
    """Test that gradient handles zero weights correctly."""
    df = fixtures.copy()

    # Create weights where some matches have zero weight
    n_matches = len(df)
    weights = np.ones(n_matches)
    weights[:10] = 0.0  # First 10 matches have zero weight

    clf = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )

    # Should fit without issues
    clf.fit(use_gradient=True)
    assert clf.fitted

    # Get gradient at fitted parameters
    fitted_params = clf._params.copy()
    gradient = clf._gradient(fitted_params)

    # Gradient should be finite
    assert np.all(
        np.isfinite(gradient)
    ), "Gradient should be finite even with zero weights"


@pytest.mark.local
def test_poisson_gradient_extreme_weights(fixtures):
    """Test that gradient handles extreme weight values correctly."""
    df = fixtures.copy()

    # Create extreme weights
    n_matches = len(df)
    weights = np.ones(n_matches)
    weights[::2] = 0.01  # Very small weights for half the matches
    weights[1::2] = 100.0  # Very large weights for the other half

    clf = pb.models.PoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )

    # Should fit without issues
    clf.fit(use_gradient=True)
    assert clf.fitted

    # Get gradient at fitted parameters
    fitted_params = clf._params.copy()
    gradient = clf._gradient(fitted_params)

    # Gradient should be finite
    assert np.all(
        np.isfinite(gradient)
    ), "Gradient should be finite even with extreme weights"
