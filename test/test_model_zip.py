import numpy as np
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_poisson_model(fixtures):
    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 1.0


@pytest.mark.local
def test_zip_minimizer_options(fixtures):
    # Test that minimizer fails with low iterations
    df = fixtures
    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Check that minimizer fails with low iterations
    with pytest.raises(ValueError) as excinfo:
        clf.fit(minimizer_options={"maxiter": 2, "disp": False})

    with pytest.raises(ValueError) as excinfo:
        clf.fit(minimizer_options={"maxiter": 2, "disp": False})
    assert "Iteration limit reached" in str(excinfo.value)

    # Test successful fit and predictions
    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()

    # Check parameters
    params = clf.get_params()
    assert 0.5 < params["attack_Man City"] < 2.0
    assert 0.1 < params["home_advantage"] < 0.4

    # Check predictions
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.6 < probs.total_goals("over", 1.5) < 0.8
    assert 0.9 < probs.asian_handicap("home", 1.5) < 1.0

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert 0.5 < params["attack_Man City"] < 2.0
    assert 0.1 < params["home_advantage"] < 0.4

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.6 < probs.total_goals("over", 1.5) < 0.8
    assert 0.9 < probs.asian_handicap("home", 1.5) < 1.0


@pytest.mark.local
def test_unfitted_raises_error(fixtures):
    df = fixtures
    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


@pytest.mark.local
def test_unfitted_repr(fixtures):
    df = fixtures
    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr


@pytest.mark.local
def test_zip_with_gradient_enabled(fixtures):
    """Test that the Zero-Inflated Poisson model works correctly with gradient enabled (default behavior)."""
    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test explicit gradient enabled
    clf.fit(use_gradient=True)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 0.5
    assert 0.0 < params["home_advantage"] < 1.5
    assert 0.0 <= params["zero_inflation"] <= 1.0  # Zero inflation parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_zip_with_gradient_disabled(fixtures):
    """Test that the Zero-Inflated Poisson model works correctly with gradient disabled."""
    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient disabled
    clf.fit(use_gradient=False)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 0.5
    assert 0.0 < params["home_advantage"] < 1.5
    assert 0.0 <= params["zero_inflation"] <= 1.0  # Zero inflation parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_zip_gradient_vs_no_gradient_consistency(fixtures):
    """Test that gradient and no-gradient approaches produce similar results."""
    df = fixtures

    # Model with gradient
    clf_grad = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)
    params_grad = clf_grad.get_params()

    # Model without gradient
    clf_no_grad = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_no_grad.fit(use_gradient=False)
    params_no_grad = clf_no_grad.get_params()

    # Parameters should be very similar (within tight tolerance for this model)
    for team in clf_grad.teams:
        attack_key = f"attack_{team}"
        defence_key = f"defence_{team}"

        # Allow for some numerical differences between gradient and no-gradient optimization
        assert (
            abs(params_grad[attack_key] - params_no_grad[attack_key]) < 0.1
        ), f"Attack parameter for {team} differs significantly"
        assert (
            abs(params_grad[defence_key] - params_no_grad[defence_key]) < 0.1
        ), f"Defence parameter for {team} differs significantly"

    # Home advantage and zero inflation should also be similar
    assert abs(params_grad["home_advantage"] - params_no_grad["home_advantage"]) < 0.1
    assert abs(params_grad["zero_inflation"] - params_no_grad["zero_inflation"]) < 0.1

    # Log likelihoods should be very close (gradient might be slightly better)
    assert abs(clf_grad.loglikelihood - clf_no_grad.loglikelihood) < 1.0


@pytest.mark.local
def test_zip_gradient_function_correctness(fixtures):
    """Test that the gradient function produces reasonable gradients."""
    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient at initial parameters
    initial_params = clf._params.copy()
    gradient = clf._gradient(initial_params)

    # Gradient should be a numpy array of the right size
    expected_size = (
        2 * clf.n_teams + 2
    )  # attack + defence + home_advantage + zero_inflation
    assert isinstance(gradient, np.ndarray)
    assert len(gradient) == expected_size

    # Gradient should contain finite values
    assert np.all(np.isfinite(gradient))

    # Test gradient at fitted parameters (should be reasonably small at optimum)
    clf.fit(use_gradient=True)
    fitted_params = clf._params.copy()
    fitted_gradient = clf._gradient(fitted_params)

    # At the optimum, gradients should be small (but not necessarily zero due to constraints)
    # However, if zero-inflation parameter hits bounds, its gradient can be large
    # Check that non-boundary gradients are reasonable
    fitted_zi_param = fitted_params[-1]
    if fitted_zi_param <= 1e-5 or fitted_zi_param >= (1 - 1e-5):
        # Zero-inflation is at boundary - check other gradients only
        assert np.all(
            np.abs(fitted_gradient[:-1]) < 5.0
        ), "Non-boundary gradients should be reasonably small"
        print(
            f"   Note: Zero-inflation parameter at boundary ({fitted_zi_param:.2e}), gradient = {fitted_gradient[-1]:.1f}"
        )
    else:
        # All gradients should be small
        assert np.all(
            np.abs(fitted_gradient) < 5.0
        ), "Gradients at optimum should be reasonably small"


@pytest.mark.local
def test_zip_gradient_numerical_consistency(fixtures):
    """Test that analytical gradient matches numerical gradient using scipy.optimize.check_grad."""
    from scipy.optimize import check_grad

    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
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
    # For a well-implemented gradient, this should be very small
    # For complex models like Zero-Inflated Poisson, 1e-3 is considered excellent accuracy
    assert gradient_error < 1e-3, f"Gradient error {gradient_error:.2e} is too large"

    # Test at a different point (middle values to avoid boundary issues)
    test_params = initial_params.copy()
    test_params[-1] = (
        0.3  # Set zero-inflation to middle value to avoid boundary effects
    )

    gradient_error_mid = check_grad(
        clf._loss_function, clf._gradient, test_params, epsilon=1e-7
    )

    assert (
        gradient_error_mid < 1e-3
    ), f"Gradient error at test point {gradient_error_mid:.2e} is too large"


@pytest.mark.local
def test_zip_default_gradient_behavior(fixtures):
    """Test that gradient is enabled by default."""
    df = fixtures

    clf = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Default behavior should use gradient (no need to specify use_gradient=True)
    clf.fit()
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 0.5
    assert "zero_inflation" in params


@pytest.mark.local
def test_zip_zero_inflation_parameter_bounds(fixtures):
    """Test that zero inflation parameter stays within reasonable bounds."""
    df = fixtures

    # Test with gradient
    clf_grad = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)
    params_grad = clf_grad.get_params()

    # Zero inflation should be between 0 and 1 (probability)
    assert (
        0.0 <= params_grad["zero_inflation"] <= 1.0
    ), f"Zero inflation {params_grad['zero_inflation']:.3f} is outside valid bounds [0, 1]"

    # Test without gradient
    clf_no_grad = pb.models.ZeroInflatedPoissonGoalsModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_no_grad.fit(use_gradient=False)
    params_no_grad = clf_no_grad.get_params()

    # Zero inflation should be between 0 and 1 (probability)
    assert (
        0.0 <= params_no_grad["zero_inflation"] <= 1.0
    ), f"Zero inflation {params_no_grad['zero_inflation']:.3f} is outside valid bounds [0, 1]"
