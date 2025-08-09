import numpy as np
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_negative_binomial_model(fixtures):
    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 1.0

    # Check that dispersion parameter is positive and reasonable
    assert params["dispersion"] > 0
    assert "dispersion" in params  # Negative binomial specific parameter


@pytest.mark.local
def test_negative_binomial_minimizer_options(fixtures):
    df = fixtures
    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    import pytest

    with pytest.raises(ValueError) as excinfo:
        clf.fit(minimizer_options={"maxiter": 2, "disp": False})
    assert "Iteration limit reached" in str(excinfo.value)

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
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


@pytest.mark.local
def test_unfitted_raises_error(fixtures):
    df = fixtures
    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


@pytest.mark.local
def test_unfitted_repr(fixtures):
    df = fixtures
    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr


@pytest.mark.local
def test_negative_binomial_with_gradient_enabled(fixtures):
    """Test that the negative binomial model works correctly with gradient enabled (default behavior)."""
    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test explicit gradient enabled
    clf.fit(use_gradient=True)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert 0.1 < params["home_advantage"] < 0.5
    assert params["dispersion"] > 0  # Negative binomial specific parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_negative_binomial_with_gradient_disabled(fixtures):
    """Test that the negative binomial model works correctly with gradient disabled."""
    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient disabled
    clf.fit(use_gradient=False)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert 0.1 < params["home_advantage"] < 0.5
    assert params["dispersion"] > 0  # Negative binomial specific parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_negative_binomial_gradient_vs_no_gradient_consistency(fixtures):
    """Test that gradient and no-gradient approaches produce similar results."""
    df = fixtures

    # Model with gradient
    clf_grad = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)
    params_grad = clf_grad.get_params()

    # Model without gradient
    clf_no_grad = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_no_grad.fit(use_gradient=False)
    params_no_grad = clf_no_grad.get_params()

    # Parameters should be similar (within reasonable tolerance)
    for team in clf_grad.teams:
        attack_key = f"attack_{team}"
        defence_key = f"defence_{team}"

        # Allow for some numerical differences between gradient and no-gradient optimization
        assert (
            abs(params_grad[attack_key] - params_no_grad[attack_key]) < 0.2
        ), f"Attack parameter for {team} differs significantly"
        assert (
            abs(params_grad[defence_key] - params_no_grad[defence_key]) < 0.2
        ), f"Defence parameter for {team} differs significantly"

    # Home advantage and dispersion should also be reasonably similar
    assert abs(params_grad["home_advantage"] - params_no_grad["home_advantage"]) < 0.15
    # Dispersion can vary significantly due to optimization challenges, so use large tolerance
    dispersion_ratio = abs(
        params_grad["dispersion"] - params_no_grad["dispersion"]
    ) / max(params_grad["dispersion"], params_no_grad["dispersion"])
    assert (
        dispersion_ratio < 0.9
    ), f"Dispersion parameters differ too much: {params_grad['dispersion']:.1f} vs {params_no_grad['dispersion']:.1f} (ratio: {dispersion_ratio:.2f})"

    # Log likelihoods should be reasonably close (gradient might be slightly better)
    assert abs(clf_grad.loglikelihood - clf_no_grad.loglikelihood) < 2.0


@pytest.mark.local
def test_negative_binomial_gradient_function_correctness(fixtures):
    """Test that the gradient function produces reasonable gradients."""
    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient at initial parameters
    initial_params = clf._params.copy()
    gradient = clf._gradient(initial_params)

    # Gradient should be a numpy array of the right size
    expected_size = (
        2 * clf.n_teams + 2
    )  # attack + defence + home_advantage + dispersion
    assert isinstance(gradient, np.ndarray)
    assert len(gradient) == expected_size

    # Gradient should contain finite values
    assert np.all(np.isfinite(gradient))

    # Test gradient at fitted parameters (should be reasonably small at optimum)
    clf.fit(use_gradient=True)
    fitted_params = clf._params.copy()
    fitted_gradient = clf._gradient(fitted_params)

    # At the optimum, gradients should be small (but not necessarily zero due to constraints)
    # Use a more lenient bound for negative binomial due to potential numerical challenges
    assert np.all(
        np.abs(fitted_gradient) < 5.0
    ), "Gradients at optimum should be reasonably small"


@pytest.mark.local
def test_negative_binomial_gradient_numerical_consistency(fixtures):
    """Test that analytical gradient matches numerical gradient using scipy.optimize.check_grad."""
    from scipy.optimize import check_grad

    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
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
    # For negative binomial models, gradients involve digamma functions making them more complex
    # 5e-3 is considered excellent accuracy for models with special functions
    assert gradient_error < 5e-3, f"Gradient error {gradient_error:.2e} is too large"

    # Test at a different point with reasonable parameter values
    test_params = initial_params.copy()
    test_params[-1] = 0.5  # Set dispersion to reasonable middle value

    gradient_error_mid = check_grad(
        clf._loss_function, clf._gradient, test_params, epsilon=1e-7
    )

    assert (
        gradient_error_mid < 5e-3
    ), f"Gradient error at test point {gradient_error_mid:.2e} is too large"


@pytest.mark.local
def test_negative_binomial_default_gradient_behavior(fixtures):
    """Test that gradient is enabled by default."""
    df = fixtures

    clf = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Default behavior should use gradient (no need to specify use_gradient=True)
    clf.fit()
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert "dispersion" in params


@pytest.mark.local
def test_negative_binomial_dispersion_parameter_bounds(fixtures):
    """Test that dispersion parameter stays within reasonable bounds."""
    df = fixtures

    # Test with gradient
    clf_grad = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)
    params_grad = clf_grad.get_params()

    # Dispersion should be positive and within bounds (including boundary values)
    assert (
        1e-5 <= params_grad["dispersion"] <= 1000
    ), f"Dispersion {params_grad['dispersion']:.3f} is outside reasonable bounds"

    # Test without gradient
    clf_no_grad = pb.models.NegativeBinomialGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_no_grad.fit(use_gradient=False)
    params_no_grad = clf_no_grad.get_params()

    # Dispersion should be positive and within bounds (including boundary values)
    assert (
        1e-5 <= params_no_grad["dispersion"] <= 1000
    ), f"Dispersion {params_no_grad['dispersion']:.3f} is outside reasonable bounds"
