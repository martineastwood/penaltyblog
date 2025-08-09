import numpy as np
import pytest
from scipy.optimize import check_grad

import penaltyblog as pb


@pytest.mark.local
def test_poisson_model(fixtures):
    df = fixtures

    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 1.0


@pytest.mark.local
def test_bivariate_minimizer_options(fixtures):
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    # With very low maxiter, expect ValueError due to iteration limit
    with pytest.raises(ValueError) as excinfo:
        clf.fit(minimizer_options={"maxiter": 2, "disp": False})
    assert "Iteration limit reached" in str(excinfo.value)

    df = fixtures

    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert 0.65 < params["attack_Man City"] < 2.0
    assert 0.1 < params["home_advantage"] < 0.4

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3
    assert 0.6 < probs.total_goals("over", 1.5) < 0.8
    assert 0.3 < probs.asian_handicap("home", 1.5) < 0.4


@pytest.mark.local
def test_unfitted_raises_error(fixtures):
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


@pytest.mark.local
def test_unfitted_repr(fixtures):
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr


@pytest.mark.local
def test_bivariate_with_gradient_enabled(fixtures):
    """Test that the model fits successfully with gradient enabled."""
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Fit with gradient enabled (default)
    clf.fit(use_gradient=True)

    assert clf.fitted is True
    assert clf.loglikelihood is not None
    assert clf.aic is not None

    params = clf.get_params()
    assert "home_advantage" in params
    assert "lambda3" in params
    assert params["lambda3"] > 0  # exp(correlation) should be positive


@pytest.mark.local
def test_bivariate_with_gradient_disabled(fixtures):
    """Test that the model fits successfully with gradient disabled."""
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Fit with gradient disabled
    clf.fit(use_gradient=False)

    assert clf.fitted is True
    assert clf.loglikelihood is not None
    assert clf.aic is not None

    params = clf.get_params()
    assert "home_advantage" in params
    assert "lambda3" in params


@pytest.mark.local
def test_bivariate_gradient_vs_no_gradient_consistency(fixtures):
    """Test that gradient and non-gradient methods produce similar results."""
    df = fixtures

    # Fit with gradient
    clf_grad = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)

    # Fit without gradient
    clf_no_grad = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_no_grad.fit(use_gradient=False)

    # Results should be similar (allowing for some numerical differences)
    assert abs(clf_grad.loglikelihood - clf_no_grad.loglikelihood) < 1e-2
    assert np.allclose(clf_grad._params, clf_no_grad._params, rtol=1e-2)


@pytest.mark.local
def test_bivariate_gradient_function_correctness(fixtures):
    """Test that gradient function returns correctly shaped output."""
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    params = clf._params.copy()
    gradient = clf._gradient(params)

    # Gradient should have same shape as parameters
    assert gradient.shape == params.shape
    assert not np.any(np.isnan(gradient))
    assert not np.any(np.isinf(gradient))


@pytest.mark.local
def test_bivariate_gradient_numerical_consistency():
    """Test gradient against numerical differentiation."""
    # Use simple test data to ensure numerical stability
    goals_home = [1, 0, 2, 1, 0, 2]
    goals_away = [0, 1, 1, 0, 1, 0]
    teams_home = ["A", "B", "A", "C", "B", "C"]
    teams_away = ["B", "C", "C", "A", "A", "B"]

    clf = pb.models.BivariatePoissonGoalModel(
        goals_home, goals_away, teams_home, teams_away
    )

    params = clf._params.copy()
    # Adjust correlation parameter to avoid extreme values
    params[-1] = 0.1

    # Test gradient accuracy using scipy's check_grad
    error = check_grad(clf._loss_function, clf._gradient, params, epsilon=1e-6)

    # Bivariate Poisson is complex, so use more lenient tolerance
    assert error < 1e-2, f"Gradient error {error} exceeds tolerance"


@pytest.mark.local
def test_bivariate_default_gradient_behavior(fixtures):
    """Test that gradient is used by default."""
    df = fixtures
    clf = pb.models.BivariatePoissonGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Default should use gradient (use_gradient=True by default)
    clf.fit()

    assert clf.fitted is True
    params = clf.get_params()
    assert "lambda3" in params


@pytest.mark.local
def test_bivariate_correlation_parameter_bounds():
    """Test that correlation parameter produces reasonable lambda3 values."""
    goals_home = [1, 0, 2, 1]
    goals_away = [0, 1, 1, 0]
    teams_home = ["A", "B", "A", "B"]
    teams_away = ["B", "A", "B", "A"]

    clf = pb.models.BivariatePoissonGoalModel(
        goals_home, goals_away, teams_home, teams_away
    )
    clf.fit()

    params = clf.get_params()

    # Lambda3 = exp(correlation) should be positive and reasonable
    assert params["lambda3"] > 0
    assert params["lambda3"] < 100  # Should not be extremely large

    # Correlation parameter should be bounded by fit constraints
    assert -3 <= params["correlation_log"] <= 3
