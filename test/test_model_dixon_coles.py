import numpy as np
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_dc_model(fixtures):
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf.fit()
    params = clf.get_params()
    assert params["attack_Man City"] > 1.0

    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid
    assert type(probs.home_draw_away) == list
    assert len(probs.home_draw_away) == 3


@pytest.mark.local
def test_dixon_coles_minimizer_options(fixtures):
    df = fixtures
    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    # With very low maxiter, expect ValueError due to iteration limit
    import pytest

    with pytest.raises(ValueError) as excinfo:
        clf.fit(minimizer_options={"maxiter": 2, "disp": False})
    assert "Iteration limit reached" in str(excinfo.value)

    df = fixtures

    clf = pb.models.DixonColesGoalModel(
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
    assert 0.4 < probs.both_teams_to_score < 0.7


@pytest.mark.local
def test_unfitted_raises_error(fixtures):
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    with pytest.raises(ValueError):
        clf.predict("Liverpool", "Wolves")

    with pytest.raises(ValueError):
        clf.get_params()


@pytest.mark.local
def test_unfitted_repr(fixtures):
    df = fixtures
    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    repr = str(clf)
    assert "Status: Model not fitted" in repr


@pytest.mark.local
def test_dixon_coles_with_gradient_enabled(fixtures):
    """Test that the Dixon-Coles model works correctly with gradient enabled (default behavior)."""
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test explicit gradient enabled
    clf.fit(use_gradient=True)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert 0.2 < params["home_advantage"] < 0.3
    assert "rho" in params  # Dixon-Coles specific parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_dixon_coles_with_gradient_disabled(fixtures):
    """Test that the Dixon-Coles model works correctly with gradient disabled."""
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient disabled
    clf.fit(use_gradient=False)
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert 0.2 < params["home_advantage"] < 0.3
    assert "rho" in params  # Dixon-Coles specific parameter

    # Test that predictions work
    probs = clf.predict("Liverpool", "Wolves")
    assert type(probs) == pb.models.FootballProbabilityGrid


@pytest.mark.local
def test_dixon_coles_gradient_vs_no_gradient_consistency(fixtures):
    """Test that gradient and no-gradient approaches produce similar results."""
    df = fixtures

    # Model with gradient
    clf_grad = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    clf_grad.fit(use_gradient=True)
    params_grad = clf_grad.get_params()

    # Model without gradient
    clf_no_grad = pb.models.DixonColesGoalModel(
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
            abs(params_grad[attack_key] - params_no_grad[attack_key]) < 0.1
        ), f"Attack parameter for {team} differs significantly"
        assert (
            abs(params_grad[defence_key] - params_no_grad[defence_key]) < 0.1
        ), f"Defence parameter for {team} differs significantly"

    # Home advantage and rho should also be similar
    assert abs(params_grad["home_advantage"] - params_no_grad["home_advantage"]) < 0.05
    assert abs(params_grad["rho"] - params_no_grad["rho"]) < 0.1

    # Log likelihoods should be similar (gradient might be slightly better)
    assert abs(clf_grad.loglikelihood - clf_no_grad.loglikelihood) < 1.0


@pytest.mark.local
def test_dixon_coles_gradient_function_correctness(fixtures):
    """Test that the gradient function produces reasonable gradients."""
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Test gradient at initial parameters
    initial_params = clf._params.copy()
    gradient = clf._gradient(initial_params)

    # Gradient should be a numpy array of the right size
    expected_size = 2 * clf.n_teams + 2  # attack + defense + home_advantage + rho
    assert isinstance(gradient, np.ndarray)
    assert len(gradient) == expected_size

    # Gradient should contain finite values
    assert np.all(np.isfinite(gradient))

    # Test gradient at fitted parameters (should be close to zero at optimum)
    clf.fit(use_gradient=True)
    fitted_params = clf._params.copy()
    fitted_gradient = clf._gradient(fitted_params)

    # At the optimum, gradients should be small (close to zero)
    # Note: exact zeros are not expected due to constraints
    assert np.all(np.abs(fitted_gradient) < 1.0), "Gradients at optimum should be small"


@pytest.mark.local
def test_dixon_coles_default_gradient_behavior(fixtures):
    """Test that gradient is enabled by default."""
    df = fixtures

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Default behavior should use gradient (no need to specify use_gradient=True)
    clf.fit()
    assert clf.fitted

    params = clf.get_params()
    assert params["attack_Man City"] > 1.0
    assert "rho" in params
