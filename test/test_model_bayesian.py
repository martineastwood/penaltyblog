import numpy as np
import pandas as pd
import pytest

import penaltyblog as pb


@pytest.mark.local
def test_bayesian_model(fixtures):
    df = fixtures.head(30)  # Use a small subset for speed

    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Use small number of samples for speed in tests
    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    assert model.fitted
    assert model.trace is not None
    assert "attack_Man City" in model.trace_dict

    params = model.get_params()
    assert "home_advantage" in params
    assert "rho" in params

    # Check diagnostics
    diag = model.get_diagnostics()
    assert isinstance(diag, pd.DataFrame)
    assert "r_hat" in diag.columns
    assert "ess" in diag.columns

    # Check predictions
    probs = model.predict("Man City", "Liverpool")
    assert isinstance(probs, pb.models.FootballProbabilityGrid)
    assert probs.home_win > 0
    assert probs.away_win > 0
    assert probs.draw > 0


@pytest.mark.local
def test_hierarchical_bayesian_model(fixtures):
    df = fixtures.head(30)  # Use a small subset for speed

    model = pb.models.HierarchicalBayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )

    # Use small number of samples for speed in tests
    model.fit(n_samples=50, burn=50, n_chains=2, thin=1)

    assert model.fitted
    assert "sigma_attack" in model.trace_dict

    params = model.get_params()
    assert "sigma_attack" in params
    assert "sigma_defense" in params

    # Check diagnostics
    diag = model.get_diagnostics()
    assert isinstance(diag, pd.DataFrame)
    assert "Sigma_Attack" in diag.index

    # Check predictions
    probs = model.predict("Man City", "Liverpool")
    assert isinstance(probs, pb.models.FootballProbabilityGrid)


@pytest.mark.local
def test_bayesian_plots(fixtures):
    df = fixtures.head(20)
    model = pb.models.BayesianGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
    )
    model.fit(n_samples=20, burn=20, n_chains=2, thin=1)

    # Test plotting methods just returns something (not None)
    # We don't check the visual correctness but that the code runs
    fig = model.plot_trace(params=["home_advantage"])
    assert fig is not None

    fig = model.plot_posterior(params=["rho"])
    assert fig is not None

    fig = model.plot_autocorr(params=["home_advantage"])
    assert fig is not None

    fig = model.plot_convergence()
    assert fig is not None

    fig = model.plot_diagnostics()
    assert fig is not None
