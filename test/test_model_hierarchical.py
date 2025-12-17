import numpy as np
import pandas as pd
import pytest

import penaltyblog as pb
from penaltyblog.models import BayesianHierarchicalModel


@pytest.fixture
def data():
    # Create a small synthetic dataset for testing
    data = {
        "goals_home": [1, 2, 0, 1, 3],
        "goals_away": [0, 1, 0, 2, 1],
        "team_home": ["A", "B", "C", "A", "C"],
        "team_away": ["B", "C", "A", "C", "B"],
    }
    return pd.DataFrame(data)


def test_initialization(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    assert model.n_teams == 3
    assert set(model.teams) == {"A", "B", "C"}
    assert not model.fitted
    assert len(model._params) == 2 * 3 + 6


def test_param_names(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    names = model._get_param_names()
    expected = (
        [f"attack_offset_{t}" for t in model.teams]
        + [f"defense_offset_{t}" for t in model.teams]
        + [
            "mu_attack",
            "log_sigma_attack",
            "mu_defense",
            "log_sigma_defense",
            "home_advantage",
            "rho",
        ]
    )
    assert names == expected


def test_unpack_params(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    nt = 3

    params = np.arange(2 * nt + 6, dtype=float)
    params[2 * nt + 1] = 0.0  # log_sigma_att
    params[2 * nt + 3] = 0.0  # log_sigma_def

    unpacked = model._unpack_params(params)

    epsilon = 1e-6
    sigma = 1.0 + epsilon

    # Check raw values
    assert unpacked["mu_attack"] == params[2 * nt]
    assert np.isclose(unpacked["sigma_attack"], sigma)
    assert unpacked["hfa"] == params[2 * nt + 4]

    expected_attack = params[2 * nt] + sigma * params[:nt]
    np.testing.assert_allclose(unpacked["attack"], expected_attack)


def test_fit_integration(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    n_walkers = 24
    n_steps = 45
    model.fit(n_walkers=n_walkers, n_steps=n_steps, n_burn=0)
    assert model.fitted

    assert model.chain.shape[0] > 0
    assert model.chain.shape[1] == 12

    assert model.aic is not None
    assert model.loglikelihood is not None


def test_predict_unfitted_raises(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )

    with pytest.raises(ValueError, match="fitted"):
        model.predict("A", "B")


def test_predict(data):
    model = BayesianHierarchicalModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    model.fitted = True
    model.n_teams = 3
    model.chain = np.zeros((10, 12))
    model.chain[:, :] = 0.01
    model.chain[:, :] = 0.01

    probs = model.predict("A", "B")
    assert isinstance(probs, pb.models.FootballProbabilityGrid)
    assert probs.home_win > 0
    assert probs.away_win > 0
    assert probs.draw > 0
