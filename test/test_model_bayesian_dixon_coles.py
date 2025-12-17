import numpy as np
import pandas as pd
import pytest

import penaltyblog as pb
from penaltyblog.models import BayesianDixonColesModel


@pytest.fixture
def data():
    data = {
        "goals_home": [1, 2, 0, 1, 3],
        "goals_away": [0, 1, 0, 2, 1],
        "team_home": ["A", "B", "C", "A", "C"],
        "team_away": ["B", "C", "A", "C", "B"],
    }
    return pd.DataFrame(data)


def test_initialization(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    assert model.n_teams == 3
    assert set(model.teams) == {"A", "B", "C"}
    assert not model.fitted


def test_param_names(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    names = model._get_param_names()
    expected = (
        [f"attack_{t}" for t in model.teams]
        + [f"defense_{t}" for t in model.teams]
        + ["home_advantage", "rho"]
    )
    assert names == expected


def test_unpack_params(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    # 3 teams: 3 attack, 3 defense, 1 hfa, 1 rho = 8 params
    params = np.arange(8, dtype=float)
    unpacked = model._unpack_params(params)

    np.testing.assert_array_equal(unpacked["attack"], [0, 1, 2])
    np.testing.assert_array_equal(unpacked["defense"], [3, 4, 5])
    assert unpacked["hfa"] == 6
    assert unpacked["rho"] == 7


def test_get_initial_params(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    init_params = model._get_initial_params()
    assert len(init_params) == 2 * 3 + 2
    assert init_params[-1] == -0.1


def test_fit_integration(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    n_walkers = 16
    n_steps = 45
    model.fit(n_walkers=n_walkers, n_steps=n_steps, n_burn=0)
    assert model.fitted

    assert model.chain.shape[0] > 0
    assert model.chain.shape[1] == 8

    assert model.aic is not None
    assert model.loglikelihood is not None


def test_predict_unfitted_raises(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )

    with pytest.raises(ValueError, match="fitted"):
        model.predict("A", "B")


def test_predict(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    model.fitted = True
    model.n_teams = 3
    model.chain = np.zeros((10, 8))
    model.chain[:, :] = 0.01

    probs = model.predict("A", "B")
    assert isinstance(probs, pb.models.FootballProbabilityGrid)
    assert probs.home_win > 0
    assert probs.away_win > 0
    assert probs.draw > 0


def test_repr_unfitted(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    assert "Status: Model not fitted" in str(model)


def test_repr_fitted(data):
    model = BayesianDixonColesModel(
        data["goals_home"], data["goals_away"], data["team_home"], data["team_away"]
    )
    model.fitted = True
    model.aic = 10.0
    model.waic = 12.0
    model.p_waic = 5.0
    model.loglikelihood = -5.0
    model.n_params = 8
    model._params = np.zeros(8)

    r = str(model)
    assert "Model: Bayesian Dixon-Coles" in r
    assert "Log Likelihood: -5.0" in r
    assert "AIC: 10.0" in r
