import numpy as np
import pytest

import penaltyblog as pb

MODELS = [
    pb.models.PoissonGoalsModel,
    pb.models.DixonColesGoalModel,
    pb.models.NegativeBinomialGoalModel,
    pb.models.ZeroInflatedPoissonGoalsModel,
    pb.models.BivariatePoissonGoalModel,
    pb.models.WeibullCopulaGoalsModel,
]

# WeibullCopula's optimiser is not bit-reproducible run-to-run (a pre-existing
# trait, unrelated to neutral_venue), so its backwards-compat check compares
# within a tolerance rather than requiring exact equality.
NON_DETERMINISTIC = {pb.models.WeibullCopulaGoalsModel}


@pytest.fixture(scope="module")
def match_data():
    """Deterministic synthetic fixtures so the suite needs no network access."""
    rng = np.random.default_rng(42)
    n_teams, n_matches = 10, 240
    teams = [f"team_{i}" for i in range(n_teams)]
    home = rng.choice(teams, n_matches)
    away = rng.choice(teams, n_matches)
    keep = home != away
    home, away = home[keep], away[keep]
    n = len(home)
    return {
        "args": (rng.poisson(1.5, n), rng.poisson(1.1, n), home, away),
        "n": n,
        "n_teams": n_teams,
    }


def _gradient_fn(model):
    """WeibullCopula names its gradient method differently from the other models."""
    return getattr(model, "_gradient", None) or model._gradient_function


@pytest.mark.parametrize("Model", MODELS)
def test_omitted_none_and_zeros_fit_identically(Model, match_data):
    """Omitting neutral_venue, passing None, or passing all-zeros must be equivalent
    — this is the backwards-compatibility guarantee for existing callers."""
    args = match_data["args"]
    zeros = np.zeros(match_data["n"], dtype=np.int64)

    m_omit = Model(*args)
    m_omit.fit()
    m_none = Model(*args, neutral_venue=None)
    m_none.fit()
    m_zeros = Model(*args, neutral_venue=zeros)
    m_zeros.fit()

    if Model in NON_DETERMINISTIC:
        assert np.allclose(m_omit._params, m_none._params, atol=1e-3)
        assert np.allclose(m_omit._params, m_zeros._params, atol=1e-3)
    else:
        assert np.array_equal(m_omit._params, m_none._params)
        assert np.array_equal(m_omit._params, m_zeros._params)


@pytest.mark.parametrize("Model", MODELS)
def test_neutral_venue_changes_the_loss(Model, match_data):
    """A non-zero home advantage must change the likelihood; flagging every match as
    neutral removes that term, so the loss must differ from the all-home case."""
    args = match_data["args"]
    n = match_data["n"]
    m_zeros = Model(*args, neutral_venue=np.zeros(n, dtype=np.int64))
    m_ones = Model(*args, neutral_venue=np.ones(n, dtype=np.int64))

    params = m_zeros._params.copy()
    params[m_zeros._get_tail_param_indices()["home_advantage"]] = 0.3

    assert not np.isclose(
        m_zeros._loss_function(params), m_ones._loss_function(params)
    )


@pytest.mark.parametrize("Model", MODELS)
def test_neutral_matches_contribute_zero_home_advantage_gradient(Model, match_data):
    """The core invariant: neutral matches must add nothing to the home advantage
    gradient (so it is estimated only from genuine home games), while team strength
    gradients keep flowing from every match."""
    args = match_data["args"]
    n = match_data["n"]
    m_zeros = Model(*args, neutral_venue=np.zeros(n, dtype=np.int64))
    m_ones = Model(*args, neutral_venue=np.ones(n, dtype=np.int64))

    hfa_idx = m_zeros._get_tail_param_indices()["home_advantage"]
    params = m_zeros._params.copy()
    params[hfa_idx] = 0.3

    grad_zeros = _gradient_fn(m_zeros)(params)
    grad_ones = _gradient_fn(m_ones)(params)

    assert not np.isclose(grad_zeros[hfa_idx], 0.0)
    assert np.isclose(grad_ones[hfa_idx], 0.0)
    assert not np.allclose(grad_ones[: 2 * match_data["n_teams"]], 0.0)


@pytest.mark.parametrize("Model", MODELS)
def test_all_neutral_fit_converges(Model, match_data):
    """Fitting with every match neutral must still converge."""
    args = match_data["args"]
    model = Model(*args, neutral_venue=np.ones(match_data["n"], dtype=np.int64))
    model.fit()
    assert model.fitted


@pytest.mark.parametrize("Model", MODELS)
def test_neutral_venue_length_mismatch_raises(Model, match_data):
    args = match_data["args"]
    bad = np.zeros(match_data["n"] - 1, dtype=np.int64)
    with pytest.raises(ValueError, match="same length"):
        Model(*args, neutral_venue=bad)


@pytest.mark.parametrize("Model", MODELS)
def test_neutral_venue_non_binary_value_raises(Model, match_data):
    args = match_data["args"]
    bad = np.zeros(match_data["n"], dtype=np.int64)
    bad[0] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        Model(*args, neutral_venue=bad)
