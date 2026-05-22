import numpy as np
import pytest

import penaltyblog as pb
from penaltyblog.bayes.likelihood import (
    football_log_prob_wrapper,
    hierarchical_log_prob_wrapper,
)

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


# --- Bayesian models -------------------------------------------------------
# The Bayesian models reach the Cython likelihood through a log-prob wrapper
# rather than the _loss_function/_gradient methods, so they are exercised
# separately. n_tail is the count of trailing params (hfa, rho [, sigmas]).
BAYESIAN_MODELS = [
    (pb.models.BayesianGoalModel, football_log_prob_wrapper, 2),
    (pb.models.HierarchicalBayesianGoalModel, hierarchical_log_prob_wrapper, 4),
]


def _bayes_data(model, neutral_venue):
    return {
        "home_idx": model.home_idx,
        "away_idx": model.away_idx,
        "goals_home": model.goals_home,
        "goals_away": model.goals_away,
        "weights": model.weights,
        "neutral_venue": neutral_venue,
        "n_teams": model.n_teams,
    }


def _bayes_params(n_teams, n_tail):
    params = np.zeros(2 * n_teams + n_tail, dtype=np.float64)
    params[2 * n_teams] = 0.3  # home advantage
    params[2 * n_teams + 1] = -0.1  # rho
    if n_tail == 4:
        params[2 * n_teams + 2] = 0.5  # sigma_attack
        params[2 * n_teams + 3] = 0.5  # sigma_defence
    return params


@pytest.mark.parametrize("Model,wrapper,n_tail", BAYESIAN_MODELS)
def test_bayesian_logprob_responds_to_neutral_venue(Model, wrapper, n_tail, match_data):
    """Flagging matches neutral must change the Bayesian log-probability, since
    the home advantage term drops out of those matches' likelihood."""
    m = Model(*match_data["args"])
    n = match_data["n"]
    params = _bayes_params(m.n_teams, n_tail)

    lp_home = wrapper(params, _bayes_data(m, np.zeros(n, dtype=np.int64)))
    lp_neutral = wrapper(params, _bayes_data(m, np.ones(n, dtype=np.int64)))

    assert np.isfinite(lp_home) and np.isfinite(lp_neutral)
    assert not np.isclose(lp_home, lp_neutral)


@pytest.mark.parametrize("Model,wrapper,n_tail", BAYESIAN_MODELS)
def test_bayesian_omitted_none_and_zeros_equivalent(Model, wrapper, n_tail, match_data):
    """Omitting neutral_venue, passing None, or all-zeros must yield the same
    log-probability — the backwards-compatibility guarantee."""
    args = match_data["args"]
    zeros = np.zeros(match_data["n"], dtype=np.int64)
    models = [
        Model(*args),
        Model(*args, neutral_venue=None),
        Model(*args, neutral_venue=zeros),
    ]
    params = _bayes_params(models[0].n_teams, n_tail)
    lp = [wrapper(params, _bayes_data(m, m.neutral_venue)) for m in models]
    assert lp[0] == lp[1] == lp[2]


@pytest.mark.parametrize("Model,wrapper,n_tail", BAYESIAN_MODELS)
def test_bayesian_all_neutral_fit_converges(Model, wrapper, n_tail, match_data):
    """A short MCMC run with every match neutral must complete — this also
    confirms fit() threads neutral_venue into the sampler's data dict — and
    home advantage must be pinned to 0 since it is unidentified."""
    model = Model(*match_data["args"], neutral_venue=np.ones(match_data["n"], dtype=np.int64))
    model.fit(n_samples=80, burn=20, n_chains=2)
    assert model.fitted
    assert model.get_params()["home_advantage"] == 0.0


# --- Prediction side -------------------------------------------------------


@pytest.mark.parametrize("Model", MODELS)
def test_predict_neutral_venue_drops_home_advantage(Model, match_data):
    """A neutral-venue prediction excludes home advantage, lowering the home
    team's expected goals — the synthetic data carries a positive home edge."""
    model = Model(*match_data["args"])
    model.fit()
    home = model.predict("team_0", "team_1")
    neutral = model.predict("team_0", "team_1", neutral_venue=True)
    assert neutral.home_goal_expectation < home.home_goal_expectation


@pytest.mark.parametrize("Model", MODELS)
def test_predict_many_applies_per_fixture_neutral_venue(Model, match_data):
    """predict_many honours a per-fixture neutral_venue array — the same
    fixture predicted neutral has a lower home expectation than non-neutral."""
    model = Model(*match_data["args"])
    model.fit()
    grids = model.predict_many(
        ["team_0", "team_0"], ["team_1", "team_1"], neutral_venue=[0, 1]
    )
    assert grids[1].home_goal_expectation < grids[0].home_goal_expectation


@pytest.mark.parametrize("Model", MODELS)
def test_all_neutral_fit_pins_home_advantage_to_zero(Model, match_data):
    """When every training match is neutral, home advantage is unidentified;
    the fit must pin it to exactly 0 rather than leave an arbitrary value."""
    model = Model(
        *match_data["args"], neutral_venue=np.ones(match_data["n"], dtype=np.int64)
    )
    model.fit()
    assert model.get_params()["home_advantage"] == 0.0


@pytest.mark.parametrize("Model,wrapper,n_tail", BAYESIAN_MODELS)
def test_bayesian_predict_neutral_venue_drops_home_advantage(
    Model, wrapper, n_tail, match_data
):
    """A neutral-venue Bayesian prediction excludes home advantage."""
    model = Model(*match_data["args"])
    model.fit(n_samples=150, burn=50, n_chains=2)
    home = model.predict("team_0", "team_1")
    neutral = model.predict("team_0", "team_1", neutral_venue=True)
    assert neutral.home_goal_expectation < home.home_goal_expectation


@pytest.mark.parametrize("Model,wrapper,n_tail", BAYESIAN_MODELS)
def test_bayesian_all_neutral_prediction_ignores_home_advantage(
    Model, wrapper, n_tail, match_data
):
    """After fitting on all-neutral data the home advantage column of the
    posterior trace is zeroed, so a prediction is identical whether or not the
    fixture is flagged neutral."""
    model = Model(
        *match_data["args"], neutral_venue=np.ones(match_data["n"], dtype=np.int64)
    )
    model.fit(n_samples=80, burn=20, n_chains=2)
    home = model.predict("team_0", "team_1", neutral_venue=False)
    neutral = model.predict("team_0", "team_1", neutral_venue=True)
    assert np.array_equal(home.grid, neutral.grid)
