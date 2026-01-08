import numpy as np
import pytest

import penaltyblog as pb
from penaltyblog.models import BayesianGoalModel


def test_bayesian_automatic_burn_thinning():
    # Create some dummy data
    teams = ["TeamA", "TeamB", "TeamC"]
    n_games = 10
    goals_h = np.random.poisson(1.5, n_games)
    goals_a = np.random.poisson(1.0, n_games)
    teams_h = np.random.choice(teams, n_games)
    teams_a = np.random.choice(teams, n_games)

    # Ensure they are different
    mask = teams_h == teams_a
    teams_a[mask] = ["TeamC" if t == "TeamA" else "TeamA" for t in teams_h[mask]]

    model = BayesianGoalModel(goals_h, goals_a, teams_h, teams_a)

    n_samples = 100
    burn = 50
    thin = 2

    # Fit the model
    # Now using the one-liner internally
    model.fit(n_samples=n_samples, burn=burn, thin=thin, n_chains=2)

    expected_steps = n_samples // thin
    n_walkers = model.sampler.chains[0].start_pos.shape[0]
    expected_total_samples = expected_steps * n_walkers * 2

    assert model.trace.shape[0] == expected_total_samples

    # Verify raw traces are trimmed
    for chain in model.sampler.chains:
        assert chain.raw_trace.shape[0] == expected_steps

    # Verify diagnostics work without arguments
    df = model.get_diagnostics()
    assert df is not None
    assert "r_hat" in df.columns
    assert "ess" in df.columns
    assert not df.isnull().values.any()


if __name__ == "__main__":
    test_bayesian_automatic_burn_thinning()
