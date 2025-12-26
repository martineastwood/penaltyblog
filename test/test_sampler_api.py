import numpy as np
import pytest

from penaltyblog.bayes.sampler_api import Chain, EnsembleSampler


def dummy_log_prob(params, data):
    """Simple multivariate normal log-prob."""
    # (x - mu)^T Sigma^-1 (x - mu)
    mu = data["mu"]
    diff = params - mu
    return -0.5 * np.sum(diff**2)


def test_ensemble_sampler_validation():
    """Test input validation in EnsembleSampler."""
    data = {"mu": np.array([0.0, 0.0])}

    # Invalid n_chains
    with pytest.raises(ValueError, match="n_chains must be greater than 0"):
        EnsembleSampler(
            n_chains=0, n_cores=1, log_prob_wrapper_func=dummy_log_prob, data_dict=data
        )

    sampler = EnsembleSampler(
        n_chains=2, n_cores=1, log_prob_wrapper_func=dummy_log_prob, data_dict=data
    )

    # Invalid n_samples
    with pytest.raises(ValueError, match="n_samples must be greater than 0"):
        sampler.run_mcmc([np.zeros((4, 2)), np.zeros((4, 2))], n_samples=0, burn=10)

    # Invalid burn
    with pytest.raises(ValueError, match="burn must be non-negative"):
        sampler.run_mcmc([np.zeros((4, 2)), np.zeros((4, 2))], n_samples=10, burn=-1)

    # Start positions length mismatch
    with pytest.raises(ValueError, match="must match n_chains"):
        sampler.run_mcmc([np.zeros((4, 2))], n_samples=10, burn=10)


def test_sampler_execution():
    """Test that the sampler actually runs and returns results."""
    data = {"mu": np.array([0.0, 0.0])}
    n_chains = 2
    n_walkers = 10
    n_dim = 2
    n_samples = 50
    burn = 10

    sampler = EnsembleSampler(
        n_chains=n_chains,
        n_cores=1,
        log_prob_wrapper_func=dummy_log_prob,
        data_dict=data,
    )

    start_positions = [np.random.randn(n_walkers, n_dim) for _ in range(n_chains)]

    sampler.run_mcmc(start_positions, n_samples=n_samples, burn=burn)

    assert len(sampler.chains) == n_chains
    for chain in sampler.chains:
        assert chain.raw_trace.shape == (n_samples + burn, n_walkers, n_dim)

    posterior = sampler.get_posterior(burn=burn, thin=1)
    # Total samples = n_chains * n_walkers * n_samples
    assert posterior.shape == (n_chains * n_walkers * n_samples, n_dim)


def test_chain_get_samples_error():
    """Test that get_samples raises error if chain hasn't run."""
    chain = Chain(
        id=0,
        seed=123,
        start_pos=np.zeros((4, 2)),
        data_dict={},
        log_prob_wrapper_func=lambda x, y: 0.0,
        n_steps=10,
    )
    with pytest.raises(ValueError, match="Chain has not been run yet"):
        chain.get_samples(burn=0, thin=1)


def test_ensemble_sampler_posterior_error():
    """Test that get_posterior raises error if sampler hasn't run."""
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=lambda x, y: 0.0, data_dict={}
    )
    with pytest.raises(RuntimeError, match="Run run_mcmc first"):
        sampler.get_posterior(burn=0)
