import numpy as np
import pytest

from penaltyblog.bayes.sampler_api import EnsembleSampler


def log_prob_inf(params, data):
    """Test that the sampler handles -inf and +inf gracefully."""
    # params is a memoryview, convert to numpy for convenience
    params_np = np.asarray(params)
    if params_np[0] > 1.0:
        return -np.inf
    if params_np[0] < -1.0:
        return np.inf
    return 0.0


def log_prob_nan(params, data):
    """Test that the sampler handles NaN returning from log_prob_func."""
    params_np = np.asarray(params)
    if params_np[0] > 0.5:
        return np.nan
    return 0.0


def log_prob_simple(params, data):
    """Test with extreme parameter values."""
    params_np = np.asarray(params)
    return -0.5 * np.sum(params_np**2)


def log_prob_extreme(params, data):
    """Test with extreme log-probability values."""
    params_np = np.asarray(params)
    # Returns very large negative values
    return -1e150 * np.sum(params_np**2)


def test_sampler_handles_inf():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_inf, data_dict=data
    )

    # Start at 0, where it is 0.0
    start_positions = [np.zeros((4, 1))]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    # Should not contain values > 1.0 (where it is -inf)
    assert np.all(posterior <= 1.0)


def test_sampler_handles_nan():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_nan, data_dict=data
    )

    start_positions = [np.zeros((4, 1))]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    # Should ideally not move to regions where it's NaN
    assert np.all(posterior <= 0.5)


def test_sampler_extreme_parameters():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_simple, data_dict=data
    )

    # Start at very large values
    # Note: 1e150 might be too large and cause issues if someone tries to square it
    # But that's exactly what we want to test - how it handles such things.
    start_positions = [np.ones((4, 2)) * 1e150]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    assert not np.any(np.isnan(posterior))


def test_sampler_extreme_log_probs():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_extreme, data_dict=data
    )

    start_positions = [np.random.randn(4, 2)]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    assert not np.any(np.isnan(posterior))


def log_prob_plus_inf(params, data):
    """Test that the sampler handles +inf log-probabilities."""
    return np.inf


def test_sampler_handles_plus_inf():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_plus_inf, data_dict=data
    )

    start_positions = [np.zeros((4, 1))]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    assert not np.any(np.isnan(posterior))


def log_prob_small(params, data):
    """Test with very small parameter values."""
    params_np = np.asarray(params)
    return -0.5 * np.sum(params_np**2) * 1e150


def test_sampler_small_log_probs():
    data = {}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_small, data_dict=data
    )

    # Start at very small values
    start_positions = [np.random.randn(4, 2) * 1e-150]
    sampler.run_mcmc(start_positions, n_samples=20, burn=0)

    posterior = sampler.get_posterior()
    assert not np.any(np.isnan(posterior))


def log_prob_high_dim(params, data):
    """Test with high dimensions."""
    params_np = np.asarray(params)
    return -0.5 * np.sum(params_np**2)


def test_sampler_high_dim():
    data = {"mu": np.zeros(100)}
    sampler = EnsembleSampler(
        n_chains=1, n_cores=1, log_prob_wrapper_func=log_prob_high_dim, data_dict=data
    )
    start_positions = [np.random.randn(200, 100)]
    sampler.run_mcmc(start_positions, n_samples=10, burn=0)

    posterior = sampler.get_posterior()
    assert posterior.shape == (10 * 200, 100)
