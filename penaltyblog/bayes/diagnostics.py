import numpy as np
import pandas as pd

from .sampler import DiffEvolEnsembleSampler


def compute_diagnostics(
    sampler: DiffEvolEnsembleSampler, burn=0, thin=1, rho_threshold=0.05
):
    """
    Computes R-hat and ESS for a fitted DiffEvolEnsembleSampler.

    Parameters
    ----------
    sampler : DiffEvolEnsembleSampler
        The fitted sampler instance containing the chains.
    burn : int
        Number of steps to discard from the beginning.
    thin : int
        Thinning factor.
    rho_threshold : float, optional
        Autocorrelation threshold for ESS calculation. The sum of autocorrelations
        is truncated when rho falls below this value. Default is 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'r_hat', 'ess' and 'autocorr' for each parameter.
    """
    if not sampler.chains:
        raise ValueError("Sampler has not been run yet.")

    first_trace = sampler.chains[0].raw_trace[burn::thin, :, :]
    n_samples = first_trace.shape[0]
    n_params = first_trace.shape[2]
    n_chains = sum(chain.raw_trace.shape[1] for chain in sampler.chains)

    print(f"Diagnostics: Analyzing {n_chains} chains of {n_samples} samples...")

    r_hats = []
    ess_vals = []
    tau_vals = []

    for p in range(n_params):
        # Allocate one parameter matrix rather than duplicating the complete
        # posterior. The row ordering matches the previous full-trace stack.
        y = np.vstack([chain.raw_trace[burn::thin, :, p].T for chain in sampler.chains])

        # r-hat
        r = _gelman_rubin(y)
        r_hats.append(r)

        # ess and autocorrelation
        ess, tau = _effective_sample_size(y, rho_threshold)
        ess_vals.append(ess)
        tau_vals.append(tau)

    df = pd.DataFrame({"r_hat": r_hats, "ess": ess_vals, "autocorr": tau_vals})

    return df


def _gelman_rubin(chain_data):
    """
    Calculates the classic Gelman-Rubin statistic (R-hat).
    chain_data: (n_chains, n_samples)
    """
    m, n = chain_data.shape
    if m < 2:
        return np.nan

    chain_means = np.mean(chain_data, axis=1)
    B = n * np.var(chain_means, ddof=1)

    chain_vars = np.var(chain_data, axis=1, ddof=1)
    W = np.mean(chain_vars)

    var_plus = ((n - 1) / n) * W + (B / n)
    if W == 0:
        return np.nan

    r_hat = np.sqrt(var_plus / W)
    return r_hat


def _effective_sample_size(chain_data, rho_threshold=0.05):
    """
    Approximates ESS using autocorrelation.

    Parameters
    ----------
    chain_data : np.ndarray
        Array of shape (n_chains, n_samples) containing chain traces.
    rho_threshold : float, optional
        Autocorrelation threshold for truncating the sum. Default is 0.05.

    Returns
    -------
    ess : float
        Effective sample size.
    tau : float
        Integrated autocorrelation time.
    """
    m, n = chain_data.shape

    # We compute autocorrelation for each chain and average them
    rhos = np.zeros(n)

    for i in range(m):
        rhos += _autocorr(chain_data[i, :])

    rhos /= m  # Average autocorrelation profile

    # Sum autocorrelations until they drop below zero (Geyer's method approx)
    # ESS = (M * N) / (1 + 2 * sum(rho))

    # Find cutoff where rho goes negative or noise dominates
    cutoff = n - 1
    for k in range(1, n):
        if rhos[k] < rho_threshold:
            cutoff = k
            break

    sum_rho = np.sum(rhos[1:cutoff])
    tau = 1 + 2 * sum_rho

    ess = (m * n) / tau
    return ess, tau


def _autocorr(x):
    """
    Compute autocorrelation function using FFT for speed.
    """
    n = len(x)
    # Pad to avoid circular correlation effects
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]  # Normalize
    return acf
