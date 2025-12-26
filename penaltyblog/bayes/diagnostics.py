import numpy as np
import pandas as pd


def compute_diagnostics(sampler, burn=0, thin=1):
    """
    Computes R-hat and ESS for a fitted EnsembleSampler.

    Parameters
    ----------
    sampler : EnsembleSampler
        The fitted sampler instance containing the chains.
    burn : int
        Number of steps to discard from the beginning.
    thin : int
        Thinning factor.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'r_hat', 'ess' and 'autocorr' for each parameter.
    """
    if not sampler.chains:
        raise ValueError("Sampler has not been run yet.")

    # 1. Aggregating Chains
    # We treat every walker in every process as a distinct chain.
    # Structure: List of (steps, walkers, ndim) -> (total_chains, steps, ndim)

    extracted = []
    for chain in sampler.chains:
        # trace shape: (n_steps, n_walkers, ndim)
        # We slice burn-in and transpose to (n_walkers, n_steps, ndim)
        clean = chain.raw_trace[burn::thin, :, :]
        transposed = np.moveaxis(clean, 0, 1)
        extracted.append(transposed)

    # Stack all processes together
    # Shape: (total_walkers, n_samples, ndim)
    full_trace = np.vstack(extracted)

    n_chains, n_samples, n_params = full_trace.shape

    # 2. Calculate Diagnostics per parameter
    print(f"Diagnostics: Analyzing {n_chains} chains of {n_samples} samples...")

    r_hats = []
    ess_vals = []
    tau_vals = []

    for p in range(n_params):
        y = full_trace[:, :, p]

        # r-hat
        r = _gelman_rubin(y)
        r_hats.append(r)

        # ess and autocorrelation
        ess, tau = _effective_sample_size(y)
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

    # 1. Calculate Between-Chain Variance (B)
    # Mean of each chain
    chain_means = np.mean(chain_data, axis=1)
    # Grand mean
    grand_mean = np.mean(chain_means)
    # B = n * Variance of chain means
    B = n * np.var(chain_means, ddof=1)

    # 2. Calculate Within-Chain Variance (W)
    # Variance of each chain
    chain_vars = np.var(chain_data, axis=1, ddof=1)
    W = np.mean(chain_vars)

    # 3. Pooled Variance
    # Var_hat = (n-1)/n * W + 1/n * B
    var_plus = ((n - 1) / n) * W + (B / n)

    # 4. R-hat
    # sqrt(Var_hat / W)
    if W == 0:
        return 0.0

    r_hat = np.sqrt(var_plus / W)
    return r_hat


def _effective_sample_size(chain_data):
    """
    Approximates ESS using autocorrelation.
    chain_data: (n_chains, n_samples)
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
        if rhos[k] < 0.05:  # Threshold near zero
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
