import multiprocessing

import numpy as np

from .sampler import MinimalSampler


# -----------------------------------------------------------------------------
# WORKER PROXY (Must be top-level for Windows Multiprocessing)
# -----------------------------------------------------------------------------
def _worker_proxy(chain_instance):
    """
    Multiprocessing helper.
    Receives a 'dormant' Chain object, runs it, and returns the 'completed' Chain object.
    """
    # We call the internal run method of the chain instance
    # This runs inside the subprocess
    chain_instance._execute()
    return chain_instance


# -----------------------------------------------------------------------------
# CHAIN CLASS (The Worker)
# -----------------------------------------------------------------------------
class Chain:
    """
    Represents a single independent MCMC chain (ensemble).
    """

    def __init__(self, id, seed, start_pos, data_dict, log_prob_wrapper_func, n_steps):
        # Configuration
        self.id = id
        self.seed = seed
        self.start_pos = start_pos  # Shape: (nwalkers, ndim)
        self.data = data_dict
        self.log_prob_func = log_prob_wrapper_func
        self.n_steps = n_steps

        # Results (initially None)
        self.raw_trace = None
        self.accept_rate = None

    def _execute(self):
        """
        Internal method run by the subprocess.
        Instantiates the Cython engine and performs sampling.
        """
        nwalkers, ndim = self.start_pos.shape

        # 1. Define the specific log_prob for this process
        # (Re-binding data to the function inside the process)
        def specific_log_prob(params):
            return self.log_prob_func(params, self.data)

        # 2. Instantiate Cython Engine
        sampler = MinimalSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_func=self.log_prob_func,  # Pass the original generic wrapper
            data_obj=self.data,  # Pass the data dict explicitly
            seed=self.seed,
        )

        # 3. Run (50/50 split of Stretch vs DE moves is robust)
        # Note: We run the full duration (burn + sample) here.
        chain, _, _ = sampler.run_mcmc(
            self.start_pos, self.n_steps, de_move_fraction=0.75
        )

        # 4. Store results
        self.raw_trace = chain  # Shape: (steps, walkers, dim)
        return self

    def get_samples(self, burn, thin):
        """Returns flattened samples for this specific chain."""
        if self.raw_trace is None:
            raise ValueError("Chain has not been run yet.")

        # Slice burn-in and thin
        kept = self.raw_trace[burn::thin, :, :]

        # Flatten (Steps * Walkers, Dims)
        return kept.reshape(-1, kept.shape[2])


# -----------------------------------------------------------------------------
# ENSEMBLE SAMPLER CLASS (The Manager)
# -----------------------------------------------------------------------------
class EnsembleSampler:
    """
    Manages parallel execution of independent MCMC chains.
    """

    def __init__(self, n_chains, n_cores, log_prob_wrapper_func, data_dict):
        self.n_chains = n_chains
        self.n_cores = n_cores
        self.log_prob_func = log_prob_wrapper_func
        self.data = data_dict
        self.chains = []

    def run_mcmc(self, start_positions, n_samples, burn):
        """
        Run the sampling.

        start_positions: list of arrays, one per chain.
                         Each array shape: (nwalkers, ndim)
        """
        total_steps = n_samples + burn

        # 1. Create the Chain objects (Dormant)
        tasks = []
        for i in range(self.n_chains):
            seed = np.random.randint(0, 1000000)

            chain = Chain(
                id=i,
                seed=seed,
                start_pos=start_positions[i],
                data_dict=self.data,
                log_prob_wrapper_func=self.log_prob_func,
                n_steps=total_steps,
            )
            tasks.append(chain)

        # 2. Parallel Execution
        with multiprocessing.Pool(processes=self.n_cores) as pool:
            self.chains = pool.map(_worker_proxy, tasks)

    def get_posterior(self, burn, thin=1) -> np.ndarray:
        """
        Aggregates samples from all chains.
        """
        if not self.chains:
            raise RuntimeError("Run run_mcmc first.")

        combined = []
        for chain in self.chains:
            combined.append(chain.get_samples(burn, thin))

        return np.vstack(combined)
