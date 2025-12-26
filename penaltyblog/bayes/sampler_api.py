import multiprocessing
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .sampler import DiffEvolEnsembleSampler


# -----------------------------------------------------------------------------
# WORKER PROXY (Must be top-level for Windows Multiprocessing)
# -----------------------------------------------------------------------------
def _worker_proxy(chain_instance: "Chain") -> "Chain":
    """Multiprocessing helper.

    Receives a 'dormant' Chain object, runs it, and returns the 'completed' Chain object.

    Args:
        chain_instance (Chain): The dormant chain instance to run.

    Returns:
        Chain: The completed chain instance.
    """
    try:
        # We call the internal run method of the chain instance
        # This runs inside the subprocess
        chain_instance._execute()
        return chain_instance
    except Exception as e:
        error_msg = (
            f"Error in chain {chain_instance.id}: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        raise RuntimeError(error_msg) from e


# -----------------------------------------------------------------------------
# CHAIN CLASS (The Worker)
# -----------------------------------------------------------------------------
class Chain:
    """Represents a single independent MCMC chain (ensemble).

    Attributes:
        id (int): Unique identifier for the chain.
        seed (int): Random seed for this chain.
        start_pos (np.ndarray): Starting positions for walkers.
        data (Dict[str, Any]): Data dictionary passed to the log-prob function.
        log_prob_func (Callable): Generic log-prob wrapper function.
        n_steps (int): Total number of MCMC steps.
        de_move_fraction (float): Fraction of Differential Evolution moves.
        raw_trace (Optional[np.ndarray]): The resulting MCMC trace.
    """

    def __init__(
        self,
        id: int,
        seed: int,
        start_pos: np.ndarray,
        data_dict: Dict[str, Any],
        log_prob_wrapper_func: Callable,
        n_steps: int,
        de_move_fraction: float = 0.75,
    ):
        """Initializes the Chain.

        Args:
            id (int): Unique identifier.
            seed (int): Random seed.
            start_pos (np.ndarray): Shape (n_walkers, n_dim).
            data_dict (Dict[str, Any]): The data to be used in likelihood.
            log_prob_wrapper_func (Callable): The likelihood function wrapper.
            n_steps (int): Number of steps to run.
            de_move_fraction (float, optional): Fraction of DE moves. Defaults to 0.75.
        """
        # Configuration
        self.id = id
        self.seed = seed
        self.start_pos = start_pos  # Shape: (nwalkers, ndim)
        self.data = data_dict
        self.log_prob_func = log_prob_wrapper_func
        self.n_steps = n_steps
        self.de_move_fraction = de_move_fraction

        # Results (initially None)
        self.raw_trace: Optional[np.ndarray] = None

    def _execute(self) -> "Chain":
        """Internal method run by the subprocess.

        Instantiates the Cython engine and performs sampling.

        Returns:
            Chain: Self with results stored in `raw_trace`.
        """
        nwalkers, ndim = self.start_pos.shape

        # 1. Instantiate Cython Engine
        sampler = DiffEvolEnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_func=self.log_prob_func,
            data_obj=self.data,
            seed=self.seed,
        )

        # 2. Run MCMC
        chain, _, _ = sampler.run_mcmc(
            self.start_pos, self.n_steps, de_move_fraction=self.de_move_fraction
        )

        # 3. Store results
        self.raw_trace = chain  # Shape: (steps, walkers, dim)
        return self

    def get_samples(self, burn: int, thin: int) -> np.ndarray:
        """Returns flattened samples for this specific chain.

        Args:
            burn (int): Number of initial samples to discard.
            thin (int): Thinning factor.

        Returns:
            np.ndarray: Flattened samples of shape (n_samples, n_dim).

        Raises:
            ValueError: If the chain has not been run yet.
        """
        if self.raw_trace is None:
            raise ValueError("Chain has not been run yet.")

        # Slice burn-in and thin
        kept = self.raw_trace[burn::thin, :, :]

        # Flatten (Steps * Walkers, Dims)
        return kept.reshape(-1, kept.shape[2])

    def trim_samples(self, burn: int, thin: int) -> np.ndarray:
        """Discard burn-in samples and apply thinning to the raw trace.

        Args:
            burn (int): Number of initial samples to discard.
            thin (int): Thinning factor.

        Returns:
            np.ndarray: The remaining flattened samples for this chain.

        Raises:
            ValueError: If the chain has not been run yet.
        """
        if self.raw_trace is None:
            raise ValueError("Chain has not been run yet.")

        self.raw_trace = self.raw_trace[burn::thin, :, :]
        return self.get_samples(burn=0, thin=1)


# -----------------------------------------------------------------------------
# ENSEMBLE SAMPLER CLASS (The Manager)
# -----------------------------------------------------------------------------
class EnsembleSampler:
    """Manages parallel execution of independent MCMC chains.

    Attributes:
        n_chains (int): Number of independent chains.
        n_cores (int): Number of CPU cores to use.
        log_prob_func (Callable): Generic log-prob wrapper function.
        data (Dict[str, Any]): Data dictionary passed to the log-prob function.
        chains (List[Chain]): List of Chain objects.
    """

    def __init__(
        self,
        n_chains: int,
        n_cores: int,
        log_prob_wrapper_func: Callable,
        data_dict: Dict[str, Any],
    ):
        """Initializes the EnsembleSampler.

        Args:
            n_chains (int): Number of independent chains.
            n_cores (int): Number of CPU cores to use.
            log_prob_wrapper_func (Callable): The likelihood function wrapper.
            data_dict (Dict[str, Any]): The data to be used in likelihood.

        Raises:
            ValueError: If `n_chains` is not positive or `n_cores` is invalid.
        """
        if n_chains <= 0:
            raise ValueError("n_chains must be greater than 0")

        max_cores = multiprocessing.cpu_count()
        if n_cores <= 0:
            n_cores = max_cores
        else:
            n_cores = min(n_cores, max_cores)

        self.n_chains = n_chains
        self.n_cores = n_cores
        self.log_prob_func = log_prob_wrapper_func
        self.data = data_dict
        self.chains: List[Chain] = []

    def run_mcmc(
        self,
        start_positions: List[np.ndarray],
        n_samples: int,
        burn: int,
        de_move_fraction: float = 0.75,
    ) -> None:
        """Run the sampling across multiple chains in parallel.

        Args:
            start_positions (List[np.ndarray]): List of arrays, one per chain.
                Each array shape: (n_walkers, n_dim).
            n_samples (int): Number of samples to collect (post-burn).
            burn (int): Number of burn-in steps to discard.
            de_move_fraction (float, optional): Fraction of DE moves. Defaults to 0.75.

        Raises:
            ValueError: If `n_samples` or `burn` are not positive, or if `start_positions`
                length doesn't match `n_chains`.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0")
        if burn < 0:
            raise ValueError("burn must be non-negative")
        if len(start_positions) != self.n_chains:
            raise ValueError(
                f"Length of start_positions ({len(start_positions)}) must match n_chains ({self.n_chains})"
            )

        total_steps = n_samples + burn

        ss = np.random.SeedSequence()
        child_seeds = ss.generate_state(self.n_chains)
        # Mask to 32-bit integer to avoid OverflowError in Cython
        child_seeds = child_seeds & 0x7FFFFFFF

        tasks = []
        for i in range(self.n_chains):
            chain = Chain(
                id=i,
                seed=int(child_seeds[i]),
                start_pos=start_positions[i],
                data_dict=self.data,
                log_prob_wrapper_func=self.log_prob_func,
                n_steps=total_steps,
                de_move_fraction=de_move_fraction,
            )
            tasks.append(chain)

        with multiprocessing.Pool(processes=self.n_cores) as pool:
            self.chains = pool.map(_worker_proxy, tasks)

    def get_posterior(self, burn: int = 0, thin: int = 1) -> np.ndarray:
        """Aggregates samples from all chains.

        Args:
            burn (int, optional): Number of steps to discard from each chain. Defaults to 0.
            thin (int, optional): Thinning factor. Defaults to 1.

        Returns:
            np.ndarray: Combined posterior samples of shape (n_total_samples, n_dim).

        Raises:
            RuntimeError: If `run_mcmc` has not been called yet.
            ValueError: If `burn` is negative.
        """
        if not self.chains:
            raise RuntimeError("Run run_mcmc first.")

        if burn < 0:
            raise ValueError("burn must be non-negative")

        combined = []
        for chain in self.chains:
            combined.append(chain.get_samples(burn, thin))

        return np.vstack(combined)

    def trim_samples(self, burn: int, thin: int = 1) -> np.ndarray:
        """Trims burn-in and applies thinning to all chains, returning the combined posterior.

        Args:
            burn (int): Number of steps to discard from each chain.
            thin (int, optional): Thinning factor. Defaults to 1.

        Returns:
            np.ndarray: Combined posterior samples of shape (n_total_samples, n_dim).

        Raises:
            RuntimeError: If `run_mcmc` has not been called yet.
            ValueError: If `burn` is negative.
        """
        if not self.chains:
            raise RuntimeError("Run run_mcmc first.")

        if burn < 0:
            raise ValueError("burn must be non-negative")

        for chain in self.chains:
            chain.trim_samples(burn, thin)

        return self.get_posterior(burn=0, thin=1)
