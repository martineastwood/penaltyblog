from __future__ import annotations

import logging
import multiprocessing
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .sampler import DiffEvolEnsembleSampler

logger = logging.getLogger(__name__)

# Choose the best multiprocessing context for the platform
# On Windows, we must use 'spawn'.
# On macOS (Python 3.8+), spawn is the default due to fork safety issues.
# On Linux, we prefer 'fork' for performance.
# NOTE: Context creation is delayed until first use to avoid import-time issues
_MP_CONTEXT = None
_MP_CONTEXT_LOCK = threading.Lock()


def _get_mp_context():
    """Get or create the multiprocessing context (lazy initialization).

    This is done lazily to avoid issues with module import in spawned processes.
    Thread-safe to prevent race conditions during concurrent access.
    """
    global _MP_CONTEXT
    if _MP_CONTEXT is None:
        with _MP_CONTEXT_LOCK:
            # Double-check after acquiring lock
            if _MP_CONTEXT is None:
                if sys.platform == "win32":
                    _MP_CONTEXT = multiprocessing.get_context("spawn")
                elif sys.platform == "darwin":
                    # macOS: Use spawn (required in Python 3.8+ for safety)
                    _MP_CONTEXT = multiprocessing.get_context("spawn")
                else:
                    # Linux and others: try fork, fall back to spawn
                    try:
                        _MP_CONTEXT = multiprocessing.get_context("fork")
                    except ValueError:
                        _MP_CONTEXT = multiprocessing.get_context("spawn")
    return _MP_CONTEXT


def _is_function_picklable(func: Callable) -> bool:
    """Check if a function can be pickled.

    This is important for spawn mode on Windows/macOS where functions must be picklable.
    """
    import pickle

    try:
        pickle.dumps(func)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


# -----------------------------------------------------------------------------
# WORKER PROXY (Must be top-level for Windows Multiprocessing)
# -----------------------------------------------------------------------------
def _worker_proxy(chain_instance: "Chain") -> "Chain":
    """Multiprocessing helper.

    Receives a 'dormant' Chain object, runs it, and returns the 'completed' Chain object.

    Args:
        chain_instance: The dormant chain instance to run.

    Returns:
        The completed chain instance.

    Raises:
        RuntimeError: If chain execution fails.
    """
    try:
        # Verify the chain instance is properly initialized
        if chain_instance is None:
            raise RuntimeError("chain_instance is None")

        # We call the internal run method of the chain instance
        # This runs inside the subprocess
        chain_instance._execute()
        return chain_instance
    except Exception as e:
        error_msg = (
            f"Error in chain {chain_instance.id}: {str(e)}\n{traceback.format_exc()}"
        )
        logger.error(error_msg)
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
        de_move_fraction: float = 0.8,
        validate_picklable: bool = False,
    ):
        """Initializes the Chain.

        Args:
            id: Unique identifier.
            seed: Random seed.
            start_pos: Shape (n_walkers, n_dim).
            data_dict: The data to be used in likelihood.
            log_prob_wrapper_func: The likelihood function wrapper. Must be picklable on
                Windows/macOS when using multiprocessing (n_cores > 1).
            n_steps: Number of steps to run.
            de_move_fraction: Fraction of DE moves. Defaults to 0.8.
            validate_picklable: If True, validate function is picklable. Used internally
                when multiprocessing will be used.

        Raises:
            TypeError: If validate_picklable=True and function is not picklable.
        """
        # Validate function is picklable only when multiprocessing will be used
        # This avoids rejecting valid test code that uses lambdas with n_cores=1
        if validate_picklable and not _is_function_picklable(log_prob_wrapper_func):
            raise TypeError(
                "log_prob_wrapper_func must be picklable when using multiprocessing "
                "(n_cores > 1) on Windows/macOS. Use a top-level function instead of "
                f"a lambda or local function. Received: {type(log_prob_wrapper_func).__name__}"
            )

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
            n_chains: Number of independent chains.
            n_cores: Number of CPU cores to use. Use 1 for single-threaded execution.
            log_prob_wrapper_func: The likelihood function wrapper. Must be picklable on
                Windows/macOS when n_cores > 1.
            data_dict: The data to be used in likelihood.

        Raises:
            ValueError: If `n_chains` is not positive or `n_cores` is invalid.
        """
        if n_chains <= 0:
            raise ValueError("n_chains must be greater than 0")

        # Safely get CPU count with fallback for restricted environments
        try:
            max_cores = multiprocessing.cpu_count()
        except (NotImplementedError, OSError):
            # Fallback for platforms where cpu_count() is not available or fails
            max_cores = 1

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
        de_move_fraction: float = 0.8,
    ) -> None:
        """Run the sampling across multiple chains in parallel.

        Args:
            start_positions: List of arrays, one per chain.
                Each array shape: (n_walkers, n_dim).
            n_samples: Number of samples to collect (post-burn).
            burn: Number of burn-in steps to discard.
            de_move_fraction: Fraction of DE moves. Defaults to 0.8.

        Raises:
            ValueError: If `n_samples` or `burn` are not positive, or if `start_positions`
                length doesn't match `n_chains`.
            RuntimeError: If chain execution fails or timeout occurs.
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

        # Create SeedSequence with proper entropy for better randomization
        ss = np.random.SeedSequence(np.random.randint(2**32, dtype=np.uint32))
        child_seeds = ss.generate_state(self.n_chains)
        # Mask to 32-bit integer to avoid OverflowError in Cython
        child_seeds = (child_seeds & 0x7FFFFFFF).astype(child_seeds.dtype)

        # Validate picklable when using multiprocessing on Windows/macOS
        validate_picklable = self.n_cores > 1 and sys.platform in ("win32", "darwin")

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
                validate_picklable=validate_picklable,
            )
            tasks.append(chain)

        # Run chains: use multiprocessing for n_cores > 1, otherwise run directly
        # This avoids spawn mode overhead and potential deadlocks on Windows/macOS
        if self.n_cores == 1:
            # Run sequentially in the main process (no multiprocessing overhead)
            try:
                self.chains = [_worker_proxy(chain) for chain in tasks]
            except Exception as e:
                # Maintain consistent error handling with multiprocessing path
                raise RuntimeError(f"Chain execution failed: {e}") from e
        else:
            # Calculate optimal chunksize for better load balancing
            # For uniform tasks (MCMC chains), use larger chunks for efficiency
            chunksize = max(1, len(tasks) // (2 * self.n_cores))

            # Use multiprocessing for parallel execution
            # On Windows/macOS (spawn mode), we need explicit pool cleanup to avoid hangs
            pool = None
            try:
                mp_context = _get_mp_context()
                pool = mp_context.Pool(processes=self.n_cores)

                # Use imap with no timeout per task, but wrap in try/except for safety
                self.chains = list(pool.imap(_worker_proxy, tasks, chunksize=chunksize))
            except Exception as e:
                # Ensure cleanup happens on error
                if pool is not None:
                    pool.terminate()
                raise RuntimeError(f"Multiprocessing pool failed: {e}") from e
            finally:
                # Explicitly close and join the pool to ensure clean termination
                # This is critical on Windows/macOS to prevent test hangs
                if pool is not None:
                    try:
                        pool.close()
                        pool.join()
                    except Exception as e:
                        # If normal cleanup fails, terminate forcefully
                        logger.error(f"Pool cleanup failed: {e}, terminating")
                        pool.terminate()
                        raise

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
