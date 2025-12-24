# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, floor, log, sqrt
from libc.stdint cimport uint64_t

# -----------------------------------------------------------------------------
# HIGH PERFORMANCE PRNG (Xoshiro128++)
# -----------------------------------------------------------------------------
# Standard C rand() is unfit for MCMC. This custom struct gives us
# scientific-grade randomness directly in C-space.

cdef struct RngState:
    uint64_t s[4]

cdef inline uint64_t rotl(const uint64_t x, int k) nogil:
    return (x << k) | (x >> (64 - k))

cdef inline uint64_t next_uint64(RngState *state) nogil:
    cdef uint64_t result = rotl(state.s[0] + state.s[3], 23) + state.s[0]
    cdef uint64_t t = state.s[1] << 17
    state.s[2] ^= state.s[0]
    state.s[3] ^= state.s[1]
    state.s[1] ^= state.s[2]
    state.s[0] ^= state.s[3]
    state.s[2] ^= t
    state.s[3] = rotl(state.s[3], 45)
    return result

cdef inline double next_double(RngState *state) nogil:
    """Returns uniform float in [0, 1)"""
    # 53 bits of randomness for double precision
    return (next_uint64(state) >> 11) * (1.0 / 9007199254740992.0)


# -----------------------------------------------------------------------------
# SAMPLER ENGINE
# -----------------------------------------------------------------------------

# Typedef for the Python log-prob function wrapper
# It takes a generic double array and some data object (usually a dict)
ctypedef double (*log_prob_fn)(object, object)

cdef class MinimalSampler:
    cdef:
        int nwalkers
        int ndim
        double a_param
        object log_prob_func
        object data_obj  # Data dict passed to likelihood
        RngState rng

    def __init__(self, int nwalkers, int ndim, log_prob_func, object data_obj, double a=2.0, int seed=1234):
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.log_prob_func = log_prob_func
        self.data_obj = data_obj
        self.a_param = a

        # Seed the internal RNG using numpy to get initial entropy
        cdef uint64_t[::1] seeds = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)
        self.rng.s[0] = seeds[0]
        self.rng.s[1] = seeds[1]
        self.rng.s[2] = seeds[2]
        self.rng.s[3] = seeds[3]

    cdef inline double evaluate_log_prob(self, double[:] params) with gil:
        """
        Calls the Python log-prob wrapper.
        We MUST acquire GIL here because we are calling a Python function.
        """
        return self.log_prob_func(np.asarray(params), self.data_obj)

    def run_mcmc(self, double[:, ::1] initial_state, int nsteps, double de_move_fraction=0.5):
        """
        Run MCMC with mixed moves (Stretch + Differential Evolution).

        Parameters
        ----------
        initial_state : array (nwalkers, ndim)
        nsteps : int
        de_move_fraction : float
            Probability (0.0 to 1.0) of using DE move per step.
            0.5 is ideal for correlated parameters.
        """
        cdef:
            int step, k, j, j2, dim, split
            int half = self.nwalkers // 2
            int start_idx, end_idx, comp_start, comp_end, n_comp

            # Buffers
            double[:, :, ::1] chain = np.empty((nsteps, self.nwalkers, self.ndim))
            double[::1] current_log_prob = np.empty(self.nwalkers)

            # Temporary state variables
            double[:, ::1] state = initial_state.copy()
            double[:] q_new = np.empty(self.ndim)
            double ln_prob_new, zz, log_accept_ratio
            double gamma_de

            # DE Constant: Ideal scale is 2.38 / sqrt(2 * ndim)
            double de_scale = 2.38 / sqrt(2 * self.ndim)

        # 1. Initialize log probs for starting positions
        for k in range(self.nwalkers):
            current_log_prob[k] = self.evaluate_log_prob(state[k])

        # 2. Main Loop
        for step in range(nsteps):

            # Store chain
            for k in range(self.nwalkers):
                for dim in range(self.ndim):
                    chain[step, k, dim] = state[k, dim]

            # Red-Blue Split (Enables vectorization logic, though we loop serially here)
            for split in range(2):
                if split == 0:
                    start_idx = 0
                    end_idx = half
                    comp_start = half
                    comp_end = self.nwalkers
                else:
                    start_idx = half
                    end_idx = self.nwalkers
                    comp_start = 0
                    comp_end = half

                n_comp = comp_end - comp_start

                # Loop walkers in this half
                for k in range(start_idx, end_idx):

                    # --- CHOOSE MOVE TYPE ---
                    if next_double(&self.rng) < de_move_fraction:
                        # === DIFFERENTIAL EVOLUTION MOVE (Ridge Killer) ===
                        # x_new = x_old + gamma * (x_r1 - x_r2)

                        # Select two distinct random walkers from the complement
                        j = <int>(next_double(&self.rng) * n_comp) + comp_start
                        j2 = <int>(next_double(&self.rng) * n_comp) + comp_start
                        while j2 == j:
                            j2 = <int>(next_double(&self.rng) * n_comp) + comp_start

                        # Jitter gamma slightly
                        gamma_de = de_scale * (1.0 + 0.1 * (next_double(&self.rng) - 0.5))

                        for dim in range(self.ndim):
                            q_new[dim] = state[k, dim] + gamma_de * (state[j, dim] - state[j2, dim])

                        log_accept_ratio = 0.0 # Symmetric move

                    else:
                        # === STRETCH MOVE (Affine Invariant Standard) ===
                        # Select one random walker from complement
                        j = <int>(next_double(&self.rng) * n_comp) + comp_start

                        # z ~ 1/sqrt(z) in [1/a, a]
                        zz = ((self.a_param - 1.0) * next_double(&self.rng) + 1.0)
                        zz = zz * zz / self.a_param

                        for dim in range(self.ndim):
                            q_new[dim] = state[j, dim] + zz * (state[k, dim] - state[j, dim])

                        log_accept_ratio = (self.ndim - 1) * log(zz)

                    # --- METROPOLIS UPDATE ---
                    ln_prob_new = self.evaluate_log_prob(q_new)

                    # Handle safety bounds
                    if ln_prob_new == -np.inf:
                         log_accept_ratio = -np.inf
                    else:
                         log_accept_ratio += (ln_prob_new - current_log_prob[k])

                    # Accept/Reject
                    if log(next_double(&self.rng)) < log_accept_ratio:
                        current_log_prob[k] = ln_prob_new
                        for dim in range(self.ndim):
                            state[k, dim] = q_new[dim]

        return np.asarray(chain), np.asarray(state), np.asarray(current_log_prob)
