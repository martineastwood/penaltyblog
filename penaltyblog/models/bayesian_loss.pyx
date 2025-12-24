# penaltyblog/models/bayesian_loss.pyx

import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport INFINITY, exp, fabs, log

from .loss cimport dixon_coles_loss_function


# Pre-computed constants for log-PDF calculations
cdef double LOG_SQRT_2_PI = 0.9189385332046727

# Keep Sigma at 1.0 (Standard normal prior)
cdef double PRIOR_SIGMA = 1.0
cdef double LOG_PRIOR_CONST = 0.9189385332046727 # Same as LOG_SQRT_2_PI for sigma=1.0

# Pre-calculated log(0.5) + LOG_SQRT_2_PI for global parameters (sigma=0.5)
cdef double GLOBAL_PRIOR_CONST = 0.2257913526447274

# Constraint: Strong enough to keep Sum~0, but not so strong it kills emcee.
# 100,000 is the pragmatic sweet spot.
cdef double CONSTRAINT_STRENGTH = 10000.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef double _bayesian_dixon_coles_log_prob_c(
    double[:] params,
    long[:] home_idx,
    long[:] away_idx,
    long[:] goals_home,
    long[:] goals_away,
    double[:] weights,
    int n_teams
) nogil:
    cdef double log_prob = 0.0
    cdef Py_ssize_t i

    cdef double[:] attack = params[0 : n_teams]
    cdef double[:] defense = params[n_teams : 2 * n_teams]
    cdef double hfa = params[2 * n_teams]
    cdef double rho = params[2 * n_teams + 1]

    # Accumulators for priors and the sum-to-zero constraint
    cdef double ss_attack = 0.0
    cdef double ss_defense = 0.0
    cdef double sum_attack = 0.0
    cdef double sum_defense = 0.0

    # 1. PHYSICAL BOUNDS CHECK (CRITICAL SAFETY)
    # You must keep this. If rho explores > 1.0, the loss function crashes.
    if rho <= -1.0 or rho >= 1.0:
        return -INFINITY

    # 2. PRIORS & CONSTRAINT ACCUMULATION
    for i in range(n_teams):
        # Accumulate values for prior calculation
        ss_attack += attack[i] * attack[i]
        ss_defense += defense[i] * defense[i]

        # Track sums for the constraint
        sum_attack += attack[i]
        sum_defense += defense[i]

    # Combine priors into log_prob
    # Prior: Normal(0, 1.0) -> -0.5 * x^2 - log(sqrt(2*pi))
    log_prob += -0.5 * (ss_attack + ss_defense) - (2 * n_teams * LOG_PRIOR_CONST)

    # 3. SOFT SUM-TO-ZERO CONSTRAINT
    # Penalizes the model if the average rating drifts away from 0.0
    log_prob -= 0.5 * CONSTRAINT_STRENGTH * (sum_attack * sum_attack)
    log_prob -= 0.5 * CONSTRAINT_STRENGTH * (sum_defense * sum_defense)

    # 4. GLOBAL PARAMETER PRIORS
    # Home Advantage ~ Normal(0.25, 0.5) -> -2.0 * (hfa - 0.25)^2 - CONST
    log_prob += -2.0 * ((hfa - 0.25) * (hfa - 0.25)) - GLOBAL_PRIOR_CONST

    # Rho ~ Normal(0, 0.5) -> -2.0 * rho^2 - CONST
    log_prob += -2.0 * (rho * rho) - GLOBAL_PRIOR_CONST

    # 5. LIKELIHOOD CALCULATION
    cdef double neg_ll = dixon_coles_loss_function(
        goals_home,
        goals_away,
        weights,
        home_idx,
        away_idx,
        attack,
        defense,
        hfa,
        rho
    )

    return log_prob - neg_ll

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def bayesian_dixon_coles_log_prob(
    double[:] params,
    long[:] home_idx,
    long[:] away_idx,
    long[:] goals_home,
    long[:] goals_away,
    double[:] weights,
    int n_teams
):
    """
    Calculates the unnormalized log posterior (log prior + log likelihood).
    Reuse the highly optimized likelihood from loss.pyx.
    """
    cdef double result
    with nogil:
        result = _bayesian_dixon_coles_log_prob_c(
            params,
            home_idx,
            away_idx,
            goals_home,
            goals_away,
            weights,
            n_teams
        )
    return result


def bayesian_hierarchical_log_prob(
    double[:] params,
    long[:] home_idx,
    long[:] away_idx,
    long[:] goals_home,
    long[:] goals_away,
    double[:] weights,
    int n_teams
):
    """
    Calculates the log posterior for the Hierarchical model entirely in Cython.
    Performs Non-Centered Parameterization transformation inline.
    """
    cdef double log_prob = 0.0
    cdef Py_ssize_t i

    cdef int offset_start_def = n_teams
    cdef int idx_mu_att = 2 * n_teams
    cdef int idx_log_sigma_att = 2 * n_teams + 1
    cdef int idx_mu_def = 2 * n_teams + 2
    cdef int idx_log_sigma_def = 2 * n_teams + 3
    cdef int idx_hfa = 2 * n_teams + 4
    cdef int idx_rho = 2 * n_teams + 5

    cdef double mu_att = params[idx_mu_att]
    cdef double log_sigma_att = params[idx_log_sigma_att]
    cdef double mu_def = params[idx_mu_def]
    cdef double log_sigma_def = params[idx_log_sigma_def]
    cdef double hfa = params[idx_hfa]
    cdef double rho = params[idx_rho]


    cdef double sigma_att = exp(log_sigma_att)
    cdef double sigma_def = exp(log_sigma_def)

    if rho <= -1.0 or rho >= 1.0:
        return -np.inf

    if fabs(mu_att) > 5.0 or fabs(mu_def) > 5.0:
        return -np.inf
    if log_sigma_att < -5.0 or log_sigma_att > 2.0:
        return -np.inf
    if log_sigma_def < -5.0 or log_sigma_def > 2.0:
        return -np.inf

    log_prob += -0.5 * mu_att**2 - LOG_SQRT_2_PI
    log_prob += -0.5 * mu_def**2 - LOG_SQRT_2_PI

    # Hyperparameters: Sigma ~ Exponential(1)
    # pdf(x) = lambda * exp(-lambda * x). With lambda=1 -> exp(-sigma)
    log_prob += -sigma_att
    log_prob += -sigma_def

    # HFA ~ N(0.25, 0.5)
    log_prob += -0.5 * ((hfa - 0.25) / 0.5)**2 - (log(0.5) + LOG_SQRT_2_PI)

    # Rho ~ N(0, 0.5)
    log_prob += -0.5 * (rho / 0.5)**2 - (log(0.5) + LOG_SQRT_2_PI)

    # Offsets ~ N(0, 1)
    # We iterate and sum their priors
    for i in range(2 * n_teams):
        log_prob += -0.5 * params[i]**2 - LOG_SQRT_2_PI

    cdef double[:] real_attack = np.empty(n_teams, dtype=np.float64)
    cdef double[:] real_defense = np.empty(n_teams, dtype=np.float64)

    for i in range(n_teams):
        real_attack[i] = mu_att + sigma_att * params[i]
        real_defense[i] = mu_def + sigma_def * params[offset_start_def + i]

    cdef double neg_ll = dixon_coles_loss_function(
        goals_home,
        goals_away,
        weights,
        home_idx,
        away_idx,
        real_attack,
        real_defense,
        hfa,
        rho
    )

    if not np.isfinite(neg_ll):
        return -np.inf

    return log_prob - neg_ll
