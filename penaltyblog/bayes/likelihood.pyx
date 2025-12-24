# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, exp, fmax, lgamma, log, pow

import cython


# Small epsilon to prevent log(0)
cdef double epsilon = 1e-10

# -----------------------------------------------------------------------------
# MODEL 1. BAYESIAN DIXON AND COLES
# -----------------------------------------------------------------------------

cdef double dixon_coles_neg_ll_c(
    long[:] goals_home,
    long[:] goals_away,
    double[:] weights,
    long[:] home_indices,
    long[:] away_indices,
    double[:] attack,
    double[:] defence,
    double hfa,
    double rho
) nogil:
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away, adjustment
    cdef int home_idx, away_idx, k_home, k_away

    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]

        # Expected goals
        lambda_home = exp(hfa + attack[home_idx] + defence[away_idx])
        lambda_away = exp(attack[away_idx] + defence[home_idx])

        k_home = goals_home[i]
        k_away = goals_away[i]

        # Poisson log-likelihood
        llk_home = -lambda_home + k_home * log(lambda_home) - lgamma(k_home + 1)
        llk_away = -lambda_away + k_away * log(lambda_away) - lgamma(k_away + 1)

        # Dixon-Coles adjustment
        if k_home == 0 and k_away == 0:
            adjustment = log(fmax(epsilon, 1 - rho * lambda_home * lambda_away))
        elif k_home == 0 and k_away == 1:
            adjustment = log(fmax(epsilon, 1 + rho * lambda_home))
        elif k_home == 1 and k_away == 0:
            adjustment = log(fmax(epsilon, 1 + rho * lambda_away))
        elif k_home == 1 and k_away == 1:
            adjustment = log(fmax(epsilon, 1 - rho))
        else:
            adjustment = 0.0

        total_llk += ((llk_home + llk_away) + adjustment) * weights[i]

    # Return NEGATIVE Log Likelihood (Lower is better for optimizers)
    return -total_llk


# -----------------------------------------------------------------------------
# 2. BAYESIAN WRAPPER (The Bridge)
# -----------------------------------------------------------------------------

def football_log_prob_wrapper(double[:] params, object data):
    """
    Wrapper to bridge Python dictionary data to Cython C-functions.
    """
    cdef long[:] home_idx = data['home_idx']
    cdef long[:] away_idx = data['away_idx']
    cdef long[:] goals_home = data['goals_home']
    cdef long[:] goals_away = data['goals_away']
    cdef double[:] weights = data['weights']
    cdef int n_teams = data['n_teams']

    return _bayesian_log_prob_c(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )

cdef double _bayesian_log_prob_c(
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

    # --- PARAMETER UNPACKING ---
    cdef double[:] attack = params[0 : n_teams]
    cdef double[:] defense = params[n_teams : 2 * n_teams]
    cdef double hfa = params[2 * n_teams]
    cdef double rho = params[2 * n_teams + 1]

    # --- 1. HARD BOUNDS ---
    if rho <= -1.0 or rho >= 1.0:
        return -INFINITY

    # --- 2. PRIORS ---

    # RELAXED PRIORS (Sigma = 10.0)
    # Variance = 100.0. Effectively flat/uninformative for football ratings.
    cdef double PRIOR_VAR = 100.0

    # Log Normalization Constant for Sigma=10.0
    # = -log(10 * sqrt(2*pi)) = -3.2215
    cdef double LOG_NORM_CONST = 3.2215

    cdef double sum_att = 0.0
    cdef double sum_def = 0.0
    cdef double ss_att = 0.0
    cdef double ss_def = 0.0

    for i in range(n_teams):
        sum_att += attack[i]
        sum_def += defense[i]
        ss_att += attack[i] * attack[i]
        ss_def += defense[i] * defense[i]

    # Add priors for all team stats (Centered at 0, Scale=10.0)
    log_prob += -0.5 * (ss_att / PRIOR_VAR) - (n_teams * LOG_NORM_CONST)
    log_prob += -0.5 * (ss_def / PRIOR_VAR) - (n_teams * LOG_NORM_CONST)

    # Soft Constraint: Sum of Attack/Defense should be ~0
    # This keeps the model identifiable (prevents drift)
    log_prob += -0.5 * (sum_att * sum_att) * 1000.0
    log_prob += -0.5 * (sum_def * sum_def) * 1000.0

    # HFA Prior ~ Normal(0.25, 0.5)
    # Keeps HFA physically grounded
    log_prob += -((hfa - 0.25) * (hfa - 0.25)) / 0.5

    # Rho Prior ~ Normal(0, 1.0)
    # Relaxed from 0.5 to 1.0 to allow for typical Rho variations
    log_prob += -0.5 * (rho * rho)

    # --- 3. LIKELIHOOD ---
    cdef double neg_ll = dixon_coles_neg_ll_c(
        goals_home, goals_away, weights,
        home_idx, away_idx, attack, defense, hfa, rho
    )

    return log_prob - neg_ll

# -----------------------------------------------------------------------------
# 3. BAYESIAN PREDICTION (Posterior Integration)
# -----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bayesian_predict_c(
    double[:, :] trace,
    int home_idx,
    int away_idx,
    int n_teams,
    int max_goals
):
    """
    Computes the Posterior Predictive probability matrix.
    Averages the Dixon-Coles matrix over all MCMC samples.
    """
    cdef int n_samples = trace.shape[0]
    cdef int n_params = trace.shape[1]

    cdef int dim = max_goals
    cdef double[:, ::1] avg_matrix = np.zeros((dim, dim), dtype=np.float64)
    cdef double avg_lambda_home = 0.0
    cdef double avg_lambda_away = 0.0

    cdef int s, x, y
    cdef double att_h, att_a, def_h, def_a, hfa, rho
    cdef double lambda_h, lambda_a
    cdef double prob, adjustment, pois_h, pois_a, term

    with nogil:
        for s in range(n_samples):
            # Extract parameters
            att_h = trace[s, home_idx]
            att_a = trace[s, away_idx]
            def_h = trace[s, n_teams + home_idx]
            def_a = trace[s, n_teams + away_idx]
            hfa   = trace[s, n_params - 2]
            rho   = trace[s, n_params - 1]

            # Lambdas
            lambda_h = exp(hfa + att_h + def_a)
            lambda_a = exp(att_a + def_h)

            avg_lambda_home += lambda_h
            avg_lambda_away += lambda_a

            # Grid
            for x in range(dim):
                for y in range(dim):

                    # Poisson PDF
                    pois_h = exp(x * log(lambda_h) - lambda_h - lgamma(x + 1))
                    pois_a = exp(y * log(lambda_a) - lambda_a - lgamma(y + 1))

                    prob = pois_h * pois_a

                    # DC Adjustment
                    adjustment = 1.0
                    if x == 0 and y == 0:
                        adjustment = 1.0 - (rho * lambda_h * lambda_a)
                    elif x == 0 and y == 1:
                        adjustment = 1.0 + (rho * lambda_h)
                    elif x == 1 and y == 0:
                        adjustment = 1.0 + (rho * lambda_a)
                    elif x == 1 and y == 1:
                        adjustment = 1.0 - rho

                    term = prob * adjustment
                    if term < 0:
                        term = 0

                    avg_matrix[x, y] += term

    # Normalize
    cdef double[:, ::1] final_matrix = np.zeros((dim, dim), dtype=np.float64)
    for x in range(dim):
        for y in range(dim):
            final_matrix[x, y] = avg_matrix[x, y] / n_samples

    return (
        np.asarray(final_matrix),
        avg_lambda_home / n_samples,
        avg_lambda_away / n_samples
    )



# =============================================================================
# MODEL 2: HIERARCHICAL MODEL (Learned Priors)
# =============================================================================

def hierarchical_log_prob_wrapper(double[:] params, object data):
    """New wrapper for HierarchicalBayesianGoalModel"""
    cdef long[:] home_idx = data['home_idx']
    cdef long[:] away_idx = data['away_idx']
    cdef long[:] goals_home = data['goals_home']
    cdef long[:] goals_away = data['goals_away']
    cdef double[:] weights = data['weights']
    cdef int n_teams = data['n_teams']

    return _hierarchical_log_prob_c(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )

cdef double _hierarchical_log_prob_c(
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

    # Unpack including Sigmas at the end
    cdef double[:] attack = params[0 : n_teams]
    cdef double[:] defense = params[n_teams : 2 * n_teams]
    cdef double hfa = params[2 * n_teams]
    cdef double rho = params[2 * n_teams + 1]
    cdef double sigma_att = params[2 * n_teams + 2]
    cdef double sigma_def = params[2 * n_teams + 3]

    if rho <= -1.0 or rho >= 1.0: return -INFINITY
    if sigma_att <= 0.0 or sigma_def <= 0.0: return -INFINITY

    # Hyperpriors for Sigma (Weak Half-Normal)
    log_prob += -0.5 * (sigma_att * sigma_att)
    log_prob += -0.5 * (sigma_def * sigma_def)

    cdef double ss_att = 0.0
    cdef double ss_def = 0.0
    cdef double sum_att = 0.0
    cdef double sum_def = 0.0

    for i in range(n_teams):
        sum_att += attack[i]
        sum_def += defense[i]
        ss_att += attack[i] * attack[i]
        ss_def += defense[i] * defense[i]

    # Hierarchical Priors (Using dynamic sigma)
    # Log PDF = -n * log(sigma) - 0.5 * sum(x^2) / sigma^2
    log_prob += (-n_teams * log(sigma_att)) - (0.5 * ss_att / (sigma_att * sigma_att))
    log_prob += (-n_teams * log(sigma_def)) - (0.5 * ss_def / (sigma_def * sigma_def))

    # Constraints & Globals
    log_prob += -0.5 * (sum_att * sum_att) * 1000.0
    log_prob += -0.5 * (sum_def * sum_def) * 1000.0
    log_prob += -((hfa - 0.25) * (hfa - 0.25)) / 0.5
    log_prob += -0.5 * (rho * rho)

    cdef double neg_ll = dixon_coles_neg_ll_c(
        goals_home, goals_away, weights,
        home_idx, away_idx, attack, defense, hfa, rho
    )

    return log_prob - neg_ll
