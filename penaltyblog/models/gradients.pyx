import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport exp, log

import scipy.special


# Inline wrapper for the Python psi (digamma) function.
cdef inline double my_psi(double x):
    return float(scipy.special.psi(x))

cdef extern from "math.h":
    double tgamma(double)
    double isnan(double)
    double isinf(double)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def poisson_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    cnp.ndarray[long, ndim=1] home_idx,
    cnp.ndarray[long, ndim=1] away_idx,
    cnp.ndarray[long, ndim=1] goals_home,
    cnp.ndarray[long, ndim=1] goals_away
):
    cdef int n_teams = attack.shape[0]
    cdef int n_games = home_idx.shape[0]

    # Allocate output arrays
    cdef cnp.ndarray[double, ndim=1] grad_attack = np.zeros(n_teams, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] grad_defence = np.zeros(n_teams, dtype=np.float64)
    cdef double grad_hfa = 0.0

    cdef int i
    cdef long h, a
    cdef double lambda_home, lambda_away

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]

        lambda_home = exp(attack[h] + defence[a] + hfa)
        lambda_away = exp(attack[a] + defence[h])

        grad_attack[h] += goals_home[i] - lambda_home
        grad_attack[a] += goals_away[i] - lambda_away
        grad_defence[a] += goals_home[i] - lambda_home
        grad_defence[h] += goals_away[i] - lambda_away
        grad_hfa += goals_home[i] - lambda_home

    return np.concatenate([grad_attack, grad_defence, [grad_hfa]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def dixon_coles_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    double rho,
    cnp.ndarray[long, ndim=1] home_idx,
    cnp.ndarray[long, ndim=1] away_idx,
    cnp.ndarray[long, ndim=1] goals_home,
    cnp.ndarray[long, ndim=1] goals_away,
    cnp.ndarray[double, ndim=1] weights
):
    cdef int n_teams = attack.shape[0]
    cdef int n_games = home_idx.shape[0]

    # Allocate gradient arrays
    cdef cnp.ndarray[double, ndim=1] grad_attack = np.zeros(n_teams, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] grad_defence = np.zeros(n_teams, dtype=np.float64)
    cdef double grad_hfa = 0.0
    cdef double grad_rho = 0.0

    cdef int i, h, a
    cdef double lambda_home, lambda_away, adj_factor
    cdef int k_home, k_away

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]

        lambda_home = exp(hfa + attack[h] + defence[a])
        lambda_away = exp(attack[a] + defence[h])

        k_home = goals_home[i]
        k_away = goals_away[i]

        # Compute attack and defense gradients
        grad_attack[h] += (k_home - lambda_home) * weights[i]
        grad_attack[a] += (k_away - lambda_away) * weights[i]
        grad_defence[a] += (k_home - lambda_home) * weights[i]
        grad_defence[h] += (k_away - lambda_away) * weights[i]
        grad_hfa += (k_home - lambda_home) * weights[i]

        # Compute rho gradient only for low-score matches
        adj_factor = 0.0
        if k_home == 0 and k_away == 0:
            adj_factor = -lambda_home * lambda_away / (1 - rho * lambda_home * lambda_away)
        elif k_home == 0 and k_away == 1:
            adj_factor = lambda_home / (1 + rho * lambda_home)
        elif k_home == 1 and k_away == 0:
            adj_factor = lambda_away / (1 + rho * lambda_away)
        elif k_home == 1 and k_away == 1:
            adj_factor = -1 / (1 - rho)

        grad_rho += adj_factor * weights[i]

    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_rho]])

cdef double neg_binom_logpmf(int k, double r, double p):
    """
    Compute the log-PMF of the Negative Binomial distribution safely.
    """
    if p <= 0 or p >= 1 or r <= 0:
        return -100.0  # Return a small value instead of -1e308
    return log(tgamma(k + r)) - log(tgamma(r)) - log(tgamma(k + 1)) + r * log(p) + k * log(1 - p)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def negative_binomial_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    double dispersion,
    cnp.ndarray[long, ndim=1] home_idx,
    cnp.ndarray[long, ndim=1] away_idx,
    cnp.ndarray[long, ndim=1] goals_home,
    cnp.ndarray[long, ndim=1] goals_away,
    cnp.ndarray[double, ndim=1] weights
):
    cdef int n_teams = attack.shape[0]
    cdef int n_games = home_idx.shape[0]

    # Allocate gradient arrays for attack, defence, and scalars for hfa and dispersion.
    cdef cnp.ndarray[double, ndim=1] grad_attack = np.zeros(n_teams, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] grad_defence = np.zeros(n_teams, dtype=np.float64)
    cdef double grad_hfa = 0.0
    cdef double grad_dispersion = 0.0

    cdef int i, h, a, k_home, k_away
    cdef double lambda_home, lambda_away
    cdef double grad_home, grad_away
    cdef double term_disp_home, term_disp_away

    # Ensure dispersion is at least a small positive value.
    if dispersion < 1e-5:
        dispersion = 1e-5

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]
        k_home = goals_home[i]
        k_away = goals_away[i]

        # Compute expected goals for home and away.
        lambda_home = exp(hfa + attack[h] + defence[a])
        lambda_away = exp(attack[a] + defence[h])

        # For parameters that affect lambda, the derivative (for one observation) is:
        #   ∂(-ℓ)/∂u = (dispersion + k)*lambda/(dispersion+lambda) - k.
        grad_home = ((dispersion + k_home) * lambda_home / (dispersion + lambda_home) - k_home) * weights[i]
        grad_away = ((dispersion + k_away) * lambda_away / (dispersion + lambda_away) - k_away) * weights[i]

        # Accumulate gradients for home match:
        grad_attack[h] += grad_home
        grad_defence[a] += grad_home
        grad_hfa += grad_home
        # For away match:
        grad_attack[a] += grad_away
        grad_defence[h] += grad_away

        # For dispersion, the derivative for one observation is:
        #   ∂(-ℓ)/∂dispersion = -ψ(k+dispersion) + ψ(dispersion)
        #                        - ln(dispersion/(dispersion+lambda)) - 1
        #                        + (dispersion+k)/(dispersion+lambda)
        term_disp_home = - my_psi(k_home + dispersion) + my_psi(dispersion) \
                         - log(dispersion / (dispersion + lambda_home)) - 1.0 \
                         + (dispersion + k_home) / (dispersion + lambda_home)
        term_disp_away = - my_psi(k_away + dispersion) + my_psi(dispersion) \
                         - log(dispersion / (dispersion + lambda_away)) - 1.0 \
                         + (dispersion + k_away) / (dispersion + lambda_away)
        grad_dispersion += (term_disp_home + term_disp_away) * weights[i]

    # Return concatenated gradient: [grad_attack, grad_defence, grad_hfa, grad_dispersion].
    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_dispersion]])
