import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport exp, log

import scipy.special

from .utils cimport (
    cdf_from_pmf,
    compute_pxy,
    precompute_alpha_table,
    weibull_count_pmf,
)


# Inline wrapper for the Python psi (digamma) function.
cdef inline double my_psi(double x):
    return float(scipy.special.psi(x))

cdef extern from "math.h":
    double tgamma(double)
    double lgamma(double)
    double isnan(double)
    double isinf(double)

# Inline wrapper for lgamma function
cdef inline double my_lgamma(double x):
    return lgamma(x)


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
def zero_inflated_poisson_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    double zero_inflation,
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
    cdef double grad_phi = 0.0

    cdef int i, h, a
    cdef long k_home, k_away
    cdef double w
    cdef double lambda_home, lambda_away
    cdef double grad_lambda_h_contrib, grad_lambda_a_contrib
    cdef double denom_h, denom_a

    # phi is zero_inflation
    cdef double phi = zero_inflation
    if phi <= 1e-5:
        phi = 1e-5
    if phi >= 1 - 1e-5:
        phi = 1 - 1e-5

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]
        k_home = goals_home[i]
        k_away = goals_away[i]
        w = weights[i]

        lambda_home = exp(attack[h] + defence[a] + hfa)
        lambda_away = exp(attack[a] + defence[h])

        # Home goal contribution to gradient
        if k_home == 0:
            denom_h = phi + (1 - phi) * exp(-lambda_home)
            grad_lambda_h_contrib = (1 - phi) * exp(-lambda_home) * lambda_home / denom_h
            grad_phi += -(1 - exp(-lambda_home)) / denom_h * w
        else:
            grad_lambda_h_contrib = lambda_home - k_home
            grad_phi += 1.0 / (1 - phi) * w

        grad_attack[h] += grad_lambda_h_contrib * w
        grad_defence[a] += grad_lambda_h_contrib * w
        grad_hfa += grad_lambda_h_contrib * w

        # Away goal contribution to gradient
        if k_away == 0:
            denom_a = phi + (1 - phi) * exp(-lambda_away)
            grad_lambda_a_contrib = (1 - phi) * exp(-lambda_away) * lambda_away / denom_a
            grad_phi += -(1 - exp(-lambda_away)) / denom_a * w
        else:
            grad_lambda_a_contrib = lambda_away - k_away
            grad_phi += 1.0 / (1 - phi) * w

        grad_attack[a] += grad_lambda_a_contrib * w
        grad_defence[h] += grad_lambda_a_contrib * w

    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_phi]])


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

        term_disp_home = - my_psi(k_home + dispersion) + my_psi(dispersion) \
                         - log(dispersion / (dispersion + lambda_home)) - 1.0 \
                         + (dispersion + k_home) / (dispersion + lambda_home)
        term_disp_away = - my_psi(k_away + dispersion) + my_psi(dispersion) \
                         - log(dispersion / (dispersion + lambda_away)) - 1.0 \
                         + (dispersion + k_away) / (dispersion + lambda_away)
        grad_dispersion += (term_disp_home + term_disp_away) * weights[i]

    # Return concatenated gradient: [grad_attack, grad_defence, grad_hfa, grad_dispersion].
    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_dispersion]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def bivariate_poisson_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    double correlation,
    cnp.ndarray[long, ndim=1] home_idx,
    cnp.ndarray[long, ndim=1] away_idx,
    cnp.ndarray[long, ndim=1] goals_home,
    cnp.ndarray[long, ndim=1] goals_away,
    cnp.ndarray[double, ndim=1] weights
):
    """
    Compute the gradient of the negative log-likelihood for the Bivariate Poisson model.

    This implements the Karlis & Ntzoufras bivariate Poisson model where:
    - X (home goals) = W1 + W3
    - Y (away goals) = W2 + W3
    - W1, W2, W3 ~ independent Poisson(λ1, λ2, λ3)
    - λ1 = exp(hfa + attack[home] + defence[away])
    - λ2 = exp(attack[away] + defence[home])
    - λ3 = exp(correlation)
    """
    cdef int n_teams = attack.shape[0]
    cdef int n_games = home_idx.shape[0]
    cdef double lambda3 = exp(correlation)

    # Allocate gradient arrays
    cdef cnp.ndarray[double, ndim=1] grad_attack = np.zeros(n_teams, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] grad_defence = np.zeros(n_teams, dtype=np.float64)
    cdef double grad_hfa = 0.0
    cdef double grad_correlation = 0.0

    cdef int i, h, a, k_home, k_away, k, kmax
    cdef double lambda1, lambda2, w
    cdef double likelihood_ij, grad_lambda1, grad_lambda2, grad_lambda3
    cdef double term_k, pmf1_k, pmf2_k, pmf3_k

    # Precompute maximum goals to avoid repeated calculation
    cdef int max_goals = 0
    for i in range(n_games):
        if goals_home[i] > max_goals:
            max_goals = goals_home[i]
        if goals_away[i] > max_goals:
            max_goals = goals_away[i]
    max_goals += 1  # Add 1 for 0-indexing

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]
        k_home = goals_home[i]
        k_away = goals_away[i]
        w = weights[i]

        # Compute lambdas for this match
        lambda1 = exp(hfa + attack[h] + defence[a])
        lambda2 = exp(attack[a] + defence[h])

        # Compute the likelihood for this observation
        likelihood_ij = 0.0
        kmax = k_home if k_home < k_away else k_away

        for k in range(kmax + 1):
            # Compute Poisson PMFs: P(W1 = k_home - k), P(W2 = k_away - k), P(W3 = k)
            pmf1_k = exp(-lambda1 + (k_home - k) * log(lambda1) - my_lgamma(k_home - k + 1))
            pmf2_k = exp(-lambda2 + (k_away - k) * log(lambda2) - my_lgamma(k_away - k + 1))
            pmf3_k = exp(-lambda3 + k * log(lambda3) - my_lgamma(k + 1))

            likelihood_ij += pmf1_k * pmf2_k * pmf3_k

        # Avoid division by zero
        if likelihood_ij < 1e-10:
            likelihood_ij = 1e-10

        # Compute gradients with respect to lambda1, lambda2, lambda3
        grad_lambda1 = 0.0
        grad_lambda2 = 0.0
        grad_lambda3 = 0.0

        for k in range(kmax + 1):
            pmf1_k = exp(-lambda1 + (k_home - k) * log(lambda1) - my_lgamma(k_home - k + 1))
            pmf2_k = exp(-lambda2 + (k_away - k) * log(lambda2) - my_lgamma(k_away - k + 1))
            pmf3_k = exp(-lambda3 + k * log(lambda3) - my_lgamma(k + 1))

            term_k = pmf1_k * pmf2_k * pmf3_k / likelihood_ij

            # ∂P(W1=k_home-k)/∂λ1 = ((k_home-k)/λ1 - 1) * P(W1=k_home-k)
            grad_lambda1 += term_k * ((k_home - k) / lambda1 - 1.0)

            # ∂P(W2=k_away-k)/∂λ2 = ((k_away-k)/λ2 - 1) * P(W2=k_away-k)
            grad_lambda2 += term_k * ((k_away - k) / lambda2 - 1.0)

            # ∂P(W3=k)/∂λ3 = (k/λ3 - 1) * P(W3=k)
            grad_lambda3 += term_k * (k / lambda3 - 1.0)

        # Apply chain rule: ∂(-log L)/∂param = -∂L/∂param / L
        # Since we computed ∂L/∂λ / L above, we multiply by -w for negative log-likelihood
        grad_lambda1 *= -w
        grad_lambda2 *= -w
        grad_lambda3 *= -w

        # Chain rule for parameters:
        # ∂λ1/∂attack[h] = λ1, ∂λ1/∂defence[a] = λ1, ∂λ1/∂hfa = λ1
        # ∂λ2/∂attack[a] = λ2, ∂λ2/∂defence[h] = λ2
        # ∂λ3/∂correlation = λ3

        grad_attack[h] += grad_lambda1 * lambda1
        grad_attack[a] += grad_lambda2 * lambda2
        grad_defence[a] += grad_lambda1 * lambda1
        grad_defence[h] += grad_lambda2 * lambda2
        grad_hfa += grad_lambda1 * lambda1
        grad_correlation += grad_lambda3 * lambda3

    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_correlation]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def weibull_copula_gradient(
    cnp.ndarray[double, ndim=1] attack,
    cnp.ndarray[double, ndim=1] defence,
    double hfa,
    double shape,
    double kappa,
    cnp.ndarray[long, ndim=1] home_idx,
    cnp.ndarray[long, ndim=1] away_idx,
    cnp.ndarray[long, ndim=1] goals_home,
    cnp.ndarray[long, ndim=1] goals_away,
    cnp.ndarray[double, ndim=1] weights,
    int max_goals
):
    """
    Compute the gradient of the negative log-likelihood for the Weibull Copula model.

    This model uses Weibull count distributions for marginals coupled with a Frank copula.
    The gradient computation involves:
    1. Computing Weibull PMFs and their derivatives w.r.t. lambda and shape
    2. Computing Frank copula derivatives w.r.t. kappa and marginal CDFs
    3. Applying chain rule for all parameters
    """
    cdef int n_teams = attack.shape[0]
    cdef int n_games = home_idx.shape[0]

    # Allocate gradient arrays
    cdef cnp.ndarray[double, ndim=1] grad_attack = np.zeros(n_teams, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] grad_defence = np.zeros(n_teams, dtype=np.float64)
    cdef double grad_hfa = 0.0
    cdef double grad_shape = 0.0
    cdef double grad_kappa = 0.0

    cdef int i, h, a, x_i, y_i
    cdef double lambda_home, lambda_away, w
    cdef double p_xy, grad_lambda_h, grad_lambda_a
    cdef double grad_p_xy_lambda_h, grad_p_xy_lambda_a, grad_p_xy_shape, grad_p_xy_kappa
    cdef double eps = 1e-6
    cdef double lambda_home_plus, lambda_away_plus, shape_plus, p_xy_plus
    cdef double grad_factor
    cdef list pmf_home, pmf_away, cdf_home, cdf_away
    cdef list pmf_home_plus, pmf_away_plus, cdf_home_plus, cdf_away_plus
    cdef list alpha_table_plus

    # Precompute alpha table for Weibull distribution
    if shape <= 0:
        # Return large gradients to push away from invalid region
        grad_shape = 1e6
        return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_shape, grad_kappa]])

    cdef list alpha_table = precompute_alpha_table(shape, max_goals)
    if alpha_table is None:
        grad_shape = 1e6
        return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_shape, grad_kappa]])

    for i in range(n_games):
        h = home_idx[i]
        a = away_idx[i]
        x_i = goals_home[i]
        y_i = goals_away[i]
        w = weights[i]

        # Compute expected goals
        lambda_home = exp(hfa + attack[h] + defence[a])
        lambda_away = exp(attack[a] + defence[h])

        # Compute Weibull PMFs and CDFs
        pmf_home = weibull_count_pmf(lambda_home, alpha_table, max_goals)
        pmf_away = weibull_count_pmf(lambda_away, alpha_table, max_goals)
        cdf_home = cdf_from_pmf(pmf_home)
        cdf_away = cdf_from_pmf(pmf_away)

        # Compute joint probability for observed outcome
        p_xy = compute_pxy(x_i, y_i, cdf_home, cdf_away, max_goals, kappa)
        if p_xy < 1e-10:
            p_xy = 1e-10

        # Compute gradients of p_xy w.r.t. parameters using finite differences
        # This is a numerical approximation due to the complexity of the Weibull-Frank copula

        # Gradient w.r.t. lambda_home
        lambda_home_plus = lambda_home + eps
        pmf_home_plus = weibull_count_pmf(lambda_home_plus, alpha_table, max_goals)
        cdf_home_plus = cdf_from_pmf(pmf_home_plus)
        p_xy_plus = compute_pxy(x_i, y_i, cdf_home_plus, cdf_away, max_goals, kappa)
        if p_xy_plus < 1e-10:
            p_xy_plus = 1e-10
        grad_p_xy_lambda_h = (p_xy_plus - p_xy) / eps

        # Gradient w.r.t. lambda_away
        lambda_away_plus = lambda_away + eps
        pmf_away_plus = weibull_count_pmf(lambda_away_plus, alpha_table, max_goals)
        cdf_away_plus = cdf_from_pmf(pmf_away_plus)
        p_xy_plus = compute_pxy(x_i, y_i, cdf_home, cdf_away_plus, max_goals, kappa)
        if p_xy_plus < 1e-10:
            p_xy_plus = 1e-10
        grad_p_xy_lambda_a = (p_xy_plus - p_xy) / eps

        # Gradient w.r.t. kappa
        p_xy_plus = compute_pxy(x_i, y_i, cdf_home, cdf_away, max_goals, kappa + eps)
        if p_xy_plus < 1e-10:
            p_xy_plus = 1e-10
        grad_p_xy_kappa = (p_xy_plus - p_xy) / eps

        # Gradient w.r.t. shape (more complex - affects alpha table)
        shape_plus = shape + eps
        if shape_plus > 0:
            alpha_table_plus = precompute_alpha_table(shape_plus, max_goals)
            if alpha_table_plus is not None:
                pmf_home_plus = weibull_count_pmf(lambda_home, alpha_table_plus, max_goals)
                pmf_away_plus = weibull_count_pmf(lambda_away, alpha_table_plus, max_goals)
                cdf_home_plus = cdf_from_pmf(pmf_home_plus)
                cdf_away_plus = cdf_from_pmf(pmf_away_plus)
                p_xy_plus = compute_pxy(x_i, y_i, cdf_home_plus, cdf_away_plus, max_goals, kappa)
                if p_xy_plus < 1e-10:
                    p_xy_plus = 1e-10
                grad_p_xy_shape = (p_xy_plus - p_xy) / eps
            else:
                grad_p_xy_shape = 0.0
        else:
            grad_p_xy_shape = 0.0

        # Apply chain rule: d(-log L)/dp_xy = -1/p_xy
        # Then d(-log L)/d(param) = d(-log L)/dp_xy * dp_xy/d(param)
        grad_factor = -w / p_xy

        # Update parameter gradients using chain rule
        # d(lambda)/d(param) = lambda for attack, defence, hfa parameters
        grad_attack[h] += grad_factor * grad_p_xy_lambda_h * lambda_home
        grad_attack[a] += grad_factor * grad_p_xy_lambda_a * lambda_away
        grad_defence[a] += grad_factor * grad_p_xy_lambda_h * lambda_home
        grad_defence[h] += grad_factor * grad_p_xy_lambda_a * lambda_away
        grad_hfa += grad_factor * grad_p_xy_lambda_h * lambda_home

        # Direct gradients for shape and kappa
        grad_shape += grad_factor * grad_p_xy_shape
        grad_kappa += grad_factor * grad_p_xy_kappa

    return np.concatenate([grad_attack, grad_defence, [grad_hfa, grad_shape, grad_kappa]])
