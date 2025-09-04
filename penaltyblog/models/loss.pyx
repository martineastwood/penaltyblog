import numpy as np

cimport numpy as np
from libc.math cimport exp, fabs, isinf, isnan, lgamma, log, tgamma

from cython.parallel import prange

cimport cython

from .utils cimport (
    cdf_from_pmf,
    compute_max_goal,
    compute_pxy,
    negBinomLogPMF,
    poisson_log_pmf,
    poisson_pmf,
    precompute_alpha_table,
    precompute_poisson_pmf,
    weibull_count_pmf,
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def poisson_loss_function(long[:] goals_home,
                          long[:] goals_away,
                          np.float64_t[:] weights,
                          long[:] home_indices,
                          long[:] away_indices,
                          np.float64_t[:] attack,
                          np.float64_t[:] defence,
                          double hfa) -> double:
    """
    Computes the negative log-likelihood for a Dixon–Coles model.

    Parameters:
      goals_home, goals_away: observed goals for home and away teams.
      weights: match weights.
      home_indices, away_indices: indices mapping each fixture to team parameters.
      attack, defence: team parameters.
      hfa: home advantage.

    Returns:
      The negative log-likelihood (a lower value indicates a better fit).
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away
    cdef int home_idx, away_idx, k_home, k_away

    # Main loop: optimized for performance
    with nogil:
        for i in prange(n):
            home_idx = home_indices[i]
            away_idx = away_indices[i]

            # Compute expected goals
            lambda_home = exp(hfa + attack[home_idx] + defence[away_idx])
            lambda_away = exp(attack[away_idx] + defence[home_idx])

            # Retrieve observed goals
            k_home = goals_home[i]
            k_away = goals_away[i]

            # Compute Poisson log-likelihood contributions
            llk_home = -lambda_home + k_home * log(lambda_home) - lgamma(k_home + 1)
            llk_away = -lambda_away + k_away * log(lambda_away) - lgamma(k_away + 1)

            total_llk += (llk_home + llk_away) * weights[i]

    return total_llk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double dixon_coles_loss_function(long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         double hfa,
                                         double rho):
    """
    Computes the negative log-likelihood for a Dixon–Coles model.

    Parameters:
      goals_home, goals_away: observed goals for home and away teams.
      weights: match weights.
      home_indices, away_indices: indices mapping each fixture to team parameters.
      attack, defence: team parameters.
      hfa: home advantage.
      rho: Dixon–Coles adjustment parameter.

    Returns:
      The negative log-likelihood (a lower value indicates a better fit).
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away, adjustment
    cdef int home_idx, away_idx, k_home, k_away

    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]

        # Compute expected goals using team parameters and home advantage.
        lambda_home = exp(hfa + attack[home_idx] + defence[away_idx])
        lambda_away = exp(attack[away_idx] + defence[home_idx])

        k_home = goals_home[i]
        k_away = goals_away[i]

        # Standard Poisson log-likelihood terms.
        llk_home = -lambda_home + k_home * log(lambda_home) - lgamma(k_home + 1)
        llk_away = -lambda_away + k_away * log(lambda_away) - lgamma(k_away + 1)

        # Dixon–Coles adjustment for low-scoring matches.
        if k_home == 0 and k_away == 0:
            adjustment = log(1 - rho * lambda_home * lambda_away)
        elif k_home == 0 and k_away == 1:
            adjustment = log(1 + rho * lambda_home)
        elif k_home == 1 and k_away == 0:
            adjustment = log(1 + rho * lambda_away)
        elif k_home == 1 and k_away == 1:
            adjustment = log(1 - rho)
        else:
            adjustment = 0.0

        total_llk += ((llk_home + llk_away) + adjustment) * weights[i]

    return -total_llk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_negative_binomial_loss(long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         double hfa,
                                         double dispersion):
    """
    Compute the negative log-likelihood for the Negative Binomial model.

    Parameters:
      params: Array of parameters of length (2*nTeams + 2), where the first nTeams
              elements are attack parameters, the next nTeams are defence parameters,
              followed by home advantage and dispersion.
      nTeams: Number of teams.
      homeIdx, awayIdx: Integer arrays (length nMatches) of team indices.
      goalsHome, goalsAway: Integer arrays (length nMatches) of goals scored.
      weights: A float array (length nMatches) of weights.
      nMatches: Number of matches.

    Returns:
      The negative log-likelihood.
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away, adjustment
    cdef int home_idx, away_idx, k_home, k_away

    # Ensure dispersion is at least 1e-5
    if dispersion < 1e-5:
        dispersion = 1e-5

    cdef double lambdaHome, lambdaAway, pHome, pAway
    cdef double logP_home, logP_away

    # Loop over each match
    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]
        # Attack parameters: first nTeams elements; Defence: next nTeams elements
        lambdaHome = exp(hfa + attack[home_idx] + defence[away_idx])
        lambdaAway = exp(attack[away_idx] + defence[home_idx])
        pHome = dispersion / (dispersion + lambdaHome)
        pAway = dispersion / (dispersion + lambdaAway)
        kHome = goals_home[i]
        kAway = goals_away[i]
        logP_home = negBinomLogPMF(kHome, dispersion, pHome)
        logP_away = negBinomLogPMF(kAway, dispersion, pAway)
        # If any computed log likelihood is NaN or Inf, return a very large number
        if isnan(logP_home) or isinf(logP_home) or isnan(logP_away) or isinf(logP_away):
            return 1e308  # Use a large number to indicate failure
        total_llk += (logP_home + logP_away) * weights[i]

    return -total_llk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_zero_inflated_poisson_loss(long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         double hfa,
                                         double zero_inflation):
    """
    Compute the negative log-likelihood for the Zero-Inflated Poisson (ZIP) model.

    Parameters:
      params: A 1D array of length (2*nTeams + 2) containing:
              - first nTeams: attack parameters,
              - next nTeams: defense parameters,
              - then: home advantage,
              - then: zero-inflation parameter.
      nTeams: Number of teams.
      homeIdx, awayIdx: Arrays of length nMatches mapping each match to a team index.
      goalsHome, goalsAway: Arrays of length nMatches containing observed goals.
      weights: Array of length nMatches with match weights.
      nMatches: Total number of matches.

    Returns:
      The negative log-likelihood.
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away, adjustment
    cdef int home_idx, away_idx, k_home, k_away

    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]

        lambda_home = exp(hfa + attack[home_idx] + defence[away_idx])
        lambda_away = exp(attack[away_idx] + defence[home_idx])

        k_home = goals_home[i]
        k_away = goals_away[i]

        # Compute ZIP log-likelihood for home team.
        if k_home == 0:
            total_llk += log(zero_inflation + (1 - zero_inflation) * exp(-lambda_home)) * weights[i]
        else:
            total_llk += (log(1 - zero_inflation) + poisson_log_pmf(k_home, lambda_home)) * weights[i]

        # Compute ZIP log-likelihood for away team.
        if k_away == 0:
            total_llk += log(zero_inflation + (1 - zero_inflation) * exp(-lambda_away)) * weights[i]
        else:
            total_llk += (log(1 - zero_inflation) + poisson_log_pmf(k_away, lambda_away)) * weights[i]

    return -total_llk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_bivariate_poisson_loss(long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         double hfa,
                                         double correlation):
    """
    Compute the negative log-likelihood for the Bivariate Poisson model.

    Parameters:
      params: 1D array of length (2*nTeams + 2) containing:
              - first nTeams: attack parameters,
              - next nTeams: defense parameters,
              - then: home advantage,
              - then: log(correlation) (last parameter).
      nTeams: Number of teams.
      homeIdx, awayIdx: Arrays (length nMatches) mapping each fixture to team indices.
      goalsHome, goalsAway: Arrays (length nMatches) with observed goals.
      weights: Array (length nMatches) with match weights.
      nMatches: Number of matches.

    Returns:
      The negative log-likelihood.
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double lambda3 = exp(correlation)

    # Determine maximum goal count (plus one)
    cdef int maxGoals = compute_max_goal(goals_home, goals_away, n)

    # Precompute the Poisson PMF vector for lambda3.
    cdef list lambda3PMF = precompute_poisson_pmf(lambda3, maxGoals)

    # Dictionary cache: maps from a lambda (double) to its precomputed PMF list.
    cdef dict pmfCache = {}

    cdef double logLikelihood = 0.0
    cdef int k, kMax, home_idx, away_idx
    cdef double lambda1, lambda2, likeIJ
    cdef list pmf1, pmf2

    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]

        # Compute lambda1 and lambda2 for the match.
        lambda1 = exp(hfa + attack[home_idx] + defence[away_idx])
        lambda2 = exp(attack[away_idx] + defence[home_idx])

        # Get or compute the PMF for lambda1.
        if lambda1 in pmfCache:
            pmf1 = pmfCache[lambda1]
        else:
            pmf1 = precompute_poisson_pmf(lambda1, maxGoals)
            pmfCache[lambda1] = pmf1

        # Get or compute the PMF for lambda2.
        if lambda2 in pmfCache:
            pmf2 = pmfCache[lambda2]
        else:
            pmf2 = precompute_poisson_pmf(lambda2, maxGoals)
            pmfCache[lambda2] = pmf2

        likeIJ = 0.0
        # kMax is the minimum of the two observed goals.
        kMax = goals_home[i] if goals_home[i] < goals_away[i] else goals_away[i]

        # Sum over k from 0 to kMax inclusive.
        for k in range(kMax + 1):
            likeIJ += pmf1[int(goals_home[i]) - k] * pmf2[int(goals_away[i]) - k] * lambda3PMF[k]

        # Avoid log(0) by enforcing a minimum.
        if likeIJ < 1e-10:
            likeIJ = 1e-10
        logLikelihood += weights[i] * log(likeIJ)

    return -logLikelihood


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_weibull_copula_loss(long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         double hfa,
                                         double shape,
                                         double kappa,
                                         int max_goals):
    """
    Computes the negative log-likelihood for the Weibull Count + Copula model.

    Parameters:
      params: 1D array of length (2*nTeams + 3) with:
              - first nTeams: attack parameters,
              - next nTeams: defense parameters,
              - then: home advantage,
              - then: shape parameter,
              - then: kappa (copula parameter).
      nTeams: Number of teams.
      homeIdx, awayIdx: Arrays (length nMatches) mapping each match to team indices.
      goalsHome, goalsAway: Arrays (length nMatches) of observed goals.
      weights: Array (length nMatches) of match weights.
      nMatches: Number of matches.
      maxGoals: Maximum number of goals to consider.

    Returns:
      The negative log-likelihood (to be minimized).
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef int home_index, away_index, x_i, y_i
    cdef double logLikelihood = 0.0
    cdef list alphaTable
    cdef double lambdaHome, lambdaAway, pXY

    if shape <= 0:
        return 1e15  # Penalize invalid shape

    alphaTable = precompute_alpha_table(shape, max_goals)
    if alphaTable is None:
        return 1e15

    # Loop over each match.
    for i in range(n):
        home_index = home_indices[i]
        away_index = away_indices[i]

        lambdaHome = exp(hfa + attack[home_index] + defence[away_index])
        lambdaAway = exp(attack[away_index] + defence[home_index])

        # Compute the Weibull PMFs.
        pmfH = weibull_count_pmf(lambdaHome, alphaTable, max_goals)
        pmfA = weibull_count_pmf(lambdaAway, alphaTable, max_goals)

        # Compute the corresponding CDFs.
        cdfH = cdf_from_pmf(pmfH)
        cdfA = cdf_from_pmf(pmfA)

        x_i = <int>goals_home[i]
        y_i = <int>goals_away[i]
        pXY = compute_pxy(x_i, y_i, cdfH, cdfA, max_goals, kappa)
        if pXY < 1e-10:
            pXY = 1e-10
        logLikelihood += weights[i] * log(pXY)

    return -logLikelihood


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double random_intercept_loss_function(
                                         long[:] goals_home,
                                         long[:] goals_away,
                                         np.float64_t[:] weights,
                                         long[:] home_indices,
                                         long[:] away_indices,
                                         np.float64_t[:] attack,
                                         np.float64_t[:] defence,
                                         np.float64_t[:] match_intercepts,
                                         double hfa,
                                         double rho):
    """
    Computes the negative log-likelihood for a Dixon–Coles model
    with a per-match random intercept.
    """
    cdef Py_ssize_t i, n = goals_home.shape[0]
    cdef double total_llk = 0.0
    cdef double lambda_home, lambda_away, llk_home, llk_away, adjustment
    cdef int home_idx, away_idx, k_home, k_away

    for i in range(n):
        home_idx = home_indices[i]
        away_idx = away_indices[i]

        # MODIFIED: Include the per-match random intercept in the lambda calculation.
        lambda_home = exp(hfa + attack[home_idx] + defence[away_idx] + match_intercepts[i])
        lambda_away = exp(attack[away_idx] + defence[home_idx] + match_intercepts[i])

        k_home = goals_home[i]
        k_away = goals_away[i]

        # Standard Poisson log-likelihood terms.
        llk_home = -lambda_home + k_home * log(lambda_home) - lgamma(k_home + 1)
        llk_away = -lambda_away + k_away * log(lambda_away) - lgamma(k_away + 1)

        # Dixon–Coles adjustment for low-scoring matches.
        if k_home == 0 and k_away == 0:
            adjustment = log(1 - rho * lambda_home * lambda_away)
        elif k_home == 0 and k_away == 1:
            adjustment = log(1 + rho * lambda_home)
        elif k_home == 1 and k_away == 0:
            adjustment = log(1 + rho * lambda_away)
        elif k_home == 1 and k_away == 1:
            adjustment = log(1 - rho)
        else:
            adjustment = 0.0

        total_llk += ((llk_home + llk_away) + adjustment) * weights[i]

    return -total_llk
