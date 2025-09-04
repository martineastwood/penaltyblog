# penaltyblog/models/probabilities.pyx
import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport exp, lgamma, log
from libc.stdlib cimport free, malloc

from .utils cimport (
    cdf_from_pmf,
    compute_pxy,
    negbinom_pmf,
    poisson_pmf,
    precompute_alpha_table,
    precompute_poisson_pmf,
    weibull_count_pmf,
    zip_poisson_pmf,
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_poisson_probabilities(double home_attack,
                                           double away_attack,
                                           double home_defense,
                                           double away_defense,
                                           double home_advantage,
                                           int max_goals,
                                           np.float64_t[:] score_matrix,
                                           np.float64_t[:] lambda_home,
                                           np.float64_t[:] lambda_away):
    """
    Compute the Poisson probabilities for each scoreline and write the results
    into the pre-allocated score_matrix array (flattened 1D array of length max_goals**2).

    Also writes the expected home and away goals (lambdas) to the provided 1-element arrays.
    """
    cdef double lh, la
    cdef int i, j, g
    cdef double* homeGoalsVector
    cdef double* awayGoalsVector

    # Compute expected goals
    lh = exp(home_advantage + home_attack + away_defense)
    la = exp(away_attack + home_defense)
    lambda_home[0] = lh
    lambda_away[0] = la

    # Allocate temporary arrays for probabilities.
    homeGoalsVector = <double*> malloc(max_goals * sizeof(double))
    awayGoalsVector = <double*> malloc(max_goals * sizeof(double))
    if homeGoalsVector == NULL or awayGoalsVector == NULL:
        raise MemoryError("Unable to allocate memory for probability vectors.")

    # Compute the Poisson probability for each goal count.
    for g in range(max_goals):
        homeGoalsVector[g] = poisson_pmf(g, lh)
        awayGoalsVector[g] = poisson_pmf(g, la)

    # Fill the score matrix (flattened in row-major order).
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i * max_goals + j] = homeGoalsVector[i] * awayGoalsVector[j]

    # Free the temporary memory.
    free(homeGoalsVector)
    free(awayGoalsVector)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_dixon_coles_probabilities(double home_attack,
                                               double away_attack,
                                               double home_defense,
                                               double away_defense,
                                               double home_advantage,
                                               double rho,
                                               int max_goals,
                                               np.float64_t[:] score_matrix,
                                               np.float64_t[:] lambda_home,
                                               np.float64_t[:] lambda_away):
    """
    Compute the Dixon–Coles adjusted scoreline probabilities.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defence parameters.
      home_advantage: Home advantage.
      rho: Dixon–Coles adjustment parameter.
      max_goals: The maximum number of goals to consider.
      score_matrix: Pre-allocated flattened (1D) array of length max_goals**2,
                    where the score matrix will be stored in row-major order.
      lambda_home, lambda_away: 1-element arrays to hold the expected goals.
    """
    cdef double lh, la
    cdef int i, j, g
    cdef double* homeGoalsVector
    cdef double* awayGoalsVector
    cdef double p, factor

    # Compute expected goals.
    lh = exp(home_advantage + home_attack + away_defense)
    la = exp(away_attack + home_defense)
    lambda_home[0] = lh
    lambda_away[0] = la

    # Allocate temporary arrays for probabilities.
    homeGoalsVector = <double*> malloc(max_goals * sizeof(double))
    awayGoalsVector = <double*> malloc(max_goals * sizeof(double))
    if homeGoalsVector == NULL or awayGoalsVector == NULL:
        raise MemoryError("Unable to allocate memory for probability vectors.")

    # Compute the Poisson probability for each possible goal count.
    for g in range(max_goals):
        homeGoalsVector[g] = poisson_pmf(g, lh)
        awayGoalsVector[g] = poisson_pmf(g, la)

    # Fill the score matrix with the Dixon–Coles adjustment.
    for i in range(max_goals):
        for j in range(max_goals):
            p = homeGoalsVector[i] * awayGoalsVector[j]
            # Apply Dixon–Coles adjustments for low scoring outcomes.
            if i == 0 and j == 0:
                factor = 1 - rho * lh * la
            elif i == 0 and j == 1:
                factor = 1 + rho * lh
            elif i == 1 and j == 0:
                factor = 1 + rho * la
            elif i == 1 and j == 1:
                factor = 1 - rho
            else:
                factor = 1.0
            score_matrix[i * max_goals + j] = p * factor

    # Free the temporary arrays.
    free(homeGoalsVector)
    free(awayGoalsVector)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_random_intercept_probabilities(double home_attack,
                                                  double away_attack,
                                                  double home_defense,
                                                  double away_defense,
                                                  double home_advantage,
                                                  double rho,
                                                  double match_intercept,
                                                  int max_goals,
                                                  np.float64_t[:] score_matrix,
                                                  np.float64_t[:] lambda_home,
                                                  np.float64_t[:] lambda_away):
    """
    Compute the Dixon–Coles adjusted scoreline probabilities with a random intercept.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defence parameters.
      home_advantage: Home advantage.
      rho: Dixon–Coles adjustment parameter.
      match_intercept: The per-match random intercept.
      max_goals: The maximum number of goals to consider.
      score_matrix: Pre-allocated flattened (1D) array of length max_goals**2,
                    where the score matrix will be stored in row-major order.
      lambda_home, lambda_away: 1-element arrays to hold the expected goals.
    """
    cdef double lh, la
    cdef int i, j, g
    cdef double* homeGoalsVector
    cdef double* awayGoalsVector
    cdef double p, factor

    # Compute expected goals.
    lh = exp(home_advantage + home_attack + away_defense + match_intercept)
    la = exp(away_attack + home_defense + match_intercept)
    lambda_home[0] = lh
    lambda_away[0] = la

    # Allocate temporary arrays for probabilities.
    homeGoalsVector = <double*> malloc(max_goals * sizeof(double))
    awayGoalsVector = <double*> malloc(max_goals * sizeof(double))
    if homeGoalsVector == NULL or awayGoalsVector == NULL:
        raise MemoryError("Unable to allocate memory for probability vectors.")

    # Compute the Poisson probability for each possible goal count.
    for g in range(max_goals):
        homeGoalsVector[g] = poisson_pmf(g, lh)
        awayGoalsVector[g] = poisson_pmf(g, la)

    # Fill the score matrix with the Dixon–Coles adjustment.
    for i in range(max_goals):
        for j in range(max_goals):
            p = homeGoalsVector[i] * awayGoalsVector[j]
            # Apply Dixon–Coles adjustments for low scoring outcomes.
            if i == 0 and j == 0:
                factor = 1 - rho * lh * la
            elif i == 0 and j == 1:
                factor = 1 + rho * lh
            elif i == 1 and j == 0:
                factor = 1 + rho * la
            elif i == 1 and j == 1:
                factor = 1 - rho
            else:
                factor = 1.0
            score_matrix[i * max_goals + j] = p * factor

    # Free the temporary arrays.
    free(homeGoalsVector)
    free(awayGoalsVector)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_negative_binomial_probabilities(double home_attack,
                                                     double away_attack,
                                                     double home_defense,
                                                     double away_defense,
                                                     double home_advantage,
                                                     double dispersion,
                                                     int max_goals,
                                                     np.float64_t[:] score_matrix,
                                                     np.float64_t[:] lambda_home,
                                                     np.float64_t[:] lambda_away):
    """
    Compute Negative Binomial probabilities for each scoreline.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defence parameters.
      home_advantage: Home advantage parameter.
      dispersion: Dispersion parameter for the negative binomial model.
      max_goals: Maximum number of goals to consider.
      score_matrix: Pre-allocated flattened (1D) array of length max_goals**2,
                    where the score matrix will be stored (row-major order).
      lambda_home, lambda_away: 1-element arrays to store the expected goals.
    """
    cdef double lh, la
    cdef int i, j, g
    cdef double* homeGoalsVector
    cdef double* awayGoalsVector

    # Compute expected goals for home and away.
    lh = exp(home_advantage + home_attack + away_defense)
    la = exp(away_attack + home_defense)
    lambda_home[0] = lh
    lambda_away[0] = la

    # Allocate temporary arrays to store the probability for each goal count.
    homeGoalsVector = <double*> malloc(max_goals * sizeof(double))
    awayGoalsVector = <double*> malloc(max_goals * sizeof(double))
    if homeGoalsVector == NULL or awayGoalsVector == NULL:
        raise MemoryError("Unable to allocate memory for probability vectors.")

    # Compute Negative Binomial probability for each goal count.
    for g in range(max_goals):
        homeGoalsVector[g] = negbinom_pmf(g, dispersion, lh)
        awayGoalsVector[g] = negbinom_pmf(g, dispersion, la)

    # Fill the score matrix (flattened in row-major order).
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i * max_goals + j] = homeGoalsVector[i] * awayGoalsVector[j]

    # Free the temporary memory.
    free(homeGoalsVector)
    free(awayGoalsVector)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_zero_inflated_poisson_probabilities(double home_attack,
                                                         double away_attack,
                                                         double home_defense,
                                                         double away_defense,
                                                         double home_advantage,
                                                         double zero_inflation,
                                                         int max_goals,
                                                         np.float64_t[:] score_matrix,
                                                         np.float64_t[:] lambda_home,
                                                         np.float64_t[:] lambda_away):
    """
    Compute Zero-Inflated Poisson probabilities for each scoreline.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defence parameters.
      home_advantage: Home advantage parameter.
      zero_inflation: Zero-inflation parameter.
      max_goals: Maximum number of goals to consider.
      score_matrix: Pre-allocated flattened (1D) NumPy array of length max_goals**2,
                    where the score matrix will be stored in row-major order.
      lambda_home, lambda_away: 1-element arrays (memoryviews) to hold the expected goals.
    """
    cdef double lh, la
    cdef int i, j, g
    cdef double* homeGoalsVector
    cdef double* awayGoalsVector

    # Compute expected goals.
    lh = exp(home_advantage + home_attack + away_defense)
    la = exp(away_attack + home_defense)
    lambda_home[0] = lh
    lambda_away[0] = la

    # Allocate temporary arrays for probabilities.
    homeGoalsVector = <double*> malloc(max_goals * sizeof(double))
    awayGoalsVector = <double*> malloc(max_goals * sizeof(double))
    if homeGoalsVector == NULL or awayGoalsVector == NULL:
        raise MemoryError("Unable to allocate memory for probability vectors.")

    # Compute the ZIP probability for each possible goal count.
    for g in range(max_goals):
        homeGoalsVector[g] = zip_poisson_pmf(g, lh, zero_inflation)
        awayGoalsVector[g] = zip_poisson_pmf(g, la, zero_inflation)

    # Fill the score matrix (flattened in row-major order).
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i * max_goals + j] = homeGoalsVector[i] * awayGoalsVector[j]

    # Free the temporary memory.
    free(homeGoalsVector)
    free(awayGoalsVector)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_bivariate_poisson_probabilities(double home_attack,
                                                     double away_attack,
                                                     double home_defense,
                                                     double away_defense,
                                                     double home_advantage,
                                                     double correlation_log,
                                                     int max_goals,
                                                     np.float64_t[:] score_matrix,
                                                     np.float64_t[:] lambda1,
                                                     np.float64_t[:] lambda2):
    """
    Compute the bivariate Poisson probability matrix.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defense parameters.
      home_advantage: Home advantage.
      correlation_log: Logarithm of the correlation parameter.
      max_goals: Maximum number of goals to consider.
      score_matrix: Pre-allocated flattened 1D NumPy array of length max_goals**2 (row-major).
      lambda1, lambda2: 1-element arrays (memoryviews) to store expected goals for home and away.
    """
    cdef double lam1, lam2, lam3
    cdef int x, y, k, idx, min_val
    cdef double p_xy
    cdef list pmf1, pmf2, pmf3

    # Compute expected goals.
    lam1 = exp(home_advantage + home_attack + away_defense)
    lam2 = exp(away_attack + home_defense)
    lam3 = exp(correlation_log)  # Convert log-correlation to lambda3

    lambda1[0] = lam1
    lambda2[0] = lam2

    # Precompute Poisson PMFs for lam1, lam2, and lam3.
    pmf1 = precompute_poisson_pmf(lam1, max_goals)
    pmf2 = precompute_poisson_pmf(lam2, max_goals)
    pmf3 = precompute_poisson_pmf(lam3, max_goals)

    # Compute the bivariate Poisson probability matrix.
    # The score_matrix is assumed to be pre-allocated as a flattened 1D array in row-major order.
    for x in range(max_goals):
        for y in range(max_goals):
            p_xy = 0.0
            # k ranges from 0 to min(x, y) inclusive.
            min_val = x if x < y else y
            for k in range(min_val + 1):
                p_xy += pmf1[x - k] * pmf2[y - k] * pmf3[k]
            idx = x * max_goals + y
            score_matrix[idx] = p_xy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_weibull_copula_probabilities(double home_attack,
                                                  double away_attack,
                                                  double home_defense,
                                                  double away_defense,
                                                  double home_advantage,
                                                  double shape,
                                                  double kappa,
                                                  int max_goals,
                                                  np.float64_t[:] score_matrix,
                                                  np.float64_t[:] lambdaH,
                                                  np.float64_t[:] lambdaA):
    """
    Compute the Weibull-Copula probability matrix for scorelines.

    Parameters:
      home_attack, away_attack: Attack parameters.
      home_defense, away_defense: Defense parameters.
      home_advantage: Home advantage parameter.
      shape: Weibull shape parameter.
      kappa: Copula (Frank) parameter.
      max_goals: Maximum number of goals to consider.
      score_matrix: Pre-allocated flattened 1D NumPy array of length max_goals**2 (row-major order).
      lambdaH, lambdaA: 1-element arrays (memoryviews) to store the expected home and away goals.
    """
    cdef double lamH, lamA
    cdef list alphaTable, pmfH, pmfA, cdfH, cdfA
    cdef int i, j, idx
    cdef double p_ij

    # Compute expected goals.
    lamH = exp(home_advantage + home_attack + away_defense)
    lamA = exp(away_attack + home_defense)
    lambdaH[0] = lamH
    lambdaA[0] = lamA

    # Precompute the Weibull alpha table.
    alphaTable = precompute_alpha_table(shape, max_goals)
    if alphaTable is None:
        print("Invalid shape value for Weibull distribution")
        return

    # Compute Weibull PMFs for home and away.
    pmfH = weibull_count_pmf(lamH, alphaTable, max_goals)
    pmfA = weibull_count_pmf(lamA, alphaTable, max_goals)

    # Compute CDFs from the PMFs.
    cdfH = cdf_from_pmf(pmfH)
    cdfA = cdf_from_pmf(pmfA)

    # Compute the score probability matrix using the Frank Copula.
    # The score_matrix is assumed to be a flattened 1D array in row-major order.
    for i in range(max_goals):
        for j in range(max_goals):
            p_ij = compute_pxy(i, j, cdfH, cdfA, max_goals, kappa)
            idx = i * max_goals + j
            score_matrix[idx] = p_ij
