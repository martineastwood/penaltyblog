import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport exp, fabs, fmax, lgamma, log, tgamma
from libc.stdlib cimport free, malloc


# Inline helper: Poisson PMF: exp(-λ) * λ^k / k!
cdef inline double poisson_pmf(int k, double lam) nogil:
    return exp(-lam + k * log(lam) - lgamma(k+1))


# Inline helper: Poisson log-PMF: -λ + k*log(λ) - lgamma(k+1)
cdef inline double poisson_log_pmf(int k, double lam):
    return -lam + k * log(lam) - lgamma(k + 1)


cdef inline double negBinomLogPMF(int k, double dispersion, double p) nogil:
    cdef double epsilon = 1e-9
    return lgamma(k + dispersion) - lgamma(k + 1) - lgamma(dispersion) + dispersion * log(fmax(epsilon, p)) + k * log(fmax(epsilon, 1 - p))


cdef inline double negbinom_pmf(int k, double disp, double lam):
    return exp(lgamma(k + disp) - lgamma(k + 1) - lgamma(disp)
               + disp * log(disp / (disp + lam)) + k * log(lam / (disp + lam)))


# Inline helper: Zero-Inflated Poisson PMF.
cdef inline double zip_poisson_pmf(int k, double lam, double zi):
    if k == 0:
        return zi + (1 - zi) * exp(-lam)
    else:
        return (1 - zi) * poisson_pmf(k, lam)


cdef inline int compute_max_goal(long[:] goalsHome, long[:] goalsAway, int nMatches):
    cdef int i, max_val = 0, temp
    for i in range(nMatches):
        temp = goalsHome[i]
        if temp > max_val:
            max_val = temp
        temp = goalsAway[i]
        if temp > max_val:
            max_val = temp
    return max_val + 1


cdef inline list precompute_poisson_pmf(double lam, int maxGoals):
    """
    Precompute the Poisson probability mass function for k = 0 ... maxGoals-1.
    Returns a Python list of doubles.
    PMF: exp(-lam) * lam^k / k!
    """
    cdef list pmf = []
    cdef int k
    for k in range(maxGoals):
        pmf.append(exp(-lam + k * log(lam) - lgamma(k + 1)))
    return pmf



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline list precompute_alpha_table(double c, int maxGoals):
    """
    Computes the alpha table A, a (maxGoals+1) x (maxGoals+1) list of lists.
    Returns None if c <= 0.
    """
    if c <= 0:
        return None
    cdef int i, j, m, maxG = maxGoals
    # Create A and alphaRaw as lists of lists.
    A = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]
    alphaRaw = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]

    for j in range(maxG+1):
        alphaRaw[0][j] = gamma_func(c * j + 1.0) / gamma_func(j + 1.0)

    for x in range(maxG):
        for j in range(x+1, maxG+1):
            tmpSum = 0.0
            for m in range(x, j):
                tmpSum += (alphaRaw[x][m] * gamma_func(c * j - c * m + 1.0)) / gamma_func(j - m + 1.0)
            alphaRaw[x+1][j] = tmpSum

    for x in range(maxG+1):
        for j in range(maxG+1):
            sign = (-1)**(x+j)
            denom = gamma_func(c * j + 1.0)
            A[x][j] = sign * (alphaRaw[x][j] / denom)

    return A


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef inline double gamma_func(double x):
    if x <= 0:
        return np.nan  # gamma is not defined for non-positive values
    return tgamma(x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline double compute_pxy(int x, int y, list cdfX, list cdfY, int maxGoals, double kappa):
    """
    Computes the joint probability p(x, y) using a Frank copula based on the CDFs.
    Boundary values for the CDFs are handled inline.
    """
    cdef double u, v, u_prev, v_prev

    # Define FX(k) inline:
    if x < 0:
        u = 0.0
    elif x > maxGoals:
        u = 1.0
    else:
        u = cdfX[x]

    if y < 0:
        v = 0.0
    elif y > maxGoals:
        v = 1.0
    else:
        v = cdfY[y]

    if x - 1 < 0:
        u_prev = 0.0
    elif x - 1 > maxGoals:
        u_prev = 1.0
    else:
        u_prev = cdfX[x - 1]

    if y - 1 < 0:
        v_prev = 0.0
    elif y - 1 > maxGoals:
        v_prev = 1.0
    else:
        v_prev = cdfY[y - 1]

    return (frank_copula_cdf(u, v, kappa)
            - frank_copula_cdf(u_prev, v, kappa)
            - frank_copula_cdf(u, v_prev, kappa)
            + frank_copula_cdf(u_prev, v_prev, kappa))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline double frank_copula_cdf(double u, double v, double kappa):
    if fabs(kappa) < 1e-8:
        return u * v  # Independence
    cdef double num = (exp(-kappa*u) - 1.0) * (exp(-kappa*v) - 1.0)
    cdef double denom = exp(-kappa) - 1.0
    cdef double inside = 1.0 + num/denom
    if inside <= 1e-14:
        if u*v > 0:
            return u * v
        else:
            return 0.0
    return -(1.0/kappa) * log(inside)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline list cdf_from_pmf(list pmf):
    cdef int i, n = len(pmf)
    cdf = [0.0]*n
    cdf[0] = pmf[0]
    for i in range(1, n):
        cdf[i] = cdf[i-1] + pmf[i]
    return cdf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline list weibull_count_pmf(double lam, list A, int maxGoals):
    """
    Computes the Weibull count probability mass function (PMF) for counts 0...maxGoals.
    Returns a list of length maxGoals+1.
    """
    cdef int j, x
    cdef double val, s
    pmf = [0.0]*(maxGoals+1)
    if lam <= 0:
        pmf[0] = 1.0
        return pmf
    lamPowers = [0.0]*(maxGoals+1)
    lamPowers[0] = 1.0
    for j in range(1, maxGoals+1):
        lamPowers[j] = lamPowers[j-1] * lam
    for x in range(maxGoals+1):
        val = 0.0
        for j in range(x, maxGoals+1):
            val += lamPowers[j] * A[x][j]
        if val > 0:
            pmf[x] = val
    s = 0.0
    for v in pmf:
        s += v
    if s < 1e-14:
        pmf = [0.0]*(maxGoals+1)
        pmf[0] = 1.0
    else:
        for x in range(len(pmf)):
            pmf[x] /= s
    return pmf
