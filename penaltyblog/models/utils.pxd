import numpy as np
import scipy.special

cimport cython
cimport numpy as np
from libc.math cimport exp, fabs, fmax, lgamma, log, tgamma
from libc.stdlib cimport free, malloc


# ─── digamma wrapper ───────────────────────────────────────────
cdef inline double _psi(double x):
    return float(scipy.special.psi(x))


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
cpdef inline tuple precompute_alpha_table_and_d_shape(double c, int maxGoals):
    """
    Computes the alpha table A AND its derivative dA/dc in a single pass.
    Returns (A, dA) as a tuple, or (None, None) if c <= 0.

    Uses digamma/polygamma functions to differentiate the Gamma terms
    analytically, avoiding a second full recomputation at shape + eps.
    """
    if c <= 0:
        return None, None
    cdef int i, j, m, maxG = maxGoals
    cdef double g_cj, g_j, g_cd, g_d, F_m, dF_m
    cdef double sign_val, psi_cj, s, ds, diff, tmpSum

    # Allocate tables
    A = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]
    dA = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]
    alphaRaw = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]
    dAlphaRaw = [[0.0 for _ in range(maxG+1)] for _ in range(maxG+1)]

    # Base case: alphaRaw[0][j] = Gamma(c*j + 1) / Gamma(j + 1)
    # d/dc alphaRaw[0][j] = j * psi(c*j + 1) * alphaRaw[0][j]
    for j in range(maxG+1):
        g_cj = gamma_func(c * j + 1.0)
        g_j = gamma_func(j + 1.0)
        alphaRaw[0][j] = g_cj / g_j
        dAlphaRaw[0][j] = j * _psi(c * j + 1.0) * alphaRaw[0][j]

    # Recursive case: alphaRaw[x+1][j] = sum_{m=x}^{j-1} alphaRaw[x][m] * F(m)
    # where F(m) = Gamma(c*(j-m) + 1) / Gamma(j-m+1)
    # dF/dc = (j-m) * psi(c*(j-m) + 1) * F(m)
    for x in range(maxG):
        for j in range(x+1, maxG+1):
            s = 0.0
            ds = 0.0
            for m in range(x, j):
                diff = <double>(j - m)
                g_cd = gamma_func(c * diff + 1.0)
                g_d = gamma_func(diff + 1.0)
                F_m = g_cd / g_d
                dF_m = diff * _psi(c * diff + 1.0) * F_m

                s += alphaRaw[x][m] * F_m
                ds += dAlphaRaw[x][m] * F_m + alphaRaw[x][m] * dF_m

            alphaRaw[x+1][j] = s
            dAlphaRaw[x+1][j] = ds

    # Final: A[x][j] = (-1)^(x+j) * alphaRaw[x][j] / Gamma(c*j + 1)
    # dA/dc = (-1)^(x+j) / G * (dAlphaRaw - alphaRaw * j * psi(c*j+1))
    for x in range(maxG+1):
        for j in range(maxG+1):
            sign_val = 1.0 if ((x + j) % 2 == 0) else -1.0
            g_cj = gamma_func(c * j + 1.0)
            A[x][j] = sign_val * (alphaRaw[x][j] / g_cj)
            if fabs(g_cj) < 1e-300:
                dA[x][j] = 0.0
            else:
                psi_cj = _psi(c * j + 1.0)
                dA[x][j] = sign_val / g_cj * (dAlphaRaw[x][j] - alphaRaw[x][j] * j * psi_cj)

    return A, dA


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
cpdef inline double frank_copula_cdf_dkappa(double u, double v, double kappa):
    """
    Analytical derivative of the Frank copula CDF w.r.t. kappa.

    C(u,v;k) = -(1/k) * log(Z)   where Z = 1 + A*B/D,
    A = e^{-ku}-1, B = e^{-kv}-1, D = e^{-k}-1.

    dC/dk = (1/k^2)*log(Z) - (1/k)*(1/Z)*dZ/dk
    """
    if fabs(kappa) < 1e-8:
        return 0.0  # Independence limit

    cdef double eku = exp(-kappa * u)
    cdef double ekv = exp(-kappa * v)
    cdef double ek = exp(-kappa)

    cdef double A = eku - 1.0
    cdef double B = ekv - 1.0
    cdef double D = ek - 1.0

    cdef double Z = 1.0 + A * B / D

    if Z <= 1e-14:
        return 0.0

    # dA/dk = -u * e^{-ku}
    cdef double dA = -u * eku
    # dB/dk = -v * e^{-kv}
    cdef double dB = -v * ekv
    # dD/dk = -e^{-k}
    cdef double dD = -ek

    cdef double dZ = (dA * B * D + A * dB * D - A * B * dD) / (D * D)

    return (1.0 / (kappa * kappa)) * log(Z) - (1.0 / kappa) * (1.0 / Z) * dZ


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline double compute_pxy_dkappa(int x, int y, list cdfX, list cdfY, int maxGoals, double kappa):
    """
    Analytical derivative of p(x, y) w.r.t. kappa.
    p(x,y) = C(u,v) - C(u_prev,v) - C(u,v_prev) + C(u_prev,v_prev)
    dp/dkappa = dC/dk(u,v) - dC/dk(u_prev,v) - dC/dk(u,v_prev) + dC/dk(u_prev,v_prev)
    """
    cdef double u, v, u_prev, v_prev

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

    return (frank_copula_cdf_dkappa(u, v, kappa)
            - frank_copula_cdf_dkappa(u_prev, v, kappa)
            - frank_copula_cdf_dkappa(u, v_prev, kappa)
            + frank_copula_cdf_dkappa(u_prev, v_prev, kappa))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline list weibull_count_pmf_d_alpha(double lam, list A, list dA, int maxGoals):
    """
    Computes the derivative of the (normalised) Weibull count PMF w.r.t. the shape
    parameter, given the alpha table A and its derivative dA/dc.

    Uses the quotient rule on pmf[x] = raw[x] / S where S = sum(raw).
    d(pmf[x])/dc = (draw[x]*S - raw[x]*dS) / S^2
    """
    cdef int j, x
    cdef double val, dval, s, ds, lp

    cdef int n = maxGoals + 1
    raw = [0.0] * n
    draw = [0.0] * n

    if lam <= 0:
        # zero-lam: pmf = [1, 0, 0, ...], derivative is 0
        return [0.0] * n

    lamPowers = [0.0] * n
    lamPowers[0] = 1.0
    for j in range(1, n):
        lamPowers[j] = lamPowers[j-1] * lam

    for x in range(n):
        val = 0.0
        dval = 0.0
        for j in range(x, n):
            lp = lamPowers[j]
            val += lp * A[x][j]
            dval += lp * dA[x][j]
        # Clamp negatives to zero (matching weibull_count_pmf)
        if val > 0:
            raw[x] = val
            draw[x] = dval
        else:
            raw[x] = 0.0
            draw[x] = 0.0

    s = 0.0
    ds = 0.0
    for x in range(n):
        s += raw[x]
        ds += draw[x]

    pmf_d = [0.0] * n
    if s < 1e-14:
        return pmf_d  # all zeros

    cdef double s_sq = s * s
    for x in range(n):
        pmf_d[x] = (draw[x] * s - raw[x] * ds) / s_sq

    return pmf_d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef inline list weibull_count_pmf_d_lambda(double lam, list A, int maxGoals):
    """
    Computes the derivative of the (normalised) Weibull count PMF w.r.t. lambda.

    raw[x] = Σ_{j≥x} λ^j · A[x][j]
    d(raw[x])/dλ = Σ_{j≥x} j · λ^(j-1) · A[x][j]

    Uses the quotient rule: d(raw[x]/S)/dλ = (draw[x]*S - raw[x]*dS) / S^2
    """
    cdef int j, x
    cdef double val, dval, s, ds

    cdef int n = maxGoals + 1
    raw = [0.0] * n
    draw = [0.0] * n

    if lam <= 0:
        return [0.0] * n

    # Compute lambda powers and their lambda-derivatives
    # lamPowers[j] = lam^j
    # dlamPowers[j] = j * lam^(j-1)
    lamPowers = [0.0] * n
    dlamPowers = [0.0] * n
    lamPowers[0] = 1.0
    dlamPowers[0] = 0.0  # d/dlam (lam^0) = 0
    for j in range(1, n):
        lamPowers[j] = lamPowers[j-1] * lam
        dlamPowers[j] = j * lamPowers[j-1]  # j * lam^(j-1)

    for x in range(n):
        val = 0.0
        dval = 0.0
        for j in range(x, n):
            val += lamPowers[j] * A[x][j]
            dval += dlamPowers[j] * A[x][j]
        if val > 0:
            raw[x] = val
            draw[x] = dval
        else:
            raw[x] = 0.0
            draw[x] = 0.0

    s = 0.0
    ds = 0.0
    for x in range(n):
        s += raw[x]
        ds += draw[x]

    pmf_d = [0.0] * n
    if s < 1e-14:
        return pmf_d

    cdef double s_sq = s * s
    for x in range(n):
        pmf_d[x] = (draw[x] * s - raw[x] * ds) / s_sq

    return pmf_d


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
