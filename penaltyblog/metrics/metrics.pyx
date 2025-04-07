# penaltyblog/models/metrics.pyx
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def compute_multiclass_brier_score(np.ndarray[np.int32_t, ndim=1] y_true,
                                    np.ndarray[np.float64_t, ndim=2] y_prob) -> float:
    """
    Compute the multiclass Brier score.

    Parameters:
        y_true: (n_samples,) array of true class indices
        y_prob: (n_samples, n_classes) array of predicted probabilities

    Returns:
        float: mean Brier score
    """
    cdef Py_ssize_t i, k
    cdef Py_ssize_t n_samples = y_true.shape[0]
    cdef Py_ssize_t n_classes = y_prob.shape[1]

    cdef double score = 0.0
    cdef double diff

    for i in range(n_samples):
        for k in range(n_classes):
            diff = y_prob[i, k] - (1.0 if y_true[i] == k else 0.0)
            score += diff * diff

    return score / n_samples



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
def compute_ignorance_score(np.ndarray[np.int32_t, ndim=1] y_true,
                    np.ndarray[np.float64_t, ndim=2] y_prob) -> float:
    """
    Compute the mean Ignorance Score for a set of probabilistic predictions.

    Parameters:
        y_true (1D np.ndarray[int]): Actual match results (0 = Home, 1 = Draw, 2 = Away).
        y_prob (2D np.ndarray[float64]): Predicted probability distributions,
                                         shape (n_samples, 3).

    Returns:
        float: Mean ignorance score.
    """
    cdef Py_ssize_t i, n = y_true.shape[0]
    cdef double prob, total = 0.0
    cdef double epsilon = 1e-15

    for i in range(n):
        prob = y_prob[i, y_true[i]]
        if prob < epsilon:
            prob = epsilon
        total += log2(prob)

    return -total / n



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef void compute_rps_array(np.float64_t[:,:] probs,
                             np.int32_t[:] outcomes,
                             int nSets,
                             int nOutcomes,
                             np.float64_t[:] out):
    """
    Compute individual RPS scores for each fixture and store them in 'out'.

    Parameters:
      probs: 2D probability array of shape (nSets, nOutcomes) in row‐major order.
      outcomes: Array (length nSets) of observed outcome indices.
      nSets: Number of fixtures.
      nOutcomes: Number of possible outcomes per fixture.
      out: Preallocated output array (length nSets) to store the RPS for each fixture.

    If an outcome is invalid (i.e. not in [0, nOutcomes-1]), the function assigns a large penalty (1e6) to that fixture.
    """
    cdef int i, j, outcome
    cdef double diffSum, d, rpsVal

    # Python lists for intermediate cumulative sums.
    cdef list cumProbs, indicator, cumOutcomes

    for i in range(nSets):
        outcome = outcomes[i]
        if outcome < 0 or outcome >= nOutcomes:
            raise ValueError("Invalid outcome index")

        # Compute cumulative probabilities for row i.
        cumProbs = [0.0] * nOutcomes
        cumProbs[0] = probs[i, 0]
        for j in range(1, nOutcomes):
            cumProbs[j] = cumProbs[j-1] + probs[i, j]

        # Build the indicator vector: 1 at the observed outcome index, 0 elsewhere.
        indicator = [0.0] * nOutcomes
        for j in range(nOutcomes):
            if j == outcome:
                indicator[j] = 1.0
            else:
                indicator[j] = 0.0

        # Compute cumulative outcomes (CDF of the indicator).
        cumOutcomes = [0.0] * nOutcomes
        cumOutcomes[0] = indicator[0]
        for j in range(1, nOutcomes):
            cumOutcomes[j] = cumOutcomes[j-1] + indicator[j]

        # Compute sum of squared differences.
        diffSum = 0.0
        for j in range(nOutcomes):
            d = cumProbs[j] - cumOutcomes[j]
            diffSum += d * d

        rpsVal = diffSum / (nOutcomes - 1.0)
        out[i] = rpsVal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef double compute_average_rps(np.float64_t[:,:] probs,
                                 np.int32_t[:] outcomes,
                                 int nSets,
                                 int nOutcomes):
    """
    Compute the average RPS over all fixtures.

    Parameters:
      probs: 2D probability array of shape (nSets, nOutcomes) in row‐major order.
      outcomes: Array (length nSets) of observed outcome indices.
      nSets: Number of fixtures.
      nOutcomes: Number of possible outcomes per fixture.

    Returns:
      The average RPS (a double).
    """
    cdef int i, j, outcome
    cdef double diffSum, d, rpsVal, sumRPS = 0.0
    cdef list cumProbs, indicator, cumOutcomes

    for i in range(nSets):
        outcome = outcomes[i]
        if outcome < 0 or outcome >= nOutcomes:
            raise ValueError("Invalid outcome index")

        # Compute cumulative probabilities for row i.
        cumProbs = [0.0] * nOutcomes
        cumProbs[0] = probs[i, 0]
        for j in range(1, nOutcomes):
            cumProbs[j] = cumProbs[j-1] + probs[i, j]

        # Build indicator vector: 1 at the observed outcome index, 0 elsewhere.
        indicator = [0.0] * nOutcomes
        for j in range(nOutcomes):
            if j == outcome:
                indicator[j] = 1.0
            else:
                indicator[j] = 0.0

        # Compute cumulative outcomes (the CDF of the indicator).
        cumOutcomes = [0.0] * nOutcomes
        cumOutcomes[0] = indicator[0]
        for j in range(1, nOutcomes):
            cumOutcomes[j] = cumOutcomes[j-1] + indicator[j]

        # Compute the sum of squared differences between cumulative probabilities and outcomes.
        diffSum = 0.0
        for j in range(nOutcomes):
            d = cumProbs[j] - cumOutcomes[j]
            diffSum += d * d

        rpsVal = diffSum / (nOutcomes - 1.0)
        sumRPS += rpsVal

    return sumRPS / nSets
