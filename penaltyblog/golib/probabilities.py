import ctypes

import numpy as np

from . import go_lib

go_lib.ComputePoissonProbabilities.argtypes = [
    ctypes.c_double,  # home_attack
    ctypes.c_double,  # away_attack
    ctypes.c_double,  # home_defense
    ctypes.c_double,  # away_defense
    ctypes.c_double,  # home_advantage
    ctypes.c_int,  # max_goals
    ctypes.POINTER(ctypes.c_double),  # score_matrix (pre-allocated output)
    ctypes.POINTER(ctypes.c_double),  # lambda_home (output)
    ctypes.POINTER(ctypes.c_double),  # lambda_away (output)
]


def compute_poisson_probabilities(
    home_attack, away_attack, home_defense, away_defense, home_advantage, max_goals
):
    """
    Calls the Go function to compute Poisson probabilities.
    Returns: score_matrix (2D NumPy array), lambda_home, lambda_away
    """
    # Pre-allocate a contiguous NumPy array for the score matrix
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Allocate memory for lambda values
    lambda_home = ctypes.c_double()
    lambda_away = ctypes.c_double()

    # Call the Go function
    go_lib.ComputePoissonProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambda_home),  # Pass pointer to store value
        ctypes.byref(lambda_away),  # Pass pointer to store value
    )

    return score_matrix, lambda_home.value, lambda_away.value


go_lib.ComputeDixonColesProbabilities.argtypes = [
    ctypes.c_double,  # home_attack
    ctypes.c_double,  # away_attack
    ctypes.c_double,  # home_defense
    ctypes.c_double,  # away_defense
    ctypes.c_double,  # home_advantage
    ctypes.c_double,  # rho
    ctypes.c_int,  # max_goals
    ctypes.POINTER(ctypes.c_double),  # score_matrix (pre-allocated output)
    ctypes.POINTER(ctypes.c_double),  # lambda_home (output)
    ctypes.POINTER(ctypes.c_double),  # lambda_away (output)
]


def compute_dixon_coles_probabilities(
    home_attack, away_attack, home_defense, away_defense, home_advantage, rho, max_goals
):
    """
    Calls the Go function to compute Poisson probabilities.
    Returns: score_matrix (2D NumPy array), lambda_home, lambda_away
    """

    # Pre-allocate a contiguous NumPy array for the score matrix
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Allocate memory for lambda values
    lambda_home = ctypes.c_double()
    lambda_away = ctypes.c_double()

    # Call the Go function
    go_lib.ComputeDixonColesProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_double(rho),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambda_home),  # Pass pointer to store value
        ctypes.byref(lambda_away),  # Pass pointer to store value
    )

    return score_matrix, lambda_home.value, lambda_away.value


# Zero-Inflated Poisson Probabilities
go_lib.ComputeZeroInflatedPoissonProbabilities.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def compute_zip_poisson_probabilities(
    home_attack,
    away_attack,
    home_defense,
    away_defense,
    home_advantage,
    zero_inflation,
    max_goals,
):
    """
    Calls Go function to compute ZIP Poisson probabilities.
    Returns: score_matrix (2D NumPy array), lambda_home, lambda_away
    """
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lambda_home = ctypes.c_double()
    lambda_away = ctypes.c_double()

    go_lib.ComputeZeroInflatedPoissonProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_double(zero_inflation),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambda_home),
        ctypes.byref(lambda_away),
    )

    return score_matrix, lambda_home.value, lambda_away.value


# Negative Binomial Probabilities
go_lib.ComputeNegativeBinomialProbabilities.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def compute_negative_binomial_probabilities(
    home_attack,
    away_attack,
    home_defense,
    away_defense,
    home_advantage,
    dispersion,
    max_goals,
):
    """
    Calls Go function to compute Negative Binomial probabilities.
    Returns: score_matrix (2D NumPy array), lambda_home, lambda_away
    """
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lambda_home = ctypes.c_double()
    lambda_away = ctypes.c_double()

    go_lib.ComputeNegativeBinomialProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_double(dispersion),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambda_home),
        ctypes.byref(lambda_away),
    )

    return score_matrix, lambda_home.value, lambda_away.value


# Bivariate Poisson Probabilities
go_lib.ComputeBivariatePoissonProbabilities.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def compute_bivariate_poisson_probabilities(
    home_attack,
    away_attack,
    home_defense,
    away_defense,
    home_advantage,
    correlation_log,
    max_goals,
):
    """
    Calls Go function to compute Bivariate Poisson probabilities.
    Returns: score_matrix (2D NumPy array), lambda1, lambda2
    """
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lambda1 = ctypes.c_double()
    lambda2 = ctypes.c_double()

    go_lib.ComputeBivariatePoissonProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_double(correlation_log),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambda1),
        ctypes.byref(lambda2),
    )

    return score_matrix, lambda1.value, lambda2.value


# Weibull Copula Probabilities
go_lib.ComputeWeibullCopulaProbabilities.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def compute_weibull_copula_probabilities(
    home_attack,
    away_attack,
    home_defense,
    away_defense,
    home_advantage,
    shape,
    kappa,
    max_goals,
):
    """
    Calls Go function to compute Weibull-Copula probabilities.
    Returns: score_matrix (2D NumPy array), lambdaH, lambdaA
    """
    score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64, order="C")
    score_matrix_ptr = score_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lambdaH = ctypes.c_double()
    lambdaA = ctypes.c_double()

    go_lib.ComputeWeibullCopulaProbabilities(
        ctypes.c_double(home_attack),
        ctypes.c_double(away_attack),
        ctypes.c_double(home_defense),
        ctypes.c_double(away_defense),
        ctypes.c_double(home_advantage),
        ctypes.c_double(shape),
        ctypes.c_double(kappa),
        ctypes.c_int(max_goals),
        score_matrix_ptr,
        ctypes.byref(lambdaH),
        ctypes.byref(lambdaA),
    )

    return score_matrix, lambdaH.value, lambdaA.value
