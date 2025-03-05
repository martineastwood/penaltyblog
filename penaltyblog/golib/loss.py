import ctypes

import numpy as np

from . import go_lib

# Poisson Loss Function
go_lib.PoissonLogLikelihood.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int),  # home_idx
    ctypes.POINTER(ctypes.c_int),  # away_idx
    ctypes.POINTER(ctypes.c_int),  # goals_home
    ctypes.POINTER(ctypes.c_int),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
)
go_lib.PoissonLogLikelihood.restype = ctypes.c_double


def poisson_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
):
    """
    Wrapper function for the PoissonLogLikelihood function in the Go library.
    """

    return go_lib.PoissonLogLikelihood(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
    )


# Poisson Loss Function
go_lib.DixonColesLogLikelihood.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int),  # home_idx
    ctypes.POINTER(ctypes.c_int),  # away_idx
    ctypes.POINTER(ctypes.c_int),  # goals_home
    ctypes.POINTER(ctypes.c_int),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
)
go_lib.DixonColesLogLikelihood.restype = ctypes.c_double


def dixon_coles_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
):
    """
    Wrapper function for the PoissonLogLikelihood function in the Go library.
    """

    return go_lib.DixonColesLogLikelihood(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
    )


# Zero-Inflated Poisson Log-Likelihood
go_lib.ComputeZeroInflatedPoissonLoss.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int32),  # home_idx
    ctypes.POINTER(ctypes.c_int32),  # away_idx
    ctypes.POINTER(ctypes.c_int32),  # goals_home
    ctypes.POINTER(ctypes.c_int32),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
)
go_lib.ComputeZeroInflatedPoissonLoss.restype = ctypes.c_double


def zero_inflated_poisson_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
):
    """
    Wrapper for ComputeZeroInflatedPoissonLoss in Go.
    """
    return go_lib.ComputeZeroInflatedPoissonLoss(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
    )


# Negative Binomial Log-Likelihood
go_lib.ComputeNegativeBinomialLoss.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int32),  # home_idx
    ctypes.POINTER(ctypes.c_int32),  # away_idx
    ctypes.POINTER(ctypes.c_int32),  # goals_home
    ctypes.POINTER(ctypes.c_int32),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
)
go_lib.ComputeNegativeBinomialLoss.restype = ctypes.c_double


def negative_binomial_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
):
    """
    Wrapper for ComputeNegativeBinomialLoss in Go.
    """
    return go_lib.ComputeNegativeBinomialLoss(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
    )


# Bivariate Poisson Log-Likelihood
go_lib.ComputeBivariatePoissonLoss.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int32),  # home_idx
    ctypes.POINTER(ctypes.c_int32),  # away_idx
    ctypes.POINTER(ctypes.c_int32),  # goals_home
    ctypes.POINTER(ctypes.c_int32),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
)
go_lib.ComputeBivariatePoissonLoss.restype = ctypes.c_double


def bivariate_poisson_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
):
    """
    Wrapper for ComputeBivariatePoissonLoss in Go.
    """

    return go_lib.ComputeBivariatePoissonLoss(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
    )


import ctypes

import numpy as np

from . import go_lib

# Define argument and return types for ComputeWeibullCopulaLoss
go_lib.ComputeWeibullCopulaLoss.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_int,  # n_teams
    ctypes.POINTER(ctypes.c_int32),  # home_idx
    ctypes.POINTER(ctypes.c_int32),  # away_idx
    ctypes.POINTER(ctypes.c_int32),  # goals_home
    ctypes.POINTER(ctypes.c_int32),  # goals_away
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.c_int,  # n_matches
    ctypes.c_int,  # max_goals
)
go_lib.ComputeWeibullCopulaLoss.restype = ctypes.c_double  # Return type is a double


def weibull_copula_loss_function(
    params_ptr,
    n_teams_ctypes,
    home_idx_ptr,
    away_idx_ptr,
    goals_home_ptr,
    goals_away_ptr,
    weights_ptr,
    n_matches_ctypes,
    max_goals_ctypes,
):

    # Call the Go function
    return go_lib.ComputeWeibullCopulaLoss(
        params_ptr,
        n_teams_ctypes,
        home_idx_ptr,
        away_idx_ptr,
        goals_home_ptr,
        goals_away_ptr,
        weights_ptr,
        n_matches_ctypes,
        max_goals_ctypes,
    )
