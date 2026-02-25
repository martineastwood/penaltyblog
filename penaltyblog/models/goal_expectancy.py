import warnings
from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def goal_expectancy(
    home: float,
    draw: float,
    away: float,
    dc_adj: bool = True,
    rho: float = 0.001,
    minimizer_options: Optional[Dict[str, Any]] = None,
    *,
    max_goals: int = 15,
    remove_overround: bool = False,
    method: str = "L-BFGS-B",
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-3.0, 3.0),
        (-3.0, 3.0),
    ),
    x0: Optional[tuple[float, float]] = None,
    renormalize_after_dc: bool = True,
    objective: Literal["brier", "cross_entropy"] = "brier",
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Infer implied goal expectancies (mu_home, mu_away) from 1X2 probabilities.

    Parameters
    ----------
    home : float
        Probability of home win
    draw : float
        Probability of draw
    away : float
        Probability of away win
    dc_adj : bool
        Whether to apply the Dixon and Coles adjustment.
    rho : float
        The value for rho within the Dixon and Coles adjustment if `dc_adj` is True.
    minimizer_options : dict, optional
        Dictionary of options to pass to `scipy.optimize.minimize` (e.g., maxiter, ftol, disp).
    max_goals : int, default 15
        Grid cutoff (0..max_goals per team).
    remove_overround : bool, default False
        If True, re-scale (home, draw, away) to sum to 1.
    method : str, default 'L-BFGS-B'
        SciPy optimizer method.
    bounds : tuple, default ((-3.0, 3.0), (-3.0, 3.0))
        Bounds on log-mu for stability (L-BFGS-B compatible).
    x0 : tuple | None, optional
        Optional initial guess for (log_mu_home, log_mu_away).
    renormalize_after_dc : bool, default True
        Re-normalise the grid after DC tweak (and clip to >=0).
    objective : {'brier','cross_entropy'}, default 'brier'
        Scoring rule to minimise.
    return_details : bool, default True
        Include predicted 1X2 and grid mass in the output.

    Returns
    -------
    dict with keys:
        home_exp, away_exp, error, success
        (plus predicted, mass if return_details=True)
    """
    # --- Input handling ---
    p = np.array([home, draw, away], dtype=float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("home/draw/away must be probabilities in [0, 1].")
    s = p.sum()
    if remove_overround:
        if s <= 0:
            raise ValueError("Sum of probabilities must be > 0 to remove_overround.")
        p = p / s  # Renormalise to sum to 1

    elif not np.isclose(s, 1.0, atol=1e-6):
        warnings.warn(
            "Input probabilities do not sum to 1.0. Consider setting `remove_overround=True`."
        )

    # Precompute constants for speed inside the objective
    k = int(max_goals)
    goals = np.arange(0, k + 1, dtype=int)

    # Objective choice
    def _loss(pred_vec: np.ndarray, obs_vec: np.ndarray) -> float:
        if objective == "brier":
            return float(np.mean((pred_vec - obs_vec) ** 2))
        elif objective == "cross_entropy":
            # Add tiny epsilon for safety
            eps = 1e-12
            q = np.clip(pred_vec, eps, 1 - eps)
            return float(-(obs_vec * np.log(q)).sum())
        else:
            raise ValueError("objective must be 'brier' or 'cross_entropy'.")

    # DC adjustment helper
    def _apply_dc(mat: np.ndarray, mu_h: float, mu_a: float) -> np.ndarray:
        if not dc_adj:
            return mat
        out = mat.copy()
        out[0, 0] *= 1 - mu_h * mu_a * rho
        out[0, 1] *= 1 + mu_h * rho
        out[1, 0] *= 1 + mu_a * rho
        out[1, 1] *= 1 - rho
        if renormalize_after_dc:
            # Clip tiny negatives then renormalise to preserve a probability measure
            np.maximum(out, 0.0, out)
            total = out.sum()
            if total > 0:
                out /= total
        return out

    # Build objective closure (minimize calls this many times)
    def _objective(params_log: np.ndarray) -> float:
        mu_h, mu_a = np.exp(params_log[0]), np.exp(params_log[1])

        mu1 = poisson(mu_h).pmf(goals)
        mu2 = poisson(mu_a).pmf(goals)

        mat = np.outer(mu1, mu2)
        mat = _apply_dc(mat, mu_h, mu_a)

        pred = np.array(
            [
                np.sum(np.tril(mat, -1)),  # home
                np.sum(np.diag(mat)),  # draw
                np.sum(np.triu(mat, 1)),  # away
            ],
            dtype=float,
        )

        return _loss(pred, p)

    # Initial guess for log-mu
    if x0 is None:
        # A mild home advantage initialisation
        x0 = (np.log(1.3), np.log(1.1))

    options = {"maxiter": 1000, "disp": False}
    if minimizer_options is not None:
        options.update(minimizer_options)

    res = minimize(
        fun=_objective,
        x0=np.array(x0, dtype=float),
        method=method,
        bounds=(
            bounds
            if method.upper() in {"L-BFGS-B", "TNC", "SLSQP", "TRUST-CONSTR"}
            else None
        ),
        options=options,
    )

    mu_h, mu_a = np.exp(res.x[0]), np.exp(res.x[1])

    # Build outputs and audit fields
    # Recompute one last time for auditing
    mu1 = poisson(mu_h).pmf(goals)
    mu2 = poisson(mu_a).pmf(goals)
    mat = np.outer(mu1, mu2)
    if dc_adj:
        mat = _apply_dc(mat, mu_h, mu_a)
    predicted = np.array(
        [
            np.sum(np.tril(mat, -1)),
            np.sum(np.diag(mat)),
            np.sum(np.triu(mat, 1)),
        ],
        dtype=float,
    )
    mass = float(mat.sum())

    out: dict[str, Any] = {
        "home_exp": float(mu_h),
        "away_exp": float(mu_a),
        "error": float(res.fun),
        "success": bool(res.success),
    }
    if return_details:
        out["predicted"] = predicted
        out["mass"] = mass
    return out


def goal_expectancy_extended(
    home: float,
    draw: float,
    away: float,
    over25: float,
    under25: float,
    minimizer_options: Optional[Dict[str, Any]] = None,
    max_goals: int = 15,
    remove_overround: bool = True,
    method: str = "L-BFGS-B",
    bounds: tuple = (
        (-3.0, 3.0),  # log(mu_home)
        (-3.0, 3.0),  # log(mu_away)
        (-0.5, 0.5),  # rho (usually falls between -0.2 and 0.2)
    ),
    x0: Optional[tuple] = None,
    objective: Literal["brier", "cross_entropy"] = "brier",
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Infer implied goal expectancies (mu_home, mu_away) AND the Dixon-Coles
    correlation parameter (rho) simultaneously from 1X2 and Over/Under 2.5 probabilities.

    Parameters
    ----------
    home : float
        Probability of home win
    draw : float
        Probability of draw
    away : float
        Probability of away win
    over25 : float
        Probability of over 2.5 goals
    under25 : float
        Probability of under 2.5 goals
    minimizer_options : dict, optional
        Dictionary of options to pass to `scipy.optimize.minimize` (e.g., maxiter, ftol, disp).
    max_goals : int, default 15
        Grid cutoff (0..max_goals per team).
    remove_overround : bool, default True
        If True, re-scale (home, draw, away) and (over25, under25) to sum to 1.
    method : str, default 'L-BFGS-B'
        SciPy optimizer method.
    bounds : tuple, default ((-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5))
        Bounds on log-mu for stability and rho (L-BFGS-B compatible).
    x0 : tuple | None, optional
        Optional initial guess for (log_mu_home, log_mu_away, rho).
    objective : {'brier','cross_entropy'}, default 'brier'
        Scoring rule to minimise.
    return_details : bool, default True
        Include predicted 1X2 and Over/Under 2.5 probabilities in the output.

    Returns
    -------
    dict with keys:
        home_exp, away_exp, implied_rho, error, success
        (plus predicted_1x2, predicted_ou if return_details=True)
    """
    # --- Input handling and Overround Removal ---
    p_1x2 = np.array([home, draw, away], dtype=float)
    p_ou = np.array([over25, under25], dtype=float)

    if np.any(p_1x2 < 0) or np.any(p_ou < 0):
        raise ValueError("Probabilities must be >= 0.")

    if remove_overround:
        p_1x2 = p_1x2 / p_1x2.sum()
        p_ou = p_ou / p_ou.sum()

    # The target vector the optimizer is trying to match
    target_probs = np.concatenate([p_1x2, p_ou])

    # Precompute grid constants
    k = int(max_goals)
    goals = np.arange(0, k + 1, dtype=int)

    # Pre-build a mask for the Over 2.5 calculation to save time in the loop
    I, J = np.indices((k + 1, k + 1))
    over25_mask = (I + J) > 2.5

    def _loss(pred_vec: np.ndarray, obs_vec: np.ndarray) -> float:
        if objective == "brier":
            return float(np.mean((pred_vec - obs_vec) ** 2))
        elif objective == "cross_entropy":
            eps = 1e-12
            q = np.clip(pred_vec, eps, 1 - eps)
            return float(-(obs_vec * np.log(q)).sum())
        raise ValueError("objective must be 'brier' or 'cross_entropy'.")

    def _apply_dc(
        mat: np.ndarray, mu_h: float, mu_a: float, rho_val: float
    ) -> np.ndarray:
        if rho_val == 0.0:
            return mat

        out = mat.copy()
        out[0, 0] *= 1 - mu_h * mu_a * rho_val
        out[1, 0] *= 1 + mu_h * rho_val
        out[0, 1] *= 1 + mu_a * rho_val
        out[1, 1] *= 1 - rho_val

        # Clip to 0 to prevent mathematically invalid negative probabilities
        np.maximum(out, 0.0, out)
        total = out.sum()
        if total > 0:
            out /= total
        return out

    def _objective(params: np.ndarray) -> float:
        # Decode the 3 parameters
        mu_h, mu_a = np.exp(params[0]), np.exp(params[1])
        rho_val = params[2]  # rho is not in log-space because it can be negative

        # Generate base Poisson
        mu1 = poisson(mu_h).pmf(goals)
        mu2 = poisson(mu_a).pmf(goals)
        mat = np.outer(mu1, mu2)

        # Apply DC correlation
        mat = _apply_dc(mat, mu_h, mu_a, rho_val)

        # Extract predicted markets from the generated matrix
        pred = np.array(
            [
                np.sum(np.tril(mat, -1)),  # Home Win
                np.sum(np.diag(mat)),  # Draw
                np.sum(np.triu(mat, 1)),  # Away Win
                np.sum(mat[over25_mask]),  # Over 2.5
                np.sum(mat[~over25_mask]),  # Under 2.5
            ],
            dtype=float,
        )

        return _loss(pred, target_probs)

    # Initial guess: slight home advantage, 0.0 rho
    if x0 is None:
        x0 = (np.log(1.3), np.log(1.1), 0.0)

    options = {"maxiter": 2000, "disp": False}
    if minimizer_options is not None:
        options.update(minimizer_options)

    res = minimize(
        fun=_objective,
        x0=np.array(x0, dtype=float),
        method=method,
        bounds=bounds,
        options=options,
    )

    # Decode final result
    mu_h, mu_a = np.exp(res.x[0]), np.exp(res.x[1])
    final_rho = res.x[2]

    # Recompute final matrix for auditing
    mat = _apply_dc(
        np.outer(poisson(mu_h).pmf(goals), poisson(mu_a).pmf(goals)),
        mu_h,
        mu_a,
        final_rho,
    )

    predicted = np.array(
        [
            np.sum(np.tril(mat, -1)),
            np.sum(np.diag(mat)),
            np.sum(np.triu(mat, 1)),
            np.sum(mat[over25_mask]),
            np.sum(mat[~over25_mask]),
        ]
    )

    out: dict[str, Any] = {
        "home_exp": float(mu_h),
        "away_exp": float(mu_a),
        "implied_rho": float(final_rho),
        "error": float(res.fun),
        "success": bool(res.success),
    }

    if return_details:
        out["predicted_1x2"] = predicted[:3]
        out["predicted_ou"] = predicted[3:]

    return out
