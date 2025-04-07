import numpy as np
from numpy.typing import ArrayLike

from .metrics import (
    compute_multiclass_brier_score as compute_multiclass_brier_score,
)


def multiclass_brier_score(y_prob: ArrayLike, y_true: ArrayLike) -> float:
    """
    Calculates multiclass Brier score.

    Args:
        y_prob: 2D array of predicted probability distributions,
                where each row corresponds to [P(Home), P(Draw), P(Away)].
        y_true: 1D array of match results (0 = Home, 1 = Draw, 2 = Away).

    Returns:
        Multiclass Brier score.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2D array")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same number of samples")
    if np.any(y_true < 0) or np.any(y_true >= y_prob.shape[1]):
        raise ValueError("y_true contains invalid class indices")

    return compute_multiclass_brier_score(y_true, y_prob)
