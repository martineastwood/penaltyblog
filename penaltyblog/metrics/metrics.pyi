import numpy as np

def compute_multiclass_brier_score(
    y_true: np.ndarray,  # 1D np.ndarray[int]
    y_prob: np.ndarray,  # 2D np.ndarray[float64]
) -> float: ...
def compute_ignorance_score(
    y_true: np.ndarray,  # 1D np.ndarray[int]
    y_prob: np.ndarray,  # 2D np.ndarray[float64]
) -> float: ...
def compute_rps_array(
    probs: np.ndarray,  # 2D np.ndarray[float64]
    outcomes: np.ndarray,  # 1D np.ndarray[int]
    nSets: int,
    nOutcomes: int,
    out: np.ndarray,  # 1D np.ndarray[float64]
) -> None: ...
def compute_average_rps(
    probs: np.ndarray,  # 2D np.ndarray[float64]
    outcomes: np.ndarray,  # 1D np.ndarray[int]
    nSets: int,
    nOutcomes: int,
) -> float: ...
