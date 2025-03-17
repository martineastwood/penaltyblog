import numpy as np
from numpy.typing import ArrayLike


def ignorance_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """
    Compute the mean Ignorance Score for a set of probabilistic predictions.

    Parameters:
        y_true (list or np.array): List of actual match results (0 = Home, 1 = Draw, 2 = Away).
        y_prob (list of lists or np.array): List of predicted probability distributions,
                                            where each sublist corresponds to [P(Home), P(Draw), P(Away)].

    Returns:
        float: Mean ignorance score.
    """
    # Extract the probability assigned to the actual outcome for each match
    actual_probs = [p[y] for p, y in zip(y_prob, y_true)]

    # Compute ignorance score (taking log2 of the probabilities)
    ign_score = -np.mean(np.log2(actual_probs))

    return ign_score
