Multiclass Briar Score
======================

The multiclass Brier score is a proper scoring rule that measures the accuracy of probabilistic predictions across multiple categories by calculating the mean squared difference between predicted probabilities and actual outcomes.

For a prediction with K possible classes, it computes the sum of squared differences between the predicted probability for each class and the actual outcome (encoded as 1 for the true class and 0 for all others), making it essentially an extension of the binary Brier score to multiple categories.

The score ranges from 0 to 2, where 0 represents perfect prediction (assigning probability 1 to the correct class) and 2 represents the worst possible prediction (assigning probability 1 to an incorrect class), though in practice scores typically fall between 0 and 1 for reasonable models.

In soccer analytics, the multiclass Brier score is commonly used to evaluate match outcome predictions (home win/draw/away win) and is particularly valued because it's a strictly proper scoring rule - meaning forecasters are incentivized to report their true beliefs rather than hedging their predictions.

Unlike metrics that only consider the predicted class, the Brier score penalizes overconfident wrong predictions more severely than uncertain predictions, making it especially useful for assessing calibration in probabilistic forecasting models where understanding prediction confidence is as important as accuracy itself.

.. code-block:: python

    import penaltyblog as pb

    predictions = [
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0.8, 0.1, 0.1],
        [0.5, 0.25, 0.25],
        [0.35, 0.3, 0.35],
        [0.6, 0.3, 0.1],
        [0.6, 0.25, 0.15],
        [0.6, 0.15, 0.25],
        [0.57, 0.33, 0.1],
        [0.6, 0.2, 0.2],
    ]

    observed = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

.. code-block:: python

    pb.metrics.multiclass_brier_score(predictions, observed)

.. code-block:: text

    0.30838
