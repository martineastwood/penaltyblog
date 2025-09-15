Ranked Probability Scores (RPS)
================================

The Ranked Probability Score (RPS) is a metric used to evaluate the accuracy of probabilistic predictions for ordinal outcomes, making it particularly useful in sports analytics where results naturally have an order (win > draw > loss).

Unlike traditional accuracy metrics that only consider whether the predicted outcome was correct, RPS accounts for the entire probability distribution across all possible outcomes and penalizes predictions based on how "wrong" they are - for instance, predicting a home win when the actual result is a draw incurs a smaller penalty than predicting a home win when the away team wins.

Mathematically, RPS is calculated as the sum of squared differences between the cumulative probability distributions of the predicted and observed outcomes, where a lower score indicates better prediction accuracy.

In soccer analytics, RPS is especially valuable for evaluating betting models and match prediction systems because it rewards models that assign high probabilities to outcomes "close" to the actual result, providing a more nuanced assessment than simple binary accuracy or log-loss metrics that don't account for the ordinal relationship between match outcomes.

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

    pb.metrics.rps_array(predictions, observed)

.. code-block:: text

    array([0.     , 0.005  , 0.025  , 0.15625, 0.1225 , 0.185  , 0.09125,
       0.11125, 0.09745, 0.1])

.. code-block:: python

    pb.metrics.rps_average(predictions, observed)

.. code-block:: text

    0.08937
