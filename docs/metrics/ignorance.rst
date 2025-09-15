Ignorance Score
======================

The Ignorance Score, also known as the logarithmic scoring rule or log-loss, is a strictly proper scoring metric that evaluates probabilistic predictions by calculating the negative logarithm of the probability assigned to the actual outcome that occurred.

In practical terms, it measures how "surprised" a model is by the actual result - assigning a probability of 0.1 to an event that actually happens yields a high ignorance score of -log(0.1) ≈ 2.3, while correctly assigning high probability like 0.9 yields a low score of -log(0.9) ≈ 0.1.

The score ranges from 0 to infinity, where lower values indicate better predictions, and it heavily penalizes overconfident wrong predictions since the logarithm approaches infinity as probabilities approach zero.

In soccer analytics, the ignorance score is particularly useful for evaluating match prediction models because it forces forecasters to be well-calibrated across their entire probability range - a model cannot achieve good scores by simply being accurate on favorites while making poor predictions on underdogs.

Unlike the Brier score which uses squared differences, the logarithmic nature of the ignorance score makes it more sensitive to extreme probability assignments, making it especially valuable when the cost of being completely wrong is high, such as in betting scenarios where confidently backing the wrong outcome can be catastrophic.

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

    pb.metrics.ignorance_score(predictions, observed)

.. code-block:: text

    0.7969725334773428
