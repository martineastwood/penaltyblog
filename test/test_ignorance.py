import numpy as np

import penaltyblog as pb


def test_basic_case():
    y_true = [0, 1, 2]
    y_prob = [
        [0.7, 0.2, 0.1],  # true = 0 → log2(0.7)
        [0.1, 0.8, 0.1],  # true = 1 → log2(0.8)
        [0.2, 0.3, 0.5],  # true = 2 → log2(0.5)
    ]
    score = pb.metrics.ignorance_score(y_true, y_prob)
    expected = -np.mean(
        [
            np.log2(0.7),
            np.log2(0.8),
            np.log2(0.5),
        ]
    )
    assert np.isclose(score, expected, atol=1e-8)


def test_perfect_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    score = pb.metrics.ignorance_score(y_true, y_prob)
    assert np.isclose(score, 0.0, atol=1e-8)


def test_uniform_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 1 / 3],
    ]
    score = pb.metrics.ignorance_score(y_true, y_prob)
    expected = -np.log2(1 / 3)  # ≈ 1.58496
    assert np.isclose(score, expected, atol=1e-5)


def test_zero_probability_handling():
    y_true = [0]
    y_prob = [[0.0, 0.5, 0.5]]  # Zero probability on actual class
    score = pb.metrics.ignorance_score(y_true, y_prob)
    assert np.isfinite(score)
    assert score > 10  # Should be a large positive number due to penalty


def test_multiple_samples_same_prediction():
    y_true = [0, 0, 0]
    y_prob = [
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
    ]
    expected = -np.log2(0.9)
    score = pb.metrics.ignorance_score(y_true, y_prob)
    assert np.isclose(score, expected, atol=1e-8)


def test_input_as_numpy_arrays():
    y_true = np.array([1, 2])
    y_prob = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )
    expected = -np.mean([np.log2(0.7), np.log2(0.6)])
    score = pb.metrics.ignorance_score(y_true, y_prob)
    assert np.isclose(score, expected, atol=1e-8)
