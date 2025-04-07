import numpy as np

import penaltyblog as pb


def test_perfect_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    score = pb.metrics.multiclass_brier_score(
        np.array(y_prob, dtype=np.float64), np.array(y_true, dtype=np.int32)
    )
    assert np.isclose(score, 0.0, atol=1e-8)


def test_uniform_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 3, 1 / 3, 1 / 3],
    ]
    expected_per_sample = 2 * ((1 / 3) ** 2) + ((2 / 3) ** 2)
    expected = expected_per_sample
    score = pb.metrics.multiclass_brier_score(
        np.array(y_prob, dtype=np.float64), np.array(y_true, dtype=np.int32)
    )
    assert np.isclose(score, expected, atol=1e-6)


def test_wrong_but_confident_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    expected = 2.0
    score = pb.metrics.multiclass_brier_score(
        np.array(y_prob, dtype=np.float64), np.array(y_true, dtype=np.int32)
    )
    assert np.isclose(score, expected, atol=1e-8)


def test_partial_confidence_predictions():
    y_true = [0, 1, 2]
    y_prob = [
        [0.6, 0.2, 0.2],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ]
    expected = np.mean(
        [
            (0.6 - 1) ** 2 + 0.2**2 + 0.2**2,
            (0.7 - 1) ** 2 + 0.1**2 + 0.2**2,
            (0.6 - 1) ** 2 + 0.2**2 + 0.2**2,
        ]
    )
    score = pb.metrics.multiclass_brier_score(
        np.array(y_prob, dtype=np.float64), np.array(y_true, dtype=np.int32)
    )
    assert np.isclose(score, expected, atol=1e-8)


def test_numpy_shapes():
    y_true = np.array([2, 0], dtype=np.int32)
    y_prob = np.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]], dtype=np.float64)
    score = pb.metrics.multiclass_brier_score(
        np.array(y_prob, dtype=np.float64), np.array(y_true, dtype=np.int32)
    )
    expected = np.mean(
        [
            (0.8 - 1) ** 2 + 0.1**2 + 0.1**2,
            (0.9 - 1) ** 2 + 0.05**2 + 0.05**2,
        ]
    )
    assert np.isclose(score, expected, atol=1e-8)
