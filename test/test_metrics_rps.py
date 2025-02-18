import pytest

import penaltyblog as pb


def test_rps():
    predictions = [0.8, 0.1, 0.1]
    observed = 0
    rps_score = pb.metrics.rps(predictions, observed)
    assert 0.024 < rps_score < 0.025


def test_rps():
    predictions = [0.8, 0.1, 0.1]

    with pytest.raises(TypeError):
        _ = pb.metrics.rps(predictions, [0, 1])

    with pytest.raises(ValueError):
        _ = pb.metrics.rps(predictions, 5)
