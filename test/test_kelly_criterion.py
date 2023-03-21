import penaltyblog as pb


def test_kelly_criterion():
    odds = 3.5
    probs = 0.3
    kc = pb.kelly.criterion(odds, probs)
    assert round(kc, 2) == 0.02


def test_kelly_criterion_fraction():
    odds = 3.5
    probs = 0.3
    kc = pb.kelly.criterion(odds, probs, 0.5)
    assert round(kc, 2) == 0.01
