import numpy as np

import penaltyblog as pb

odds = [2.7, 2.3, 4.4]


def test_multiplicative():
    normalised = pb.implied.multiplicative(odds)
    expected = np.array([0.35873804, 0.42112726, 0.2201347])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001


def test_additive():
    normalised = pb.implied.additive(odds)
    expected = np.array([0.3595618, 0.423974, 0.2164642])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001


def test_power():
    normalised = pb.implied.power(odds)
    expected = np.array([0.3591708, 0.4237291, 0.2171001])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001


def test_shin():
    normalised = pb.implied.shin(odds)
    expected = np.array([0.3593461, 0.4232517, 0.2174022])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001


def test_differential_margin_weighting():
    normalised = pb.implied.differential_margin_weighting(odds)
    expected = np.array([0.3595618, 0.423974, 0.2164642])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001


def test_odds_ratio():
    normalised = pb.implied.odds_ratio(odds)
    expected = np.array([0.3588103, 0.4225611, 0.2186286])
    assert (normalised["implied_probabilities"] - expected).sum() < 0.001
