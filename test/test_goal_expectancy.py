import numpy as np

import penaltyblog as pb


def test_goal_expectancy():
    probs = (0.093897, 0.158581, 0.747522)
    exp = pb.models.goal_expectancy(*probs, dc_adj=False)

    assert exp["success"] is True
    assert 0.72 < exp["home_exp"] < 0.75
    assert 2.225 < exp["away_exp"] < 2.5
    assert 0.99 < np.array(probs).sum() < 1.01


def test_goal_expectancy_dc_adj():
    probs = (0.093897, 0.158581, 0.747522)
    exp = pb.models.goal_expectancy(*probs)

    assert exp["success"] is True
    assert 0.72 < exp["home_exp"] < 0.75
    assert 2.225 < exp["away_exp"] < 2.5
    assert 0.99 < np.array(probs).sum() < 1.01
