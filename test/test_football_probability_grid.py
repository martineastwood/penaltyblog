import numpy as np
import pytest

import penaltyblog as pb


def test_grid():
    m = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    fpg = pb.models.FootballProbabilityGrid(m, 1, 1)
    assert type(fpg.home_win) == np.float64
    assert type(fpg.draw) == np.float64
    assert type(fpg.away_win) == np.float64
    assert type(fpg.both_teams_to_score) == np.float64
    assert type(fpg.home_draw_away) == list
    assert type(fpg.total_goals("over", 1.5)) == np.float64
    assert type(fpg.total_goals("under", 1.5)) == np.float64
    assert type(fpg.asian_handicap("home", 1.5)) == np.float64
    assert type(fpg.asian_handicap("away", 1.5)) == np.float64

    with pytest.raises(ValueError):
        type(fpg.total_goals("wrong", 1.5)) == np.float64

    with pytest.raises(ValueError):
        type(fpg.asian_handicap("wrong", 1.5)) == np.float64


def test_str():
    m = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    fpg = pb.models.FootballProbabilityGrid(m, 1, 2)
    assert type(str(fpg)) == str
    assert "Class: FootballProbabilityGrid" in str(fpg)
    assert "Home Goal Expectation: 1" in str(fpg)
    assert "Away Goal Expectation: 2" in str(fpg)
