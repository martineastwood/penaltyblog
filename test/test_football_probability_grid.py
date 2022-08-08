import pytest

import penaltyblog as pb


def test_multiplicative():
    m = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    fpg = pb.models.FootballProbabilityGrid(m, 1, 1)
    assert type(fpg.home_win) == float
    assert type(fpg.draw) == float
    assert type(fpg.away_win) == float
    assert type(fpg.both_teams_to_score) == float
    assert type(fpg.home_draw_away) == list
    assert type(fpg.total_goals("over", 1.5)) == float
    assert type(fpg.asian_handicap("home", 1.5)) == float

    with pytest.raises(ValueError):
        type(fpg.total_goals("wrong", 1.5)) == float

    with pytest.raises(ValueError):
        type(fpg.asian_handicap("wrong", 1.5)) == float
