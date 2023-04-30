import pytest

import penaltyblog as pb


def test_elo_rating():
    elo = pb.ratings.Elo(k=32)
    elo.add_player("Bob", 2000)
    elo.add_player("Clare", 1800)
    elo.update_ratings("Bob", "Clare", 0)

    r_a = elo.get_rating("Bob")
    r_b = elo.get_rating("Clare")

    expected_diff = 7.68

    assert pytest.approx(r_a, 0.1) == 2000 + expected_diff
    assert pytest.approx(r_b, 0.1) == 1800 - expected_diff

    elo = pb.ratings.Elo(k=32)
    elo.add_player("Bob", 2000)
    elo.add_player("Clare", 1800)
    elo.update_ratings("Bob", "Clare", 1)

    r_a = elo.get_rating("Bob")
    r_b = elo.get_rating("Clare")

    expected_diff = 24.32

    assert pytest.approx(r_a, 0.1) == 2000 - expected_diff
    assert pytest.approx(r_b, 0.1) == 1800 + expected_diff
