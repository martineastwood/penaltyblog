import pytest

import penaltyblog as pb


def test_lp_equalizes_profits_three_way():
    # Three-way market example
    existing_stakes = [50, 30, 0]
    existing_odds = [2.5, 4.0, 3.0]
    hedge_odds = [2.4, 3.8, 2.9]

    res = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)

    # LP should succeed and practical stakes should be non-negative
    assert getattr(res, "lp_success", True) is True
    stakes = res.practical_hedge_stakes
    assert all(s >= 0 for s in stakes)

    # Per-outcome profits should be (approximately) equal to the guaranteed profit
    existing_payouts = [s * o for s, o in zip(existing_stakes, existing_odds)]
    total_existing = sum(existing_stakes)
    total_practical = sum(stakes)

    profits = [
        existing_payouts[i]
        + stakes[i] * hedge_odds[i]
        - total_existing
        - total_practical
        for i in range(len(stakes))
    ]

    for p in profits:
        assert abs(p - res.guaranteed_profit) < 1e-6


def test_infeasible_target_sets_lp_success_false():
    existing_stakes = [100, 0]
    existing_odds = [2.0, 2.0]
    hedge_odds = [2.0, 2.0]

    # Very large target which is infeasible
    res = pb.betting.arbitrage_hedge(
        existing_stakes, existing_odds, hedge_odds, target_profit=1e6
    )
    assert getattr(res, "lp_success", False) is False


def test_allow_lay_returns_negative_raw_when_allowed():
    existing_stakes = [100, 0]
    existing_odds = [3.0, 2.0]
    hedge_odds = [3.0, 2.0]

    # By default laying is not allowed -> practical stakes non-negative
    res_default = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)
    assert all(s >= 0 for s in res_default.practical_hedge_stakes)

    # When allow_lay=True, raw_hedge_stakes may contain negative values
    res_lay = pb.betting.arbitrage_hedge(
        existing_stakes, existing_odds, hedge_odds, allow_lay=True
    )
    assert any(s < 0 for s in res_lay.raw_hedge_stakes)
