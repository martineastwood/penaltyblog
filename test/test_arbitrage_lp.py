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


def _outcome_profits(existing_stakes, existing_odds, hedge_odds, result):
    paid = sum(existing_stakes) + sum(result.practical_hedge_stakes)
    return [
        stake * odds + hedge * hedge_odd - paid
        for stake, odds, hedge, hedge_odd in zip(
            existing_stakes, existing_odds, result.practical_hedge_stakes, hedge_odds
        )
    ]


@pytest.mark.parametrize(
    "existing_stakes,existing_odds,hedge_odds,expected_hedges,expected_profit",
    [
        ([100, 0], [3.0, 1.5], [2.0, 1.9], [0, 0], -100),
        ([100, 0, 50], [2.0, 3.0, 4.0], [1.9, 2.8, 3.8], [0, 0, 0], -150),
        ([100, 50], [3.0, 2.0], [2.0, 1.9], [0, 200 / 1.9], 150 - 200 / 1.9),
        ([50, 50], [2.0, 2.0], [1.9, 1.9], [0, 0], 0),
        ([0, 0], [3.0, 1.5], [2.0, 1.9], [0, 0], 0),
    ],
)
def test_partial_hedge_optimizes_allowed_back_positions(
    existing_stakes, existing_odds, hedge_odds, expected_hedges, expected_profit
):
    result = pb.betting.arbitrage_hedge(
        existing_stakes, existing_odds, hedge_odds, hedge_all=False
    )
    profits = _outcome_profits(existing_stakes, existing_odds, hedge_odds, result)
    assert result.lp_success
    assert result.practical_hedge_stakes == pytest.approx(expected_hedges)
    assert result.guaranteed_profit == pytest.approx(expected_profit)
    assert result.guaranteed_profit == pytest.approx(min(profits))
    assert (
        min(profits)
        >= min(s * o for s, o in zip(existing_stakes, existing_odds))
        - sum(existing_stakes)
        - 1e-9
    )
    assert all(
        h == 0 for s, h in zip(existing_stakes, result.practical_hedge_stakes) if s == 0
    )


@pytest.mark.parametrize("target_profit,success", [(-100, True), (0, False)])
def test_partial_hedge_target_respects_restrictions(target_profit, success):
    result = pb.betting.arbitrage_hedge(
        [100, 0],
        [3.0, 1.5],
        [2.0, 1.9],
        hedge_all=False,
        target_profit=target_profit,
    )
    assert result.lp_success is success
    assert result.practical_hedge_stakes == [0, 0]
    assert result.guaranteed_profit == -100
    if not success:
        assert result.lp_message


def test_partial_hedge_solver_failure_returns_original_position(monkeypatch):
    from penaltyblog.betting import arbitrage

    monkeypatch.setattr(
        arbitrage, "_solve_hedge_lp", lambda *args, **kwargs: ([], 0, False, "failed")
    )
    result = arbitrage.arbitrage_hedge(
        [100, 50], [3.0, 2.0], [2.0, 1.9], hedge_all=False
    )
    assert result.practical_hedge_stakes == [0, 0]
    assert result.guaranteed_profit == -50
    assert not result.lp_success
    assert result.lp_message == "failed"


def test_partial_hedge_signed_positions_stay_on_existing_outcomes():
    result = pb.betting.arbitrage_hedge(
        [100, 0], [3.0, 1.5], [2.0, 1.9], hedge_all=False, allow_lay=True
    )
    assert result.practical_hedge_stakes == pytest.approx([-150, 0])
    profits = _outcome_profits([100, 0], [3.0, 1.5], [2.0, 1.9], result)
    assert result.guaranteed_profit == pytest.approx(min(profits))
    assert result.guaranteed_profit == pytest.approx(50)


def test_full_hedge_issue_50_control():
    result = pb.betting.arbitrage_hedge([100, 0], [3.0, 1.5], [2.0, 1.9])
    assert result.practical_hedge_stakes == pytest.approx([0, 3000 / 19])
    profits = _outcome_profits([100, 0], [3.0, 1.5], [2.0, 1.9], result)
    assert profits == pytest.approx([800 / 19, 800 / 19])
    assert result.guaranteed_profit == pytest.approx(min(profits))


def test_reported_profit_is_actual_minimum_even_when_target_is_lower():
    result = pb.betting.arbitrage_hedge(
        [100, 100], [2.0, 2.0], [1.9, 1.9], target_profit=-50
    )
    profits = _outcome_profits([100, 100], [2.0, 2.0], [1.9, 1.9], result)
    assert result.lp_success
    assert result.guaranteed_profit == pytest.approx(min(profits))
    assert result.guaranteed_profit == 0


def test_partial_hedge_excludes_stakes_below_tolerance():
    result = pb.betting.arbitrage_hedge(
        [100, 1e-12], [3.0, 1.5], [2.0, 1.9], hedge_all=False
    )
    assert result.practical_hedge_stakes == [0, 0]
    profits = _outcome_profits([100, 1e-12], [3.0, 1.5], [2.0, 1.9], result)
    assert result.guaranteed_profit == pytest.approx(min(profits))
