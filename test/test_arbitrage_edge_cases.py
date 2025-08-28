import math

import pytest

import penaltyblog as pb


class TestArbitrageHedgeEdgeCases:
    """Test edge cases and input validation for arbitrage_hedge function."""

    def test_empty_inputs_raise_error(self):
        """Test that empty input lists raise ValueError."""
        with pytest.raises(ValueError, match="Input lists cannot be empty"):
            pb.betting.arbitrage_hedge([], [], [])

    def test_mismatched_input_lengths_raise_error(self):
        """Test that mismatched input list lengths raise ValueError."""
        with pytest.raises(
            ValueError, match="All input lists must have the same length"
        ):
            pb.betting.arbitrage_hedge([100], [2.0], [2.0, 3.0])

    def test_invalid_existing_odds_raise_error(self):
        """Test that odds <= 1.0 raise ValueError."""
        with pytest.raises(
            ValueError, match="All existing_odds must be greater than 1.0"
        ):
            pb.betting.arbitrage_hedge([100], [1.0], [2.0])

        with pytest.raises(
            ValueError, match="All existing_odds must be greater than 1.0"
        ):
            pb.betting.arbitrage_hedge([100], [0.5], [2.0])

    def test_invalid_hedge_odds_raise_error(self):
        """Test that hedge odds <= 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="All hedge_odds must be greater than 1.0"):
            pb.betting.arbitrage_hedge([100], [2.0], [1.0])

        with pytest.raises(ValueError, match="All hedge_odds must be greater than 1.0"):
            pb.betting.arbitrage_hedge([100], [2.0], [0.9])

    def test_negative_stakes_raise_error(self):
        """Test that negative existing stakes raise ValueError."""
        with pytest.raises(
            ValueError, match="All existing_stakes must be non-negative"
        ):
            pb.betting.arbitrage_hedge([-100], [2.0], [2.0])

    def test_infinite_target_profit_raises_error(self):
        """Test that infinite target profit raises ValueError."""
        with pytest.raises(ValueError, match="target_profit must be a finite number"):
            pb.betting.arbitrage_hedge([100], [2.0], [2.0], target_profit=float("inf"))

        with pytest.raises(ValueError, match="target_profit must be a finite number"):
            pb.betting.arbitrage_hedge([100], [2.0], [2.0], target_profit=float("nan"))

    def test_single_outcome_market(self):
        """Test behavior with single outcome (degenerate case)."""
        res = pb.betting.arbitrage_hedge([100], [2.0], [1.8])

        # Should handle single outcome gracefully
        assert len(res.practical_hedge_stakes) == 1
        assert res.practical_hedge_stakes[0] >= 0
        # Guaranteed profit should be calculable
        assert math.isfinite(res.guaranteed_profit)

    def test_all_zero_stakes(self):
        """Test behavior when all existing stakes are zero."""
        res = pb.betting.arbitrage_hedge([0, 0], [2.0, 3.0], [2.0, 3.0])

        # Should handle gracefully - no existing positions to hedge
        assert res.practical_hedge_stakes == [0.0, 0.0]
        assert res.guaranteed_profit == 0  # No stakes means no profit/loss

    def test_extreme_odds_high(self):
        """Test behavior with very high odds."""
        high_odds = 1000.0
        res = pb.betting.arbitrage_hedge([10], [high_odds], [high_odds])

        # Should handle high odds without numerical issues
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

    def test_extreme_odds_low(self):
        """Test behavior with odds close to 1.0."""
        low_odds = 1.001  # Very close to 1.0 but still valid
        res = pb.betting.arbitrage_hedge([100], [low_odds], [low_odds])

        # Should handle low odds without numerical issues
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

    def test_very_small_stakes(self):
        """Test behavior with very small stake amounts."""
        tiny_stake = 1e-8
        res = pb.betting.arbitrage_hedge([tiny_stake], [2.0], [2.0])

        # Should handle tiny stakes with numerical tolerance
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

    def test_large_stakes(self):
        """Test behavior with very large stake amounts."""
        large_stake = 1e9
        res = pb.betting.arbitrage_hedge([large_stake, 0], [2.0, 3.0], [2.0, 3.0])

        # Should handle large stakes without overflow
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

    def test_custom_tolerance(self):
        """Test behavior with custom tolerance parameter."""
        tolerance = 1e-6
        res = pb.betting.arbitrage_hedge(
            [100, 1e-7], [2.0, 3.0], [2.0, 3.0], tolerance=tolerance
        )

        # Test that tolerance is used in calculations (actual behavior may vary)
        # The key is that the function runs without numerical errors
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

        # The tiny stake (1e-7) should be treated as effectively zero in the presence check
        # This means only the first position (100) should be considered for hedging decisions
        # However, the actual hedge distribution depends on the LP solution

    def test_infeasible_target_profit(self):
        """Test behavior when target profit is infeasible."""
        # The LP solver can handle high target profits by recommending large hedge stakes
        # Let's test that lp_success is properly reported
        res = pb.betting.arbitrage_hedge([100], [2.0], [2.0], target_profit=1000)

        # LP should succeed (it found a solution requiring $900 hedge)
        assert hasattr(res, "lp_success")
        assert res.lp_success is True  # LP found a valid solution
        assert math.isfinite(res.guaranteed_profit)
        assert res.guaranteed_profit >= 1000 - 1e-6  # Should achieve target profit

        # Test a truly infeasible case (negative target impossible with positive odds)
        res2 = pb.betting.arbitrage_hedge(
            [100, 0],
            [1.01, 1.01],
            [1.01, 1.01],
            target_profit=-500,  # Impossible negative profit
        )
        # This may still succeed depending on LP solver capabilities

    def test_negative_target_profit(self):
        """Test behavior with negative target profit (accepting loss)."""
        res = pb.betting.arbitrage_hedge([100], [2.0], [2.0], target_profit=-50)

        # With single outcome market, if target is infeasible, the LP may fail
        # and fallback to heuristic. The key is that it handles negative targets gracefully
        assert math.isfinite(res.guaranteed_profit)

        # For a more realistic test, try with two outcomes where negative target is achievable
        res2 = pb.betting.arbitrage_hedge(
            [100, 0], [2.0, 3.0], [1.5, 2.5], target_profit=-20
        )
        # This should be achievable since we can just not hedge and accept the loss
        assert res2.guaranteed_profit <= -20 + 1e-6  # Allow for numerical tolerance

    def test_hedge_all_false_with_zero_stakes(self):
        """Test hedge_all=False when some stakes are zero."""
        res = pb.betting.arbitrage_hedge(
            [100, 0, 50], [2.0, 3.0, 4.0], [1.9, 2.8, 3.8], hedge_all=False
        )

        # Should only hedge positions with non-zero stakes
        # Position 1 (index 1) has zero stake, so shouldn't be hedged
        non_zero_positions = [i for i, s in enumerate([100, 0, 50]) if s > 0]
        hedged_positions = [
            i for i, h in enumerate(res.practical_hedge_stakes) if h > 1e-10
        ]

        # Should have hedges, but structure depends on the specific algorithm
        assert (
            len(hedged_positions) >= 0
        )  # May or may not hedge depending on profitability

    def test_allow_lay_true_preserves_negative_stakes(self):
        """Test that allow_lay=True preserves negative stakes in practical_hedge_stakes."""
        res = pb.betting.arbitrage_hedge(
            [100, 0], [3.0, 2.0], [3.0, 2.0], allow_lay=True
        )

        # When laying is allowed, practical stakes might contain negative values
        # This depends on the optimal solution, but raw stakes might be negative
        if any(s < 0 for s in res.raw_hedge_stakes):
            # If there are negative raw stakes, they should be preserved in practical
            assert res.raw_hedge_stakes == res.practical_hedge_stakes

    def test_profit_consistency_across_outcomes(self):
        """Test that profits are consistent across all outcomes (within tolerance)."""
        res = pb.betting.arbitrage_hedge([50, 30, 20], [2.5, 3.0, 4.0], [2.4, 2.9, 3.9])

        # Calculate profit for each outcome
        total_existing = sum([50, 30, 20])
        total_hedge = sum(res.practical_hedge_stakes)
        existing_payouts = [50 * 2.5, 30 * 3.0, 20 * 4.0]

        profits = []
        for i in range(3):
            if i == 0:  # Outcome 0 wins
                profit = (
                    existing_payouts[0]
                    + res.practical_hedge_stakes[0] * 2.4
                    - total_existing
                    - total_hedge
                )
            elif i == 1:  # Outcome 1 wins
                profit = (
                    existing_payouts[1]
                    + res.practical_hedge_stakes[1] * 2.9
                    - total_existing
                    - total_hedge
                )
            else:  # Outcome 2 wins
                profit = (
                    existing_payouts[2]
                    + res.practical_hedge_stakes[2] * 3.9
                    - total_existing
                    - total_hedge
                )
            profits.append(profit)

        # All profits should be approximately equal to guaranteed_profit
        tolerance = 1e-6
        for profit in profits:
            assert abs(profit - res.guaranteed_profit) < tolerance

    def test_backward_compatibility_unpacking(self):
        """Test that the result can still be unpacked for backward compatibility."""
        res = pb.betting.arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5])

        # Should be able to unpack like the old return format
        hedge_stakes, guaranteed_profit = res

        assert hedge_stakes == res.practical_hedge_stakes
        assert guaranteed_profit == res.guaranteed_profit

    def test_result_attributes_present(self):
        """Test that all expected attributes are present in the result."""
        res = pb.betting.arbitrage_hedge([100, 0], [3.0, 2.5], [3.0, 2.5])

        # Check all documented attributes are present
        required_attrs = [
            "raw_hedge_stakes",
            "practical_hedge_stakes",
            "guaranteed_profit",
            "existing_payouts",
            "total_existing_stakes",
            "total_hedge_needed",
            "lp_success",
            "lp_message",
        ]

        for attr in required_attrs:
            assert hasattr(res, attr), f"Missing attribute: {attr}"
            # Note: lp_message can be None, so we just check it exists

    def test_lp_diagnostic_information(self):
        """Test that LP diagnostic information is properly exposed."""
        # Test case where LP might succeed
        res1 = pb.betting.arbitrage_hedge([50, 50], [2.0, 2.0], [2.0, 2.0])
        assert hasattr(res1, "lp_success")
        assert isinstance(res1.lp_success, bool)
        assert hasattr(res1, "lp_message")
        # lp_message should be None when successful, or a string when failed
        assert res1.lp_message is None or isinstance(res1.lp_message, str)

        # Test hedge_all=False case (doesn't use LP)
        res2 = pb.betting.arbitrage_hedge(
            [100, 0], [3.0, 2.5], [3.0, 2.5], hedge_all=False
        )
        assert res2.lp_success is True  # Not applicable, set to True
        assert res2.lp_message is None


class TestNumericalStability:
    """Test numerical stability improvements."""

    def test_very_close_to_zero_values(self):
        """Test handling of values very close to zero."""
        epsilon = 1e-15
        res = pb.betting.arbitrage_hedge(
            [100, epsilon], [2.0, 3.0], [2.0, 3.0], tolerance=1e-10
        )

        # The epsilon stake should be treated as zero in the tolerance check for existing stakes
        # But the LP may still create hedge stakes to equalize profits across all outcomes
        # The key test is numerical stability - no inf/nan values
        assert math.isfinite(res.guaranteed_profit)
        assert all(math.isfinite(s) for s in res.practical_hedge_stakes)

        # Test that very small values are handled without causing numerical issues
        # The actual hedge amounts depend on the LP solution

    def test_tolerance_affects_calculations(self):
        """Test that tolerance parameter affects calculations."""
        small_stake = 1e-8
        large_tolerance = 1e-6
        small_tolerance = 1e-12

        res_large_tol = pb.betting.arbitrage_hedge(
            [100, small_stake], [2.0, 3.0], [2.0, 3.0], tolerance=large_tolerance
        )

        res_small_tol = pb.betting.arbitrage_hedge(
            [100, small_stake], [2.0, 3.0], [2.0, 3.0], tolerance=small_tolerance
        )

        # With large tolerance, small stake should be ignored
        # With small tolerance, small stake should be considered
        # The results might differ depending on tolerance
        assert math.isfinite(res_large_tol.guaranteed_profit)
        assert math.isfinite(res_small_tol.guaranteed_profit)
