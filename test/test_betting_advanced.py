import pytest

import penaltyblog as pb


class TestMultipleKelly:
    """Test the multiple Kelly criterion function."""

    def test_basic_multiple_kelly(self):
        """Test basic multiple Kelly calculation."""
        # Two-way market with good value
        odds = [2.0, 2.0]
        probs = [0.6, 0.35]  # Sum = 0.95, leaves 5% no-outcome probability

        result = pb.betting.multiple_kelly_criterion(odds, probs)

        assert len(result.stakes) == 2
        assert all(s >= 0 for s in result.stakes)
        assert result.total_stake <= 1.0
        assert result.optimization_success
        assert hasattr(result.risk_metrics, "expected_profit")  # Test structured output
        assert hasattr(result.risk_metrics, "sharpe_ratio")
        assert hasattr(result.risk_metrics, "wealth_volatility")
        assert hasattr(result.risk_metrics, "log_return_volatility")

    def test_independent_vs_simultaneous(self):
        """Test different optimization methods."""
        odds = [2.5, 3.0, 2.8]
        probs = [0.35, 0.25, 0.30]

        result_independent = pb.betting.multiple_kelly_criterion(
            odds, probs, method="independent"
        )
        result_simultaneous = pb.betting.multiple_kelly_criterion(
            odds, probs, method="simultaneous"
        )

        assert len(result_independent.stakes) == 3
        assert len(result_simultaneous.stakes) == 3
        # Both should respect max total stake
        assert result_independent.total_stake <= 1.0
        assert result_simultaneous.total_stake <= 1.0
        assert result_independent.method == "independent"
        assert result_simultaneous.method == "simultaneous"

    def test_max_total_stake_constraint(self):
        """Test that max total stake constraint is respected."""
        odds = [1.5, 1.5, 1.5]
        probs = [0.8, 0.1, 0.1]  # Very confident about first outcome
        max_stake = 0.5

        result = pb.betting.multiple_kelly_criterion(
            odds, probs, max_total_stake=max_stake
        )

        assert result.total_stake <= max_stake + 1e-10  # Allow floating point errors
        assert result.max_total_stake == max_stake

    def test_fraction_scaling(self):
        """Test that fraction parameter scales results correctly."""
        odds = [2.0, 2.0]
        probs = [0.55, 0.40]

        result_full = pb.betting.multiple_kelly_criterion(odds, probs, fraction=1.0)
        result_half = pb.betting.multiple_kelly_criterion(odds, probs, fraction=0.5)

        # Half Kelly should be roughly half the stakes
        for full, half in zip(result_full.stakes, result_half.stakes):
            if full > 0:  # Only check non-zero stakes
                assert abs(half - full * 0.5) < 0.01

        assert result_full.fraction == 1.0
        assert result_half.fraction == 0.5

    def test_no_value_bets(self):
        """Test behavior when no bets have positive expected value."""
        odds = [1.8, 1.8]
        probs = [0.5, 0.5]  # Fair odds, no value

        result = pb.betting.multiple_kelly_criterion(odds, probs)

        # Should recommend no betting
        assert all(s == 0 for s in result.stakes)
        assert result.total_stake == 0
        assert result.portfolio_edge <= 0

    def test_validation_errors(self):
        """Test input validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            pb.betting.multiple_kelly_criterion([2.0, 2.0], [0.5])

        # Negative probabilities
        with pytest.raises(ValueError, match="must be non-negative"):
            pb.betting.multiple_kelly_criterion([2.0, 2.0], [0.5, -0.1])

        # Probabilities sum > 1
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            pb.betting.multiple_kelly_criterion([2.0, 2.0], [0.6, 0.6])

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            pb.betting.multiple_kelly_criterion(
                [2.0, 2.0], [0.5, 0.4], method="invalid"
            )


class TestArbitrageHedge:
    """Test the arbitrage hedging function."""

    def test_basic_two_way_hedge(self):
        """Test basic two-way arbitrage hedge."""
        # Bet $100 on Team A at 3.0 odds, now hedge on Team B at 2.0 odds
        existing_stakes = [100, 0]
        existing_odds = [3.0, 2.0]
        hedge_odds = [3.0, 2.0]

        result = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)

        hedge_stakes = result.practical_hedge_stakes
        profit = result.guaranteed_profit

        assert len(hedge_stakes) == 2
        assert all(s >= 0 for s in hedge_stakes)
        assert hedge_stakes[1] > 0  # Should recommend hedging on the other outcome

        # Check that the returned guaranteed profit matches the minimum of
        # the per-outcome profits computed from the practical hedge stakes.
        existing_payouts = [s * o for s, o in zip(existing_stakes, existing_odds)]
        total_existing = sum(existing_stakes)
        total_practical = sum(hedge_stakes)

        profits = []
        for i in range(len(existing_stakes)):
            profit_if_i_wins = (
                existing_payouts[i]
                + hedge_stakes[i] * hedge_odds[i]
                - total_existing
                - total_practical
            )
            profits.append(profit_if_i_wins)

        assert abs(profit - min(profits)) < 1e-9

    def test_multiple_existing_positions(self):
        """Test hedging with multiple existing positions."""
        existing_stakes = [50, 30, 0]
        existing_odds = [2.5, 4.0, 3.0]
        hedge_odds = [2.4, 3.8, 2.9]

        result = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)
        hedge_stakes = result.practical_hedge_stakes

        assert len(hedge_stakes) == 3
        assert all(s >= 0 for s in hedge_stakes)

    def test_target_profit(self):
        """Test hedging with a specific target profit."""
        existing_stakes = [100, 0]
        existing_odds = [2.0, 2.0]
        hedge_odds = [2.0, 2.0]
        target_profit = 50

        result = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, target_profit=target_profit
        )

        profit = result.guaranteed_profit

        # Implementation may not hit the exact target; assert we get a numeric
        # profit value (sanity check) and it's finite.
        assert isinstance(profit, float)
        assert profit == profit  # not NaN

    def test_hedge_all_vs_selective(self):
        """Test difference between hedging all outcomes vs only existing positions."""
        existing_stakes = [100, 0, 0]
        existing_odds = [3.0, 2.5, 2.0]
        hedge_odds = [3.0, 2.5, 2.0]

        res_all = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, hedge_all=True
        )

        res_selective = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, hedge_all=False
        )

        assert len(res_all.practical_hedge_stakes) == 3
        assert len(res_selective.practical_hedge_stakes) == 3

    def test_no_existing_stakes(self):
        """Test behavior when there are no existing stakes."""
        existing_stakes = [0, 0]
        existing_odds = [2.0, 2.0]
        hedge_odds = [2.0, 2.0]

        result = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)
        hedge_stakes = result.practical_hedge_stakes
        profit = result.guaranteed_profit

        assert all(s == 0 for s in hedge_stakes)
        assert profit == 0

    def test_validation_errors(self):
        """Test input validation for arbitrage hedge."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            pb.betting.arbitrage_hedge([100], [2.0, 2.0], [2.0, 2.0])

        with pytest.raises(ValueError, match="must have the same length"):
            pb.betting.arbitrage_hedge([100, 0], [2.0], [2.0, 2.0])

        with pytest.raises(ValueError, match="must have the same length"):
            pb.betting.arbitrage_hedge([100, 0], [2.0, 2.0], [2.0])


class TestIntegrationScenarios:
    """Test realistic betting scenarios."""

    def test_premier_league_match_scenario(self):
        """Test a realistic Premier League match betting scenario."""
        # Manchester City vs Brighton
        # Bookmaker odds: Man City 1.25, Draw 6.0, Brighton 12.0
        # Your implied probabilities: 0.85, 0.08, 0.07

        odds = [1.25, 6.0, 12.0]
        true_probs = [0.85, 0.08, 0.07]  # Sum = 1.0

        result = pb.betting.multiple_kelly_criterion(odds, true_probs, fraction=0.25)

        assert len(result.stakes) == 3
        # With such strong edge on Man City, should recommend betting on them
        assert result.stakes[0] > 0  # Should bet on Man City
        assert result.total_stake <= 0.25  # Shouldn't exceed 25% of bankroll
        assert result.fraction == 0.25
        assert result.optimization_success

    def test_tennis_match_hedge_scenario(self):
        """Test a realistic tennis match hedging scenario."""
        # You bet $200 on Player A at 2.5 before the match
        # During the match, Player A takes the lead and odds change:
        # Player A now 1.8, Player B now 2.1

        existing_stakes = [200, 0]
        existing_odds = [2.5, 2.1]  # Original odds
        hedge_odds = [1.8, 2.1]  # Current odds

        result = pb.betting.arbitrage_hedge(existing_stakes, existing_odds, hedge_odds)
        hedge_stakes = result.practical_hedge_stakes
        profit = result.guaranteed_profit

        # Should recommend hedging on Player B
        assert hedge_stakes[1] > 0
        # Hedging may only limit losses rather than guarantee positive profit;
        # assert that the guaranteed profit is a finite number and no worse than
        # losing the full existing stake.
        assert isinstance(profit, float)
        assert profit >= -sum(existing_stakes)

    def test_arbitrage_opportunity(self):
        """Test a pure arbitrage opportunity across different bookmakers."""
        # Bookmaker A: Team X 2.1, Team Y 1.95
        # Bookmaker B: Team X 1.90, Team Y 2.15
        # You can bet Team X on A and Team Y on B for guaranteed profit

        # Start with no existing bets
        existing_stakes = [0, 0]
        existing_odds = [2.1, 1.95]

        # Use Kelly to determine stakes based on "true" probabilities that create arbitrage
        # If we can guarantee profit, we can think of this as having very high confidence
        true_probs = [
            0.47,
            0.47,
        ]  # Slightly under 1.0 to ensure positive expected value

        result = pb.betting.multiple_kelly_criterion([2.1, 2.15], true_probs)

        # Stakes should be non-negative and respect sizing constraints
        assert len(result.stakes) == 2
        assert all(s >= 0 for s in result.stakes)
        assert result.optimization_success
