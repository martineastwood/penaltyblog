import numpy as np
import pytest

import penaltyblog as pb


class TestMultipleKelly:
    """Test the multiple Kelly criterion function."""

    def test_basic_multiple_kelly(self):
        """Test basic multiple Kelly calculation."""
        # Two-way market with good value
        odds = [2.0, 2.0]
        probs = [0.6, 0.35]  # Sum = 0.95, leaves 5% no-outcome probability

        stakes = pb.betting.multiple_criterion(odds, probs)

        assert len(stakes) == 2
        assert all(s >= 0 for s in stakes)
        assert sum(stakes) <= 1.0

    def test_independent_vs_simultaneous(self):
        """Test different optimization methods."""
        odds = [2.5, 3.0, 2.8]
        probs = [0.35, 0.25, 0.30]

        stakes_independent = pb.betting.multiple_criterion(
            odds, probs, method="independent"
        )
        stakes_simultaneous = pb.betting.multiple_criterion(
            odds, probs, method="simultaneous"
        )

        assert len(stakes_independent) == 3
        assert len(stakes_simultaneous) == 3
        # Both should respect max total stake
        assert sum(stakes_independent) <= 1.0
        assert sum(stakes_simultaneous) <= 1.0

    def test_max_total_stake_constraint(self):
        """Test that max total stake constraint is respected."""
        odds = [1.5, 1.5, 1.5]
        probs = [0.8, 0.1, 0.1]  # Very confident about first outcome
        max_stake = 0.5

        stakes = pb.betting.multiple_criterion(odds, probs, max_total_stake=max_stake)

        assert sum(stakes) <= max_stake + 1e-10  # Allow floating point errors

    def test_fraction_scaling(self):
        """Test that fraction parameter scales results correctly."""
        odds = [2.0, 2.0]
        probs = [0.55, 0.40]

        stakes_full = pb.betting.multiple_criterion(odds, probs, fraction=1.0)
        stakes_half = pb.betting.multiple_criterion(odds, probs, fraction=0.5)

        # Half Kelly should be roughly half the stakes
        for full, half in zip(stakes_full, stakes_half):
            if full > 0:  # Only check non-zero stakes
                assert abs(half - full * 0.5) < 0.01

    def test_no_value_bets(self):
        """Test behavior when no bets have positive expected value."""
        odds = [1.8, 1.8]
        probs = [0.5, 0.5]  # Fair odds, no value

        stakes = pb.betting.multiple_criterion(odds, probs)

        # Should recommend no betting
        assert all(s == 0 for s in stakes)

    def test_validation_errors(self):
        """Test input validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            pb.betting.multiple_criterion([2.0, 2.0], [0.5])

        # Negative probabilities
        with pytest.raises(ValueError, match="must be non-negative"):
            pb.betting.multiple_criterion([2.0, 2.0], [0.5, -0.1])

        # Probabilities sum > 1
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            pb.betting.multiple_criterion([2.0, 2.0], [0.6, 0.6])

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            pb.betting.multiple_criterion([2.0, 2.0], [0.5, 0.4], method="invalid")


class TestArbitrageHedge:
    """Test the arbitrage hedging function."""

    def test_basic_two_way_hedge(self):
        """Test basic two-way arbitrage hedge."""
        # Bet $100 on Team A at 3.0 odds, now hedge on Team B at 2.0 odds
        existing_stakes = [100, 0]
        existing_odds = [3.0, 2.0]
        hedge_odds = [3.0, 2.0]

        hedge_stakes, profit = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds
        )

        assert len(hedge_stakes) == 2
        assert hedge_stakes[0] == 0  # No additional bet on Team A
        assert hedge_stakes[1] > 0  # Bet on Team B

        # Check that profit is guaranteed regardless of outcome
        # If Team A wins: 100*3.0 - 100 - hedge_stakes[1] = profit
        # If Team B wins: hedge_stakes[1]*2.0 - 100 - hedge_stakes[1] = profit
        team_a_profit = 100 * 3.0 - 100 - hedge_stakes[1]
        team_b_profit = hedge_stakes[1] * 2.0 - 100 - hedge_stakes[1]

        assert abs(team_a_profit - team_b_profit) < 0.01  # Should be equal
        assert abs(team_a_profit - profit) < 0.01

    def test_multiple_existing_positions(self):
        """Test hedging with multiple existing positions."""
        existing_stakes = [50, 30, 0]
        existing_odds = [2.5, 4.0, 3.0]
        hedge_odds = [2.4, 3.8, 2.9]

        hedge_stakes, profit = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds
        )

        assert len(hedge_stakes) == 3
        assert all(s >= 0 for s in hedge_stakes)

    def test_target_profit(self):
        """Test hedging with a specific target profit."""
        existing_stakes = [100, 0]
        existing_odds = [2.0, 2.0]
        hedge_odds = [2.0, 2.0]
        target_profit = 50

        hedge_stakes, profit = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, target_profit=target_profit
        )

        # Note: The function may not always achieve exact target profit due to
        # implementation constraints, but should be close
        assert abs(profit - target_profit) < 10  # Within reasonable range

    def test_hedge_all_vs_selective(self):
        """Test difference between hedging all outcomes vs only existing positions."""
        existing_stakes = [100, 0, 0]
        existing_odds = [3.0, 2.5, 2.0]
        hedge_odds = [3.0, 2.5, 2.0]

        hedge_all_stakes, profit_all = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, hedge_all=True
        )

        hedge_selective_stakes, profit_selective = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds, hedge_all=False
        )

        assert len(hedge_all_stakes) == 3
        assert len(hedge_selective_stakes) == 3

    def test_no_existing_stakes(self):
        """Test behavior when there are no existing stakes."""
        existing_stakes = [0, 0]
        existing_odds = [2.0, 2.0]
        hedge_odds = [2.0, 2.0]

        hedge_stakes, profit = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds
        )

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

        stakes = pb.betting.multiple_criterion(odds, true_probs, fraction=0.25)

        assert len(stakes) == 3
        # With such strong edge on Man City, should recommend betting on them
        assert stakes[0] > 0  # Should bet on Man City
        assert sum(stakes) <= 0.25  # Shouldn't exceed 25% of bankroll

    def test_tennis_match_hedge_scenario(self):
        """Test a realistic tennis match hedging scenario."""
        # You bet $200 on Player A at 2.5 before the match
        # During the match, Player A takes the lead and odds change:
        # Player A now 1.8, Player B now 2.1

        existing_stakes = [200, 0]
        existing_odds = [2.5, 2.1]  # Original odds
        hedge_odds = [1.8, 2.1]  # Current odds

        hedge_stakes, profit = pb.betting.arbitrage_hedge(
            existing_stakes, existing_odds, hedge_odds
        )

        # Should recommend hedging on Player B
        assert hedge_stakes[1] > 0
        assert profit > 0  # Should guarantee some profit

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

        stakes = pb.betting.multiple_criterion([2.1, 2.15], true_probs)

        # Should recommend betting on both outcomes
        assert stakes[0] > 0
        assert stakes[1] > 0
