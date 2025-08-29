import numpy as np
import pytest

import penaltyblog as pb
from penaltyblog.betting.value_bets import (
    ArbitrageResult,
    MultipleValueBetResult,
    ValueBetResult,
    calculate_bet_value,
    find_arbitrage_opportunities,
    identify_value_bet,
)


class TestValueBetIdentification:
    """Test the identify_value_bet function."""

    def test_single_value_bet(self):
        """Test identification of a single value bet."""
        # Bookmaker odds 2.5 (implied prob 40%), our estimate 50% -> value bet
        result = identify_value_bet(2.5, 0.50)

        assert isinstance(result, ValueBetResult)
        assert result.bookmaker_odds == 2.5
        assert result.estimated_probability == 0.5
        assert result.implied_probability == pytest.approx(0.4, abs=1e-6)
        assert result.edge == pytest.approx(0.1, abs=1e-6)  # 50% - 40%
        assert result.expected_value > 0
        assert result.is_value_bet is True
        assert result.recommended_stake_kelly > 0
        assert result.potential_profit == 1.5  # 2.5 - 1
        assert result.potential_loss == 1.0

    def test_single_fair_bet(self):
        """Test a fair bet with no value."""
        # Bookmaker odds 2.0 (implied prob 50%), our estimate 50% -> fair bet
        result = identify_value_bet(2.0, 0.50)

        assert isinstance(result, ValueBetResult)
        assert result.expected_value == pytest.approx(0.0, abs=1e-6)
        assert result.edge == pytest.approx(0.0, abs=1e-6)
        assert result.is_value_bet is False
        assert result.recommended_stake_kelly == pytest.approx(0.0, abs=1e-6)

    def test_single_negative_value_bet(self):
        """Test a bet with negative expected value."""
        # Bookmaker odds 2.0 (implied prob 50%), our estimate 40% -> negative value
        result = identify_value_bet(2.0, 0.40)

        assert isinstance(result, ValueBetResult)
        assert result.expected_value < 0
        assert result.edge < 0
        assert result.is_value_bet is False
        assert result.recommended_stake_kelly == 0.0

    def test_multiple_value_bets(self):
        """Test identification of multiple value bets."""
        odds = [2.0, 3.0, 1.8]
        probs = [0.6, 0.4, 0.5]  # First two are value bets, third is not

        result = identify_value_bet(odds, probs)

        assert isinstance(result, MultipleValueBetResult)
        assert len(result.individual_results) == 3
        assert result.total_value_bets == 2

        # Check individual results
        assert result.individual_results[0].is_value_bet is True  # 2.0 odds, 60% prob
        assert result.individual_results[1].is_value_bet is True  # 3.0 odds, 40% prob
        assert result.individual_results[2].is_value_bet is False  # 1.8 odds, 50% prob

        # Portfolio metrics
        assert result.average_edge > 0
        assert result.total_expected_value > 0
        assert result.best_value_bet_index in [0, 1]  # One of the value bets

    def test_kelly_fraction_scaling(self):
        """Test that Kelly fraction parameter scales stakes correctly."""
        odds = 2.5
        prob = 0.5

        result_full = identify_value_bet(odds, prob, kelly_fraction=1.0)
        result_half = identify_value_bet(odds, prob, kelly_fraction=0.5)

        assert result_half.recommended_stake_fraction == pytest.approx(
            result_full.recommended_stake_fraction * 0.5, abs=1e-6
        )

    def test_min_edge_threshold(self):
        """Test minimum edge threshold parameter."""
        odds = 2.1
        prob = 0.49  # Small positive expected value

        # Without threshold, should be value bet
        result_no_threshold = identify_value_bet(odds, prob, min_edge_threshold=0.0)
        assert result_no_threshold.is_value_bet is True

        # With high threshold, should not be value bet
        result_with_threshold = identify_value_bet(odds, prob, min_edge_threshold=0.05)
        assert result_with_threshold.is_value_bet is False

    def test_numpy_array_inputs(self):
        """Test that numpy arrays work as inputs."""
        odds_array = np.array([2.0, 3.0, 1.5])
        probs_array = np.array([0.6, 0.4, 0.7])

        result = identify_value_bet(odds_array, probs_array)

        assert isinstance(result, MultipleValueBetResult)
        assert len(result.individual_results) == 3

    def test_validation_errors(self):
        """Test validation errors for arbitrage function."""
        # Empty list
        result = find_arbitrage_opportunities([])
        assert isinstance(result, ArbitrageResult)
        assert result.has_arbitrage is False

        # Mismatched outcome counts
        with pytest.raises(ValueError, match="same number of outcomes"):
            find_arbitrage_opportunities([[2.0, 1.8], [2.1]])

        # Invalid odds
        with pytest.raises(ValueError, match="must be > 1.0"):
            find_arbitrage_opportunities([[2.0, 0.8], [2.1, 1.9]])

    def test_extreme_values(self):
        """Test extreme but valid values."""
        # Very high odds, low probability
        result = identify_value_bet(100.0, 0.02)
        assert isinstance(result, ValueBetResult)
        assert result.expected_value > 0  # Should be value bet

        # Very low odds, high probability
        result = identify_value_bet(1.01, 0.99)
        assert isinstance(result, ValueBetResult)
        assert result.expected_value == pytest.approx(0.0, abs=0.01)

    def test_portfolio_metrics(self):
        """Test portfolio-level metrics are calculated correctly."""
        odds = [2.0, 2.0, 2.0]
        probs = [0.6, 0.5, 0.4]  # One value, one fair, one negative

        result = identify_value_bet(odds, probs)

        # Should have one value bet
        assert result.total_value_bets == 1

        # Portfolio overround should be sum of implied probabilities
        expected_overround = sum(1.0 / odd for odd in odds)
        assert result.portfolio_overround == pytest.approx(expected_overround, abs=1e-6)

        # Total expected value should sum individual EVs
        expected_total_ev = sum(r.expected_value for r in result.individual_results)
        assert result.total_expected_value == pytest.approx(expected_total_ev, abs=1e-6)


class TestCalculateBetValue:
    """Test the calculate_bet_value utility function."""

    def test_positive_value(self):
        """Test calculation of positive expected value."""
        value = calculate_bet_value(2.0, 0.6)  # 60% chance at 2.0 odds
        expected = (0.6 * 1.0) - (0.4 * 1.0)  # 0.6 profit - 0.4 loss = 0.2
        assert value == pytest.approx(expected, abs=1e-6)

    def test_zero_value(self):
        """Test calculation of zero expected value."""
        value = calculate_bet_value(2.0, 0.5)  # Fair odds
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_negative_value(self):
        """Test calculation of negative expected value."""
        value = calculate_bet_value(2.0, 0.4)  # 40% chance at 2.0 odds
        assert value < 0

    def test_validation_errors(self):
        """Test input validation for calculate_bet_value."""
        with pytest.raises(ValueError, match="must be greater than 1.0"):
            calculate_bet_value(1.0, 0.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            calculate_bet_value(2.0, -0.1)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            calculate_bet_value(2.0, 1.5)


class TestArbitrageOpportunities:
    """Test the find_arbitrage_opportunities function."""

    def test_arbitrage_exists(self):
        """Test detection of arbitrage opportunity."""
        # Two bookmakers with complementary better odds
        odds = [
            [2.1, 1.85],  # Bookmaker 1: better odds on outcome 1
            [1.95, 2.0],  # Bookmaker 2: better odds on outcome 2
        ]

        result = find_arbitrage_opportunities(odds, ["Home", "Away"])

        assert isinstance(result, ArbitrageResult)
        assert result.has_arbitrage is True
        assert result.total_implied_probability < 1.0
        assert result.guaranteed_return > 0
        assert len(result.best_odds) == 2
        assert len(result.stake_percentages) == 2
        assert sum(result.stake_percentages) == pytest.approx(1.0, abs=1e-6)

    def test_no_arbitrage(self):
        """Test when no arbitrage opportunity exists."""
        odds = [[2.0, 1.8], [1.9, 1.7]]  # Both bookmakers have similar unfavorable odds

        result = find_arbitrage_opportunities(odds)

        assert isinstance(result, ArbitrageResult)
        assert result.has_arbitrage is False
        assert result.total_implied_probability >= 1.0
        assert result.guaranteed_return == 0.0
        assert all(stake == 0.0 for stake in result.stake_percentages)

    def test_three_way_arbitrage(self):
        """Test arbitrage detection in three-way market."""
        odds = [
            [2.2, 3.5, 4.0],  # Bookmaker 1
            [2.0, 3.8, 3.8],  # Bookmaker 2
            [2.1, 3.4, 4.2],  # Bookmaker 3
        ]

        result = find_arbitrage_opportunities(odds, ["Home", "Draw", "Away"])

        assert isinstance(result, ArbitrageResult)
        assert len(result.best_odds) == 3
        assert len(result.outcome_labels) == 3
        assert result.num_bookmakers == 3
        assert result.num_outcomes == 3
        # May or may not be arbitrage, but should handle three outcomes

    def test_validation_errors(self):
        """Test validation errors for arbitrage function."""
        # Empty list
        result = find_arbitrage_opportunities([])
        assert isinstance(result, ArbitrageResult)
        assert result.has_arbitrage is False

        # Mismatched outcome counts
        with pytest.raises(ValueError, match="same number of outcomes"):
            find_arbitrage_opportunities([[2.0, 1.8], [2.1]])

        # Invalid odds
        with pytest.raises(ValueError, match="must be > 1.0"):
            find_arbitrage_opportunities([[2.0, 0.8], [2.1, 1.9]])

    def test_outcome_labels(self):
        """Test outcome label handling."""
        odds = [[2.0, 1.8], [1.9, 1.9]]

        # With custom labels
        result = find_arbitrage_opportunities(odds, ["Team A", "Team B"])
        assert isinstance(result, ArbitrageResult)
        assert result.outcome_labels == ["Team A", "Team B"]

        # Without labels (should auto-generate)
        result = find_arbitrage_opportunities(odds)
        assert result.outcome_labels == ["Outcome_1", "Outcome_2"]


class TestIntegrationScenarios:
    """Test realistic value betting scenarios."""

    def test_football_match_value_betting(self):
        """Test a realistic football match value betting scenario."""
        # Premier League match: Man City vs Brighton
        # Bookmaker odds: City 1.3, Draw 5.5, Brighton 9.0
        # Your model: City 80%, Draw 12%, Brighton 8%

        bookmaker_odds = [1.3, 5.5, 9.0]
        estimated_probs = [0.80, 0.12, 0.08]

        result = identify_value_bet(bookmaker_odds, estimated_probs)

        assert isinstance(result, MultipleValueBetResult)

        # Check if any are identified as value bets
        # Man City: 1.3 odds (76.9% implied) vs 80% estimated -> small value
        # Draw: 5.5 odds (18.2% implied) vs 12% estimated -> negative value
        # Brighton: 9.0 odds (11.1% implied) vs 8% estimated -> negative value

        city_result = result.individual_results[0]
        assert city_result.edge > 0  # Should have small positive edge

    def test_tennis_match_hedging_scenario(self):
        """Test value betting in the context of an existing position."""
        # You have a position on Player A at 2.5 odds
        # Current odds: Player A 1.8, Player B 2.1
        # Your probabilities: A 60%, B 40%

        current_odds = [1.8, 2.1]
        estimated_probs = [0.60, 0.40]

        result = identify_value_bet(current_odds, estimated_probs)

        # Player A: 1.8 odds (55.6% implied) vs 60% estimated -> value bet
        # Player B: 2.1 odds (47.6% implied) vs 40% estimated -> negative value

        player_a_result = result.individual_results[0]
        player_b_result = result.individual_results[1]

        assert player_a_result.is_value_bet is True
        assert player_b_result.is_value_bet is False

    def test_cross_bookmaker_arbitrage_scenario(self):
        """Test arbitrage detection across multiple bookmakers."""
        # Real-world scenario: Soccer match with slight arbitrage
        bookmaker_odds = [
            [2.05, 3.40, 3.80],  # Bookmaker 1
            [2.10, 3.30, 3.60],  # Bookmaker 2
            [1.98, 3.50, 3.90],  # Bookmaker 3
        ]

        result = find_arbitrage_opportunities(
            bookmaker_odds, ["Home Win", "Draw", "Away Win"]
        )

        # Should find best odds: [2.10, 3.50, 3.90]
        expected_best_odds = [2.10, 3.50, 3.90]
        assert isinstance(result, ArbitrageResult)
        assert result.best_odds == expected_best_odds

        # Calculate if this creates arbitrage
        total_implied = sum(1.0 / odds for odds in expected_best_odds)
        assert result.total_implied_probability == pytest.approx(
            total_implied, abs=1e-6
        )

        if total_implied < 1.0:
            assert result.has_arbitrage is True
            assert result.guaranteed_return > 0

    def test_portfolio_optimization_scenario(self):
        """Test value bet identification for portfolio construction."""
        # Multiple matches with varying value betting opportunities
        matches = [
            ([2.1, 3.2], [0.52, 0.30]),  # Match 1: slight value on favorite
            ([1.8, 2.1], [0.60, 0.35]),  # Match 2: value on favorite
            ([3.0, 1.4], [0.25, 0.70]),  # Match 3: value on underdog
            ([2.0, 2.0], [0.50, 0.50]),  # Match 4: fair odds
        ]

        all_results = []
        for odds, probs in matches:
            result = identify_value_bet(odds, probs)
            all_results.append(result)

        # Count total value bets across all matches
        total_value_bets = sum(r.total_value_bets for r in all_results)

        # Should find multiple value betting opportunities
        assert total_value_bets >= 2

        # Calculate total Kelly stake across portfolio
        total_kelly = sum(sum(r.kelly_stakes) for r in all_results)

        # Should recommend reasonable total exposure
        assert 0 <= total_kelly <= 2.0  # Reasonable range for Kelly stakes
