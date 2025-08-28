"""
Unit tests for the improved implied odds API.
"""

import numpy as np
import pytest

from penaltyblog.implied import (
    ImpliedMethod,
    ImpliedProbabilities,
    OddsFormat,
    OddsInput,
    calculate_implied,
)


class TestOddsInput:
    """Test the OddsInput helper class."""

    def test_decimal_odds_conversion(self):
        """Test decimal odds pass through unchanged."""
        odds_input = OddsInput([2.7, 2.3, 4.4], OddsFormat.DECIMAL)
        decimal = odds_input.to_decimal()
        assert decimal == [2.7, 2.3, 4.4]

    def test_american_odds_conversion(self):
        """Test American odds conversion to decimal."""
        # Positive American odds
        odds_input = OddsInput([170, 130, 340], OddsFormat.AMERICAN)
        decimal = odds_input.to_decimal()
        expected = [2.7, 2.3, 4.4]
        assert np.allclose(decimal, expected, atol=0.01)

        # Negative American odds
        odds_input = OddsInput([-200, -150, -110], OddsFormat.AMERICAN)
        decimal = odds_input.to_decimal()
        expected = [1.5, 1.67, 1.91]
        assert np.allclose(decimal, expected, atol=0.02)

    def test_fractional_odds_conversion(self):
        """Test fractional odds conversion to decimal."""
        odds_input = OddsInput(["7/4", "13/10", "7/2"], OddsFormat.FRACTIONAL)
        decimal = odds_input.to_decimal()
        expected = [2.75, 2.3, 4.5]
        assert np.allclose(decimal, expected, atol=0.01)

    def test_invalid_fractional_format(self):
        """Test error handling for invalid fractional odds."""
        odds_input = OddsInput(["invalid"], OddsFormat.FRACTIONAL)
        with pytest.raises(ValueError, match="Invalid fractional odds format"):
            odds_input.to_decimal()


class TestImpliedProbabilities:
    """Test the ImpliedProbabilities dataclass."""

    def test_valid_probabilities(self):
        """Test creating valid probabilities object."""
        probs = ImpliedProbabilities(
            probabilities=[0.4, 0.4, 0.2],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
        )
        assert len(probs) == 3
        assert probs[0] == 0.4
        assert probs.as_percentages == [40.0, 40.0, 20.0]

    def test_invalid_probability_sum(self):
        """Test validation of probability sums."""
        with pytest.raises(ValueError, match="Probabilities should sum to ~1.0"):
            ImpliedProbabilities(
                probabilities=[0.5, 0.5, 0.5],  # Sums to 1.5
                method=ImpliedMethod.MULTIPLICATIVE,
                margin=0.05,
            )

    def test_backward_compatibility_indexing(self):
        """Test list-like access for backward compatibility."""
        probs = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
        )
        assert probs[0] == 0.35
        assert probs[1] == 0.42
        assert probs[2] == 0.23
        assert len(probs) == 3

    def test_with_market_names(self):
        """Test probabilities with market names."""
        probs = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
            market_names=["Home", "Draw", "Away"],
        )
        assert probs.market_names == ["Home", "Draw", "Away"]
        assert probs["Home"] == 0.35
        assert probs["Draw"] == 0.42
        assert probs["Away"] == 0.23

    def test_most_likely_methods(self):
        """Test most/least likely outcome detection."""
        probs = ImpliedProbabilities(
            probabilities=[0.25, 0.55, 0.20],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
            market_names=["Home", "Draw", "Away"],
        )
        assert probs.most_likely_index == 1
        assert probs.most_likely_probability == 0.55
        assert probs.most_likely_name == "Draw"

        assert probs.least_likely_index == 2
        assert probs.least_likely_probability == 0.20
        assert probs.least_likely_name == "Away"

    def test_get_probability_by_name(self):
        """Test getting probability by outcome name."""
        probs = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
            market_names=["Home", "Draw", "Away"],
        )
        assert probs.get_probability_by_name("Home") == 0.35
        assert probs.get_probability_by_name("Draw") == 0.42
        assert probs.get_probability_by_name("Away") == 0.23

    def test_get_probability_by_name_errors(self):
        """Test error cases for get_probability_by_name."""
        # No market names provided
        probs = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
        )
        with pytest.raises(ValueError, match="No market names provided"):
            probs.get_probability_by_name("Home")

        # Invalid name
        probs_with_names = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
            market_names=["Home", "Draw", "Away"],
        )
        with pytest.raises(ValueError, match="Market name 'Invalid' not found"):
            probs_with_names.get_probability_by_name("Invalid")

    def test_market_names_length_validation(self):
        """Test validation of market names length."""
        with pytest.raises(
            ValueError, match="market_names length .* must match probabilities length"
        ):
            ImpliedProbabilities(
                probabilities=[0.35, 0.42, 0.23],
                method=ImpliedMethod.MULTIPLICATIVE,
                margin=0.05,
                market_names=["Home", "Draw"],  # Only 2 names for 3 probabilities
            )

    def test_to_dict_conversion(self):
        """Test conversion to legacy dict format."""
        probs = ImpliedProbabilities(
            probabilities=[0.35, 0.42, 0.23],
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.05,
            method_params={"k": 1.5},
        )
        result_dict = probs.to_dict()

        expected = {
            "implied_probabilities": [0.35, 0.42, 0.23],
            "method": "multiplicative",
            "margin": 0.05,
            "k": 1.5,
        }
        assert result_dict == expected


class TestCalculateImplied:
    """Test the main calculate_implied function."""

    def test_multiplicative_method(self):
        """Test multiplicative method calculation."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method=ImpliedMethod.MULTIPLICATIVE)

        assert isinstance(result, ImpliedProbabilities)
        assert result.method == ImpliedMethod.MULTIPLICATIVE
        assert np.isclose(sum(result.probabilities), 1.0, atol=1e-10)
        assert result.margin > 0  # Should have positive margin

    def test_all_methods_produce_valid_results(self):
        """Test that all methods produce valid probability distributions."""
        odds = [2.7, 2.3, 4.4]
        methods = [
            ImpliedMethod.MULTIPLICATIVE,
            ImpliedMethod.ADDITIVE,
            ImpliedMethod.POWER,
            ImpliedMethod.SHIN,
            ImpliedMethod.DIFFERENTIAL_MARGIN_WEIGHTING,
            ImpliedMethod.ODDS_RATIO,
            ImpliedMethod.LOGARITHMIC,
        ]

        for method in methods:
            result = calculate_implied(odds, method=method)
            assert isinstance(result, ImpliedProbabilities)
            assert result.method == method
            assert np.isclose(sum(result.probabilities), 1.0, atol=1e-8)
            assert all(0 <= p <= 1 for p in result.probabilities)

    def test_string_method_parameter(self):
        """Test using string method names."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method="multiplicative")
        assert result.method == ImpliedMethod.MULTIPLICATIVE

    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        odds = [2.7, 2.3, 4.4]
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_implied(odds, method="invalid_method")

    def test_different_odds_formats(self):
        """Test calculation with different odds formats."""
        decimal_odds = [2.7, 2.3, 4.4]
        american_odds = [170, 130, 340]

        result1 = calculate_implied(decimal_odds, odds_format=OddsFormat.DECIMAL)
        result2 = calculate_implied(american_odds, odds_format=OddsFormat.AMERICAN)

        # Should produce similar results (within conversion tolerance)
        assert np.allclose(result1.probabilities, result2.probabilities, atol=0.01)

    def test_odds_input_object(self):
        """Test using OddsInput object."""
        odds_input = OddsInput([2.7, 2.3, 4.4], OddsFormat.DECIMAL)
        result = calculate_implied(odds_input)

        assert isinstance(result, ImpliedProbabilities)
        assert np.isclose(sum(result.probabilities), 1.0, atol=1e-10)

    def test_with_market_names(self):
        """Test calculate_implied with market names."""
        odds = [2.7, 2.3, 4.4]
        market_names = ["Home", "Draw", "Away"]
        result = calculate_implied(odds, market_names=market_names)

        assert result.market_names == market_names
        assert result["Home"] == result.probabilities[0]
        assert result["Draw"] == result.probabilities[1]
        assert result["Away"] == result.probabilities[2]
        assert result.most_likely_name == "Draw"

    def test_two_way_market(self):
        """Test calculate_implied with two-way market."""
        odds = [2.0, 1.8]
        market_names = ["Over 2.5", "Under 2.5"]
        result = calculate_implied(odds, market_names=market_names)

        assert len(result.probabilities) == 2
        assert result.market_names == market_names
        assert result["Over 2.5"] == result.probabilities[0]
        assert result["Under 2.5"] == result.probabilities[1]
        assert result.most_likely_name == "Under 2.5"

    def test_many_outcomes_market(self):
        """Test calculate_implied with many outcomes."""
        odds = [9.0, 7.5, 12.0, 6.5, 8.0, 15.0]
        market_names = ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2"]
        result = calculate_implied(odds, market_names=market_names)

        assert len(result.probabilities) == 6
        assert result.market_names == market_names
        assert result.most_likely_name == "1-1"  # Lowest odds = highest probability
        assert result.least_likely_name == "0-2"  # Highest odds = lowest probability

        # Test all names accessible
        for name in market_names:
            assert name in result.market_names
            assert isinstance(result[name], float)

    def test_asian_handicap_market(self):
        """Test calculate_implied with Asian handicap market."""
        odds = [1.95, 1.85]
        market_names = ["Team A -0.5", "Team B +0.5"]
        result = calculate_implied(odds, market_names=market_names)

        assert result.market_names == market_names
        assert result["Team A -0.5"] < result["Team B +0.5"]  # Team B is favorite
        assert result.most_likely_name == "Team B +0.5"


class TestMethodParameters:
    """Test method-specific parameters."""

    def test_power_method_parameters(self):
        """Test that power method returns k parameter."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method=ImpliedMethod.POWER)

        assert result.method_params is not None
        assert "k" in result.method_params
        assert isinstance(result.method_params["k"], float)

    def test_shin_method_parameters(self):
        """Test that Shin method returns z parameter."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method=ImpliedMethod.SHIN)

        assert result.method_params is not None
        assert "z" in result.method_params
        assert isinstance(result.method_params["z"], float)

    def test_odds_ratio_method_parameters(self):
        """Test that odds ratio method returns c parameter."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method=ImpliedMethod.ODDS_RATIO)

        assert result.method_params is not None
        assert "c" in result.method_params
        assert isinstance(result.method_params["c"], float)

    def test_logarithmic_method_parameters(self):
        """Test that logarithmic method returns alpha parameter."""
        odds = [2.7, 2.3, 4.4]
        result = calculate_implied(odds, method=ImpliedMethod.LOGARITHMIC)

        assert result.method_params is not None
        assert "alpha" in result.method_params
        assert isinstance(result.method_params["alpha"], float)
        assert result.method_params["alpha"] > 0  # Should be positive
