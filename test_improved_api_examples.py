"""
Examples demonstrating the improved implied odds API.

This file shows the difference between the old dict-based API and the new
type-safe dataclass-based API, plus examples of multi-format odds support.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "penaltyblog"))

from penaltyblog.implied import (
    multiplicative as old_multiplicative,  # Original API
)
from penaltyblog.implied.improved import (
    calculate_1x2,
    calculate_implied,
    calculate_over_under,
    calculate_two_way,
)
from penaltyblog.implied.models import ImpliedMethod, OddsFormat, OddsInput


def demo_current_vs_improved_api():
    """Demonstrate the difference between current and improved APIs."""
    print("=== Current API vs Improved API ===\n")

    odds = [2.7, 2.3, 4.4]  # Home, Draw, Away

    # Current dict-based API
    print("CURRENT API (dict-based):")
    old_result = old_multiplicative(odds)
    print(f"Type: {type(old_result)}")
    print(f"Keys: {list(old_result.keys())}")
    print(f"Home win: {old_result['implied_probabilities'][0]:.4f}")
    print(f"Draw: {old_result['implied_probabilities'][1]:.4f}")
    print(f"Away win: {old_result['implied_probabilities'][2]:.4f}")
    print(f"Method: {old_result['method']}")
    print(f"Margin: {old_result['margin']:.4f}")
    print(f"Access pattern: result['implied_probabilities'][0]")
    print()

    # New type-safe API
    print("IMPROVED API (dataclass-based):")
    new_result = calculate_1x2(odds)
    print(f"Type: {type(new_result)}")
    print(f"Home win: {new_result.home_win:.4f}")
    print(f"Draw: {new_result.draw:.4f}")
    print(f"Away win: {new_result.away_win:.4f}")
    print(f"Method: {new_result.method}")
    print(f"Margin: {new_result.margin:.4f}")
    print(f"Access pattern: result.home_win")
    print(f"Favorite: {new_result.favorite}")
    print(f"Double chance 1X: {new_result.double_chance_1x:.4f}")
    print()


def demo_different_odds_formats():
    """Show support for different odds formats."""
    print("=== Different Odds Formats ===\n")

    # Same match odds in different formats
    decimal_odds = [2.7, 2.3, 4.4]
    american_odds = [170, -130, 340]
    fractional_odds = ["7/4", "13/10", "7/2"]

    print("DECIMAL ODDS:")
    result1 = calculate_1x2(decimal_odds, odds_format=OddsFormat.DECIMAL)
    print(
        f"Home: {result1.home_win:.4f}, Draw: {result1.draw:.4f}, Away: {result1.away_win:.4f}"
    )
    print()

    print("AMERICAN ODDS:")
    result2 = calculate_1x2(american_odds, odds_format=OddsFormat.AMERICAN)
    print(
        f"Home: {result2.home_win:.4f}, Draw: {result2.draw:.4f}, Away: {result2.away_win:.4f}"
    )
    print()

    print("FRACTIONAL ODDS:")
    result3 = calculate_1x2(fractional_odds, odds_format=OddsFormat.FRACTIONAL)
    print(
        f"Home: {result3.home_win:.4f}, Draw: {result3.draw:.4f}, Away: {result3.away_win:.4f}"
    )
    print()

    # Show they're equivalent (within rounding)
    print("All formats produce equivalent results:")
    print(
        f"Decimal vs American difference: {abs(result1.home_win - result2.home_win):.6f}"
    )
    print(
        f"Decimal vs Fractional difference: {abs(result1.home_win - result3.home_win):.6f}"
    )
    print()


def demo_specialized_market_functions():
    """Demonstrate specialized functions for different bet types."""
    print("=== Specialized Market Functions ===\n")

    # 1X2 Market
    print("1X2 MARKET (Home-Draw-Away):")
    match_odds = [2.7, 2.3, 4.4]
    result_1x2 = calculate_1x2(match_odds)
    print(f"Home win: {result_1x2.home_win:.4f} ({result_1x2.home_win_pct:.1f}%)")
    print(f"Draw: {result_1x2.draw:.4f} ({result_1x2.draw_pct:.1f}%)")
    print(f"Away win: {result_1x2.away_win:.4f} ({result_1x2.away_win_pct:.1f}%)")
    print(f"Most likely outcome: {result_1x2.favorite}")
    print()

    # Over/Under Market
    print("OVER/UNDER MARKET:")
    ou_odds = [2.0, 1.8]  # Over 2.5, Under 2.5
    result_ou = calculate_over_under(ou_odds, 2.5)
    print(f"Over 2.5: {result_ou.outcome_a:.4f}")
    print(f"Under 2.5: {result_ou.outcome_b:.4f}")
    print(f"More likely: {result_ou.favorite}")
    print()

    # Both Teams To Score
    print("BOTH TEAMS TO SCORE:")
    btts_odds = [1.95, 1.85]  # Yes, No
    result_btts = calculate_two_way(btts_odds, ("Yes", "No"))
    print(f"BTTS Yes: {result_btts.outcome_a:.4f}")
    print(f"BTTS No: {result_btts.outcome_b:.4f}")
    print(f"More likely: {result_btts.favorite}")
    print()


def demo_different_methods():
    """Show different calculation methods."""
    print("=== Different Calculation Methods ===\n")

    odds = [2.7, 2.3, 4.4]
    methods = [
        ImpliedMethod.MULTIPLICATIVE,
        ImpliedMethod.ADDITIVE,
        ImpliedMethod.SHIN,
        ImpliedMethod.POWER,
    ]

    for method in methods:
        result = calculate_1x2(odds, method=method)
        print(f"{method.value.upper()}:")
        print(f"  Home: {result.home_win:.4f}")
        print(f"  Draw: {result.draw:.4f}")
        print(f"  Away: {result.away_win:.4f}")
        print(f"  Margin: {result.margin:.4f}")
        if result.method_params:
            for key, value in result.method_params.items():
                print(f"  {key}: {value:.4f}")
        print()


def demo_type_safety_and_ide_support():
    """Demonstrate type safety benefits."""
    print("=== Type Safety & IDE Support ===\n")

    odds = [2.7, 2.3, 4.4]

    # Type-safe access with autocomplete
    result = calculate_1x2(odds)

    print("With the improved API, IDEs can provide:")
    print("- Autocomplete for attributes")
    print("- Type hints for all return values")
    print("- Validation of probability sums")
    print("- Clear method documentation")
    print()

    # Show automatic validation
    print("AUTOMATIC VALIDATION:")
    try:
        from penaltyblog.implied.models import ImpliedMethod, ThreeWayOdds

        # This will fail validation (probabilities don't sum to ~1.0)
        invalid = ThreeWayOdds(
            home_win=0.5,
            draw=0.3,
            away_win=0.5,  # Total = 1.3, too high!
            method=ImpliedMethod.MULTIPLICATIVE,
            margin=0.1,
        )
    except ValueError as e:
        print(f"Validation caught error: {e}")
    print()


def demo_backward_compatibility():
    """Show how improved API maintains backward compatibility."""
    print("=== Backward Compatibility ===\n")

    odds = [2.7, 2.3, 4.4]
    result = calculate_1x2(odds)

    # Can still access as list/array for backward compatibility
    print("Access as list (backward compatibility):")
    print(f"Home (index 0): {result.probabilities[0]:.4f}")
    print(f"Draw (index 1): {result.probabilities[1]:.4f}")
    print(f"Away (index 2): {result.probabilities[2]:.4f}")
    print()

    # Can use in numpy operations
    probs_array = np.array(result.probabilities)
    print(f"As numpy array: {probs_array}")
    print(f"Sum: {probs_array.sum():.4f}")
    print()


def demo_odds_input_object():
    """Demonstrate the OddsInput helper class."""
    print("=== OddsInput Helper Class ===\n")

    # Create structured odds input
    odds_input = OddsInput(
        values=[2.7, 2.3, 4.4],
        format=OddsFormat.DECIMAL,
        market_names=["Manchester City", "Draw", "Liverpool"],
    )

    print("OddsInput object:")
    print(f"Values: {odds_input.values}")
    print(f"Format: {odds_input.format}")
    print(f"Market names: {odds_input.market_names}")
    print()

    # Use with improved API
    result = calculate_1x2(odds_input)
    print(f"Result: {result.favorite} is most likely")
    print()


if __name__ == "__main__":
    demo_current_vs_improved_api()
    demo_different_odds_formats()
    demo_specialized_market_functions()
    demo_different_methods()
    demo_type_safety_and_ide_support()
    demo_backward_compatibility()
    demo_odds_input_object()
