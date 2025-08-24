"""
Demo of the simplified calculate_implied function.

This shows how the single generic function handles different market types
with optional market names for better user experience.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "penaltyblog"))

from penaltyblog.implied.improved import calculate_implied
from penaltyblog.implied.models import ImpliedMethod, OddsFormat


def demo_basic_usage():
    """Basic usage without market names."""
    print("=== Basic Usage (No Market Names) ===")

    odds = [2.7, 2.3, 4.4]
    result = calculate_implied(odds)

    print(f"Probabilities: {result.probabilities}")
    print(f"Most likely index: {result.most_likely_index}")
    print(f"Most likely probability: {result.most_likely_probability:.4f}")
    print(f"Margin: {result.margin:.4f}")

    # Access by index (backward compatibility)
    print(f"Home (index 0): {result[0]:.4f}")
    print(f"Draw (index 1): {result[1]:.4f}")
    print(f"Away (index 2): {result[2]:.4f}")
    print()


def demo_with_market_names():
    """Usage with market names for better UX."""
    print("=== With Market Names ===")

    # 1X2 Market
    odds = [2.7, 2.3, 4.4]
    market_names = ["Home", "Draw", "Away"]
    result = calculate_implied(odds, market_names=market_names)

    print(f"Market names: {result.market_names}")
    print(f"Most likely outcome: {result.most_likely_name}")
    print(f"Least likely outcome: {result.least_likely_name}")

    # Access by name
    print(f"Home probability: {result['Home']:.4f}")
    print(f"Draw probability: {result['Draw']:.4f}")
    print(f"Away probability: {result['Away']:.4f}")
    print()


def demo_over_under_market():
    """Over/Under market with 2 outcomes."""
    print("=== Over/Under Market ===")

    odds = [2.0, 1.8]  # Over 2.5, Under 2.5
    market_names = ["Over 2.5", "Under 2.5"]
    result = calculate_implied(odds, market_names=market_names)

    print(f"Over 2.5: {result['Over 2.5']:.4f}")
    print(f"Under 2.5: {result['Under 2.5']:.4f}")
    print(f"Most likely: {result.most_likely_name}")
    print()


def demo_asian_handicap():
    """Asian handicap market."""
    print("=== Asian Handicap Market ===")

    odds = [1.95, 1.85]  # Team A -0.5, Team B +0.5
    market_names = ["Team A -0.5", "Team B +0.5"]
    result = calculate_implied(odds, market_names=market_names)

    print(f"Team A -0.5: {result['Team A -0.5']:.4f}")
    print(f"Team B +0.5: {result['Team B +0.5']:.4f}")
    print(f"Favorite: {result.most_likely_name}")
    print()


def demo_many_outcomes():
    """Market with many outcomes (e.g., correct score)."""
    print("=== Many Outcomes (Correct Score) ===")

    odds = [9.0, 7.5, 12.0, 6.5, 8.0, 15.0]
    market_names = ["0-0", "1-0", "0-1", "1-1", "2-0", "0-2"]
    result = calculate_implied(odds, market_names=market_names)

    print("Correct score probabilities:")
    for name, prob in zip(result.market_names, result.probabilities):
        print(f"  {name}: {prob:.4f} ({prob*100:.1f}%)")

    print(
        f"\nMost likely score: {result.most_likely_name} ({result.most_likely_probability:.4f})"
    )
    print(
        f"Least likely score: {result.least_likely_name} ({result.least_likely_probability:.4f})"
    )
    print()


def demo_different_formats():
    """Different odds formats."""
    print("=== Different Odds Formats ===")

    # Decimal odds
    decimal_result = calculate_implied(
        [2.7, 2.3, 4.4], market_names=["Home", "Draw", "Away"]
    )
    print(f"Decimal odds result: Home={decimal_result['Home']:.4f}")

    # American odds
    american_result = calculate_implied(
        [170, 130, 340],
        odds_format=OddsFormat.AMERICAN,
        market_names=["Home", "Draw", "Away"],
    )
    print(f"American odds result: Home={american_result['Home']:.4f}")

    # Fractional odds
    fractional_result = calculate_implied(
        ["7/4", "13/10", "7/2"],
        odds_format=OddsFormat.FRACTIONAL,
        market_names=["Home", "Draw", "Away"],
    )
    print(f"Fractional odds result: Home={fractional_result['Home']:.4f}")
    print()


def demo_backward_compatibility():
    """Show backward compatibility with dict format."""
    print("=== Backward Compatibility ===")

    result = calculate_implied([2.7, 2.3, 4.4])

    # Convert to old dict format
    old_dict = result.to_dict()
    print("Old dict format:")
    print(f"  Keys: {list(old_dict.keys())}")
    print(f"  implied_probabilities: {old_dict['implied_probabilities']}")
    print(f"  method: {old_dict['method']}")
    print(f"  margin: {old_dict['margin']}")
    print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_with_market_names()
    demo_over_under_market()
    demo_asian_handicap()
    demo_many_outcomes()
    demo_different_formats()
    demo_backward_compatibility()
