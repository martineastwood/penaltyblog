"""
Demo of all overround removal methods including the three new ones:
- Logarithmic
- Maximum Entropy
- Least Squares
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "penaltyblog"))

import numpy as np

from penaltyblog.implied.improved import calculate_implied
from penaltyblog.implied.models import ImpliedMethod


def compare_all_methods():
    """Compare all 9 overround removal methods."""

    print("=== Comparison of All Overround Removal Methods ===\n")

    # Test odds with a reasonable margin
    odds = [2.7, 2.3, 4.4]
    raw_probs = [1 / odd for odd in odds]
    margin = sum(raw_probs) - 1

    print(f"Original odds: {odds}")
    print(f"Raw probabilities: {[f'{p:.4f}' for p in raw_probs]}")
    print(f"Raw probability sum: {sum(raw_probs):.4f}")
    print(f"Margin (overround): {margin:.4f} ({margin*100:.2f}%)")
    print()

    methods = [
        ImpliedMethod.MULTIPLICATIVE,
        ImpliedMethod.ADDITIVE,
        ImpliedMethod.POWER,
        ImpliedMethod.SHIN,
        ImpliedMethod.DIFFERENTIAL_MARGIN_WEIGHTING,
        ImpliedMethod.ODDS_RATIO,
        ImpliedMethod.LOGARITHMIC,  # New
        ImpliedMethod.MAXIMUM_ENTROPY,  # New
        ImpliedMethod.LEAST_SQUARES,  # New
    ]

    results = []

    for method in methods:
        result = calculate_implied(odds, method=method)
        results.append(result)

        print(f"{method.value.upper().replace('_', ' ')}:")
        print(f"  Probabilities: {[f'{p:.4f}' for p in result.probabilities]}")
        print(f"  Sum: {sum(result.probabilities):.8f}")

        if result.method_params:
            params_str = ", ".join(
                [f"{k}={v:.4f}" for k, v in result.method_params.items()]
            )
            print(f"  Parameters: {params_str}")
        print()

    return results


def analyze_method_differences():
    """Analyze how different methods compare."""

    print("=== Method Differences Analysis ===\n")

    odds = [2.7, 2.3, 4.4]

    # Get results for comparison
    multiplicative = calculate_implied(odds, method=ImpliedMethod.MULTIPLICATIVE)
    logarithmic = calculate_implied(odds, method=ImpliedMethod.LOGARITHMIC)
    max_entropy = calculate_implied(odds, method=ImpliedMethod.MAXIMUM_ENTROPY)
    least_squares = calculate_implied(odds, method=ImpliedMethod.LEAST_SQUARES)

    print("Comparing new methods to multiplicative (baseline):")
    print()

    print("Method differences from MULTIPLICATIVE:")
    for method_name, result in [
        ("LOGARITHMIC", logarithmic),
        ("MAXIMUM ENTROPY", max_entropy),
        ("LEAST SQUARES", least_squares),
    ]:
        diffs = np.array(result.probabilities) - np.array(multiplicative.probabilities)
        print(f"{method_name}:")
        print(f"  Differences: {[f'{d:+.6f}' for d in diffs]}")
        print(f"  Max absolute difference: {max(abs(diffs)):.6f}")
        print()

    # Note about Maximum Entropy
    print("📝 Note: Maximum Entropy is mathematically equivalent to Multiplicative")
    print(
        "   when the only constraint is Σp_i = 1. Differences are due to numerical precision."
    )
    print()


def test_extreme_odds():
    """Test methods with extreme odds to see how they handle edge cases."""

    print("=== Extreme Odds Test ===\n")

    # Very unbalanced odds
    extreme_odds = [1.1, 15.0, 100.0]  # Heavy favorite, longshot, very long shot
    raw_probs = [1 / odd for odd in extreme_odds]
    margin = sum(raw_probs) - 1

    print(f"Extreme odds: {extreme_odds}")
    print(f"Margin: {margin:.4f} ({margin*100:.2f}%)")
    print()

    methods_to_test = [
        ImpliedMethod.MULTIPLICATIVE,
        ImpliedMethod.LOGARITHMIC,
        ImpliedMethod.LEAST_SQUARES,
    ]

    for method in methods_to_test:
        result = calculate_implied(extreme_odds, method=method)
        print(f"{method.value.upper()}:")
        print(f"  Probabilities: {[f'{p:.4f}' for p in result.probabilities]}")
        print(f"  Percentages: {[f'{p*100:.1f}%' for p in result.probabilities]}")
        print()


def demonstrate_market_names_with_new_methods():
    """Show how market names work with new methods."""

    print("=== Market Names with New Methods ===\n")

    odds = [2.7, 2.3, 4.4]
    market_names = ["Manchester City", "Draw", "Liverpool"]

    new_methods = [
        ImpliedMethod.LOGARITHMIC,
        ImpliedMethod.MAXIMUM_ENTROPY,
        ImpliedMethod.LEAST_SQUARES,
    ]

    for method in new_methods:
        result = calculate_implied(odds, method=method, market_names=market_names)

        print(f"{method.value.upper()}:")
        print(
            f"  Most likely: {result.most_likely_name} ({result.most_likely_probability:.1%})"
        )
        print(
            f"  Least likely: {result.least_likely_name} ({result.least_likely_probability:.1%})"
        )

        # Access by name
        print(f"  Manchester City: {result['Manchester City']:.1%}")
        print(f"  Draw: {result['Draw']:.1%}")
        print(f"  Liverpool: {result['Liverpool']:.1%}")
        print()


def performance_comparison():
    """Basic performance comparison (not comprehensive benchmarking)."""

    print("=== Method Characteristics ===\n")

    characteristics = {
        "MULTIPLICATIVE": {
            "complexity": "O(n)",
            "parameters": "None",
            "use_case": "General purpose, fast",
        },
        "ADDITIVE": {
            "complexity": "O(n)",
            "parameters": "None",
            "use_case": "Equal margin removal",
        },
        "POWER": {
            "complexity": "O(n log n)",
            "parameters": "k (power coefficient)",
            "use_case": "Non-linear adjustment",
        },
        "SHIN": {
            "complexity": "O(n log n)",
            "parameters": "z (insider trading parameter)",
            "use_case": "Market efficiency modeling",
        },
        "LOGARITHMIC": {
            "complexity": "O(n log n)",
            "parameters": "α (adjustment factor)",
            "use_case": "Better for extreme probabilities",
        },
        "MAXIMUM_ENTROPY": {
            "complexity": "O(n)",
            "parameters": "entropy",
            "use_case": "Information theory based",
        },
        "LEAST_SQUARES": {
            "complexity": "O(n)",
            "parameters": "λ (Lagrange multiplier)",
            "use_case": "Minimal deviation from original",
        },
    }

    for method, info in characteristics.items():
        print(f"{method}:")
        print(f"  Complexity: {info['complexity']}")
        print(f"  Parameters: {info['parameters']}")
        print(f"  Use case: {info['use_case']}")
        print()


if __name__ == "__main__":
    compare_all_methods()
    analyze_method_differences()
    test_extreme_odds()
    demonstrate_market_names_with_new_methods()
    performance_comparison()
