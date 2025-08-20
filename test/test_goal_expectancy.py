import numpy as np
import pytest

import penaltyblog as pb


# Test fixtures
@pytest.fixture
def balanced_probs():
    """Balanced probabilities for a fairly even match."""
    return (0.35, 0.30, 0.35)  # Home, Draw, Away


@pytest.fixture
def home_favored_probs():
    """Probabilities where home team is strongly favored."""
    return (0.60, 0.25, 0.15)


@pytest.fixture
def away_favored_probs():
    """Probabilities where away team is strongly favored."""
    return (0.093897, 0.158581, 0.747522)


@pytest.fixture
def unnormalized_probs():
    """Probabilities that don't sum to 1.0 (overround)."""
    return (0.45, 0.35, 0.35)  # Sum = 1.15


class TestGoalExpectancyBasicFunctionality:
    """Test basic functionality and core features."""

    def test_basic_functionality(self, away_favored_probs):
        """Test basic functionality with default parameters."""
        home, draw, away = away_favored_probs
        result = pb.models.goal_expectancy(home, draw, away, dc_adj=False)

        # Check return type and required keys
        assert isinstance(result, dict)
        assert "home_exp" in result
        assert "away_exp" in result
        assert "error" in result
        assert "success" in result

        # Check default return_details=True includes additional keys
        assert "predicted" in result
        assert "mass" in result

        # Check optimization success
        assert result["success"] is True

        # Check reasonable expectation values
        assert 0.5 < result["home_exp"] < 1.0  # Lower for away-favored
        assert 2.0 < result["away_exp"] < 3.0  # Higher for away-favored
        assert result["error"] >= 0  # Non-negative error

    def test_return_details_false(self, balanced_probs):
        """Test return_details=False excludes detailed information."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, dc_adj=False, return_details=False
        )

        # Should have basic keys
        assert "home_exp" in result
        assert "away_exp" in result
        assert "error" in result
        assert "success" in result

        # Should NOT have detailed keys
        assert "predicted" not in result
        assert "mass" not in result

    def test_dixon_coles_adjustment(self, balanced_probs):
        """Test Dixon-Coles adjustment functionality."""
        home, draw, away = balanced_probs

        # Test with DC adjustment (default)
        result_dc = pb.models.goal_expectancy(home, draw, away, dc_adj=True)

        # Test without DC adjustment
        result_no_dc = pb.models.goal_expectancy(home, draw, away, dc_adj=False)

        # Both should be successful
        assert result_dc["success"] is True
        assert result_no_dc["success"] is True

        # Results should be different (DC adjustment affects low-scoring games)
        assert result_dc["home_exp"] != result_no_dc["home_exp"]
        assert result_dc["away_exp"] != result_no_dc["away_exp"]

    def test_rho_parameter(self, balanced_probs):
        """Test different rho values in Dixon-Coles adjustment."""
        home, draw, away = balanced_probs

        result_low_rho = pb.models.goal_expectancy(
            home, draw, away, dc_adj=True, rho=0.001
        )
        result_high_rho = pb.models.goal_expectancy(
            home, draw, away, dc_adj=True, rho=0.1
        )

        # Both should be successful
        assert result_low_rho["success"] is True
        assert result_high_rho["success"] is True

        # Different rho values should give different results
        assert result_low_rho["home_exp"] != result_high_rho["home_exp"]
        assert result_low_rho["away_exp"] != result_high_rho["away_exp"]


class TestGoalExpectancyInputValidation:
    """Test input validation and error handling."""

    def test_negative_probabilities(self):
        """Test that negative probabilities raise ValueError."""
        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(-0.1, 0.5, 0.5)

        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(0.3, -0.2, 0.9)

        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(0.3, 0.2, -0.1)

    def test_probabilities_greater_than_one(self):
        """Test that probabilities > 1 raise ValueError."""
        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(1.1, 0.3, 0.2)

        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(0.3, 1.5, 0.2)

        with pytest.raises(
            ValueError, match=r"home/draw/away must be probabilities in \[0, 1\]"
        ):
            pb.models.goal_expectancy(0.3, 0.2, 1.2)

    def test_edge_case_probabilities(self):
        """Test edge cases with extreme probabilities."""
        # Test with zero probabilities
        result = pb.models.goal_expectancy(0.0, 0.1, 0.9, dc_adj=False)
        assert result["success"] is True
        assert result["home_exp"] > 0  # Should still be positive

        # Test with one probability = 1
        result = pb.models.goal_expectancy(0.0, 0.0, 1.0, dc_adj=False)
        assert result["success"] is True

        # Test boundary values
        result = pb.models.goal_expectancy(1.0, 0.0, 0.0, dc_adj=False)
        assert result["success"] is True

    def test_invalid_objective(self, balanced_probs):
        """Test that invalid objective raises ValueError."""
        home, draw, away = balanced_probs
        with pytest.raises(
            ValueError, match="objective must be 'brier' or 'cross_entropy'"
        ):
            pb.models.goal_expectancy(home, draw, away, objective="invalid")


class TestGoalExpectancyDeoverround:
    """Test deoverround functionality."""

    def test_deoverround_functionality(self, unnormalized_probs):
        """Test that deoverround properly normalizes probabilities."""
        home, draw, away = unnormalized_probs
        original_sum = home + draw + away
        assert original_sum > 1.0  # Verify overround exists

        # Test with remove_overround=True
        result = pb.models.goal_expectancy(
            home, draw, away, remove_overround=True, dc_adj=False
        )
        assert result["success"] is True

        # Test with remove_overround=False (should still work but may have higher error)
        # This should trigger a warning about probabilities not summing to 1.0
        with pytest.warns(UserWarning, match="Input probabilities do not sum to 1.0"):
            result_no_deoverround = pb.models.goal_expectancy(
                home, draw, away, remove_overround=False, dc_adj=False
            )
        assert result_no_deoverround["success"] is True

        # Remove overround should generally give better fit
        assert result["error"] <= result_no_deoverround["error"]

    def test_deoverround_zero_sum_error(self):
        """Test that zero-sum probabilities with remove_overround=True raise error."""
        with pytest.raises(
            ValueError, match="Sum of probabilities must be > 0 to remove_overround"
        ):
            pb.models.goal_expectancy(0.0, 0.0, 0.0, remove_overround=True)

    def test_deoverround_preserves_ratios(self, unnormalized_probs):
        """Test that remove_overround preserves relative probability ratios."""
        home, draw, away = unnormalized_probs
        original_ratio = home / away

        result = pb.models.goal_expectancy(
            home, draw, away, remove_overround=True, dc_adj=False, return_details=True
        )

        predicted_home, predicted_draw, predicted_away = result["predicted"]
        new_ratio = predicted_home / predicted_away

        # Ratios should be approximately preserved
        assert abs(original_ratio - new_ratio) < 0.1


class TestGoalExpectancyObjectiveFunctions:
    """Test different objective functions."""

    def test_brier_objective(self, balanced_probs):
        """Test Brier score (MSE) objective function."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, objective="brier", dc_adj=False
        )

        assert result["success"] is True
        assert result["error"] >= 0  # MSE is non-negative

    def test_cross_entropy_objective(self, balanced_probs):
        """Test cross-entropy objective function."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, objective="cross_entropy", dc_adj=False
        )

        assert result["success"] is True
        assert result["error"] >= 0  # Cross-entropy is non-negative

    def test_objective_comparison(self, balanced_probs):
        """Test that different objectives give different results."""
        home, draw, away = balanced_probs

        result_brier = pb.models.goal_expectancy(
            home, draw, away, objective="brier", dc_adj=False
        )
        result_ce = pb.models.goal_expectancy(
            home, draw, away, objective="cross_entropy", dc_adj=False
        )

        # Both should be successful
        assert result_brier["success"] is True
        assert result_ce["success"] is True

        # Results should be different (different objectives)
        # Note: They might be similar but generally not identical
        assert (
            result_brier["home_exp"] != result_ce["home_exp"]
            or result_brier["away_exp"] != result_ce["away_exp"]
        )


class TestGoalExpectancyOptimizationParameters:
    """Test optimization-related parameters."""

    def test_max_goals_parameter(self, balanced_probs):
        """Test different max_goals values."""
        home, draw, away = balanced_probs

        result_small = pb.models.goal_expectancy(
            home, draw, away, max_goals=5, dc_adj=False
        )
        result_large = pb.models.goal_expectancy(
            home, draw, away, max_goals=20, dc_adj=False
        )

        # Both should be successful
        assert result_small["success"] is True
        assert result_large["success"] is True

        # Results should be different but reasonably close
        assert abs(result_small["home_exp"] - result_large["home_exp"]) < 0.5
        assert abs(result_small["away_exp"] - result_large["away_exp"]) < 0.5

    def test_method_parameter(self, balanced_probs):
        """Test different optimization methods."""
        home, draw, away = balanced_probs

        # Test L-BFGS-B (default)
        result_lbfgs = pb.models.goal_expectancy(
            home, draw, away, method="L-BFGS-B", dc_adj=False
        )

        # Test SLSQP
        result_slsqp = pb.models.goal_expectancy(
            home, draw, away, method="SLSQP", dc_adj=False
        )

        # Both should be successful
        assert result_lbfgs["success"] is True
        assert result_slsqp["success"] is True

        # Results should be similar (same problem, different methods)
        assert abs(result_lbfgs["home_exp"] - result_slsqp["home_exp"]) < 0.1
        assert abs(result_lbfgs["away_exp"] - result_slsqp["away_exp"]) < 0.1

    def test_bounds_parameter(self, balanced_probs):
        """Test custom bounds parameter."""
        home, draw, away = balanced_probs

        # Test with tighter bounds
        result = pb.models.goal_expectancy(
            home, draw, away, bounds=((-1.0, 1.0), (-1.0, 1.0)), dc_adj=False
        )

        assert result["success"] is True

        # Results should respect bounds (exp(-1) ≈ 0.37, exp(1) ≈ 2.72)
        assert 0.3 <= result["home_exp"] <= 3.0
        assert 0.3 <= result["away_exp"] <= 3.0

    def test_x0_initial_guess(self, balanced_probs):
        """Test custom initial guess parameter."""
        home, draw, away = balanced_probs

        # Test with custom initial guess
        result = pb.models.goal_expectancy(
            home,
            draw,
            away,
            x0=(np.log(2.0), np.log(1.5)),  # Initial guess: mu_h=2.0, mu_a=1.5
            dc_adj=False,
        )

        assert result["success"] is True
        assert result["home_exp"] > 0
        assert result["away_exp"] > 0

    def test_minimizer_options(self, balanced_probs):
        """Test minimizer_options parameter."""
        home, draw, away = balanced_probs

        # Test with very low iterations (may not converge)
        result_low_iter = pb.models.goal_expectancy(
            home,
            draw,
            away,
            minimizer_options={"maxiter": 2, "disp": False},
            dc_adj=False,
        )

        # Should return a result (may not be successful)
        assert isinstance(result_low_iter, dict)
        assert "home_exp" in result_low_iter
        assert "away_exp" in result_low_iter

        # Test with high iterations (should converge)
        result_high_iter = pb.models.goal_expectancy(
            home, draw, away, minimizer_options={"maxiter": 5000}, dc_adj=False
        )

        assert result_high_iter["success"] is True


class TestGoalExpectancyDixonColesOptions:
    """Test Dixon-Coles specific parameters."""

    def test_renormalize_after_dc_true(self, balanced_probs):
        """Test renormalize_after_dc=True (default)."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, dc_adj=True, renormalize_after_dc=True
        )

        assert result["success"] is True

        # Mass should be close to 1.0 when renormalized
        assert abs(result["mass"] - 1.0) < 1e-6

    def test_renormalize_after_dc_false(self, balanced_probs):
        """Test renormalize_after_dc=False."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, dc_adj=True, renormalize_after_dc=False
        )

        assert result["success"] is True

        # Mass may not be exactly 1.0 when not renormalized
        # but should be positive and reasonable
        assert result["mass"] > 0
        assert result["mass"] < 2.0  # Should be reasonable

    def test_dc_with_different_rho_values(self, balanced_probs):
        """Test Dixon-Coles with various rho values."""
        home, draw, away = balanced_probs

        rho_values = [-0.1, 0.0, 0.001, 0.05, 0.1]
        results = []

        for rho in rho_values:
            result = pb.models.goal_expectancy(home, draw, away, dc_adj=True, rho=rho)
            results.append(result)
            assert result["success"] is True

        # Different rho values should produce different results
        home_exps = [r["home_exp"] for r in results]
        away_exps = [r["away_exp"] for r in results]

        # Should have some variation across rho values
        assert max(home_exps) - min(home_exps) > 1e-6
        assert max(away_exps) - min(away_exps) > 1e-6


class TestGoalExpectancyResultConsistency:
    """Test consistency and mathematical properties of results."""

    def test_predicted_probabilities_consistency(self, balanced_probs):
        """Test that predicted probabilities are mathematically consistent."""
        home, draw, away = balanced_probs
        result = pb.models.goal_expectancy(
            home, draw, away, dc_adj=False, return_details=True
        )

        predicted = result["predicted"]

        # Predicted probabilities should sum to approximately 1
        assert abs(predicted.sum() - 1.0) < 1e-6

        # All predicted probabilities should be non-negative
        assert all(p >= 0 for p in predicted)

        # All predicted probabilities should be <= 1
        assert all(p <= 1 for p in predicted)

    def test_mass_property(self, balanced_probs):
        """Test that grid mass is reasonable."""
        home, draw, away = balanced_probs

        # Test without DC adjustment
        result_no_dc = pb.models.goal_expectancy(
            home, draw, away, dc_adj=False, return_details=True
        )

        # Test with DC adjustment and renormalization
        result_dc_norm = pb.models.goal_expectancy(
            home,
            draw,
            away,
            dc_adj=True,
            renormalize_after_dc=True,
            return_details=True,
        )

        # Both should have reasonable mass
        assert 0.95 <= result_no_dc["mass"] <= 1.05
        assert (
            abs(result_dc_norm["mass"] - 1.0) < 1e-6
        )  # Should be exactly 1 when renormalized

    def test_error_decreases_with_better_fit(self):
        """Test that error decreases when we have better initial conditions."""
        # Use probabilities that correspond to known expectancies
        home, draw, away = 0.3, 0.3, 0.4

        # Test with bad initial guess
        result_bad_init = pb.models.goal_expectancy(
            home,
            draw,
            away,
            x0=(np.log(0.5), np.log(3.0)),  # Far from likely solution
            dc_adj=False,
        )

        # Test with good initial guess
        result_good_init = pb.models.goal_expectancy(
            home,
            draw,
            away,
            x0=(np.log(1.0), np.log(1.2)),  # Closer to likely solution
            dc_adj=False,
        )

        # Both should be successful
        assert result_bad_init["success"] is True
        assert result_good_init["success"] is True

        # Final errors should be similar (optimization should find global minimum)
        assert abs(result_bad_init["error"] - result_good_init["error"]) < 0.01

    def test_expectancy_reasonableness(self, home_favored_probs, away_favored_probs):
        """Test that goal expectancies make intuitive sense."""
        # Home favored case
        home_h, draw_h, away_h = home_favored_probs
        result_home_fav = pb.models.goal_expectancy(
            home_h, draw_h, away_h, dc_adj=False
        )

        # Away favored case
        home_a, draw_a, away_a = away_favored_probs
        result_away_fav = pb.models.goal_expectancy(
            home_a, draw_a, away_a, dc_adj=False
        )

        # When home is favored, home expectancy should be higher
        assert result_home_fav["home_exp"] > result_home_fav["away_exp"]

        # When away is favored, away expectancy should be higher
        assert result_away_fav["away_exp"] > result_away_fav["home_exp"]

        # Home-favored case should have higher home expectancy than away-favored case
        assert result_home_fav["home_exp"] > result_away_fav["home_exp"]

        # Away-favored case should have higher away expectancy than home-favored case
        assert result_away_fav["away_exp"] > result_home_fav["away_exp"]


class TestGoalExpectancyEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_extreme_probabilities(self):
        """Test with extreme probability distributions."""
        # Heavily skewed toward away win
        result = pb.models.goal_expectancy(0.01, 0.04, 0.95, dc_adj=False)
        assert result["success"] is True
        assert result["away_exp"] > result["home_exp"]

        # Heavily skewed toward home win
        result = pb.models.goal_expectancy(0.95, 0.04, 0.01, dc_adj=False)
        assert result["success"] is True
        assert result["home_exp"] > result["away_exp"]

        # Draw heavy
        result = pb.models.goal_expectancy(0.1, 0.8, 0.1, dc_adj=False)
        assert result["success"] is True
        # For draw-heavy, expectancies should be relatively similar and low-ish
        assert abs(result["home_exp"] - result["away_exp"]) < 0.5

    def test_very_low_scoring_game(self):
        """Test scenario that should result in very low scoring expectations."""
        # High draw probability suggests low-scoring game
        result = pb.models.goal_expectancy(0.2, 0.6, 0.2, dc_adj=False)
        assert result["success"] is True

        # Both expectancies should be relatively low
        assert result["home_exp"] < 2.0
        assert result["away_exp"] < 2.0

    def test_high_scoring_game(self):
        """Test scenario suggesting a high-scoring game."""
        # Low draw probability might suggest higher scoring
        result = pb.models.goal_expectancy(0.45, 0.1, 0.45, dc_adj=False)
        assert result["success"] is True

        # The exact values depend on the optimization, but both should be positive
        assert result["home_exp"] > 0
        assert result["away_exp"] > 0


# Legacy tests (updated for better coverage)
def test_goal_expectancy():
    """Legacy test updated with better assertions."""
    probs = (0.093897, 0.158581, 0.747522)
    exp = pb.models.goal_expectancy(*probs, dc_adj=False)

    assert exp["success"] is True
    assert 0.72 < exp["home_exp"] < 0.75
    assert 2.225 < exp["away_exp"] < 2.5
    assert 0.99 < np.array(probs).sum() < 1.01

    # Additional checks for completeness
    assert exp["error"] >= 0
    assert "predicted" in exp  # default return_details=True
    assert "mass" in exp
    assert len(exp["predicted"]) == 3


def test_goal_expectancy_minimizer_options():
    """Legacy test for minimizer options."""
    probs = (0.093897, 0.158581, 0.747522)
    exp = pb.models.goal_expectancy(
        *probs, dc_adj=False, minimizer_options={"maxiter": 2, "disp": False}
    )

    # Just check that a result dict is returned with expected keys
    assert isinstance(exp, dict)
    assert "home_exp" in exp and "away_exp" in exp
    assert "error" in exp and "success" in exp

    # Test with normal conditions
    exp_normal = pb.models.goal_expectancy(*probs, dc_adj=False)
    assert exp_normal["success"] is True
    assert 0.72 < exp_normal["home_exp"] < 0.75
    assert 2.225 < exp_normal["away_exp"] < 2.5
    assert 0.99 < np.array(probs).sum() < 1.01


def test_goal_expectancy_dc_adj():
    """Legacy test for Dixon-Coles adjustment."""
    probs = (0.093897, 0.158581, 0.747522)
    exp = pb.models.goal_expectancy(*probs)  # dc_adj=True by default

    assert exp["success"] is True
    assert 0.72 < exp["home_exp"] < 0.75
    assert 2.225 < exp["away_exp"] < 2.5
    assert 0.99 < np.array(probs).sum() < 1.01

    # Additional checks
    assert exp["error"] >= 0
    assert "predicted" in exp
    assert "mass" in exp
    assert abs(exp["mass"] - 1.0) < 0.1  # Should be close to 1
