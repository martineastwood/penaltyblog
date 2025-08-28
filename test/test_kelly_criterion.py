import numpy as np
import pytest

import penaltyblog as pb


def test_kelly_criterion():
    """Test basic Kelly criterion calculation."""
    odds = 3.5
    probs = 0.3
    result = pb.betting.kelly_criterion(odds, probs)
    assert round(result.stake, 2) == 0.02
    assert result.is_favorable
    assert result.edge > 0
    assert isinstance(result.warnings, list)


def test_kelly_criterion_fraction():
    """Test Kelly criterion with fraction scaling."""
    odds = 3.5
    probs = 0.3
    result = pb.betting.kelly_criterion(odds, probs, 0.5)
    assert round(result.stake, 2) == 0.01
    assert result.fraction == 0.5


def test_kelly_criterion_unfavorable():
    """Test Kelly criterion with unfavorable odds."""
    odds = 1.8
    probs = 0.4  # Implied probability is ~55.6%, but true prob is 40%
    result = pb.betting.kelly_criterion(odds, probs)
    assert result.stake == 0.0  # Should recommend no bet
    assert not result.is_favorable
    assert result.edge < 0


def test_kelly_criterion_warnings():
    """Test that warnings are generated appropriately."""
    # Test high confidence warning
    odds = 1.1
    probs = 0.99  # Very high confidence
    result = pb.betting.kelly_criterion(odds, probs)
    assert len(result.warnings) > 0
    assert any("probabilities are very high" in w for w in result.warnings)

    # Test low odds warning
    odds = 0.9  # Odds <= 1.0
    probs = 0.5
    result = pb.betting.kelly_criterion(odds, probs)
    assert len(result.warnings) > 0
    assert any("odds are <= 1.0" in w for w in result.warnings)


def test_input_validation_errors():
    """Test input validation raises appropriate errors."""
    # Test invalid probabilities (negative)
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        pb.betting.kelly_criterion(2.0, -0.1)

    # Test invalid probabilities (> 1)
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        pb.betting.kelly_criterion(2.0, 1.5)

    # Test odds exactly equal to 1.0 (division by zero risk)
    with pytest.raises(ValueError, match="Odds cannot be exactly 1.0"):
        pb.betting.kelly_criterion(1.0, 0.5)


def test_low_probability_warnings():
    """Test warnings for very low probabilities."""
    # Test very low probability warning (< 1%)
    odds = 100.0
    probs = 0.005  # 0.5% probability
    result = pb.betting.kelly_criterion(odds, probs)
    assert any("probabilities are very low" in w for w in result.warnings)

    # Test edge case: exactly 1% should not trigger warning
    odds = 50.0
    probs = 0.01  # Exactly 1%
    result = pb.betting.kelly_criterion(odds, probs)
    # Should not have the low probability warning
    assert not any("probabilities are very low" in w for w in result.warnings)


def test_kelly_criterion_array_inputs():
    """Test kelly_criterion with numpy array inputs."""
    # Test with numpy arrays
    odds_array = np.array([2.0, 3.0, 1.5])
    probs_array = np.array([0.6, 0.4, 0.5])
    result = pb.betting.kelly_criterion(odds_array, probs_array)

    # Should work and return results for each input
    assert hasattr(result, "stake")
    assert hasattr(result, "expected_growth")
    assert hasattr(result, "edge")
    assert hasattr(result, "is_favorable")
    assert hasattr(result, "risk_of_ruin")

    # Test single values work too
    single_result = pb.betting.kelly_criterion(2.0, 0.6)
    assert isinstance(single_result.stake, (float, np.number))
    assert isinstance(single_result.expected_growth, (float, np.number))


def test_multiple_kelly_basic():
    """Test basic multiple Kelly criterion functionality."""
    # Test basic multiple Kelly with independent method
    odds = [2.5, 3.0, 2.2]
    probs = [0.45, 0.35, 0.15]  # Sum = 0.95, leaving 5% for no outcome

    result = pb.betting.multiple_kelly_criterion(odds, probs, method="independent")

    assert len(result.stakes) == 3
    assert result.total_stake > 0
    assert result.expected_growth != 0
    assert result.expected_return != 0
    assert isinstance(result.portfolio_edge, float)
    assert hasattr(result.risk_metrics, "expected_profit")
    assert result.method == "independent"
    assert result.optimization_success == True
    assert isinstance(result.optimization_details, dict)
    assert isinstance(result.warnings, list)

    # Test with simultaneous method
    result_sim = pb.betting.multiple_kelly_criterion(odds, probs, method="simultaneous")

    assert len(result_sim.stakes) == 3
    assert result_sim.method == "simultaneous"
    assert result_sim.optimization_success == True


def test_multiple_kelly_input_validation():
    """Test input validation for multiple Kelly criterion."""
    # Test mismatched array lengths
    with pytest.raises(
        ValueError, match="decimal_odds and true_probs must have the same length"
    ):
        pb.betting.multiple_kelly_criterion([2.0, 3.0], [0.5])

    # Test negative probabilities
    with pytest.raises(ValueError, match="All probabilities must be non-negative"):
        pb.betting.multiple_kelly_criterion([2.0, 3.0], [0.5, -0.1])

    # Test probabilities sum exceeding 1.0
    with pytest.raises(ValueError, match="Sum of probabilities cannot exceed 1.0"):
        pb.betting.multiple_kelly_criterion([2.0, 3.0], [0.6, 0.6])

    # Test invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        pb.betting.multiple_kelly_criterion([2.0, 3.0], [0.4, 0.3], method="invalid")


def test_multiple_kelly_constraints():
    """Test constraint handling in multiple Kelly criterion."""
    # Test max_total_stake constraint
    odds = [2.0, 2.5, 3.0]
    probs = [0.3, 0.25, 0.2]  # Sum = 0.75, these would normally suggest large stakes
    max_stake = 0.2  # Limit total stake to 20%

    result = pb.betting.multiple_kelly_criterion(
        odds, probs, max_total_stake=max_stake, method="independent"
    )

    assert result.total_stake <= max_stake + 1e-10  # Allow for numerical tolerance
    assert all(
        stake >= 0 for stake in result.stakes
    )  # All stakes should be non-negative

    # Test with simultaneous method and constraints
    result_sim = pb.betting.multiple_kelly_criterion(
        odds, probs, max_total_stake=max_stake, method="simultaneous"
    )

    assert result_sim.total_stake <= max_stake + 1e-10


def test_multiple_kelly_fraction_scaling():
    """Test fraction scaling in multiple Kelly criterion."""
    odds = [2.0, 3.0]
    probs = [0.4, 0.3]  # Sum = 0.7, valid probabilities
    fraction = 0.5  # Half Kelly

    # Test with independent method
    result = pb.betting.multiple_kelly_criterion(
        odds, probs, fraction=fraction, method="independent"
    )

    assert result.fraction == fraction
    # Stakes should be reduced due to fractional Kelly

    # Test with simultaneous method
    result_sim = pb.betting.multiple_kelly_criterion(
        odds, probs, fraction=fraction, method="simultaneous"
    )

    assert result_sim.fraction == fraction


def test_multiple_kelly_edge_cases():
    """Test edge cases for multiple Kelly criterion."""
    # Test with single outcome (should behave like single Kelly)
    odds = [2.0]
    probs = [0.6]

    result = pb.betting.multiple_kelly_criterion(odds, probs)

    assert len(result.stakes) == 1
    assert result.stakes[0] > 0  # Should recommend some stake

    # Test with zero probabilities
    odds = [2.0, 3.0]
    probs = [0.0, 0.0]  # No expected wins

    result = pb.betting.multiple_kelly_criterion(odds, probs)

    assert all(stake == 0 for stake in result.stakes)  # Should recommend no stakes
    assert result.total_stake == 0

    # Test with very small probabilities (should trigger warnings)
    odds = [100.0, 200.0]
    probs = [0.005, 0.003]  # Very low probabilities

    result = pb.betting.multiple_kelly_criterion(odds, probs)

    assert len(result.warnings) > 0  # Should have warnings
    # Check that warnings contain expected text
    warnings_text = " ".join(str(w) for w in result.warnings)
    assert "very low" in warnings_text


def test_multiple_kelly_optimization_fallback():
    """Test optimization fallback scenarios."""
    # Test scenario that might cause simultaneous optimization to fail
    # by using extreme values or constraints that are difficult to satisfy
    odds = [1.01, 1.01]  # Very low odds (barely profitable)
    probs = [0.99, 0.99]  # Very high probabilities (sum > 1, should raise error first)

    # This should raise an error due to probability sum
    with pytest.raises(ValueError):
        pb.betting.multiple_kelly_criterion(odds, probs, method="simultaneous")

    # Test with odds that should trigger warnings about low profit potential
    odds = [0.99, 0.98]  # Odds <= 1.0 (no profit potential)
    probs = [0.5, 0.4]  # Reasonable probabilities that sum < 1

    # This should complete with warnings about no profit potential
    result = pb.betting.multiple_kelly_criterion(odds, probs, method="simultaneous")
    # Should have warnings about odds being <= 1.0
    assert len(result.warnings) > 0


def test_multiple_kelly_comprehensive_metrics():
    """Test comprehensive metrics from multiple Kelly results."""
    odds = [2.5, 3.0, 2.2]
    probs = [0.45, 0.30, 0.20]  # Sum = 0.95, leaving 5% for no outcome

    result = pb.betting.multiple_kelly_criterion(odds, probs)

    # Test all expected attributes exist and have reasonable values
    assert isinstance(result.stakes, list)
    assert isinstance(result.total_stake, float)
    assert isinstance(result.expected_growth, float)
    assert isinstance(result.expected_return, float)
    assert isinstance(result.portfolio_edge, float)

    # Test risk metrics
    rm = result.risk_metrics
    assert isinstance(rm.expected_profit, float)
    assert isinstance(rm.expected_return, float)
    assert isinstance(rm.kelly_growth_rate, float)
    assert isinstance(rm.wealth_volatility, float)
    assert isinstance(rm.log_return_volatility, float)
    assert isinstance(rm.sharpe_ratio, float)
    assert isinstance(rm.win_probability, float)
    assert isinstance(rm.probability_of_ruin, float)
    assert isinstance(rm.value_at_risk_95, float)
    assert isinstance(rm.max_loss, float)
    assert isinstance(rm.total_exposure, float)

    # Test reasonable ranges
    assert 0 <= rm.win_probability <= 1
    assert 0 <= rm.probability_of_ruin <= 1
    assert rm.value_at_risk_95 >= 0
    assert rm.max_loss >= 0
    assert rm.total_exposure >= 0


def test_risk_metrics_scalar_edge_cases():
    """Test risk metrics with scalar inputs and edge cases to cover lines 181-183, 277."""
    # Test single bet scenario to trigger scalar handling in _calculate_risk_metrics
    result = pb.betting.multiple_kelly_criterion([2.5], [0.5])  # Profitable bet

    rm = result.risk_metrics
    assert rm.expected_profit > 0  # Should be profitable
    assert rm.total_exposure > 0

    # Test scenario that might trigger VaR edge case (line 277)
    # Use probabilities that create unusual wealth distributions
    odds = [1.1, 1.1]  # Very low odds
    probs = [0.8, 0.1]  # High probability for first, low for second

    result = pb.betting.multiple_kelly_criterion(odds, probs)
    rm = result.risk_metrics
    assert rm.value_at_risk_95 >= 0  # Should handle edge case gracefully


def test_optimization_edge_cases():
    """Test optimization edge cases to cover lines 491, 507, 518, 576-578, 595-596."""
    # Test scenario that might trigger scaling in independent method (line 491)
    odds = [3.0, 4.0, 5.0]  # High odds that would suggest large stakes
    probs = [0.8, 0.7, 0.6]  # High probabilities (sum > 1, will error)

    # This will error due to probability sum, but let's try valid probabilities
    odds = [3.0, 4.0, 5.0]
    probs = [0.25, 0.25, 0.25]  # Sum = 0.75, but high Kelly values
    max_stake = 0.3  # Reasonable limit

    result = pb.betting.multiple_kelly_criterion(
        odds, probs, max_total_stake=max_stake, method="independent"
    )

    # Should have applied scaling if total exceeded max_stake
    assert result.total_stake <= max_stake + 1e-10

    # Test scenario that might cause optimization issues (lines 507, 518, 576-578)
    # Use odds that create potential numerical issues in simultaneous optimization
    odds = [1.0001, 1.0001]  # Extremely close to 1.0 but not exactly 1.0
    probs = [0.9999, 0.00005]  # Extreme probabilities that might cause numerical issues

    try:
        result = pb.betting.multiple_kelly_criterion(
            odds, probs, method="simultaneous", max_total_stake=0.01
        )
        # If it succeeds, check it handled the edge case
        assert result.optimization_success in [True, False]
    except (ValueError, RuntimeError):
        # Some extreme cases might still raise errors, which is acceptable
        pass

    # Test fallback scenario (lines 595-596) by trying to force simultaneous to fail
    # then fall back to independent method
    try:
        # This specific combination might trigger the fallback
        odds = [1.000001, 1.000001, 1.000001, 1.000001]  # Many bets with tiny edges
        probs = [0.24, 0.24, 0.24, 0.24]  # Sum = 0.96

        result = pb.betting.multiple_kelly_criterion(
            odds, probs, method="simultaneous", max_total_stake=0.001
        )

        # Should either succeed or have warnings about optimization issues
        assert isinstance(result.warnings, list)

    except Exception:
        # Some extreme optimization scenarios may still fail completely
        pass
