"""
Tests for the public params_array and param_indices API.

This API provides stable access to fitted model parameters, enabling
downstream tools to work with parameter vectors without relying on
private implementation details.
"""

import numpy as np
import pytest

import penaltyblog as pb


@pytest.fixture
def sample_data():
    """Create sample match data for testing."""
    return {
        "goals_home": [3, 1, 0, 2, 1, 3],
        "goals_away": [0, 1, 2, 0, 0, 0],
        "teams_home": ["A", "B", "A", "B", "A", "B"],
        "teams_away": ["B", "A", "B", "A", "B", "A"],
    }


class TestParamsArrayProperty:
    """Tests for the params_array property."""

    def test_params_array_returns_copy(self, sample_data):
        """params_array should return a copy, not a reference."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        arr1 = model.params_array
        arr2 = model.params_array

        # Should be equal but not the same object
        np.testing.assert_array_equal(arr1, arr2)
        assert arr1 is not arr2

    def test_params_array_modification_does_not_affect_model(self, sample_data):
        """Modifying params_array should not affect the model."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        original = model.params_array.copy()
        modified = model.params_array
        modified[0] = 999.0

        # Model's internal params should be unchanged
        np.testing.assert_array_equal(model.params_array, original)

    def test_params_array_raises_when_not_fitted(self, sample_data):
        """params_array should raise ValueError when model is not fitted."""
        model = pb.models.PoissonGoalsModel(**sample_data)

        with pytest.raises(ValueError, match="not yet fitted"):
            _ = model.params_array

    def test_params_array_length_matches_n_params(self, sample_data):
        """params_array length should match n_params after fitting."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        assert len(model.params_array) == model.n_params


class TestParamIndicesMethod:
    """Tests for the param_indices method."""

    def test_param_indices_contains_attack_defense_slices(self, sample_data):
        """param_indices should contain attack and defense slices."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert "attack" in indices
        assert "defense" in indices
        assert isinstance(indices["attack"], slice)
        assert isinstance(indices["defense"], slice)

    def test_param_indices_slices_are_correct(self, sample_data):
        """Attack and defense slices should match expected ranges."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()
        n_teams = model.n_teams

        assert indices["attack"] == slice(0, n_teams)
        assert indices["defense"] == slice(n_teams, 2 * n_teams)

    def test_param_indices_raises_when_not_fitted(self, sample_data):
        """param_indices should raise ValueError when model is not fitted."""
        model = pb.models.PoissonGoalsModel(**sample_data)

        with pytest.raises(ValueError, match="not yet fitted"):
            _ = model.param_indices()


class TestPoissonParamIndices:
    """Tests for Poisson model parameter indices."""

    def test_poisson_tail_indices(self, sample_data):
        """Poisson should have home_advantage at -1."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -1
        assert model.params_array[indices["home_advantage"]] == model.params_array[-1]


class TestDixonColesParamIndices:
    """Tests for Dixon-Coles model parameter indices."""

    def test_dixon_coles_tail_indices(self, sample_data):
        """DixonColes should have home_advantage at -2 and rho at -1."""
        model = pb.models.DixonColesGoalModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -2
        assert indices["rho"] == -1
        assert model.params_array[indices["home_advantage"]] == model.params_array[-2]
        assert model.params_array[indices["rho"]] == model.params_array[-1]


class TestNegativeBinomialParamIndices:
    """Tests for Negative Binomial model parameter indices."""

    def test_negative_binomial_tail_indices(self, sample_data):
        """NegativeBinomial should have home_advantage at -2 and dispersion at -1."""
        model = pb.models.NegativeBinomialGoalModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -2
        assert indices["dispersion"] == -1


class TestZeroInflatedPoissonParamIndices:
    """Tests for Zero-Inflated Poisson model parameter indices."""

    def test_zip_tail_indices(self, sample_data):
        """ZIP should have home_advantage at -2 and zero_inflation at -1."""
        model = pb.models.ZeroInflatedPoissonGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -2
        assert indices["zero_inflation"] == -1


class TestBivariateParamIndices:
    """Tests for Bivariate Poisson model parameter indices."""

    def test_bivariate_tail_indices(self, sample_data):
        """Bivariate should have home_advantage at -2 and correlation at -1."""
        model = pb.models.BivariatePoissonGoalModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -2
        assert indices["correlation"] == -1


class TestWeibullCopulaParamIndices:
    """Tests for Weibull Copula model parameter indices."""

    def test_weibull_copula_tail_indices(self, sample_data):
        """WeibullCopula should have home_advantage at -3, shape at -2, kappa at -1."""
        model = pb.models.WeibullCopulaGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()

        assert indices["home_advantage"] == -3
        assert indices["shape"] == -2
        assert indices["kappa"] == -1


class TestParamsArrayIntegration:
    """Integration tests demonstrating real-world usage patterns."""

    def test_extract_team_parameters_with_indices(self, sample_data):
        """Demonstrate extracting team-specific parameters using indices."""
        model = pb.models.PoissonGoalsModel(**sample_data)
        model.fit()

        indices = model.param_indices()
        params = model.params_array

        # Extract all attack parameters
        attacks = params[indices["attack"]]
        assert len(attacks) == model.n_teams

        # Extract all defense parameters
        defenses = params[indices["defense"]]
        assert len(defenses) == model.n_teams

    def test_apply_factor_to_parameters(self, sample_data):
        """Demonstrate applying factors to parameters (climate adjustment use case)."""
        model = pb.models.DixonColesGoalModel(**sample_data)
        model.fit()

        indices = model.param_indices()
        params = model.params_array

        # Get attack parameters
        attacks = params[indices["attack"]]

        # Apply a factor (e.g., climate adjustment)
        factor = 1.1
        adjusted_attacks = attacks * factor

        # Verify the adjustment
        np.testing.assert_array_almost_equal(adjusted_attacks, attacks * factor)

    def test_consistent_indices_across_fits(self, sample_data):
        """Indices should be consistent across multiple fits."""
        model = pb.models.PoissonGoalsModel(**sample_data)

        model.fit()
        indices1 = model.param_indices()

        # Refit the model
        model.fit()
        indices2 = model.param_indices()

        assert indices1 == indices2
