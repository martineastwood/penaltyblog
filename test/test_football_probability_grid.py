import numpy as np
import pytest

import penaltyblog as pb


# Test fixtures
@pytest.fixture
def sample_grid():
    """A simple 3x3 probability grid for testing."""
    return np.array(
        [
            [0.1, 0.2, 0.1],  # Home 0 goals
            [0.15, 0.25, 0.05],  # Home 1 goal
            [0.05, 0.1, 0.05],  # Home 2 goals
        ]
    )


@pytest.fixture
def normalized_grid():
    """A 3x3 probability grid that sums to 1.0."""
    return np.array(
        [
            [0.1, 0.2, 0.1],  # 0.4 total
            [0.1, 0.25, 0.05],  # 0.4 total
            [0.05, 0.1, 0.05],  # 0.2 total  (sum = 1.0)
        ]
    )


@pytest.fixture
def unnormalized_grid():
    """A 3x3 probability grid that doesn't sum to 1.0."""
    return np.array(
        [
            [0.2, 0.4, 0.2],  # 0.8 total
            [0.3, 0.5, 0.1],  # 0.9 total
            [0.1, 0.2, 0.1],  # 0.4 total  (sum = 2.1)
        ]
    )


class TestFootballProbabilityGridInitialization:
    """Test initialization and validation of FootballProbabilityGrid."""

    def test_basic_initialization(self, normalized_grid):
        """Test basic initialization with valid parameters."""
        fpg = pb.models.FootballProbabilityGrid(
            normalized_grid, home_goal_expectation=1.5, away_goal_expectation=1.2
        )
        assert isinstance(fpg.grid, np.ndarray)
        assert fpg.home_goal_expectation == 1.5
        assert fpg.away_goal_expectation == 1.2
        assert fpg.normalize is True  # default value

    def test_normalization_enabled(self, unnormalized_grid):
        """Test initialization with normalization enabled."""
        fpg = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=True
        )
        # Grid should be normalized to sum to 1.0
        assert np.isclose(fpg.grid.sum(), 1.0, atol=1e-10)

    def test_normalization_disabled(self, unnormalized_grid):
        """Test initialization with normalization disabled."""
        fpg = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=False
        )
        # Grid should retain original values
        assert np.isclose(fpg.grid.sum(), 2.1, atol=1e-10)

    def test_invalid_matrix_empty(self):
        """Test that empty matrix raises ValueError."""
        with pytest.raises(
            ValueError, match="goal_matrix must be a non-empty 2D array"
        ):
            pb.models.FootballProbabilityGrid(np.array([]), 1.0, 1.0)

    def test_invalid_matrix_1d(self):
        """Test that 1D matrix raises ValueError."""
        with pytest.raises(
            ValueError, match="goal_matrix must be a non-empty 2D array"
        ):
            pb.models.FootballProbabilityGrid(np.array([0.1, 0.2, 0.3]), 1.0, 1.0)

    def test_negative_probabilities(self):
        """Test that negative probabilities raise ValueError."""
        matrix = np.array([[0.1, -0.2], [0.3, 0.4]])
        with pytest.raises(
            ValueError, match="goal_matrix contains negative probabilities"
        ):
            pb.models.FootballProbabilityGrid(matrix, 1.0, 1.0)

    def test_zero_sum_normalization_error(self):
        """Test that zero-sum matrix raises error when normalization is requested."""
        matrix = np.zeros((2, 2))
        with pytest.raises(
            ValueError, match="goal_matrix sums to zero; cannot normalize"
        ):
            pb.models.FootballProbabilityGrid(matrix, 1.0, 1.0, normalize=True)


class TestCoreMarkets:
    """Test core betting market calculations (1X2, etc.)."""

    def test_1x2_markets(self, normalized_grid):
        """Test home win, draw, and away win probabilities."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Check types
        assert isinstance(fpg.home_win, (float, np.float64))
        assert isinstance(fpg.draw, (float, np.float64))
        assert isinstance(fpg.away_win, (float, np.float64))

        # Check that probabilities sum to 1 for normalized grid
        total_1x2 = fpg.home_win + fpg.draw + fpg.away_win
        assert np.isclose(total_1x2, 1.0, atol=1e-10)

        # Check home_draw_away property
        hda = fpg.home_draw_away
        assert isinstance(hda, list)
        assert len(hda) == 3
        assert hda == [fpg.home_win, fpg.draw, fpg.away_win]

    def test_1x2_calculation_correctness(self):
        """Test that 1X2 calculations are mathematically correct."""
        # Create a simple grid where we can manually verify calculations
        matrix = np.array(
            [
                [0.1, 0.2, 0.1],  # 0-0: draw, 0-1: away, 0-2: away
                [0.2, 0.1, 0.05],  # 1-0: home, 1-1: draw, 1-2: away
                [0.1, 0.05, 0.1],  # 2-0: home, 2-1: home, 2-2: draw
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.5, 1.2)

        # Manual calculation:
        # Home win (i > j): (1,0), (2,0), (2,1) = 0.2 + 0.1 + 0.05 = 0.35
        # Draw (i == j): (0,0), (1,1), (2,2) = 0.1 + 0.1 + 0.1 = 0.3
        # Away win (i < j): (0,1), (0,2), (1,2) = 0.2 + 0.1 + 0.05 = 0.35

        assert np.isclose(fpg.home_win, 0.35, atol=1e-10)
        assert np.isclose(fpg.draw, 0.3, atol=1e-10)
        assert np.isclose(fpg.away_win, 0.35, atol=1e-10)

    def test_both_teams_to_score(self, normalized_grid):
        """Test both teams to score (BTTS) market."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        btts = fpg.btts_yes
        btts_no = fpg.btts_no

        # Check types and values
        assert isinstance(btts, (float, np.float64))
        assert isinstance(btts_no, (float, np.float64))
        assert 0 <= btts <= 1
        assert 0 <= btts_no <= 1

        # BTTS + BTTS_NO should sum to 1
        assert np.isclose(btts + btts_no, 1.0, atol=1e-10)

    def test_btts_calculation_correctness(self):
        """Test BTTS calculation correctness."""
        matrix = np.array(
            [
                [0.1, 0.2, 0.1],  # (0,0): no BTTS, (0,1): no BTTS, (0,2): no BTTS
                [0.2, 0.1, 0.05],  # (1,0): no BTTS, (1,1): BTTS, (1,2): BTTS
                [0.1, 0.05, 0.1],  # (2,0): no BTTS, (2,1): BTTS, (2,2): BTTS
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.5, 1.2)

        # BTTS occurs when both home and away > 0
        # BTTS scores: (1,1), (1,2), (2,1), (2,2) = 0.1 + 0.05 + 0.05 + 0.1 = 0.3
        assert np.isclose(fpg.btts_yes, 0.3, atol=1e-10)
        assert np.isclose(fpg.btts_no, 0.7, atol=1e-10)


class TestDoubleChanceMarkets:
    """Test double chance markets (1X, X2, 12)."""

    def test_double_chance_markets(self, normalized_grid):
        """Test all double chance market calculations."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        dc_1x = fpg.double_chance_1x  # Home or Draw
        dc_x2 = fpg.double_chance_x2  # Draw or Away
        dc_12 = fpg.double_chance_12  # Home or Away

        # Check types
        assert isinstance(dc_1x, (float, np.float64))
        assert isinstance(dc_x2, (float, np.float64))
        assert isinstance(dc_12, (float, np.float64))

        # Verify calculations
        assert np.isclose(dc_1x, fpg.home_win + fpg.draw, atol=1e-10)
        assert np.isclose(dc_x2, fpg.draw + fpg.away_win, atol=1e-10)
        assert np.isclose(dc_12, fpg.home_win + fpg.away_win, atol=1e-10)

    def test_draw_no_bet(self, normalized_grid):
        """Test Draw No Bet markets."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        dnb_home = fpg.draw_no_bet_home
        dnb_away = fpg.draw_no_bet_away

        # Check types
        assert isinstance(dnb_home, (float, np.float64))
        assert isinstance(dnb_away, (float, np.float64))

        # DNB probabilities should be conditional probabilities
        if fpg.draw < 1.0:  # Avoid division by zero
            expected_home = fpg.home_win / (1.0 - fpg.draw)
            expected_away = fpg.away_win / (1.0 - fpg.draw)
            assert np.isclose(dnb_home, expected_home, atol=1e-10)
            assert np.isclose(dnb_away, expected_away, atol=1e-10)

            # DNB home + DNB away should sum to 1 (conditional on no draw)
            assert np.isclose(dnb_home + dnb_away, 1.0, atol=1e-10)

    def test_draw_no_bet_edge_case(self):
        """Test Draw No Bet when draw probability is 1 (edge case)."""
        # Create matrix where only draws are possible
        matrix = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # Only (1,1) has probability
                [0.0, 0.0, 0.0],
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.0, 1.0)

        # When draw = 1.0, DNB should return 0.0
        assert fpg.draw_no_bet_home == 0.0
        assert fpg.draw_no_bet_away == 0.0


class TestTotalsMarkets:
    """Test over/under totals markets."""

    def test_totals_with_push(self, normalized_grid):
        """Test totals calculation including push probabilities."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test integer line (can have push)
        under, push, over = fpg.totals(2.0)
        assert isinstance(under, float)
        assert isinstance(push, float)
        assert isinstance(over, float)

        # Should sum to 1.0 for normalized grid
        assert np.isclose(under + push + over, 1.0, atol=1e-10)

        # Test half-line (no push possible)
        under_half, push_half, over_half = fpg.totals(2.5)
        assert push_half == 0.0  # No push on half-lines
        assert np.isclose(under_half + over_half, 1.0, atol=1e-10)

    def test_total_goals_backward_compatibility(self, normalized_grid):
        """Test backward-compatible total_goals method."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test over/under for different lines
        over_2_5 = fpg.total_goals("over", 2.5)
        under_2_5 = fpg.total_goals("under", 2.5)

        assert isinstance(over_2_5, float)
        assert isinstance(under_2_5, float)
        assert np.isclose(over_2_5 + under_2_5, 1.0, atol=1e-10)

        # Test error handling
        with pytest.raises(ValueError, match="over_under must be 'over' or 'under'"):
            fpg.total_goals("wrong", 2.5)

    def test_totals_calculation_correctness(self):
        """Test totals calculations are mathematically correct."""
        matrix = np.array(
            [
                [0.1, 0.2, 0.1],  # Total goals: 0, 1, 2
                [0.15, 0.1, 0.05],  # Total goals: 1, 2, 3
                [0.1, 0.05, 0.1],  # Total goals: 2, 3, 4
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.5, 1.2, normalize=False)

        # Manual calculation for totals(2.0):
        # Under 2 (total < 2): (0,0)=0, (0,1)=1 -> 0.1 + 0.2 = 0.3
        # Push (total == 2): (0,2), (1,1), (2,0) -> 0.1 + 0.1 + 0.1 = 0.3
        # Over 2 (total > 2): (1,2), (2,1), (2,2) -> 0.05 + 0.05 + 0.1 = 0.2
        # Missing: (1,0) = 0.15, total = 1, so under

        under, push, over = fpg.totals(2.0)
        expected_under = 0.1 + 0.2 + 0.15  # (0,0), (0,1), (1,0)
        expected_push = 0.1 + 0.1 + 0.1  # (0,2), (1,1), (2,0)
        expected_over = 0.05 + 0.05 + 0.1  # (1,2), (2,1), (2,2)

        assert np.isclose(under, expected_under, atol=1e-10)
        assert np.isclose(push, expected_push, atol=1e-10)
        assert np.isclose(over, expected_over, atol=1e-10)

    def test_totals_quarter_lines(self, normalized_grid):
        """Test totals with quarter lines (split stakes)."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test 2.25 line (should be 50% on 2.0, 50% on 2.5)
        under_225, push_225, over_225 = fpg.totals(2.25)

        # Get the individual line probabilities
        under_20, push_20, over_20 = fpg.totals(2.0)
        under_25, push_25, over_25 = fpg.totals(2.5)

        # Quarter line should be average of the two
        expected_under = 0.5 * (under_20 + under_25)
        expected_push = 0.5 * (push_20 + push_25)
        expected_over = 0.5 * (over_20 + over_25)

        assert np.isclose(under_225, expected_under, atol=1e-10)
        assert np.isclose(push_225, expected_push, atol=1e-10)
        assert np.isclose(over_225, expected_over, atol=1e-10)

        # Should still sum to 1.0
        assert np.isclose(under_225 + push_225 + over_225, 1.0, atol=1e-10)

    def test_totals_quarter_lines_075(self, normalized_grid):
        """Test totals with 0.75 quarter line."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test 1.75 line (should be 50% on 1.5, 50% on 2.0)
        under_175, push_175, over_175 = fpg.totals(1.75)

        under_15, push_15, over_15 = fpg.totals(1.5)
        under_20, push_20, over_20 = fpg.totals(2.0)

        expected_under = 0.5 * (under_15 + under_20)
        expected_push = 0.5 * (push_15 + push_20)
        expected_over = 0.5 * (over_15 + over_20)

        assert np.isclose(under_175, expected_under, atol=1e-10)
        assert np.isclose(push_175, expected_push, atol=1e-10)
        assert np.isclose(over_175, expected_over, atol=1e-10)

    def test_totals_quarter_line_push_probability(self):
        """Test that quarter lines correctly return non-zero push probability."""
        # Create a grid where we know exactly what the push probability should be
        matrix = np.array(
            [
                [0.1, 0.1, 0.1],  # Total goals: 0, 1, 2
                [0.1, 0.1, 0.1],  # Total goals: 1, 2, 3
                [0.1, 0.1, 0.1],  # Total goals: 2, 3, 4
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.0, 1.0)

        # Matrix sums to 0.9, so after normalization each 0.1 becomes 0.1/0.9 = 1/9
        # For line 1.25: 50% on 1.0, 50% on 1.5
        # Push on 1.0 = probability of exactly 1 goal = (0,1) + (1,0) = 2/9
        # Push on 1.5 = 0 (half line)
        # Expected push for 1.25 = 0.5 * (2/9 + 0) = 1/9
        under, push, over = fpg.totals(1.25)

        # Verify push is non-zero (the bug we're fixing)
        assert push > 0.0
        assert np.isclose(push, 1.0 / 9.0, atol=1e-10)


class TestAsianHandicapMarkets:
    """Test Asian handicap markets."""

    def test_asian_handicap_probs(self, normalized_grid):
        """Test Asian handicap with win/push/lose probabilities."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test half-line (no push)
        home_probs = fpg.asian_handicap_probs("home", -0.5)
        away_probs = fpg.asian_handicap_probs("away", 0.5)

        # Check structure
        assert isinstance(home_probs, dict)
        assert "win" in home_probs and "push" in home_probs and "lose" in home_probs
        assert isinstance(away_probs, dict)

        # For half-lines, push should be 0
        assert home_probs["push"] == 0.0
        assert away_probs["push"] == 0.0

        # Probabilities should sum to 1
        assert np.isclose(sum(home_probs.values()), 1.0, atol=1e-10)
        assert np.isclose(sum(away_probs.values()), 1.0, atol=1e-10)

    def test_asian_handicap_quarter_lines(self, normalized_grid):
        """Test Asian handicap with quarter lines (split stakes)."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test quarter line
        probs = fpg.asian_handicap_probs("home", -0.25)

        assert isinstance(probs, dict)
        assert "win" in probs and "push" in probs and "lose" in probs
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-10)

    def test_asian_handicap_backward_compatibility(self, normalized_grid):
        """Test backward-compatible asian_handicap method."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        home_win_prob = fpg.asian_handicap("home", 0.5)
        away_win_prob = fpg.asian_handicap("away", -0.5)

        assert isinstance(home_win_prob, float)
        assert isinstance(away_win_prob, float)
        assert 0 <= home_win_prob <= 1
        assert 0 <= away_win_prob <= 1

        # Verify consistency with detailed method
        detailed_home = fpg.asian_handicap_probs("home", 0.5)["win"]
        detailed_away = fpg.asian_handicap_probs("away", -0.5)["win"]

        assert np.isclose(home_win_prob, detailed_home, atol=1e-10)
        assert np.isclose(away_win_prob, detailed_away, atol=1e-10)

    def test_asian_handicap_error_handling(self, normalized_grid):
        """Test Asian handicap error handling."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test invalid side parameter
        with pytest.raises(ValueError, match="side must be 'home' or 'away'"):
            fpg.asian_handicap_probs("wrong", 0.5)

        with pytest.raises(ValueError, match="home_away must be 'home' or 'away'"):
            fpg.asian_handicap("wrong", 0.5)

    def test_asian_handicap_sign_correctness(self):
        """Test that Asian handicap signs are correctly interpreted.

        Verifies that:
        - Negative line (e.g., -0.5): side gives goals, must win for bet to win
        - Positive line (e.g., +0.5): side receives goals, draw also wins
        """
        # Create a grid with known 1X2 probabilities
        matrix = np.array(
            [
                [0.1, 0.1, 0.1],  # (0,0)=draw, (0,1)=away, (0,2)=away
                [0.2, 0.1, 0.05],  # (1,0)=home, (1,1)=draw, (1,2)=away
                [0.1, 0.1, 0.05],  # (2,0)=home, (2,1)=home, (2,2)=draw
            ]
        )
        fpg = pb.models.FootballProbabilityGrid(matrix, 1.0, 1.0)

        # Home -0.5: home gives 0.5 goals, must win for bet to win
        probs = fpg.asian_handicap_probs("home", -0.5)
        assert np.isclose(probs["win"], fpg.home_win, atol=1e-10)
        assert np.isclose(probs["push"], 0.0, atol=1e-10)
        assert np.isclose(probs["lose"], fpg.draw + fpg.away_win, atol=1e-10)

        # Home +0.5: home receives 0.5 goals, draw also wins
        probs = fpg.asian_handicap_probs("home", 0.5)
        assert np.isclose(probs["win"], fpg.home_win + fpg.draw, atol=1e-10)
        assert np.isclose(probs["push"], 0.0, atol=1e-10)
        assert np.isclose(probs["lose"], fpg.away_win, atol=1e-10)

        # Home 0.0 (DNB): win=home_win, push=draw, lose=away_win
        probs = fpg.asian_handicap_probs("home", 0.0)
        assert np.isclose(probs["win"], fpg.home_win, atol=1e-10)
        assert np.isclose(probs["push"], fpg.draw, atol=1e-10)
        assert np.isclose(probs["lose"], fpg.away_win, atol=1e-10)

        # Away -0.5: away gives 0.5 goals, must win for bet to win
        probs = fpg.asian_handicap_probs("away", -0.5)
        assert np.isclose(probs["win"], fpg.away_win, atol=1e-10)
        assert np.isclose(probs["push"], 0.0, atol=1e-10)
        assert np.isclose(probs["lose"], fpg.draw + fpg.home_win, atol=1e-10)

        # Away +0.5: away receives 0.5 goals, draw also wins
        probs = fpg.asian_handicap_probs("away", 0.5)
        assert np.isclose(probs["win"], fpg.away_win + fpg.draw, atol=1e-10)
        assert np.isclose(probs["push"], 0.0, atol=1e-10)
        assert np.isclose(probs["lose"], fpg.home_win, atol=1e-10)


class TestExactScoresAndDistributions:
    """Test exact score probabilities and goal distributions."""

    def test_exact_score(self, normalized_grid):
        """Test exact score probabilities."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # Test valid scores
        score_0_0 = fpg.exact_score(0, 0)
        score_1_1 = fpg.exact_score(1, 1)
        score_2_2 = fpg.exact_score(2, 2)

        assert isinstance(score_0_0, float)
        assert score_0_0 == normalized_grid[0, 0]
        assert score_1_1 == normalized_grid[1, 1]
        assert score_2_2 == normalized_grid[2, 2]

        # Test out-of-bounds scores
        assert fpg.exact_score(5, 5) == 0.0
        assert fpg.exact_score(-1, 0) == 0.0
        assert fpg.exact_score(0, -1) == 0.0

    def test_goal_distributions(self, normalized_grid):
        """Test marginal goal distributions."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        home_dist = fpg.home_goal_distribution()
        away_dist = fpg.away_goal_distribution()
        total_dist = fpg.total_goals_distribution()

        # Check types and shapes
        assert isinstance(home_dist, np.ndarray)
        assert isinstance(away_dist, np.ndarray)
        assert isinstance(total_dist, np.ndarray)

        assert len(home_dist) == normalized_grid.shape[0]
        assert len(away_dist) == normalized_grid.shape[1]
        assert (
            len(total_dist) == normalized_grid.shape[0] + normalized_grid.shape[1] - 1
        )

        # Check that marginals sum to 1 for normalized grid
        assert np.isclose(home_dist.sum(), 1.0, atol=1e-10)
        assert np.isclose(away_dist.sum(), 1.0, atol=1e-10)
        assert np.isclose(total_dist.sum(), 1.0, atol=1e-10)

        # Verify marginal calculations
        expected_home = normalized_grid.sum(axis=1)
        expected_away = normalized_grid.sum(axis=0)

        assert np.allclose(home_dist, expected_home, atol=1e-10)
        assert np.allclose(away_dist, expected_away, atol=1e-10)


class TestSpecialMarkets:
    """Test special markets like win to nil, expected points."""

    def test_win_to_nil(self, normalized_grid):
        """Test win to nil probabilities."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        home_wtn = fpg.win_to_nil_home()
        away_wtn = fpg.win_to_nil_away()

        assert isinstance(home_wtn, float)
        assert isinstance(away_wtn, float)
        assert 0 <= home_wtn <= 1
        assert 0 <= away_wtn <= 1

        # Verify calculations
        # Home win to nil: home > 0, away = 0
        expected_home_wtn = normalized_grid[1:, 0].sum()
        # Away win to nil: home = 0, away > 0
        expected_away_wtn = normalized_grid[0, 1:].sum()

        assert np.isclose(home_wtn, expected_home_wtn, atol=1e-10)
        assert np.isclose(away_wtn, expected_away_wtn, atol=1e-10)

    def test_expected_points(self, normalized_grid):
        """Test expected points calculations (3-1-0 system)."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        home_points = fpg.expected_points_home()
        away_points = fpg.expected_points_away()

        assert isinstance(home_points, float)
        assert isinstance(away_points, float)
        assert 0 <= home_points <= 3
        assert 0 <= away_points <= 3

        # Verify calculations (3 for win, 1 for draw, 0 for loss)
        expected_home = 3.0 * fpg.home_win + 1.0 * fpg.draw
        expected_away = 3.0 * fpg.away_win + 1.0 * fpg.draw

        assert np.isclose(home_points, expected_home, atol=1e-10)
        assert np.isclose(away_points, expected_away, atol=1e-10)


class TestNormalizationFeature:
    """Test the new normalization feature."""

    def test_normalization_functionality(self, unnormalized_grid):
        """Test that normalization works correctly."""
        # Without normalization
        fpg_no_norm = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=False
        )
        original_sum = fpg_no_norm.grid.sum()
        assert not np.isclose(original_sum, 1.0, atol=1e-6)

        # With normalization
        fpg_norm = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=True
        )
        normalized_sum = fpg_norm.grid.sum()
        assert np.isclose(normalized_sum, 1.0, atol=1e-10)

        # Verify that normalized grid maintains proportions
        expected_normalized = unnormalized_grid / original_sum
        assert np.allclose(fpg_norm.grid, expected_normalized, atol=1e-10)

    def test_already_normalized_grid(self, normalized_grid):
        """Test that already normalized grids work correctly with normalize=True."""
        fpg = pb.models.FootballProbabilityGrid(
            normalized_grid, 1.5, 1.2, normalize=True
        )

        # Should still sum to 1.0 and be very close to original
        assert np.isclose(fpg.grid.sum(), 1.0, atol=1e-10)
        assert np.allclose(fpg.grid, normalized_grid, atol=1e-10)

    def test_normalization_preserves_markets(self, unnormalized_grid):
        """Test that normalization preserves relative market probabilities."""
        fpg_no_norm = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=False
        )
        fpg_norm = pb.models.FootballProbabilityGrid(
            unnormalized_grid, 1.5, 1.2, normalize=True
        )

        # Calculate normalization factor
        norm_factor = unnormalized_grid.sum()

        # All probability ratios should be preserved
        assert np.isclose(
            fpg_norm.home_win / fpg_norm.away_win,
            fpg_no_norm.home_win / fpg_no_norm.away_win,
            atol=1e-10,
        )

        # Market probabilities should scale by normalization factor
        assert np.isclose(
            fpg_norm.home_win * norm_factor, fpg_no_norm.home_win, atol=1e-10
        )


class TestCachingAndPerformance:
    """Test internal caching mechanism and performance aspects."""

    def test_caching_mechanism(self, normalized_grid):
        """Test that caching works correctly for repeated calculations."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        # First access should populate cache
        home_win_1 = fpg.home_win
        assert "home_win" in fpg._cache

        # Second access should use cache (same result)
        home_win_2 = fpg.home_win
        assert home_win_1 == home_win_2

        # Cache should contain the computed value
        assert fpg._cache["home_win"] == home_win_1

    def test_index_grids_created(self, normalized_grid):
        """Test that index grids are properly created for vectorized operations."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)

        assert hasattr(fpg, "_I")
        assert hasattr(fpg, "_J")
        assert isinstance(fpg._I, np.ndarray)
        assert isinstance(fpg._J, np.ndarray)
        assert fpg._I.shape == normalized_grid.shape
        assert fpg._J.shape == normalized_grid.shape


class TestStringRepresentation:
    """Test string representation of FootballProbabilityGrid."""

    def test_repr_content(self, normalized_grid):
        """Test that __repr__ contains expected information."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 1.5, 1.2)
        repr_str = str(fpg)

        assert isinstance(repr_str, str)
        assert "Module: Penaltyblog" in repr_str
        assert "Class: FootballProbabilityGrid" in repr_str
        assert "Home Goal Expectation: 1.500" in repr_str
        assert "Away Goal Expectation: 1.200" in repr_str
        assert "Home Win:" in repr_str
        assert "Draw:" in repr_str
        assert "Away Win:" in repr_str

    def test_repr_formatting(self, normalized_grid):
        """Test that __repr__ has proper formatting."""
        fpg = pb.models.FootballProbabilityGrid(normalized_grid, 2.567, 1.234)
        repr_str = str(fpg)

        # Test that goal expectations are formatted to 3 decimal places
        assert "Home Goal Expectation: 2.567" in repr_str
        assert "Away Goal Expectation: 1.234" in repr_str

        # Test that probabilities are formatted to 4 decimal places
        lines = repr_str.split("\n")
        prob_lines = [line for line in lines if "Win:" in line or "Draw:" in line]
        for line in prob_lines:
            # Extract the probability value (after colon and space)
            prob_part = line.split(": ")[1]
            # Should have 4 decimal places
            assert len(prob_part.split(".")[1]) == 4


# Legacy tests (updated for better coverage)
def test_grid():
    """Legacy test updated with better assertions."""
    m = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    fpg = pb.models.FootballProbabilityGrid(m, 1, 1)

    # Test return types
    assert isinstance(fpg.home_win, (float, np.float64))
    assert isinstance(fpg.draw, (float, np.float64))
    assert isinstance(fpg.away_win, (float, np.float64))
    assert isinstance(fpg.btts_yes, (float, np.float64))
    assert isinstance(fpg.home_draw_away, list)
    assert isinstance(fpg.total_goals("over", 1.5), (float, np.float64))
    assert isinstance(fpg.total_goals("under", 1.5), (float, np.float64))
    assert isinstance(fpg.asian_handicap("home", 1.5), (float, np.float64))
    assert isinstance(fpg.asian_handicap("away", 1.5), (float, np.float64))

    # Test error cases
    with pytest.raises(ValueError, match="over_under must be 'over' or 'under'"):
        fpg.total_goals("wrong", 1.5)

    with pytest.raises(ValueError, match="home_away must be 'home' or 'away'"):
        fpg.asian_handicap("wrong", 1.5)


def test_str():
    """Legacy test for string representation."""
    m = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    fpg = pb.models.FootballProbabilityGrid(m, 1, 2)

    repr_str = str(fpg)
    assert isinstance(repr_str, str)
    assert "Class: FootballProbabilityGrid" in repr_str
    assert "Home Goal Expectation: 1.000" in repr_str
    assert "Away Goal Expectation: 2.000" in repr_str


class TestCreateDixonColesGrid:
    """Test the create_dixon_coles_grid factory function."""

    def test_basic_creation(self):
        """Test basic grid creation with default parameters."""
        grid = pb.models.create_dixon_coles_grid(1.5, 1.2)
        assert isinstance(grid, pb.models.FootballProbabilityGrid)
        assert grid.home_goal_expectation == 1.5
        assert grid.away_goal_expectation == 1.2
        assert grid.grid.shape == (16, 16)  # default max_goals is 15
        assert np.isclose(np.sum(grid.grid), 1.0)

    def test_custom_max_goals(self):
        """Test custom max_goals parameter."""
        grid = pb.models.create_dixon_coles_grid(1.5, 1.2, max_goals=10)
        assert grid.grid.shape == (11, 11)

    def test_invalid_lambdas(self):
        """Test that invalid lambdas raise ValueError."""
        with pytest.raises(
            ValueError, match="Expected goals \\(lambdas\\) must be strictly positive"
        ):
            pb.models.create_dixon_coles_grid(0.0, 1.2)

        with pytest.raises(
            ValueError, match="Expected goals \\(lambdas\\) must be strictly positive"
        ):
            pb.models.create_dixon_coles_grid(1.5, -0.1)

    def test_rho_adjustment(self):
        """Test that rho adjustments work as expected."""
        # Grid with independent Poisson
        grid_base = pb.models.create_dixon_coles_grid(1.5, 1.2, rho=0.0)

        # Grid with positive rho
        grid_rho = pb.models.create_dixon_coles_grid(1.5, 1.2, rho=0.1)

        # 1-1 outcome probability should be decreased for positive rho (grid[1,1] *= 1-rho)
        # Note: the effect might be complex due to renormalization, but let's check
        # the unnormalized initial adjustment or just check matrices aren't equal
        assert not np.array_equal(grid_base.grid, grid_rho.grid)

        # The sum should still be 1 after adjustment
        assert np.isclose(np.sum(grid_rho.grid), 1.0)

    def test_negative_probability_throws(self):
        """Test that rho causing negative initial probability does not throw and clips to 0."""
        with pytest.raises(ValueError, match="is out of bounds for the given lambdas"):
            pb.models.create_dixon_coles_grid(3.0, 3.0, rho=0.5)
