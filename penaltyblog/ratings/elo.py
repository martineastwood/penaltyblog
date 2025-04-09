"""Football Elo Ratings System"""

from typing import Dict

import numpy as np


class Elo:
    """
    Elo rating system implementation designed for football matches by including
    home field advantage and draw probability.
    """

    def __init__(self, k: float = 20.0, home_field_advantage: float = 100.0):
        """
        Initialize the Elo rating system with default parameters.

        Args:
            k (float): K-factor for rating updates.
            home_field_advantage (float): Home field advantage in Elo points.
        """
        self.k = k
        self.hfa = home_field_advantage
        self.ratings: Dict[str, float] = {}

    def get_team_rating(self, team: str) -> float:
        """
        Get the Elo rating for a team.
        Args:
            team (str): Team name.
        Returns:
            float: Elo rating for the team.
        """
        if team not in self.ratings:
            self.ratings[team] = 1500.0
        return self.ratings[team]

    def home_win_probability(self, home: str, away: str) -> float:
        """
        Calculate the expected score for a match between two teams.
        Args:
            home (str): Home team name.
            away (str): Away team name.
        Returns:
            float: Expected score for the home team.
        """
        r_home = self.get_team_rating(home) + self.hfa
        r_away = self.get_team_rating(away)
        return 1 / (1 + 10 ** ((r_away - r_home) / 400))

    def calculate_match_probabilities(
        self,
        home: str,
        away: str,
        draw_base: float = 0.3,
        draw_width: float = 200.0,
    ) -> Dict[str, float]:
        """
        Predicts probabilities for home win, draw, away win.
        Draw probability is modeled as a Gaussian-shaped function around Elo difference.

        Args:
            home (str): Home team name.
            away (str): Away team name.
            draw_base (float): Base probability for draw.
            draw_width (float): Width of Gaussian for draw probability.

        Returns:
            Tuple[float, float, float]: Probabilities for home win, draw, away win.
        """
        r_home = self.get_team_rating(home) + self.hfa
        r_away = self.get_team_rating(away)
        elo_diff = r_home - r_away

        p_draw = draw_base * np.exp(-(elo_diff**2) / (2 * draw_width**2))
        p_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        p_away = 1 - p_home

        # Normalize to sum to 1
        z = p_home + p_away + p_draw
        return {"home_win": p_home / z, "draw": p_draw / z, "away_win": p_away / z}

    def update_ratings(self, home: str, away: str, result: int) -> None:
        """
        Updates Elo ratings based on match result.
        result = 0 → home win
        result = 1 → draw
        result = 2 → away win

        Args:
            home (str): Home team name.
            away (str): Away team name.
            result (int): Match result (0 for home win, 1 for draw, 2 for away win).
        """
        r_home = self.get_team_rating(home)
        r_away = self.get_team_rating(away)

        expected_home = self.home_win_probability(home, away)
        expected_away = 1 - expected_home

        if result == 0:
            actual_home, actual_away = 1.0, 0.0
        elif result == 1:
            actual_home, actual_away = 0.5, 0.5
        elif result == 2:
            actual_home, actual_away = 0.0, 1.0
        else:
            raise ValueError("Invalid result: must be 0, 1, or 2")

        self.ratings[home] = r_home + self.k * (actual_home - expected_home)
        self.ratings[away] = r_away + self.k * (actual_away - expected_away)
