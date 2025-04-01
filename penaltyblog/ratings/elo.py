"""
Elo Rating System

Calculates the Elo ratings for a group of teams.
"""

from typing import Dict, Tuple

import numpy as np


class Elo:
    """Used to calculate ELO ratings for a group of players"""

    def __init__(self, k: int = 32):
        self.k = k
        self.ratings: Dict[str, float] = {}

    def add_player(self, name: str, rating: float = 1500) -> None:
        """Add a new player with initial rating"""
        if name in self.ratings:
            raise ValueError("Player already exists")
        self.ratings[name] = rating

    def get_rating(self, name: str) -> float:
        """
        Get a player's current rating

        Args:
            name (str): Name of the player

        Returns:
            float: Current rating of the player

        Raises:
            ValueError: If the player is not found
        """
        if name not in self.ratings:
            raise ValueError("Player not found")
        return self.ratings[name]

    def expected_results(self, p_a: str, p_b: str) -> Tuple[float, float]:
        """
        Calculate expected win probabilities for both players

        Args:
            p_a (str): Name of player A
            p_b (str): Name of player B

        Returns:
            Tuple[float, float]: Probability of player A winning, probability of player B winning
        """
        r_a = self.ratings[p_a]
        r_b = self.ratings[p_b]
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        return e_a, 1 - e_a

    def expected_results_with_draw(
        self, p_a: str, p_b: str, draw_base: float = 0.3, draw_width: float = 200
    ) -> Tuple[float, float, float]:
        """
        Calculate expected win probabilities for both players, including draw

        Args:
            p_a (str): Name of player A
            p_b (str): Name of player B
            draw_base (float, optional): Base probability of draw. Defaults to 0.3.
            draw_width (float, optional): Width of draw probability curve. Defaults to 200.

        Returns:
            Tuple[float, float, float]: Probability of player A winning, probability of player B winning, probability of draw
        """
        r_a = self.ratings[p_a]
        r_b = self.ratings[p_b]
        elo_diff = r_a - r_b
        P_draw = draw_base * np.exp(-(elo_diff**2) / (2 * draw_width**2))
        P_home_win = 1 - P_draw
        P_away_win = 1 - P_draw
        return P_home_win, P_away_win, P_draw

    def update_ratings(self, p_a: str, p_b: str, outcome: int) -> None:
        """
        Update ratings based on game outcome (0: p_a wins, 1: p_b wins)

        Args:
            p_a (str): Name of player A
            p_b (str): Name of player B
            outcome (int): Outcome of the game (0: p_a wins, 1: p_b wins)

        Raises:
            ValueError: If the outcome is not 0 or 1
        """
        if outcome not in (0, 1):
            raise ValueError("Outcome must be 0 or 1")

        e_a, e_b = self.expected_results(p_a, p_b)
        actual_a = 1 - outcome
        actual_b = outcome

        self.ratings[p_a] += self.k * (actual_a - e_a)
        self.ratings[p_b] += self.k * (actual_b - e_b)
