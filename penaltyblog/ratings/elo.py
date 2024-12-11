from typing import Dict, Tuple


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
        """Get a player's current rating"""
        if name not in self.ratings:
            raise ValueError("Player not found")
        return self.ratings[name]

    def expected_results(self, p_a: str, p_b: str) -> Tuple[float, float]:
        """Calculate expected win probabilities for both players"""
        r_a = self.ratings[p_a]
        r_b = self.ratings[p_b]
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        return e_a, 1 - e_a

    def update_ratings(self, p_a: str, p_b: str, outcome: int) -> None:
        """Update ratings based on game outcome (0: p_a wins, 1: p_b wins)"""
        if outcome not in (0, 1):
            raise ValueError("Outcome must be 0 or 1")

        e_a, e_b = self.expected_results(p_a, p_b)
        actual_a = 1 - outcome
        actual_b = outcome

        self.ratings[p_a] += self.k * (actual_a - e_a)
        self.ratings[p_b] += self.k * (actual_b - e_b)
