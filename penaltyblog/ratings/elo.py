from typing import Tuple


class Elo:
    """Used to calculate ELO ratings for a group of players

    Methods
    -------
    add_player(name, rating)
        Adds a new player to the ratings and sets their initial ratings

    expected_results(p_a, p_b)
        Gets the probability of player a and player b winning a game against each other

    update_ratings(p_a, p_b, outcome)
        Updates player A and player B's ratings after they play against each other

    get_rating(name)
        Gets a player's current rating
    """

    def __init__(self, k: int = 32):
        self.k = k
        self.ratings = dict()

    def add_player(self, name: str, rating: float = 1500):
        """
        Adds a new player to the ratings

        Parameters
        ----------
        name : str
            The name of the player

        rating : float
            The initial rating for theplayer
        """
        if name in self.ratings:
            raise ValueError("Player already exists")

        self.ratings[name] = rating

    def get_rating(self, name: str) -> float:
        if name not in self.ratings:
            raise ValueError("Player not found")
        return self.ratings[name]

    def expected_results(self, p_a, p_b) -> Tuple[float, float]:
        """
        Get the expected probabilities for each player winning

        Parameters
        ----------
        p_a : str
            The name of player one

        p_b : str
            The name of player two

        Returns
        -------
            A tuple of (probability_for_player_a, probability_for_player_b)
        """
        r_a = self.ratings[p_a]
        r_b = self.ratings[p_b]
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        return e_a, 1 - e_a

    def update_ratings(self, p_a: str, p_b: str, outcome):
        """
        Updates the Elo ratings for two players based on the outcome of a game

        Parameters
        ----------
        p_a : str
            The name of player one

        p_b : str
            The name of player two

        outcome : int
            0 if player one won, 1 if player two won

        """
        e_a, e_b = self.expected_results(p_a, p_b)

        if outcome == 0:
            self.ratings[p_a] = self.ratings[p_a] + self.k * (1 - e_a)
            self.ratings[p_b] = self.ratings[p_b] + self.k * (0 - e_b)

        elif outcome == 1:
            self.ratings[p_a] = self.ratings[p_a] + self.k * (0 - e_a)
            self.ratings[p_b] = self.ratings[p_b] + self.k * (1 - e_b)

        else:
            raise ValueError("Outcome not recognised")
