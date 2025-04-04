"""
Pi Rating System

Calculates the Pi ratings for a group of teams.
"""

from typing import Any, Dict, List

from scipy.stats import norm


class PiRatingSystem:
    """
    Pi-Rating system parameters based on
    http://www.constantinou.info/downloads/papers/pi-ratings.pdf
    """

    def __init__(
        self,
        alpha: float = 0.15,
        beta: float = 0.10,
        k: float = 0.75,
        sigma: float = 1.0,
    ):
        """
        Initialize the Pi-Rating system parameters based on
        http://www.constantinou.info/downloads/papers/pi-ratings.pdf

        Parameters:
        - alpha (float): Learning rate for direct rating updates.
        - beta (float): Learning rate for cross-updates (home affects away, and vice versa).
        - k (float): Diminishing factor for goal difference impact.
        - sigma (float): Standard deviation used in probability calculation.
        """
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.sigma = sigma
        self.team_ratings: Dict[str, Dict[str, float]] = {}
        self.rating_history: List[Dict[str, Any]] = []

    def initialize_team(self, team: str):
        """
        Initialize a team with a home and away rating of 0 if not already present.

        Args:
            team (str): Name of the team.
        """
        if team not in self.team_ratings:
            self.team_ratings[team] = {"home": 0.0, "away": 0.0}

    def expected_goal_difference(self, home_team: str, away_team: str) -> float:
        """
        Calculate the expected goal difference based on current ratings.

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.

        Returns:
            float: Expected goal difference.
        """
        if home_team not in self.team_ratings:
            self.initialize_team(home_team)

        if away_team not in self.team_ratings:
            self.initialize_team(away_team)

        home_rating = self.team_ratings[home_team]["home"]
        away_rating = self.team_ratings[away_team]["away"]

        return home_rating - away_rating

    def diminishing_error(self, error: float) -> float:
        """
        Apply diminishing returns to large score discrepancies.

        Args:
            error (float): The raw error between observed and expected goal difference.

        Returns:
            float: Adjusted error with diminishing returns.
        """
        return error / (1 + self.k * abs(error))

    def update_ratings(
        self, home_team: str, away_team: str, observed_goal_difference: int, date=None
    ):
        """
        Update pi-ratings based on the observed goal difference.

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.
            observed_goal_difference (int): Actual goal difference (home - away).
            date (datetime): Date of the match (optional).
        """
        self.initialize_team(home_team)
        self.initialize_team(away_team)

        # Calculate expected goal difference
        expected_diff = self.expected_goal_difference(home_team, away_team)

        # Compute error and apply diminishing effect
        error = observed_goal_difference - expected_diff
        adjusted_error = self.diminishing_error(error)

        # Update ratings with learning rates
        self.team_ratings[home_team]["home"] += self.alpha * adjusted_error
        self.team_ratings[home_team]["away"] += self.beta * adjusted_error
        self.team_ratings[away_team]["home"] -= self.beta * adjusted_error
        self.team_ratings[away_team]["away"] -= self.alpha * adjusted_error

        if date is not None:
            self.rating_history.append(
                {
                    "date": date,
                    "team": home_team,
                    "home_rating": self.team_ratings[home_team]["home"],
                    "away_rating": self.team_ratings[home_team]["away"],
                }
            )
            self.rating_history.append(
                {
                    "date": date,
                    "team": away_team,
                    "home_rating": self.team_ratings[away_team]["home"],
                    "away_rating": self.team_ratings[away_team]["away"],
                }
            )

    def get_team_rating(self, team: str) -> float:
        """
        Return the average rating of a team (home and away).

        Args:
            team (str): Name of the team.

        Returns:
            float: Average rating of the team.
        """
        if team in self.team_ratings:
            return (
                self.team_ratings[team]["home"] + self.team_ratings[team]["away"]
            ) / 2
        self.initialize_team(team)
        return 0.0

    def calculate_match_probabilities(self, home_team: str, away_team: str) -> dict:
        """
        Calculate the probabilities of a home win, draw, and away win.

        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.

        Returns:
            dict: Probabilities of home win, draw, and away win.
        """
        expected_diff = self.expected_goal_difference(home_team, away_team)
        draw_margin = 0.5

        prob_draw = norm.cdf(
            draw_margin, loc=expected_diff, scale=self.sigma
        ) - norm.cdf(-draw_margin, loc=expected_diff, scale=self.sigma)
        prob_home_win = 1 - norm.cdf(draw_margin, loc=expected_diff, scale=self.sigma)
        prob_away_win = norm.cdf(-draw_margin, loc=expected_diff, scale=self.sigma)

        return {"home_win": prob_home_win, "draw": prob_draw, "away_win": prob_away_win}

    def display_ratings(self):
        """
        Print the current team ratings.
        """
        for team, ratings in self.team_ratings.items():
            print(
                f"{team}: Home = {ratings['home']:.2f}, Away = {ratings['away']:.2f}, Average = {self.get_team_rating(team):.2f}"
            )
