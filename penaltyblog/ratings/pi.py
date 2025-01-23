import numpy as np
from scipy.stats import norm


class PiRatingSystem:
    def __init__(self, alpha=0.15, beta=0.10, k=0.75, sigma=1.0):
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
        self.team_ratings = {}

    def initialize_team(self, team):
        """Initialize a team with a home and away rating of 0 if not already present."""
        if team not in self.team_ratings:
            self.team_ratings[team] = {"home": 0.0, "away": 0.0}

    def expected_goal_difference(self, home_team, away_team):
        """Calculate the expected goal difference based on current ratings."""
        home_rating = self.team_ratings[home_team]["home"]
        away_rating = self.team_ratings[away_team]["away"]
        return home_rating - away_rating

    def diminishing_error(self, error):
        """Apply diminishing returns to large score discrepancies."""
        return error / (1 + self.k * abs(error))

    def update_ratings(self, home_team, away_team, observed_goal_difference):
        """
        Update pi-ratings based on the observed goal difference.

        Parameters:
        - home_team (str): Name of the home team.
        - away_team (str): Name of the away team.
        - observed_goal_difference (int): Actual goal difference (home - away).
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

    def get_team_rating(self, team):
        """Return the average rating of a team (home and away)."""
        if team in self.team_ratings:
            return (
                self.team_ratings[team]["home"] + self.team_ratings[team]["away"]
            ) / 2
        return 0.0

    def calculate_match_probabilities(self, home_team, away_team):
        """
        Calculate the probabilities of a home win, draw, and away win.
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
        """Print the current team ratings."""
        for team, ratings in self.team_ratings.items():
            print(
                f"{team}: Home = {ratings['home']:.2f}, Away = {ratings['away']:.2f}, Average = {self.get_team_rating(team):.2f}"
            )


# Example usage
# pi_rating = PiRatingSystem()

# # Simulated match results: (home_team, away_team, home_goals - away_goals)
# match_results = [
#     ("Team A", "Team B", 2),
#     ("Team B", "Team C", -1),
#     ("Team C", "Team A", 0),
#     ("Team A", "Team B", 1),
# ]

# # Process match results
# for home_team, away_team, goal_diff in match_results:
#     pi_rating.update_ratings(home_team, away_team, goal_diff)

# # Display the final ratings
# pi_rating.display_ratings()

# # Calculate and display match probabilities
# probabilities = pi_rating.calculate_match_probabilities("Team A", "Team B")
# print(f"Probabilities - Home Win: {probabilities['home_win']:.2%}, Draw: {probabilities['draw']:.2%}, Away Win: {probabilities['away_win']:.2%}")
