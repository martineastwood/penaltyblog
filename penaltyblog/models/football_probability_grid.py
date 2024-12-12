from typing import Callable, List

import numpy as np
from numpy.typing import NDArray


class FootballProbabilityGrid:
    """
    Class for calculating probabilities of football outcomes.
    """

    def __init__(
        self,
        goal_matrix: NDArray,
        home_goal_expectation: float,
        away_goal_expectation: float,
    ):
        """
        Calculate probabilities of football outcomes.

        Parameters
        ----------
        goal_matrix : List[List[float]] Matrix of probabilities for each goal difference
        home_goal_expectation : float Expected number of goals for home
        away_goal_expectation : float Expected number of goals for away
        """
        self.grid = np.array(goal_matrix)
        self.home_goal_expectation = home_goal_expectation
        self.away_goal_expectation = away_goal_expectation

    def __repr__(self) -> str:
        return (
            f"Module: Penaltyblog\n\n"
            f"Class: FootballProbabilityGrid\n\n"
            f"Home Goal Expectation: {self.home_goal_expectation}\n"
            f"Away Goal Expectation: {self.away_goal_expectation}\n\n"
            f"Home Win: {self.home_win}\n"
            f"Draw: {self.draw}\n"
            f"Away Win: {self.away_win}\n"
        )

    def _sum(self, condition: Callable[[int, int], bool]) -> float:
        rows, cols = self.grid.shape
        return sum(
            self.grid[i, j] for i in range(rows) for j in range(cols) if condition(i, j)
        )

    @property
    def home_win(self) -> float:
        """Probability of home win"""
        return self._sum(lambda a, b: a > b)

    @property
    def draw(self) -> float:
        """Probability of draw"""
        return self._sum(lambda a, b: a == b)

    @property
    def away_win(self) -> float:
        """Probability of away win"""
        return self._sum(lambda a, b: a < b)

    @property
    def both_teams_to_score(self) -> float:
        """Probability of both teams scoring"""
        return self._sum(lambda a, b: a > 0 and b > 0)

    @property
    def home_draw_away(self) -> List[float]:
        """1x2 Probabilities"""
        return [self.home_win, self.draw, self.away_win]

    def total_goals(self, over_under: str, strike: float) -> float:
        """
        Calculate probabilities for total goals market

        Parameters
        ----------
        over_under : str
            'over' or 'under'
        strike : float
            Total goals value
        """
        conditions = {
            "over": lambda a, b: a + b > strike,
            "under": lambda a, b: a + b < strike,
        }
        if over_under not in conditions:
            raise ValueError("over_under must be 'over' or 'under'")
        return self._sum(conditions[over_under])

    def asian_handicap(self, home_away: str, strike: float) -> float:
        """
        Calculate probabilities for asian handicap market

        Parameters
        ----------
        home_away : str
            'home' or 'away'
        strike : float
            Handicap value
        """
        conditions = {
            "home": lambda a, b: a - b > strike,
            "away": lambda a, b: b - a > strike,
        }
        if home_away not in conditions:
            raise ValueError("home_away must be 'home' or 'away'")
        return self._sum(conditions[home_away])
