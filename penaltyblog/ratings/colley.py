"""
Colley Rating System

Calculates the Colley ratings for a group of teams.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Colley:  # pylint: disable=too-few-public-methods
    """
    Calculates each team's Colley ratings

    Parameters
    ----------
    goals_home : array-like
        List of goals scored by the home teams

    goals_away : array-like
        List of goals scored by the away teams

    teams_home : array-like
        List of names of the home teams

    teams_away : array-like
        List of names of the away teams

    include_draws : bool
        Should tied results be included in the ratings?

    draw_weight : float
        if include_draws is `True` then this sets the weighting applied to tied scores.
        For example `0.5` means a draw is worth half a win, `0.333` means a draw
        is a third of a win etc
    """

    def __init__(
        self,
        goals_home: Union[Sequence[int], NDArray],
        goals_away: Union[Sequence[int], NDArray],
        teams_home: Sequence[str],
        teams_away: Sequence[str],
        include_draws: bool = True,
        draw_weight: float = 0.5,
    ):
        self.goals_home = goals_home
        self.goals_away = goals_away
        self.teams_home = teams_home
        self.teams_away = teams_away
        self.include_draws = include_draws
        self.draw_weight = draw_weight

    def get_ratings(self) -> pd.DataFrame:
        """
        Gets the Colley ratings

        Returns
        -------
            Returns a dataframe containing colley ratings per team
        """
        teams = np.sort(np.unique(np.concatenate([self.teams_home, self.teams_away])))

        fixtures = _build_fixtures(
            self.goals_home, self.goals_away, self.teams_home, self.teams_away
        )

        c, b = _build_c_b(fixtures, teams, self.include_draws, self.draw_weight)

        r = _solve_r(c, b)
        r = pd.DataFrame([teams, r]).T
        r.columns = ["team", "rating"]
        r = r.sort_values("rating", ascending=False)
        r = r.reset_index(drop=True)

        return r


def _build_fixtures(goals_home, goals_away, teams_home, teams_away):
    fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
    fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
    fixtures["goals_home"] = fixtures["goals_home"].astype(int)
    fixtures["goals_away"] = fixtures["goals_away"].astype(int)
    return fixtures


def _solve_r(c, b):
    r = np.linalg.solve(c, b)
    return r


def _build_c_b(fixtures, teams, include_draws, draw_weight):
    n_teams = len(teams)
    c = np.zeros([n_teams, n_teams])
    b = np.zeros([n_teams])

    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        c[h, a] = c[h, a] - 1
        c[a, h] = c[a, h] - 1

        if row["goals_home"] > row["goals_away"]:
            b[h] += 1
            b[a] -= 1

        elif row["goals_home"] < row["goals_away"]:
            b[h] -= 1
            b[a] += 1

        else:
            if include_draws:
                b[h] += draw_weight
                b[a] += draw_weight

    np.fill_diagonal(c, np.abs(c.sum(axis=1)) + 2)
    b = 1 + b * 0.5

    return c, b
