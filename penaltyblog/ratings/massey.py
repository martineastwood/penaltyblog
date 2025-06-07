"""
Massey Rating System

Calculates the Massey ratings for a group of teams.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Massey:  # pylint: disable=too-few-public-methods
    """
    Calculates each team's Massey ratings

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
    """

    def __init__(
        self,
        goals_home: Union[Sequence[int], NDArray],
        goals_away: Union[Sequence[int], NDArray],
        teams_home: Sequence[str],
        teams_away: Sequence[str],
    ):
        """
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
        """
        self.goals_home = goals_home
        self.goals_away = goals_away
        self.teams_home = teams_home
        self.teams_away = teams_away

    def get_ratings(self) -> pd.DataFrame:
        """
        Gets the Massey ratings

        Returns
        -------
            Returns a dataframe containing colley ratings per team
        """
        teams = np.sort(np.unique(np.concatenate([self.teams_home, self.teams_away])))

        fixtures = pd.DataFrame(
            [self.goals_home, self.goals_away, self.teams_home, self.teams_away]
        ).T
        fixtures.columns = pd.Index(
            ["goals_home", "goals_away", "team_home", "team_away"]
        )
        fixtures["goals_home"] = fixtures["goals_home"].astype(int)
        fixtures["goals_away"] = fixtures["goals_away"].astype(int)

        m = _build_m(fixtures, teams)
        p = _build_p(fixtures, teams)
        r = _solve_ratings(m, p)

        t = _build_t(fixtures, teams)
        f = _build_f(fixtures, teams)
        tr_f = (np.diag(t) * r) - f
        d = _solve_d(t, tr_f)
        o = r - d

        res = pd.DataFrame([teams, r, o, d]).T
        res.columns = pd.Index(["team", "rating", "offence", "defence"])
        res = res.sort_values("rating", ascending=False)
        res = res.reset_index(drop=True)
        return res


def _build_m(fixtures, teams):
    n_teams = len(teams)
    m = np.zeros([n_teams, n_teams])
    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        m[h, a] = m[h, a] - 1
        m[a, h] = m[a, h] - 1

    for i in range(len(m)):
        m[i, i] = np.abs(np.sum(m[i,]))

    m = np.vstack((m, [1 for x in range(n_teams)]))

    return m


def _build_p(fixtures, teams):
    p = []
    for team in teams:
        _ = team  # keeps pylint happy
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")

        goals_for = home["goals_home"].sum() + away["goals_away"].sum()
        goals_away = home["goals_away"].sum() + away["goals_home"].sum()

        p.append(goals_for - goals_away)
    p.append(0)

    return p


def _solve_ratings(m, p):
    ratings = np.linalg.lstsq(m, p, rcond=None)[0]
    return ratings


def _solve_d(t, tr_f):
    ratings = np.linalg.lstsq(t, tr_f, rcond=None)[0]
    return ratings


def _build_t(fixtures, teams):
    n_teams = len(teams)
    t = np.zeros([n_teams, n_teams])

    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        t[h, a] = t[h, a] + 1
        t[a, h] = t[a, h] + 1

    for i in range(len(t)):
        t[i, i] = np.sum(t[i,])

    return t


def _build_f(fixtures, teams):
    f = []
    for team in teams:
        _ = team  # keeps pylint happy
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")
        goals_for = home["goals_home"].sum() + away["goals_away"].sum()
        f.append(goals_for)
    return f
