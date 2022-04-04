import numpy as np
import pandas as pd


def _build_m(fixtures, teams):
    n_teams = len(teams)
    M = np.zeros([n_teams, n_teams])
    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        M[h, a] = M[h, a] - 1
        M[a, h] = M[a, h] - 1

    for i in range(len(M)):
        M[i, i] = np.abs(
            np.sum(
                M[
                    i,
                ]
            )
        )

    M = np.vstack((M, [1 for x in range(n_teams)]))

    return M


def _build_p(fixtures, teams):
    p = list()
    for team in teams:
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")

        goals_for = home["goals_home"].sum() + away["goals_away"].sum()
        goals_away = home["goals_away"].sum() + away["goals_home"].sum()

        p.append(goals_for - goals_away)
    p.append(0)

    return p


def _solve_ratings(M, p):
    ratings = np.linalg.lstsq(M, p, rcond=None)[0]
    return ratings


def _solve_d(t, Tr_f):
    ratings = np.linalg.lstsq(t, Tr_f, rcond=None)[0]
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
        t[i, i] = np.sum(
            t[
                i,
            ]
        )

    return t


def _build_f(fixtures, teams):
    f = list()
    for team in teams:
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")
        goals_for = home["goals_home"].sum() + away["goals_away"].sum()
        f.append(goals_for)
    return f


def ratings(goals_home, goals_away, teams_home, teams_away) -> pd.DataFrame:
    """
    Calculates each team's Massey ratings, and splits the Massey rating into defence and offence ratings

    Parameters
    ----------
    goals_home : list
        List of goals scored by the home teams

    goals_away : list
        List of goals scored by the away teams

    teams_home : list
        List of names of the home teams

    teams_away : list
        List of names of the away teams

    Returns
    -------
        Returns a dataframe containing overall ratings, offence ratings and defence ratings per team

    Examples
    --------
    >>> import penaltyblog as pb
    >>> df = pb.footballdata.fetch_data("england", 2020, 0)
    >>> pb.massey.ratings(df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"])
    """
    teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))

    fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
    fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
    fixtures["goals_home"] = fixtures["goals_home"].astype(int)
    fixtures["goals_away"] = fixtures["goals_away"].astype(int)

    M = _build_m(fixtures, teams)
    p = _build_p(fixtures, teams)
    r = _solve_ratings(M, p)

    t = _build_t(fixtures, teams)
    f = _build_f(fixtures, teams)
    Tr_f = (np.diag(t) * r) - f
    d = _solve_d(t, Tr_f)
    o = r - d

    res = pd.DataFrame([teams, r, o, d]).T
    res.columns = ["team", "rating", "offence", "defence"]
    res = res.sort_values("rating", ascending=False)
    res = res.reset_index(drop=True)
    return res
