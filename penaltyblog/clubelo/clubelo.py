import pandas as pd
from datetime import datetime


def list_all_teams():
    """
    Fetches all the available teams from clubelo.com

    Returns
    -------
    Returns list of team names

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.clubelo.list_all_teams()
    """
    base_url = "http://api.clubelo.com/{y}-{m}-{d}"

    today = datetime.now().date()

    y = str(today.year)
    m = str(today.month).zfill(2)
    d = str(today.day).zfill(2)

    url = base_url.format(y=y, m=m, d=d)

    df = pd.read_csv(url)
    teams = df["Club"].tolist()

    return teams


def fetch_rankings_by_date(year, month, day) -> pd.DataFrame:
    """
    Fetches all the club rankings from http://clubelo.com/ for a given date

    Parameters
    ----------
    year : int
        The year of interest
    month : int
        The month of interest
    day : int
        The day of interest

    Returns
    -------
    Returns a Pandas dataframe of ratings

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.clubelo.fetch_rankings_by_date(2010, 1, 1)
    """
    base_url = "http://api.clubelo.com/{y}-{m}-{d}"

    y = str(year)
    m = str(month).zfill(2)
    d = str(day).zfill(2)

    url = base_url.format(y=y, m=m, d=d)

    df = pd.read_csv(url)
    df["From"] = pd.to_datetime(df["From"])
    df["To"] = pd.to_datetime(df["To"])

    return df


def fetch_rankings_by_team(team) -> pd.DataFrame:
    """
    Fetches all the club rankings from http://clubelo.com/ for a given date

    Parameters
    ----------
    team : str
        The team of interest

    Returns
    -------
    Returns a Pandas dataframe of ratings

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.clubelo.fetch_rankings_by_team("barcelona")
    """
    base_url = "http://api.clubelo.com/{team}"

    url = base_url.format(team=team)

    df = pd.read_csv(url)
    df["From"] = pd.to_datetime(df["From"])
    df["To"] = pd.to_datetime(df["To"])

    return df
