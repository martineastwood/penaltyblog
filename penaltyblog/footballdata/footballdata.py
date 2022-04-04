import pandas as pd

COUNTRIES = {
    "belgium": "B",
    "england": "E",
    "france": "F",
    "germany": "D",
    "greece": "G",
    "italy": "I",
    "portugal": "P",
    "scotland": "SC",
    "spain": "SP",
    "turkey": "T",
}


def list_countries() -> list:
    """
    Lists all the countries currently available
    """
    countries = list(COUNTRIES.keys())
    return countries


def _season_code(season_start_year):
    season_str = str(season_start_year)[-2:] + str(season_start_year + 1)[-2:]
    return season_str


def fetch_data(country, season_start_year, division) -> pd.DataFrame:
    """
    Fetches the requested data from football-data.co.uk

    Parameters
    ----------
    country : string
        The name of the country of interest
    season_start_year : int
        The year the season started, e.g. `2018` for the 2018/2019 season
    division : int
        The division's level, where `0` is the top tier, `1` is the second tier etc

    Examples
    --------
    >>> import penaltyblog as pb
    >>> pb.footballdata.fetch("England", 2018, 0)

    Returns
    -------
    Returns a Pandas dataframe containing the requested data
    """
    country_code = COUNTRIES.get(country.lower())
    if country_code is None:
        raise ValueError("Country not recognised")

    season_str = _season_code(season_start_year)

    base_url = "https://www.football-data.co.uk/mmz4281/{season}/{country}{division}.csv"

    url = base_url.format(
        season=season_str, country=country_code, division=str(division)
    )

    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    return df