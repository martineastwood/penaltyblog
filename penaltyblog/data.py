import pandas as pd


def get_example_data() -> pd.DataFrame:
    """
    Downloads a pandas dataframe containing data from http://www.football-data.co.uk

    return:
        pandas dataframe
    """
    url = "http://www.football-data.co.uk/mmz4281/1819/E0.csv"
    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    return df