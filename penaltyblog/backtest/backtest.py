import numpy as np
import pandas as pd

from .account import Account
from .context import Context


class Backtest:
    def __init__(self, data, start_date, end_date, stop_at_negative=False):

        self.stop_at_negative = stop_at_negative
        self.start_date = pd.to_datetime(start_date).to_pydatetime().date()
        self.end_date = pd.to_datetime(end_date).to_pydatetime().date()

        # validate the dataframe we are passed
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas dataframe")

        if "date" not in data.columns:
            raise ValueError("Data must contain a column called `date`")

        self.df = data
        try:
            self.df["date"] = pd.to_datetime(self.df.date).dt.date
        except ValueError:
            pass

        # get the unique dates during the test window
        self.window = (
            self.df[
                (self.df["date"] >= self.start_date)
                & (self.df["date"] <= self.end_date)
            ]["date"]
            .sort_values()
            .unique()
        )

    def start(self, bankroll, logic, trainer=None):
        self.account = Account(bankroll, self.stop_at_negative)

        for date in self.window:
            self.account.current_date = date
            lookback = self.df[self.df["date"] < date]
            test = self.df[self.df["date"] == date]

            for _, row in test.iterrows():
                ctx = Context(self.account, lookback, row)
                try:
                    logic(ctx)
                except ValueError:
                    return None

    def results(self):
        total_bets = len(self.account.history)
        total_profit = self.account.current_bankroll - self.account.bankroll
        successful_bets = sum([x["outcome"] for x in self.account.history])

        output = {
            "Total Bets": total_bets,
            "Successful Bets": successful_bets,
            "Successful Bet %": successful_bets / total_bets * 100,
            "Max Bankroll": np.max(self.account.tracker),
            "Profit": total_profit,
            "ROI": total_profit / self.account.bankroll * 100,
        }

        return output
