"""
Backtest module

Used to backtest different betting strategies.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .account import Account
from .context import Context


class Backtest:
    """Used to backtest different betting strategies.

    Methods
    -------
    start(bankroll, logic, trainer)
        Runs the backtest using the logic function (and optionally
        the trainer function)

    results()
        Calculates how well the backtest has performed
    """

    def __init__(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        stop_at_negative: bool = False,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            A dataframe containing the data to run the backtest over. Must contain a column
            called `date`

        start_date : str
            A string containing the date for the start of the test window

        stop_at_negative : bool
            If True then the backtest will stop as soon as the bankroll goes below zero
        """

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

    def start(
        self, bankroll: float, logic: Callable, trainer: Optional[Callable] = None
    ):
        """
        Parameters
        ----------
        bankroll : float
            The initial starting value for the bankroll

        logic : callable
            The function to apply to each individual fixture. The function should have one
            argument called `ctx`, which contains the the information required to
            run the strategy. See the example notebooks for more examples of the
            `logic` function and how to use the `ctx` object. `ctx` will contain an instance
            of the `Account` class, which contains functions for placing virtual bets,
            `lookback` which contains all the fuxtures prior to the date of the current fixture,
            `fixture` which is the current fixture being processed, and optionally `model` if
            a trainer function is used.

        trainer : callable
            The function used to train a model, which is then added to the `ctx` object
            passed to the `logic` function. This function should have one
            argument called `ctx`, which contains the the information required to
            train the model and should return the trained model. See the example notebooks
            for more examples of the `trainer` function and how to use the `ctx` object. The
            trainer function gets called once per unique date and then is made available to
            all fixtures for that date.

        Returns
        -------
            None
        """
        self.account = Account(bankroll)

        # for date in self.window:
        for date in tqdm(self.window):
            self.account.current_date = date
            lookback = self.df[self.df["date"] < date]
            test = self.df[self.df["date"] == date]

            ctx = Context(self.account, lookback, None)

            if trainer is not None:
                ctx.model = trainer(ctx)

            for _, row in test.iterrows():
                ctx.fixture = row

                logic(ctx)

                if self.stop_at_negative and self.account.current_bankroll < 0:
                    return None
        return None

    def results(self) -> dict:
        """
        Calculates the results of the backtest and returns them as a dict

        Returns
        -------
            Dictionary containing metrics about the backtest
        """
        total_bets = len(self.account.history)
        total_profit = self.account.current_bankroll - self.account.bankroll
        successful_bets = sum([x["outcome"] for x in self.account.history])
        successful_bet_pc = 0 if total_bets == 0 else successful_bets / total_bets * 100
        max_bankroll = (
            None if len(self.account.tracker) == 0 else np.max(self.account.tracker)
        )
        min_bankroll = (
            None if len(self.account.tracker) == 0 else np.min(self.account.tracker)
        )
        roi = total_profit / self.account.bankroll * 100

        output = {
            "Total Bets": total_bets,
            "Successful Bets": successful_bets,
            "Successful Bet %": successful_bet_pc,
            "Max Bankroll": max_bankroll,
            "Min Bankroll": min_bankroll,
            "Profit": total_profit,
            "ROI": roi,
        }

        return output
