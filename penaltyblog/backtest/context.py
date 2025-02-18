"""
Context object passed into the `logic` and `trainer` functions.
Contains the account object, lookback data, fixture being processed
and optionally a trained model
"""

from typing import Optional

import pandas as pd

from .account import Account


class Context:
    """
    Object passed into the `logic` and `trainer` functions. Contains
    the account object, lookback data, fixture being processed and
    optionally a trained model
    """

    def __init__(
        self,
        account: Account,
        lookback: pd.DataFrame,
        fixture: Optional[pd.Series],
        model=None,
    ):
        """
        Parameters
        ----------
        account : Account
            The account object to track the virtual bets

        lookback : pd.DataFrame
            Dataframe containing all fixtures prior to the date being processed

        fixture : pd.Series
            The current fixture being processed

        model :
            Optional model that has been trained
        """
        self.account = account
        self.lookback = lookback
        self.fixture = fixture
        self.model = model
