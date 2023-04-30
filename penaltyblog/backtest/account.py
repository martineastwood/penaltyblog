class Account:
    """Used to make and track bets made during the backtest

    Methods
    -------
    place_bet(odds, stake, outcome)
        Places a virtual bet
    """

    def __init__(self, bankroll: float):
        """
        Parameters
        ----------
        bankroll : float
            The initial starting bankroll
        """

        self.bankroll = self.current_bankroll = bankroll
        self.current_date = None

        self.history = list()
        self.tracker = list()

    def place_bet(self, odds: float, stake: float, outcome: int):
        """
        Parameters
        ----------
        odds : float
            The odds for the bet in European decimal format

        stake : float
            The number of units to be staked on the bet

        outcome : int
            The outcome of the bet, 1 for successful and 0 for failed
        """

        bet = {
            "odds": odds,
            "stake": stake,
            "outcome": outcome,
            "date": self.current_date,
            "profit": (stake * odds * outcome) - stake,
        }
        self.history.append(bet)

        if self.tracker:
            self.current_bankroll = self.tracker[-1] + bet["profit"]
        else:
            self.current_bankroll = self.bankroll + bet["profit"]

        self.tracker.append(self.current_bankroll)
