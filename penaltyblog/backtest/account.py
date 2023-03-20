class Account:
    def __init__(self, bankroll, stop_at_negative=False):
        self.bankroll = self.current_bankroll = bankroll
        self.stop_at_negative = stop_at_negative
        self.current_date = None

        self.history = list()
        self.tracker = list()

    def place_bet(self, odds, stake, outcome):
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

        if self.stop_at_negative and self.current_bankroll < 0:
            raise ValueError("Bankroll below zero")
