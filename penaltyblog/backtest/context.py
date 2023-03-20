class Context:
    def __init__(self, account, lookback, fixture, model=None):
        self.account = account
        self.lookback = lookback
        self.fixture = fixture
        self.model = model
