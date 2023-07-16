import penaltyblog as pb


def test_backtest_simple():
    df = pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures()

    def logic(ctx):
        fixture = ctx.fixture
        account = ctx.account

        if 2.5 <= fixture["b365_a"] <= 4.0:
            account.place_bet(
                fixture["b365_a"],
                account.current_bankroll * 0.025,
                1 if ctx.fixture["ftr"] == "A" else 0,
            )

    backtest = pb.backtest.Backtest(df, "2020-01-01", "2020-05-01", True)
    backtest.start(100, logic)
    res = backtest.results()
    assert isinstance(res, dict)


def test_backtest_trainer():
    df = pb.scrapers.FootballData("ENG Premier League", "2019-2020").get_fixtures()

    def trainer(ctx):
        weights = pb.models.dixon_coles_weights(ctx.lookback["date"], 0.001)

        model = pb.models.DixonColesGoalModel(
            teams_home=ctx.lookback["team_home"],
            teams_away=ctx.lookback["team_away"],
            goals_home=ctx.lookback["goals_home"],
            goals_away=ctx.lookback["goals_away"],
            weights=weights,
        )

        model.fit()

        return model

    def logic(ctx):
        fixture = ctx.fixture
        account = ctx.account
        model = ctx.model

        pred = model.predict(fixture["team_home"], fixture["team_away"])

        kc = pb.kelly.criterion(fixture["b365_h"], pred.home_win, 0.3)
        if kc > 0:
            account.place_bet(
                fixture["b365_h"],
                account.current_bankroll * kc,
                1 if ctx.fixture["ftr"] == "H" else 0,
            )

        kc = pb.kelly.criterion(fixture["b365_a"], pred.away_win, 0.3)
        if kc > 0:
            account.place_bet(
                fixture["b365_a"],
                account.current_bankroll * kc,
                1 if ctx.fixture["ftr"] == "A" else 0,
            )

        kc = pb.kelly.criterion(fixture["b365_d"], pred.draw, 0.3)
        if kc > 0:
            account.place_bet(
                fixture["b365_d"],
                account.current_bankroll * kc,
                1 if ctx.fixture["ftr"] == "D" else 0,
            )

    backtest = pb.backtest.Backtest(df, "2020-03-01", "2020-04-10", True)
    backtest.start(100, logic, trainer)
    res = backtest.results()
    assert isinstance(res, dict)

    backtest = pb.backtest.Backtest(df, "2020-06-01", "2020-06-10", True)
    backtest.start(100, logic, trainer)
    res = backtest.results()
    assert isinstance(res, dict)
