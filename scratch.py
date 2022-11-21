import penaltyblog as pb

fd = pb.scrapers.FootballData("ENG Premier League", "2020-2021")
df = fd.get_fixtures()
model = pb.models.BayesianRandomInterceptGoalModel(
    df["goals_home"], df["goals_away"], df["team_home"], df["team_away"]
)

model.fit()

print(model.predict("Liverpool", "Wolves"))

print(model)
