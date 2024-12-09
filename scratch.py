import pandas as pd

import penaltyblog as pb

if __name__ == "__main__":

    df = pd.read_csv("/Users/martin/Downloads/E0.csv")

    print(df.head())

    clf = pb.models.BayesianBivariateGoalModel(
        df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"]
    )
    clf.fit(draws=2500)
    grid = clf.predict("Liverpool", "Wolves")
    print(grid)
    print(clf)
