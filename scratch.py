import pandas as pd

import penaltyblog as pb

if __name__ == "__main__":

    df = pd.read_csv("/Users/martin/Downloads/E0.csv")

    print(df.head())

    clf = pb.models.BayesianSkellamGoalModel(
        df["FTHG"], df["FTAG"], df["HomeTeam"], df["AwayTeam"]
    )
    clf.fit(draws=5000)
    grid = clf.predict("Liverpool", "Wolves")
    print(grid)
    print(clf)
    print(clf.get_params())
