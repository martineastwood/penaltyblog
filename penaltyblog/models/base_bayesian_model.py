import os
import tempfile
from typing import Dict, Sequence, Union

import cmdstanpy
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class BaseBayesianGoalModel:
    """Base class for Bayesian Goal Models with shared functionality."""

    def __init__(
        self,
        goals_home: Union[Sequence[int], NDArray],
        goals_away: Union[Sequence[int], NDArray],
        teams_home: Union[Sequence[int], NDArray],
        teams_away: Union[Sequence[int], NDArray],
        weights: Union[float, Sequence[float], NDArray] = 1.0,
    ):
        self.fixtures = pd.DataFrame(
            {
                "goals_home": goals_home,
                "goals_away": goals_away,
                "team_home": teams_home,
                "team_away": teams_away,
                "weights": weights,
            }
        )
        self._setup_teams()
        self.model = None
        self.fit_result = None
        self.fitted = False

    def _setup_teams(self):
        unique_teams = pd.DataFrame(
            {
                "team": pd.concat(
                    [self.fixtures["team_home"], self.fixtures["team_away"]]
                ).unique()
            }
        )
        unique_teams = (
            unique_teams.sort_values("team")
            .reset_index(drop=True)
            .assign(team_index=lambda x: np.arange(len(x)) + 1)
        )

        self.n_teams = len(unique_teams)
        self.teams = unique_teams
        self.fixtures = (
            self.fixtures.merge(unique_teams, left_on="team_home", right_on="team")
            .rename(columns={"team_index": "home_index"})
            .drop("team", axis=1)
            .merge(unique_teams, left_on="team_away", right_on="team")
            .rename(columns={"team_index": "away_index"})
            .drop("team", axis=1)
        )

    def _compile_and_fit_stan_model(
        self, stan_file: str, data: Dict, draws: int, warmup: int
    ) -> cmdstanpy.CmdStanMCMC:
        """
        Compiles and fits the Stan model.

        Args:
            stan_model (str): The Stan model code as a string.
            data (dict): The data dictionary for the model.
            draws (int): Number of posterior draws.
            warmup (int): Number of warmup draws.

        Returns:
            cmdstanpy.CmdStanMCMC: The fit result object.
        """
        self.model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        self.fit_result = self.model.sample(
            data=data, iter_sampling=draws, iter_warmup=warmup
        )
        self.fitted = True
        return self.fit_result

    def fit(self, draws: int, warmup: int):
        raise NotImplementedError("The 'fit' method must be implemented in subclasses.")

    def predict(self, home_team: str, away_team: str, max_goals: int, n_samples: int):
        raise NotImplementedError(
            "The 'predict' method must be implemented in subclasses."
        )

    def __repr__(self):
        raise NotImplementedError(
            "The '__repr__' method must be implemented in subclasses."
        )

    def _get_team_index(self, team_name):
        return self.teams.loc[self.teams["team"] == team_name, "team_index"].iloc[0]
