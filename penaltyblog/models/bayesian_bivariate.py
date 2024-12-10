import tempfile

import cmdstanpy
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid


class BayesianBivariateGoalModel:
    """Bayesian Bivariate Poisson Model for Predicting Soccer Matches"""

    STAN_MODEL = """
    data {
        int<lower=0> N;                 // Number of matches
        int<lower=1> n_teams;           // Number of teams
        array[N] int goals_home;         // home goals scored
        array[N] int goals_away;         // away goals scored
        array[N] int<lower=1,upper=n_teams> home_team;  // home team indices
        array[N] int<lower=1,upper=n_teams> away_team;  // away team indices
        vector[N] weights;               // match weights
    }

    parameters {
        real home;
        vector[n_teams] attack;
        vector[n_teams] defence;
        real<lower=0,upper=1> rho;
    }

    transformed parameters {
        vector[N] lambda_home;
        vector[N] lambda_away;

        for (i in 1:N) {
            lambda_home[i] = exp(home + attack[home_team[i]] - defence[away_team[i]]);
            lambda_away[i] = exp(attack[away_team[i]] - defence[home_team[i]]);
        }
    }

    model {
        // Priors
        home ~ normal(0, 1);
        attack ~ normal(0, 1);
        defence ~ normal(0, 1);
        rho ~ beta(2, 2);

        // Likelihood
        for (i in 1:N) {
            target += weights[i] * (poisson_log_lpmf(goals_home[i] | log(lambda_home[i])) +
                      poisson_log_lpmf(goals_away[i] | log(lambda_away[i])));
        }
    }
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
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

        self.n_teams = len(self.fixtures["team_home"].unique())

        self.teams = unique_teams
        self.fixtures = (
            self.fixtures.merge(unique_teams, left_on="team_home", right_on="team")
            .rename(columns={"team_index": "home_index"})
            .drop("team", axis=1)
            .merge(unique_teams, left_on="team_away", right_on="team")
            .rename(columns={"team_index": "away_index"})
            .drop("team", axis=1)
        )

    def _get_model_parameters(self):
        draws = self.fit_result.draws_pd()
        att_params = [x for x in draws.columns if "attack" in x]
        defs_params = [x for x in draws.columns if "defence" in x]
        return draws, att_params, defs_params

    def _format_team_parameters(self, draws, att_params, defs_params):
        attack = [None] * self.n_teams
        defence = [None] * self.n_teams
        team = self.teams["team"].tolist()

        atts = draws[att_params].mean()
        defs = draws[defs_params].mean()

        for idx, _ in enumerate(team):
            attack[idx] = round(atts.iloc[idx], 3)
            defence[idx] = round(defs.iloc[idx], 3)

        return team, attack, defence

    def __repr__(self):
        repr_str = "Module: Penaltyblog\n\nModel: Bayesian Bivariate (Stan)\n\n"

        if not self.fitted:
            return repr_str + "Status: Model not fitted"

        draws, att_params, defs_params = self._get_model_parameters()
        team, attack, defence = self._format_team_parameters(
            draws, att_params, defs_params
        )

        repr_str += f"Number of parameters: {len(att_params) + len(defs_params) + 2}\n"
        repr_str += "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defence")
        repr_str += "\n" + "-" * 60 + "\n"

        for t, a, d in zip(team, attack, defence):
            repr_str += "{0: <20} {1:<20} {2:<20}\n".format(t, a, d)

        repr_str += "-" * 60 + "\n"
        repr_str += f"Home Advantage: {round(draws['home'].mean(), 3)}\n"
        # repr_str += f"Intercept: {round(draws['intercept'].mean(), 3)}\n"

        return repr_str

    def fit(self, draws=5000, warmup=2000):
        data = {
            "N": len(self.fixtures),
            "n_teams": len(self.teams),
            "goals_home": self.fixtures["goals_home"].values,
            "goals_away": self.fixtures["goals_away"].values,
            "home_team": self.fixtures["home_index"].values,
            "away_team": self.fixtures["away_index"].values,
            "weights": self.fixtures["weights"].values,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".stan") as tmp:
            tmp.write(self.STAN_MODEL)
            tmp.flush()
            self.model = cmdstanpy.CmdStanModel(stan_file=tmp.name)
            self.fit_result = self.model.sample(
                data=data, iter_sampling=draws, iter_warmup=warmup
            )

        self.fitted = True
        return self

    def get_params(self):
        if not self.fitted:
            raise ValueError("Model must be fit before getting parameters")

        draws = self.fit_result.draws_pd()
        team_names = self.teams["team"].tolist()
        attack = draws.filter(like="attack").mean().values
        defence = draws.filter(like="defence").mean().values

        params = {
            "teams": team_names,
            "attack": dict(zip(team_names, np.round(attack, 3))),
            "defence": dict(zip(team_names, np.round(defence, 3))),
            "home_advantage": round(draws["home"].mean(), 3),
            "rho": round(draws["rho"].mean(), 3),
        }
        return params

    def predict(self, home_team, away_team, max_goals=15, n_samples=1000):
        if not self.fitted:
            raise ValueError("Model must be fit before making predictions")

        draws = self.fit_result.draws_pd()
        home_idx = self._get_team_index(home_team)
        away_idx = self._get_team_index(away_team)
        samples = draws.sample(n=n_samples, replace=True)

        lambda_home = np.exp(
            samples["home"]
            + samples[f"attack[{home_idx}]"]
            - samples[f"defence[{away_idx}]"]
        )

        lambda_away = np.exp(
            samples[f"attack[{away_idx}]"] - samples[f"defence[{home_idx}]"]
        )

        home_probs = poisson.pmf(np.arange(max_goals + 1)[:, None], lambda_home.values)
        away_probs = poisson.pmf(
            np.arange(max_goals + 1)[None, :], lambda_away.values[:, None]
        )
        score_probs = np.tensordot(home_probs, away_probs, axes=(1, 0)) / n_samples

        home_expectancy = np.sum(score_probs.sum(axis=1) * np.arange(max_goals + 1))
        away_expectancy = np.sum(score_probs.sum(axis=0) * np.arange(max_goals + 1))

        return FootballProbabilityGrid(score_probs, home_expectancy, away_expectancy)

    def _get_team_index(self, team_name):
        idx = self.teams.loc[self.teams["team"] == team_name, "team_index"]
        if idx.empty:
            raise ValueError(f"Team {team_name} not found.")
        return idx.iloc[0] - 1
