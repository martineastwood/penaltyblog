import tempfile

import cmdstanpy
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .football_probability_grid import FootballProbabilityGrid


class StanHierarchicalGoalModel:
    """Bayesian Hierarchical model for predicting outcomes of football
    (soccer) matches. Based on the paper by Baio and Blangiardo from
    https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf

    Methods
    -------
    fit()
        fits a Bayesian Hierarchical model to the data to calculate the team strengths.
        Must be called before the model can be used to predict game outcomes

    predict(home_team, away_team, max_goals=15)
        predict the outcome of a football (soccer) game between the home_team and
        away_team
    """

    STAN_MODEL = """
    data {
        int N;                           // number of matches
        int n_teams;                     // number of teams
        array[N] int goals_home;         // home goals scored
        array[N] int goals_away;         // away goals scored
        array[N] int<lower=1,upper=n_teams> home_team;  // home team indices
        array[N] int<lower=1,upper=n_teams> away_team;  // away team indices
        vector[N] weights;               // match weights
    }

    parameters {
        real home;                    // home advantage
        real intercept;               // global intercept
        real<lower=0.001, upper=100> tau_att;  // Add upper bound
        real<lower=0.001, upper=100> tau_def;  // Add upper bound
        vector[n_teams] atts_star;   // raw attack parameters
        vector[n_teams] def_star;    // raw defense parameters
    }

    transformed parameters {
        vector[n_teams] atts;        // centered attack parameters
        vector[n_teams] defs;        // centered defense parameters
        vector[N] home_theta;        // home scoring rates
        vector[N] away_theta;        // away scoring rates

        // Center the attack and defense parameters
        atts = atts_star - mean(atts_star);
        defs = def_star - mean(def_star);

        // Calculate scoring rates
        for (i in 1:N) {
            home_theta[i] = exp(intercept + home + atts[home_team[i]] + defs[away_team[i]]);
            away_theta[i] = exp(intercept + atts[away_team[i]] + defs[home_team[i]]);
        }
    }

    model {
        // Priors
        tau_att ~ gamma(0.1, 0.1);
        tau_def ~ gamma(0.1, 0.1);

        // Team effects
        atts_star ~ normal(0, inv_sqrt(tau_att));
        def_star ~ normal(0, inv_sqrt(tau_def));

        // Likelihood
        for (i in 1:N) {
            target += weights[i] * poisson_lpmf(goals_home[i] | home_theta[i]);
            target += weights[i] * poisson_lpmf(goals_away[i] | away_theta[i]);
        }
    }

    generated quantities {
        vector[N] log_lik;

        for (i in 1:N) {
            log_lik[i] = poisson_lpmf(goals_home[i] | home_theta[i]) +
                        poisson_lpmf(goals_away[i] | away_theta[i]);
        }
    }
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=1):
        self._initialize_fixtures(
            goals_home, goals_away, teams_home, teams_away, weights
        )
        self._initialize_teams()
        self.model = None
        self.fit_result = None
        self.fitted = False

    def _initialize_fixtures(
        self, goals_home, goals_away, teams_home, teams_away, weights
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

    def _initialize_teams(self):
        self.teams = (
            pd.DataFrame({"team": self.fixtures["team_home"].unique()})
            .sort_values("team")
            .reset_index(drop=True)
            .assign(team_index=lambda x: np.arange(len(x)) + 1)
        )

        self.n_teams = len(self.fixtures["team_home"].unique())

        self.fixtures = (
            self.fixtures.merge(self.teams, left_on="team_home", right_on="team")
            .rename(columns={"team_index": "home_index"})
            .drop("team", axis=1)
            .merge(self.teams, left_on="team_away", right_on="team")
            .rename(columns={"team_index": "away_index"})
            .drop("team", axis=1)
        )

    def _get_model_parameters(self):
        draws = self.fit_result.draws_pd()
        att_params = [x for x in draws.columns if "atts_star" in x]
        defs_params = [x for x in draws.columns if "def_star" in x]
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

    def get_params(self):
        if not self.fitted:
            raise ValueError("Model must be fit before getting parameters")

        draws, att_params, defs_params = self._get_model_parameters()
        team, attack, defence = self._format_team_parameters(
            draws, att_params, defs_params
        )

        params = {
            "teams": team,
            "attack": dict(zip(team, attack)),
            "defence": dict(zip(team, defence)),
            "home_advantage": round(draws["home"].mean(), 3),
            "intercept": round(draws["intercept"].mean(), 3),
        }

        return params

    def __repr__(self):
        repr_str = "Module: Penaltyblog\n\nModel: Bayesian Hierarchical (Stan)\n\n"

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
        repr_str += f"Intercept: {round(draws['intercept'].mean(), 3)}\n"

        return repr_str

    def fit(self, draws=5000):
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
                data=data, iter_sampling=draws, iter_warmup=2000
            )

        self.fitted = True
        return self

    def _get_team_index(self, team_name):
        return self.teams.loc[self.teams["team"] == team_name, "team_index"].iloc[0] - 1

    def _calculate_theta(self, draws, home_team, away_team):
        home = draws["home"].mean()
        intercept = draws["intercept"].mean()

        home_idx = self._get_team_index(home_team)
        away_idx = self._get_team_index(away_team)

        home_atts = draws[f"atts[{home_idx}]"].mean()
        away_atts = draws[f"atts[{away_idx}]"].mean()
        home_defs = draws[f"defs[{home_idx}]"].mean()
        away_defs = draws[f"defs[{away_idx}]"].mean()

        home_theta = np.exp(intercept + home + home_atts + away_defs)
        away_theta = np.exp(intercept + away_atts + home_defs)

        return home_theta, away_theta

    def predict(self, home_team, away_team, max_goals=15):
        if not self.fit_result:
            raise ValueError("Model must be fit before making predictions")

        draws = self.fit_result.draws_pd()
        home_theta, away_theta = self._calculate_theta(draws, home_team, away_team)

        home_goals_vector = poisson.pmf(np.arange(max_goals), home_theta)
        away_goals_vector = poisson.pmf(np.arange(max_goals), away_theta)

        m = np.outer(home_goals_vector, away_goals_vector)

        return FootballProbabilityGrid(m, home_theta, away_theta)
