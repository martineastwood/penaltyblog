from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from penaltyblog.bayes.diagnostics import compute_diagnostics
from penaltyblog.bayes.likelihood import (
    bayesian_predict_c,
    football_log_prob_wrapper,
)
from penaltyblog.bayes.sampler_api import EnsembleSampler
from penaltyblog.models.base_bayesian_model import BaseBayesianModel
from penaltyblog.models.custom_types import GoalInput, TeamInput, WeightInput
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)


class BayesianGoalModel(BaseBayesianModel):
    """
    Bayesian Football Model using Dixon-Coles methodology.

    This class extends BaseBayesianModel to provide Bayesian inference for
    football match predictions using MCMC sampling. Instead of point
    estimates from MLE, this model produces full posterior distributions
    for all parameters.

    Parameters
    ----------
    goals_home : GoalInput
        Goals scored by the home team in each match.
    goals_away : GoalInput
        Goals scored by the away team in each match.
    teams_home : TeamInput
        Names of home teams for each match.
    teams_away : TeamInput
        Names of away teams for each match.
    weights : WeightInput, optional
        Match weights (e.g., from time decay). If None, all matches are weighted equally.
    """

    def __init__(
        self,
        goals_home: GoalInput,
        goals_away: GoalInput,
        teams_home: TeamInput,
        teams_away: TeamInput,
        weights: WeightInput = None,
    ):
        """
        Initialize a Bayesian goal model.

        Parameters
        ----------
        goals_home : GoalInput
            Goals scored by the home team in each match.
        goals_away : GoalInput
            Goals scored by the away team in each match.
        teams_home : TeamInput
            Names of home teams for each match.
        teams_away : TeamInput
            Names of away teams for each match.
        weights : WeightInput, optional
            Match weights (e.g., from time decay). If None, all matches are weighted equally.
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

    def _generate_start_positions(
        self, n_walkers: int, mle_params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate starting positions for MCMC walkers.

        If mle_params is provided, cluster walkers around that solution
        AFTER recentering to match Bayesian constraints.

        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers.
        mle_params : np.ndarray, optional
            MLE parameter estimates to use as starting point. If None,
            random starting positions are generated.

        Returns
        -------
        np.ndarray
            Starting positions for all walkers.
        """
        ndim = (self.n_teams * 2) + 2
        start_pos = np.zeros((n_walkers, ndim))

        if mle_params is not None:
            if len(mle_params) != ndim:
                raise ValueError(
                    f"MLE params shape {len(mle_params)} != Model ndim {ndim}"
                )

            att_mle = mle_params[: self.n_teams]
            def_mle = mle_params[self.n_teams : 2 * self.n_teams]
            hfa_rho = mle_params[2 * self.n_teams :]

            att_centered = att_mle - np.mean(att_mle)
            def_centered = def_mle - np.mean(def_mle)

            centered_params = np.concatenate([att_centered, def_centered, hfa_rho])

            for w in range(n_walkers):
                start_pos[w, :] = centered_params + np.random.normal(0, 0.05, ndim)

        else:
            start_pos = np.random.normal(0, 0.1, size=(n_walkers, ndim))
            start_pos[:, -2] = np.random.normal(0.25, 0.1, size=n_walkers)
            start_pos[:, -1] = np.random.normal(0.0, 0.1, size=n_walkers)

        return start_pos

    def fit(
        self,
        minimizer_options: Optional[dict] = None,
        n_samples: int = 2000,
        burn: int = 1000,
        n_chains: int = 4,
        thin: int = 1,
    ) -> None:
        """
        Fit the model using parallel MCMC chains.

        Parameters
        ----------
        minimizer_options : dict, optional
            Options to pass to the minimizer. Ignored for Bayesian models.
        n_samples : int, default=2000
            Number of MCMC samples per chain.
        burn : int, default=1000
            Number of burn-in samples to discard.
        n_chains : int, default=4
            Number of parallel MCMC chains to run.
        thin : int, default=1
            Thinning interval to reduce autocorrelation.
        """
        data_dict = {
            "home_idx": self.home_idx,
            "away_idx": self.away_idx,
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "weights": self.weights,
            "n_teams": self.n_teams,
        }

        mle_params = self._get_initial_params()
        mle_params = np.concatenate([mle_params, [-0.1]])

        self.sampler = EnsembleSampler(
            n_chains=n_chains,
            n_cores=n_chains,
            log_prob_wrapper_func=football_log_prob_wrapper,
            data_dict=data_dict,
        )

        ndim = (self.n_teams * 2) + 2
        n_walkers = max(50, 2 * ndim + 10)

        start_positions = [
            self._generate_start_positions(n_walkers, mle_params=mle_params)
            for _ in range(n_chains)
        ]

        self.sampler.run_mcmc(start_positions, n_samples, burn)

        self.trace = self.sampler.trim_samples(burn=burn, thin=thin)
        self._map_trace_to_dict()

        if self.trace_dict is not None and self.trace is not None:
            self.trace_dict["home_advantage"] = self.trace[:, -2]
            self.trace_dict["rho"] = self.trace[:, -1]

        self._params = np.mean(self.trace, axis=0)
        self.fitted = True

    def get_diagnostics(self, burn: int = 0, thin: int = 1) -> pd.DataFrame:
        """
        Returns a DataFrame of R-hat and ESS for all parameters,
        including home_advantage and rho.

        Args:
            burn (int, optional): Additional burn-in to discard. Defaults to 0 since
                primary burn-in is handled during .fit().
            thin (int, optional): Additional thinning factor. Defaults to 1.
        """
        if not hasattr(self, "sampler") or self.sampler is None:
            raise ValueError("Model has not been fitted.")

        df = compute_diagnostics(self.sampler, burn=burn, thin=thin)

        labels = []
        for team in self.teams:
            labels.append(f"Attack_{team}")
        for team in self.teams:
            labels.append(f"Defense_{team}")

        labels.append("Home_Advantage")
        labels.append("Rho")

        if len(df) != len(labels):
            print(
                f"Warning: Label mismatch ({len(df)} params vs {len(labels)} labels). Returning raw indices."
            )
            return df

        df.index = labels
        return df

    def _get_initial_params(self) -> np.ndarray:
        """
        Get initial parameters for MCMC sampling.

        Returns:
        --------
        array-like
            Initial parameter values
        """
        from penaltyblog.models import PoissonGoalsModel

        simple_model = PoissonGoalsModel(
            self.goals_home,
            self.goals_away,
            self.teams_home,
            self.teams_away,
            self.weights,
        )
        simple_model.fit()

        return simple_model._params

    def _compute_probabilities(
        self, home_idx: int, away_idx: int, max_goals: int, normalize: bool = True
    ) -> FootballProbabilityGrid:
        """
        Compute posterior predictive probabilities for a fixture.

        Uses the full posterior distribution to average over parameter uncertainty.
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted. Call .fit() first.")

        matrix, avg_lam_h, avg_lam_a = bayesian_predict_c(
            self.trace, home_idx, away_idx, self.n_teams, max_goals
        )
        return FootballProbabilityGrid(
            matrix, avg_lam_h, avg_lam_a, normalize=normalize
        )

    def _get_param_names(self) -> List[str]:
        """
        Return the parameter names for this model.

        Returns
        -------
        list[str]
            Parameter names: attack_*, defense_*, home_advantage, rho
        """
        names = []
        for team in self.teams:
            names.append(f"attack_{team}")
        for team in self.teams:
            names.append(f"defense_{team}")
        names.extend(["home_advantage", "rho"])
        return names

    def _get_tail_param_indices(self) -> Dict[str, int]:
        """
        Return indices for model-specific trailing parameters.

        Returns
        -------
        dict
            Parameter names mapped to their index in the params array.
        """
        return {"home_advantage": -2, "rho": -1}
