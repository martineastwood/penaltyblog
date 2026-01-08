from typing import Dict, List

import numpy as np
import pandas as pd

from penaltyblog.bayes.diagnostics import compute_diagnostics
from penaltyblog.bayes.likelihood import hierarchical_log_prob_wrapper
from penaltyblog.bayes.sampler_api import EnsembleSampler
from penaltyblog.models.bayesian_goal_model import BayesianGoalModel
from penaltyblog.models.custom_types import GoalInput, TeamInput, WeightInput


class HierarchicalBayesianGoalModel(BayesianGoalModel):
    """
    Advanced Bayesian Model with Hierarchical Priors.

    Extends BayesianGoalModel to learn the variance of the league
    automatically by placing hierarchical priors on team strength parameters.

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

    def fit(
        self,
        minimizer_options: dict = None,
        n_samples: int = 3000,
        burn: int = 1500,
        n_chains: int = 4,
        thin: int = 1,
    ) -> None:
        """
        Fit the hierarchical model using parallel MCMC chains.

        Parameters
        ----------
        minimizer_options : dict, optional
            Options to pass to the optimiser. Ignored for Bayesian models.
        n_samples : int, default=3000
            Number of MCMC samples per chain.
        burn : int, default=1500
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
            log_prob_wrapper_func=hierarchical_log_prob_wrapper,
            data_dict=data_dict,
        )

        ndim = (self.n_teams * 2) + 4
        n_walkers = max(50, 2 * ndim + 10)

        start_positions = [
            self._generate_hierarchical_starts(n_walkers, mle_params)
            for _ in range(n_chains)
        ]
        self.sampler.run_mcmc(start_positions, n_samples, burn)

        self.trace = self.sampler.trim_samples(burn=burn, thin=thin)
        self._map_trace_to_dict()

        self._params = np.mean(self.trace, axis=0)
        self.fitted = True

    def _generate_hierarchical_starts(
        self, n_walkers: int, mle_params: np.ndarray
    ) -> np.ndarray:
        """
        Generate starting positions including the 2 extra Sigma parameters.

        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers.
        mle_params : np.ndarray
            MLE parameter estimates.

        Returns
        -------
        np.ndarray
            Starting positions for all walkers.
        """
        ndim = (self.n_teams * 2) + 4
        start_pos = np.zeros((n_walkers, ndim))

        att_mle = mle_params[: self.n_teams]
        def_mle = mle_params[self.n_teams : 2 * self.n_teams]
        hfa_rho = mle_params[2 * self.n_teams :]

        att_centered = att_mle - np.mean(att_mle)
        def_centered = def_mle - np.mean(def_mle)

        sig_att_est = np.std(att_centered)
        sig_def_est = np.std(def_centered)

        base_params = np.concatenate(
            [att_centered, def_centered, hfa_rho, [sig_att_est, sig_def_est]]
        )

        for w in range(n_walkers):
            start_pos[w, :] = base_params + np.random.normal(0, 0.05, ndim)

        return start_pos

    def _map_trace_to_dict(self) -> None:
        """
        Map trace to dictionary including the new sigma parameters.

        Extends the base class method to include sigma_attack and sigma_defense
        parameters from the hierarchical model.
        """
        super()._map_trace_to_dict()

        idx_sig_att = 2 * self.n_teams + 2
        idx_sig_def = 2 * self.n_teams + 3

        if self.trace_dict is not None and self.trace is not None:
            self.trace_dict["sigma_attack"] = self.trace[:, idx_sig_att]
            self.trace_dict["sigma_defense"] = self.trace[:, idx_sig_def]

    def get_diagnostics(self, burn: int = 0, thin: int = 1) -> pd.DataFrame:
        """
        Returns a DataFrame of R-hat and ESS for all parameters,
        including the hierarchical sigmas.

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

        labels.append("Sigma_Attack")
        labels.append("Sigma_Defense")

        if len(df) != len(labels):
            print(
                f"Warning: Label mismatch ({len(df)} params vs {len(labels)} labels). Returning raw indices."
            )
            return df

        df.index = labels
        return df

    def _get_param_names(self) -> List[str]:
        """
        Return the parameter names for this model, including hierarchical sigmas.

        Returns
        -------
        List[str]
            Parameter names: attack_*, defense_*, home_advantage, rho,
            sigma_attack, sigma_defense.
        """
        names = super()._get_param_names()
        names.extend(["sigma_attack", "sigma_defense"])
        return names

    def _get_tail_param_indices(self) -> Dict[str, int]:
        """
        Return indices for hierarchical-specific trailing parameters.

        Returns
        -------
        Dict[str, int]
            Parameter names mapped to their index in the params array.
            Includes home_advantage, rho, sigma_attack, and sigma_defense.
        """
        indices = super()._get_tail_param_indices()
        indices.update({"sigma_attack": -2, "sigma_defense": -1})

        indices["home_advantage"] = -4
        indices["rho"] = -3
        return indices
