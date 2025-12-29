from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from penaltyblog.bayes.diagnostics import compute_diagnostics
from penaltyblog.bayes.likelihood import (
    bayesian_predict_c,
    football_log_prob_wrapper,
    hierarchical_log_prob_wrapper,
)
from penaltyblog.bayes.sampler_api import EnsembleSampler
from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.custom_types import GoalInput, TeamInput, WeightInput
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)


class BayesianGoalModel(BaseGoalsModel):
    """
    Bayesian Football Model using Dixon-Coles methodology.

    This class extends BaseGoalsModel to provide Bayesian inference for
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
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        # Bayesian-specific attributes
        self.trace: Optional[np.ndarray] = None
        self.trace_dict: Optional[Dict[str, np.ndarray]] = None
        self.sampler: Optional[EnsembleSampler] = None

        # team_map for backward compatibility (parent uses team_to_idx)
        self.team_map = self.team_to_idx

    def _generate_start_positions(self, n_walkers, mle_params=None):
        """
        Generate starting positions.
        If mle_params is provided, cluster walkers around that solution
        AFTER recentering to match Bayesian constraints.
        """
        ndim = (self.n_teams * 2) + 2
        start_pos = np.zeros((n_walkers, ndim))

        if mle_params is not None:
            if len(mle_params) != ndim:
                raise ValueError(
                    f"MLE params shape {len(mle_params)} != Model ndim {ndim}"
                )

            # Decompose parameters
            # Structure: [Att (n), Def (n), HFA, Rho]
            att_mle = mle_params[: self.n_teams]
            def_mle = mle_params[self.n_teams : 2 * self.n_teams]
            hfa_rho = mle_params[2 * self.n_teams :]

            # --- Recenter to Mean=0 ---
            # MLE has Mean(Att)=1. Bayesian wants Mean(Att)=0.
            # We subtract the mean from both Attack and Defense to satisfy
            # the sum-to-zero priors.
            att_centered = att_mle - np.mean(att_mle)
            def_centered = def_mle - np.mean(def_mle)

            # Reconstruct the vector
            centered_params = np.concatenate([att_centered, def_centered, hfa_rho])

            for w in range(n_walkers):
                # Add small Gaussian noise to ensure walkers are distinct
                start_pos[w, :] = centered_params + np.random.normal(0, 0.05, ndim)

        else:
            # === COLD START ===
            start_pos = np.random.normal(0, 0.1, size=(n_walkers, ndim))
            start_pos[:, -2] = np.random.normal(0.25, 0.1, size=n_walkers)
            start_pos[:, -1] = np.random.normal(0.0, 0.1, size=n_walkers)

        return start_pos

    def fit(self, n_samples=2000, burn=1000, n_chains=4, thin=5):
        """
        Fit the model using parallel MCMC chains.
        """
        # 1. Pack data for Cython (Must be pure types)
        data_dict = {
            "home_idx": self.home_idx,
            "away_idx": self.away_idx,
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "weights": self.weights,
            "n_teams": self.n_teams,
        }

        # 1. Get Initial Parameters
        mle_params = self._get_initial_params()
        mle_params = np.concatenate([mle_params, [-0.1]])

        # 2. Setup Sampler Manager
        # Using 1 core per chain is standard
        self.sampler = EnsembleSampler(
            n_chains=n_chains,
            n_cores=n_chains,
            log_prob_wrapper_func=football_log_prob_wrapper,
            data_dict=data_dict,
        )

        # 3. Generate Starting Positions
        # Rule of thumb: Walkers > 2 * ndim
        ndim = (self.n_teams * 2) + 2
        n_walkers = max(50, 2 * ndim + 10)

        start_positions = [
            self._generate_start_positions(n_walkers, mle_params=mle_params)
            for _ in range(n_chains)
        ]

        # 4. Run Execution
        self.sampler.run_mcmc(start_positions, n_samples, burn)

        # 5. Extract Results
        self.trace = self.sampler.trim_samples(burn=burn, thin=thin)
        self._map_trace_to_dict()

        # 6. Set fitted state and point estimates (posterior mean) for Base API compatibility
        self._params = np.mean(self.trace, axis=0)
        self.fitted = True

    def _map_trace_to_dict(self):
        """Helper to convert raw matrix to user-friendly dict"""
        self.trace_dict = {}

        # Map Teams
        for i, team in enumerate(self.teams):
            self.trace_dict[f"attack_{team}"] = self.trace[:, i]
            self.trace_dict[f"defence_{team}"] = self.trace[:, self.n_teams + i]

        # Map Globals
        self.trace_dict["home_advantage"] = self.trace[:, -2]
        self.trace_dict["rho"] = self.trace[:, -1]

    def get_diagnostics(self, burn: int = 0, thin: int = 1):
        """
        Returns a DataFrame of R-hat and ESS for all parameters.

        Args:
            burn (int, optional): Additional burn-in to discard. Defaults to 0 since
                primary burn-in is handled during .fit().
            thin (int, optional): Additional thinning factor. Defaults to 1.
        """
        if not hasattr(self, "sampler") or self.sampler is None:
            raise ValueError("Model has not been fitted.")

        df = compute_diagnostics(self.sampler, burn=burn, thin=thin)

        # Add nice labels
        labels = []
        # Teams
        for team in self.teams:
            labels.append(f"Attack_{team}")
        for team in self.teams:
            labels.append(f"Defense_{team}")
        # Globals
        labels.append("Home_Advantage")
        labels.append("Rho")

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

    # Diagnostic Plotting Methods
    def plot_trace(
        self,
        params=None,
        chains: bool = True,
        **kwargs,
    ):
        """
        Plot MCMC trace for convergence diagnostics.

        Visualizes the evolution of parameter values across MCMC iterations.
        Well-mixed chains should look like fuzzy caterpillars with no trends.

        Parameters
        ----------
        params : str, list of str, optional
            Parameter name(s) to plot. If None, plots key parameters.
        chains : bool, default=True
            If True, show individual chains. If False, combine all chains.        **kwargs
            Additional keyword arguments passed to plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive trace plot figure.

        Examples
        --------
        >>> model.fit(n_samples=1000, burn=500)
        >>> fig = model.plot_trace(params=['home_advantage', 'rho'])
        >>> fig.show()
        """
        from penaltyblog.viz.diagnostics import plot_trace

        return plot_trace(
            self,
            params=params,
            chains=chains,
            **kwargs,
        )

    def plot_autocorr(
        self,
        params=None,
        max_lag: int = 50,
        **kwargs,
    ):
        """
        Plot autocorrelation function for MCMC parameters.

        Helps identify the thinning interval needed for independent samples.
        Autocorrelation should decay to near zero within a reasonable lag.

        Parameters
        ----------
        params : str, list of str, optional
            Parameter name(s) to plot. If None, plots key parameters.
        max_lag : int, default=50
            Maximum lag to compute autocorrelation for.        **kwargs
            Additional keyword arguments passed to plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive autocorrelation plot.

        Examples
        --------
        >>> fig = model.plot_autocorr(params=['home_advantage'], max_lag=100)
        >>> fig.show()
        """
        from penaltyblog.viz.diagnostics import plot_autocorr

        return plot_autocorr(
            self,
            params=params,
            max_lag=max_lag,
            **kwargs,
        )

    def plot_posterior(
        self,
        params=None,
        kind: str = "density",
        **kwargs,
    ):
        """
        Plot posterior distributions for model parameters.

        Visualizes the marginal posterior distributions, showing the uncertainty
        in parameter estimates.

        Parameters
        ----------
        params : str, list of str, optional
            Parameter name(s) to plot. If None, plots key parameters.
        kind : str, default="density"
            Type of plot: "density" for KDE or "histogram" for binned counts.        **kwargs
            Additional keyword arguments passed to plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive posterior distribution plot.

        Examples
        --------
        >>> fig = model.plot_posterior(params=['home_advantage', 'rho'])
        >>> fig.show()
        """
        from penaltyblog.viz.diagnostics import plot_posterior

        return plot_posterior(
            self,
            params=params,
            kind=kind,
            **kwargs,
        )

    def plot_convergence(self, **kwargs):
        """
        Plot convergence diagnostics (R-hat and ESS) for all parameters.

        Visualizes R-hat (Gelman-Rubin statistic) and effective sample size (ESS)
        for all model parameters. R-hat < 1.1 indicates good convergence.

        Parameters
        ----------        **kwargs
            Additional keyword arguments passed to plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive convergence diagnostic plot.

        Examples
        --------
        >>> fig = model.plot_convergence()
        >>> fig.show()
        """
        from penaltyblog.viz.diagnostics import plot_convergence

        return plot_convergence(self, **kwargs)

    def plot_diagnostics(self, params=None, **kwargs):
        """
        Create a comprehensive diagnostic dashboard.

        Combines trace plots, autocorrelation, posterior distributions, and
        convergence metrics into a single multi-panel figure.

        Parameters
        ----------
        params : str, list of str, optional
            Parameter name(s) to include in detailed plots. If None, uses key parameters.        **kwargs
            Additional keyword arguments passed to plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Comprehensive diagnostic dashboard.

        Examples
        --------
        >>> model.fit(n_samples=1000, burn=500)
        >>> fig = model.plot_diagnostics()
        >>> fig.show()
        """
        from penaltyblog.viz.diagnostics import plot_diagnostics

        return plot_diagnostics(self, params=params, **kwargs)


class HierarchicalBayesianGoalModel(BayesianGoalModel):
    """
    Advanced Bayesian Model with Hierarchical Priors.
    Learns the 'variance' of the league automatically.
    """

    def fit(self, n_samples=3000, burn=1500, n_chains=4, thin=5):
        # 1. Prepare Data (Same as base)
        data_dict = {
            "home_idx": self.home_idx,
            "away_idx": self.away_idx,
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "weights": self.weights,
            "n_teams": self.n_teams,
        }

        # 2. Get MLE Init (Same as base)
        mle_params = self._get_initial_params()
        mle_params = np.concatenate([mle_params, [-0.1]])  # Add Rho

        # 3. Setup Sampler
        self.sampler = EnsembleSampler(
            n_chains=n_chains,
            n_cores=n_chains,
            log_prob_wrapper_func=hierarchical_log_prob_wrapper,
            data_dict=data_dict,
        )

        # 4. Generate Starts (Hierarchical specific)
        # We need 2 extra dimensions for Sigma_Att and Sigma_Def
        ndim = (self.n_teams * 2) + 4
        n_walkers = max(50, 2 * ndim + 10)

        start_positions = [
            self._generate_hierarchical_starts(n_walkers, mle_params)
            for _ in range(n_chains)
        ]
        self.sampler.run_mcmc(start_positions, n_samples, burn)

        # 5. Extract
        self.trace = self.sampler.trim_samples(burn=burn, thin=thin)
        self._map_trace_to_dict()

        # 6. Set fitted state and point estimates (posterior mean) for Base API compatibility
        self._params = np.mean(self.trace, axis=0)
        self.fitted = True

        return self.trace_dict

    def _generate_hierarchical_starts(self, n_walkers, mle_params) -> np.ndarray:
        """
        Generates starts including the 2 extra Sigma parameters

        Args:
            n_walkers (int): Number of walkers
            mle_params (array-like): MLE parameters

        Returns:
            array-like: Start positions
        """
        ndim = (self.n_teams * 2) + 4
        start_pos = np.zeros((n_walkers, ndim))

        # Unpack MLE
        att_mle = mle_params[: self.n_teams]
        def_mle = mle_params[self.n_teams : 2 * self.n_teams]
        hfa_rho = mle_params[2 * self.n_teams :]

        # Center MLE
        att_centered = att_mle - np.mean(att_mle)
        def_centered = def_mle - np.mean(def_mle)

        # Estimate Initial Sigmas from MLE spread
        sig_att_est = np.std(att_centered)
        sig_def_est = np.std(def_centered)

        # Construct Vector: [Att, Def, HFA, Rho, SigAtt, SigDef]
        base_params = np.concatenate(
            [att_centered, def_centered, hfa_rho, [sig_att_est, sig_def_est]]
        )

        # Add Jitter
        for w in range(n_walkers):
            start_pos[w, :] = base_params + np.random.normal(0, 0.05, ndim)

        return start_pos

    def _map_trace_to_dict(self):
        """
        Map including the new sigma parameters
        """
        # Call parent to do the heavy lifting
        super()._map_trace_to_dict()

        # Indices: [Att(n), Def(n), HFA(1), Rho(1), SigAtt(1), SigDef(1)]
        idx_sig_att = 2 * self.n_teams + 2
        idx_sig_def = 2 * self.n_teams + 3

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

        # 1. Compute raw stats
        df = compute_diagnostics(self.sampler, burn=burn, thin=thin)

        # 2. Generate Labels (Must match 50 rows)
        labels = []
        # Teams
        for team in self.teams:
            labels.append(f"Attack_{team}")
        for team in self.teams:
            labels.append(f"Defense_{team}")

        # Globals
        labels.append("Home_Advantage")
        labels.append("Rho")

        # --- NEW LABELS ---
        labels.append("Sigma_Attack")
        labels.append("Sigma_Defense")

        # 3. Apply
        if len(df) != len(labels):
            # Fallback if something is misaligned to prevent crash
            print(
                f"Warning: Label mismatch ({len(df)} params vs {len(labels)} labels). Returning raw indices."
            )
            return df

        df.index = labels
        return df

    def _get_param_names(self) -> List[str]:
        """
        Return the parameter names for this model, including hierarchical sigmas.
        """
        names = super()._get_param_names()
        names.extend(["sigma_attack", "sigma_defense"])
        return names

    def _get_tail_param_indices(self) -> Dict[str, int]:
        """
        Return indices for hierarchical-specific trailing parameters.
        """
        indices = super()._get_tail_param_indices()
        indices.update({"sigma_attack": -2, "sigma_defense": -1})

        # Update base indices because hierarchical adds params at the end
        # Base indices were -2 (hfa) and -1 (rho)
        # Now they are -4 (hfa) and -3 (rho)
        indices["home_advantage"] = -4
        indices["rho"] = -3
        return indices
