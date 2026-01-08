from typing import Dict, Optional

import numpy as np
import pandas as pd

from penaltyblog.bayes.diagnostics import compute_diagnostics
from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.custom_types import GoalInput, TeamInput, WeightInput


class BaseBayesianModel(BaseGoalsModel):
    """
    Base class for Bayesian football models.

    This class extends BaseGoalsModel to provide common Bayesian inference
    functionality including MCMC sampling, trace management, and diagnostic
    plotting methods.
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
        Initialize a Bayesian football model.

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

        self.trace: Optional[np.ndarray] = None
        self.trace_dict: Optional[Dict[str, np.ndarray]] = None
        self.sampler: Optional[EnsembleSampler] = None

        self.team_map = self.team_to_idx

    def _map_trace_to_dict(self) -> None:
        """
        Helper to convert raw matrix to user-friendly dict.

        Converts the raw trace matrix into a dictionary with meaningful
        parameter names as keys.
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before mapping trace.")

        self.trace_dict = {}

        for i, team in enumerate(self.teams):
            self.trace_dict[f"attack_{team}"] = self.trace[:, i]
            self.trace_dict[f"defense_{team}"] = self.trace[:, self.n_teams + i]

    def get_diagnostics(self, burn: int = 0, thin: int = 1) -> pd.DataFrame:
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

        labels = []
        for team in self.teams:
            labels.append(f"Attack_{team}")
        for team in self.teams:
            labels.append(f"Defense_{team}")

        df.index = labels
        return df

    def _get_tail_param_indices(self) -> Dict[str, int]:
        """
        Return indices for model-specific trailing parameters.

        Returns
        -------
        dict
            Parameter names mapped to their index in the params array.
        """
        return {}

    def plot_trace(
        self,
        params: Optional[str] = None,
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
        params: Optional[str] = None,
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
            Maximum lag to compute autocorrelation for.
        **kwargs
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
        params: Optional[str] = None,
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
            Type of plot: "density" for KDE or "histogram" for binned counts.
        **kwargs
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

    def plot_diagnostics(self, params: Optional[str] = None, **kwargs):
        """
        Create a comprehensive diagnostic dashboard.

        Combines trace plots, autocorrelation, posterior distributions, and
        convergence metrics into a single multi-panel figure.

        Parameters
        ----------
        params : str, list of str, optional
            Parameter name(s) to include in detailed plots. If None, uses key parameters.
        **kwargs
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
