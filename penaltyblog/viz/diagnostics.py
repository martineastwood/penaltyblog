"""
Diagnostic plotting for Bayesian MCMC models.

This module provides interactive diagnostic plots for assessing MCMC convergence,
including trace plots, autocorrelation, posterior distributions, and convergence metrics.
"""

from typing import List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models import BayesianGoalModel, HierarchicalBayesianGoalModel


def plot_trace(
    model: Union[BayesianGoalModel, HierarchicalBayesianGoalModel],
    params: Optional[Union[str, List[str]]] = None,
    chains: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Plot MCMC trace ("fuzzy caterpillar") for convergence diagnostics.

    Visualizes the evolution of parameter values across MCMC iterations.
    Well-mixed chains should look like fuzzy caterpillars with no trends.

    Parameters
    ----------
    model : BayesianGoalModel or a HierarchicalBayesianGoalModel
        A fitted Bayesian model with trace data.
    params : str, list of str, optional
        Parameter name(s) to plot. If None, plots key parameters
        (home_advantage, rho, hierarchical sigmas if present, and all team parameters).
    chains : bool, default=True
        If True, show individual chains. If False, combine all chains.    **kwargs
        Additional keyword arguments passed to plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive trace plot figure.

    Raises
    ------
    ValueError
        If model has not been fitted or if specified params don't exist.

    Examples
    --------
    >>> model = BayesianGoalModel(...)
    >>> model.fit(n_samples=1000, burn=500)
    >>> fig = plot_trace(model, params=['home_advantage', 'rho'])
    >>> fig.show()
    """
    if not hasattr(model, "sampler") or model.sampler is None:
        raise ValueError("Model has not been fitted. Call .fit() first.")

    if not model.sampler.chains:
        raise ValueError("No MCMC chains found in sampler.")

    # Get parameter names
    param_names = model._get_param_names()

    # Select parameters to plot
    if params is None:
        # Default: plot global parameters + all team parameters
        selected_params = ["home_advantage", "rho"]
        # Add hierarchical params if present
        if "sigma_attack" in param_names:
            selected_params.extend(["sigma_attack", "sigma_defense"])
        # Add grouped team parameters (all teams)
        selected_params.extend(["_team_attacks_", "_team_defenses_"])
    elif isinstance(params, str):
        selected_params = [params]
    else:
        selected_params = params

    # Separate individual params from grouped team params
    individual_params = [p for p in selected_params if not p.startswith("_team_")]
    has_team_attacks = "_team_attacks_" in selected_params
    has_team_defenses = "_team_defenses_" in selected_params

    # Validate individual parameter names
    for param in individual_params:
        if param not in param_names:
            raise ValueError(f"Parameter '{param}' not found. Available: {param_names}")

    # Build subplot structure
    subplot_params = individual_params.copy()
    subplot_titles = individual_params.copy()

    if has_team_attacks:
        subplot_params.append("_team_attacks_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Attack Parameters (n={n_teams_to_show})")

    if has_team_defenses:
        subplot_params.append("_team_defenses_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Defense Parameters (n={n_teams_to_show})")

    # Create subplots
    n_subplots = len(subplot_params)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    # Get theme

    # Color palette for team overlays
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot each subplot
    for subplot_idx, param_name in enumerate(subplot_params, 1):
        if param_name == "_team_attacks_":
            # Plot overlaid attack parameters for all teams
            for team_idx in range(len(model.teams)):
                team = model.teams[team_idx]
                param_full_name = f"attack_{team}"
                param_idx = param_names.index(param_full_name)
                color = colors[team_idx % len(colors)]

                if chains:
                    # Plot each chain
                    for chain_id, chain in enumerate(model.sampler.chains):
                        trace_data = chain.raw_trace[:, :, param_idx]
                        # Average across walkers for cleaner overlay
                        mean_trace = trace_data.mean(axis=1)
                        fig.add_trace(
                            go.Scatter(
                                y=mean_trace,
                                mode="lines",
                                line=dict(width=1.5, color=color),
                                opacity=0.7,
                                name=team,
                                legendgroup=team,
                                showlegend=(chain_id == 0),
                            ),
                            row=subplot_idx,
                            col=1,
                        )
                else:
                    # Combine all chains
                    all_traces = []
                    for chain in model.sampler.chains:
                        trace_data = chain.raw_trace[:, :, param_idx]
                        all_traces.append(trace_data.mean(axis=1))
                    combined = np.vstack(all_traces).mean(axis=0)

                    fig.add_trace(
                        go.Scatter(
                            y=combined,
                            mode="lines",
                            line=dict(width=1.5, color=color),
                            opacity=0.7,
                            name=team,
                            showlegend=True,
                        ),
                        row=subplot_idx,
                        col=1,
                    )

        elif param_name == "_team_defenses_":
            # Plot overlaid defense parameters for all teams
            for team_idx in range(len(model.teams)):
                team = model.teams[team_idx]
                param_full_name = f"defense_{team}"
                param_idx = param_names.index(param_full_name)
                color = colors[team_idx % len(colors)]

                if chains:
                    # Plot each chain
                    for chain_id, chain in enumerate(model.sampler.chains):
                        trace_data = chain.raw_trace[:, :, param_idx]
                        # Average across walkers for cleaner overlay
                        mean_trace = trace_data.mean(axis=1)
                        fig.add_trace(
                            go.Scatter(
                                y=mean_trace,
                                mode="lines",
                                line=dict(width=1.5, color=color),
                                opacity=0.7,
                                name=team,
                                legendgroup=team,
                                showlegend=(chain_id == 0),
                            ),
                            row=subplot_idx,
                            col=1,
                        )
                else:
                    # Combine all chains
                    all_traces = []
                    for chain in model.sampler.chains:
                        trace_data = chain.raw_trace[:, :, param_idx]
                        all_traces.append(trace_data.mean(axis=1))
                    combined = np.vstack(all_traces).mean(axis=0)

                    fig.add_trace(
                        go.Scatter(
                            y=combined,
                            mode="lines",
                            line=dict(width=1.5, color=color),
                            opacity=0.7,
                            name=team,
                            showlegend=True,
                        ),
                        row=subplot_idx,
                        col=1,
                    )
        else:
            # Individual parameter - plot as before
            param_idx = param_names.index(param_name)
            if chains:
                # Plot each chain separately
                for chain_id, chain in enumerate(model.sampler.chains):
                    # Extract trace for this parameter across all walkers
                    # Shape: (n_steps, n_walkers)
                    trace_data = chain.raw_trace[:, :, param_idx]

                    # Plot each walker
                    for walker_id in range(trace_data.shape[1]):
                        fig.add_trace(
                            go.Scatter(
                                y=trace_data[:, walker_id],
                                mode="lines",
                                line=dict(width=0.5, color="#1f77b4"),
                                opacity=0.3,
                                showlegend=(subplot_idx == 1 and walker_id == 0),
                                name=f"Chain {chain_id}" if walker_id == 0 else None,
                                legendgroup=f"chain_{chain_id}",
                            ),
                            row=subplot_idx,
                            col=1,
                        )
            else:
                # Combine all chains
                all_traces = []
                for chain in model.sampler.chains:
                    trace_data = chain.raw_trace[:, :, param_idx]
                    all_traces.append(trace_data)

                # Concatenate along walker dimension
                combined = np.hstack(all_traces)  # Shape: (n_steps, total_walkers)

                # Plot all walkers
                for walker_id in range(combined.shape[1]):
                    fig.add_trace(
                        go.Scatter(
                            y=combined[:, walker_id],
                            mode="lines",
                            line=dict(width=0.5, color="#1f77b4"),
                            opacity=0.3,
                            showlegend=False,
                        ),
                        row=subplot_idx,
                        col=1,
                    )

        # Update axes
        fig.update_xaxes(title_text="Iteration", row=subplot_idx, col=1)
        fig.update_yaxes(title_text="Value", row=subplot_idx, col=1)

    # Update layout
    fig.update_layout(
        title="MCMC Trace Plot",
        height=300 * n_subplots,
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_layout(**kwargs)

    return fig


def plot_autocorr(
    model: Union[BayesianGoalModel, HierarchicalBayesianGoalModel],
    params: Optional[Union[str, List[str]]] = None,
    max_lag: int = 50,
    **kwargs,
) -> go.Figure:
    """
    Plot autocorrelation function for MCMC parameters.

    Helps identify the thinning interval needed for independent samples.
    Autocorrelation should decay to near zero within a reasonable lag.

    Parameters
    ----------
    model : BayesianGoalModel or HierarchicalBayesianGoalModel
        A fitted Bayesian model with trace data.
    params : str, list of str, optional
        Parameter name(s) to plot. If None, plots key parameters.
    max_lag : int, default=50
        Maximum lag to compute autocorrelation for.    **kwargs
        Additional keyword arguments passed to plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive autocorrelation plot.

    Raises
    ------
    ValueError
        If model has not been fitted or if specified params don't exist.

    Examples
    --------
    >>> fig = plot_autocorr(model, params=['home_advantage'], max_lag=100)
    >>> fig.show()
    """
    if not hasattr(model, "sampler") or model.sampler is None:
        raise ValueError("Model has not been fitted. Call .fit() first.")

    # Get parameter names
    param_names = model._get_param_names()

    # Select parameters
    if params is None:
        selected_params = ["home_advantage", "rho"]
        if "sigma_attack" in param_names:
            selected_params.extend(["sigma_attack", "sigma_defense"])
        # Add grouped team parameters if requested
        selected_params.extend(["_team_attacks_", "_team_defenses_"])
    elif isinstance(params, str):
        selected_params = [params]
    else:
        selected_params = params

    # Separate individual params from grouped team params
    individual_params = [p for p in selected_params if not p.startswith("_team_")]
    has_team_attacks = "_team_attacks_" in selected_params
    has_team_defenses = "_team_defenses_" in selected_params

    # Validate individual parameter names
    for param in individual_params:
        if param not in param_names:
            raise ValueError(f"Parameter '{param}' not found. Available: {param_names}")

    # Build subplot structure
    subplot_params = individual_params.copy()
    subplot_titles = individual_params.copy()

    if has_team_attacks:
        subplot_params.append("_team_attacks_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Attack Parameters (n={n_teams_to_show})")

    if has_team_defenses:
        subplot_params.append("_team_defenses_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Defense Parameters (n={n_teams_to_show})")

    # Create subplots
    n_subplots = len(subplot_params)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Compute and plot autocorrelation for each subplot
    for subplot_idx, param_name in enumerate(subplot_params, 1):
        if param_name == "_team_attacks_" or param_name == "_team_defenses_":
            # Plot overlaid team parameters
            n_teams_to_show = len(model.teams)
            param_type = "attack" if param_name == "_team_attacks_" else "defense"

            for team_idx in range(n_teams_to_show):
                team = model.teams[team_idx]
                param_full_name = f"{param_type}_{team}"
                param_idx = param_names.index(param_full_name)
                color = colors[team_idx % len(colors)]

                # Collect samples from all chains
                all_samples = []
                for chain in model.sampler.chains:
                    trace_data = chain.raw_trace[:, :, param_idx].flatten()
                    all_samples.append(trace_data)
                combined_samples = np.concatenate(all_samples)

                # Compute autocorrelation
                acf = _compute_autocorr(combined_samples, max_lag)

                # Plot
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(acf))),
                        y=acf,
                        mode="lines",
                        line=dict(color=color, width=2),
                        opacity=0.7,
                        name=team,
                        showlegend=True,
                    ),
                    row=subplot_idx,
                    col=1,
                )
        else:
            # Individual parameter
            param_idx = param_names.index(param_name)

            # Collect samples from all chains
            all_samples = []
            for chain in model.sampler.chains:
                trace_data = chain.raw_trace[:, :, param_idx].flatten()
                all_samples.append(trace_data)
            combined_samples = np.concatenate(all_samples)

            # Compute autocorrelation
            acf = _compute_autocorr(combined_samples, max_lag)

            # Plot
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf))),
                    y=acf,
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=4),
                    showlegend=False,
                ),
                row=subplot_idx,
                col=1,
            )

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=subplot_idx,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Lag", row=subplot_idx, col=1)
        fig.update_yaxes(
            title_text="Autocorrelation", row=subplot_idx, col=1, range=[-0.1, 1.0]
        )

    # Update layout
    fig.update_layout(
        title="Autocorrelation Function",
        height=300 * n_subplots,
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_layout(**kwargs)

    return fig


def plot_posterior(
    model: Union[BayesianGoalModel, HierarchicalBayesianGoalModel],
    params: Optional[Union[str, List[str]]] = None,
    kind: str = "density",
    **kwargs,
) -> go.Figure:
    """
    Plot posterior distributions for model parameters.

    Visualizes the marginal posterior distributions, showing the uncertainty
    in parameter estimates.

    Parameters
    ----------
    model : BayesianGoalModel or HierarchicalBayesianGoalModel
        A fitted Bayesian model with trace data.
    params : str, list of str, optional
        Parameter name(s) to plot. If None, plots key parameters.
    kind : str, default="density"
        Type of plot: "density" for KDE or "histogram" for binned counts.    **kwargs
        Additional keyword arguments passed to plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive posterior distribution plot.

    Raises
    ------
    ValueError
        If model has not been fitted or if specified params don't exist.

    Examples
    --------
    >>> fig = plot_posterior(model, params=['home_advantage', 'rho'])
    >>> fig.show()
    """
    if not hasattr(model, "trace") or model.trace is None:
        raise ValueError("Model has not been fitted. Call .fit() first.")

    # Get parameter names
    param_names = model._get_param_names()

    # Select parameters
    if params is None:
        selected_params = ["home_advantage", "rho"]
        if "sigma_attack" in param_names:
            selected_params.extend(["sigma_attack", "sigma_defense"])
        # Add grouped team parameters if requested
        selected_params.extend(["_team_attacks_", "_team_defenses_"])
    elif isinstance(params, str):
        selected_params = [params]
    else:
        selected_params = params

    # Separate individual params from grouped team params
    individual_params = [p for p in selected_params if not p.startswith("_team_")]
    has_team_attacks = "_team_attacks_" in selected_params
    has_team_defenses = "_team_defenses_" in selected_params

    # Validate individual parameter names
    for param in individual_params:
        if param not in param_names:
            raise ValueError(f"Parameter '{param}' not found. Available: {param_names}")

    # Build subplot structure
    subplot_params = individual_params.copy()
    subplot_titles = individual_params.copy()

    if has_team_attacks:
        subplot_params.append("_team_attacks_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Attack Parameters (n={n_teams_to_show})")

    if has_team_defenses:
        subplot_params.append("_team_defenses_")
        n_teams_to_show = len(model.teams)
        subplot_titles.append(f"Team Defense Parameters (n={n_teams_to_show})")

    # Create subplots
    n_subplots = len(subplot_params)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot each subplot
    for subplot_idx, param_name in enumerate(subplot_params, 1):
        if param_name == "_team_attacks_" or param_name == "_team_defenses_":
            # Plot overlaid team parameters
            n_teams_to_show = len(model.teams)
            param_type = "attack" if param_name == "_team_attacks_" else "defense"

            for team_idx in range(n_teams_to_show):
                team = model.teams[team_idx]
                param_full_name = f"{param_type}_{team}"
                param_idx = param_names.index(param_full_name)
                color = colors[team_idx % len(colors)]
                samples = model.trace[:, param_idx]

                if kind == "histogram":
                    fig.add_trace(
                        go.Histogram(
                            x=samples,
                            nbinsx=50,
                            marker=dict(color=color, opacity=0.5),
                            name=team,
                            showlegend=True,
                        ),
                        row=subplot_idx,
                        col=1,
                    )
                else:  # density
                    from scipy import stats

                    kde = stats.gaussian_kde(samples)
                    x_range = np.linspace(samples.min(), samples.max(), 200)
                    density = kde(x_range)

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=density,
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.7,
                            name=team,
                            showlegend=True,
                        ),
                        row=subplot_idx,
                        col=1,
                    )
        else:
            # Individual parameter
            param_idx = param_names.index(param_name)
            samples = model.trace[:, param_idx]

            if kind == "histogram":
                fig.add_trace(
                    go.Histogram(
                        x=samples,
                        nbinsx=50,
                        marker=dict(color="#1f77b4", opacity=0.7),
                        showlegend=False,
                    ),
                    row=subplot_idx,
                    col=1,
                )
            else:  # density
                # Compute KDE
                from scipy import stats

                kde = stats.gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 200)
                density = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=density,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#1f77b4", width=2),
                        fillcolor="#1f77b4",
                        opacity=0.5,
                        showlegend=False,
                    ),
                    row=subplot_idx,
                    col=1,
                )

            # Add posterior mean line (only for individual params)
            mean_val = np.mean(samples)
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.3f}",
                annotation_position="top",
                row=subplot_idx,
                col=1,
            )

        # Update axes
        fig.update_xaxes(title_text="Value", row=subplot_idx, col=1)
        fig.update_yaxes(
            title_text="Density" if kind == "density" else "Count",
            row=subplot_idx,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title="Posterior Distributions",
        height=300 * n_subplots,
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_layout(**kwargs)

    return fig


def plot_convergence(
    model: Union[BayesianGoalModel, HierarchicalBayesianGoalModel],
    **kwargs,
) -> go.Figure:
    """
    Plot convergence diagnostics (R-hat and ESS) for all parameters.

    Visualizes R-hat (Gelman-Rubin statistic) and effective sample size (ESS)
    for all model parameters. R-hat < 1.1 indicates good convergence.

    Parameters
    ----------
    model : BayesianGoalModel or HierarchicalBayesianGoalModel
        A fitted Bayesian model with trace data.    **kwargs
        Additional keyword arguments passed to plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive convergence diagnostic plot.

    Raises
    ------
    ValueError
        If model has not been fitted.

    Examples
    --------
    >>> fig = plot_convergence(model)
    >>> fig.show()
    """
    if not hasattr(model, "sampler") or model.sampler is None:
        raise ValueError("Model has not been fitted. Call .fit() first.")

    # Get diagnostics
    diag_df = model.get_diagnostics()

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["R-hat (Gelman-Rubin)", "Effective Sample Size"],
        horizontal_spacing=0.15,
    )

    # R-hat plot
    r_hat_colors = [
        "green" if r < 1.01 else "orange" if r < 1.1 else "red"
        for r in diag_df["r_hat"]
    ]

    fig.add_trace(
        go.Bar(
            y=diag_df.index,
            x=diag_df["r_hat"],
            orientation="h",
            marker=dict(color=r_hat_colors),
            showlegend=False,
            text=diag_df["r_hat"].round(3),
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Add R-hat threshold lines
    fig.add_vline(
        x=1.01,
        line_dash="dash",
        line_color="green",
        annotation_text="Excellent (1.01)",
        annotation_position="top",
        row=1,
        col=1,
    )
    fig.add_vline(
        x=1.1,
        line_dash="dash",
        line_color="orange",
        annotation_text="Acceptable (1.1)",
        annotation_position="bottom",
        row=1,
        col=1,
    )

    # ESS plot
    ess_colors = [
        "green" if ess > 400 else "orange" if ess > 100 else "red"
        for ess in diag_df["ess"]
    ]

    fig.add_trace(
        go.Bar(
            y=diag_df.index,
            x=diag_df["ess"],
            orientation="h",
            marker=dict(color=ess_colors),
            showlegend=False,
            text=diag_df["ess"].fillna(0).round(0).astype(int),
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text="R-hat", row=1, col=1)
    fig.update_xaxes(title_text="ESS", row=1, col=2)
    fig.update_yaxes(title_text="Parameter", row=1, col=1)

    # Update layout
    fig.update_layout(
        title="Convergence Diagnostics",
        height=max(400, len(diag_df) * 20),
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_layout(**kwargs)

    return fig


def plot_diagnostics(
    model: Union[BayesianGoalModel, HierarchicalBayesianGoalModel],
    params: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> go.Figure:
    """
    Create a comprehensive diagnostic dashboard.

    Combines trace plots, autocorrelation, posterior distributions, and
    convergence metrics into a single multi-panel figure.

    Parameters
    ----------
    model : BayesianGoalModel or HierarchicalBayesianGoalModel
        A fitted Bayesian model with trace data.
    params : str, list of str, optional
        Parameter name(s) to include in detailed plots. If None, uses key parameters.    **kwargs
        Additional keyword arguments passed to plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
        Comprehensive diagnostic dashboard.

    Raises
    ------
    ValueError
        If model has not been fitted.

    Examples
    --------
    >>> fig = plot_diagnostics(model)
    >>> fig.show()
    """
    if not hasattr(model, "sampler") or model.sampler is None:
        raise ValueError("Model has not been fitted. Call .fit() first.")

    # Get parameter names
    param_names = model._get_param_names()

    # Select parameters
    if params is None:
        selected_params = ["home_advantage", "rho"]
        if "sigma_attack" in param_names:
            selected_params.extend(["sigma_attack", "sigma_defense"])
        # Add grouped team parameters if requested
        selected_params.extend(["_team_attacks_", "_team_defenses_"])
    elif isinstance(params, str):
        selected_params = [params]
    else:
        selected_params = params

    # Separate individual params from grouped team params
    individual_params = [p for p in selected_params if not p.startswith("_team_")]
    has_team_attacks = "_team_attacks_" in selected_params
    has_team_defenses = "_team_defenses_" in selected_params

    # Build subplot structure
    subplot_params = individual_params.copy()

    if has_team_attacks:
        subplot_params.append("_team_attacks_")

    if has_team_defenses:
        subplot_params.append("_team_defenses_")

    n_subplots = len(subplot_params)

    # Create subplot titles
    subplot_titles = []
    for p in subplot_params:
        if p == "_team_attacks_":
            n_teams_to_show = len(model.teams)
            base_title = f"Team Attacks (n={n_teams_to_show})"
        elif p == "_team_defenses_":
            n_teams_to_show = len(model.teams)
            base_title = f"Team Defenses (n={n_teams_to_show})"
        else:
            base_title = p
        subplot_titles.extend(
            [
                f"{base_title} - Trace",
                f"{base_title} - Autocorr",
                f"{base_title} - Posterior",
            ]
        )

    # Create subplots: trace, autocorr, posterior for each param
    fig = make_subplots(
        rows=n_subplots,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot each parameter
    for subplot_idx, param_name in enumerate(subplot_params, 1):
        if param_name == "_team_attacks_" or param_name == "_team_defenses_":
            # Plot overlaid team parameters
            n_teams_to_show = len(model.teams)
            param_type = "attack" if param_name == "_team_attacks_" else "defense"

            for team_idx in range(n_teams_to_show):
                team = model.teams[team_idx]
                param_full_name = f"{param_type}_{team}"
                param_idx = param_names.index(param_full_name)
                color = colors[team_idx % len(colors)]

                # Trace plot (column 1)
                for chain in model.sampler.chains:
                    trace_data = chain.raw_trace[:, :, param_idx]
                    mean_trace = trace_data.mean(axis=1)
                    fig.add_trace(
                        go.Scatter(
                            y=mean_trace,
                            mode="lines",
                            line=dict(width=1, color=color),
                            opacity=0.6,
                            name=team,
                            legendgroup=team,
                            showlegend=(subplot_idx == 1),
                        ),
                        row=subplot_idx,
                        col=1,
                    )

                # Autocorrelation plot (column 2)
                all_samples = []
                for chain in model.sampler.chains:
                    trace_data = chain.raw_trace[:, :, param_idx].flatten()
                    all_samples.append(trace_data)
                combined_samples = np.concatenate(all_samples)
                acf = _compute_autocorr(combined_samples, max_lag=50)

                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(acf))),
                        y=acf,
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        opacity=0.6,
                        name=team,
                        legendgroup=team,
                        showlegend=False,
                    ),
                    row=subplot_idx,
                    col=2,
                )

                # Posterior plot (column 3)
                samples = model.trace[:, param_idx]
                from scipy import stats

                kde = stats.gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 100)
                density = kde(x_range)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=density,
                        mode="lines",
                        line=dict(color=color, width=2),
                        opacity=0.6,
                        name=team,
                        legendgroup=team,
                        showlegend=False,
                    ),
                    row=subplot_idx,
                    col=3,
                )
        else:
            # Individual parameter
            param_idx = param_names.index(param_name)

            # Trace plot (column 1)
            for chain in model.sampler.chains:
                trace_data = chain.raw_trace[:, :, param_idx]
                for walker_id in range(trace_data.shape[1]):
                    fig.add_trace(
                        go.Scatter(
                            y=trace_data[:, walker_id],
                            mode="lines",
                            line=dict(width=0.5, color="#1f77b4"),
                            opacity=0.3,
                            showlegend=False,
                        ),
                        row=subplot_idx,
                        col=1,
                    )

            # Autocorrelation plot (column 2)
            all_samples = []
            for chain in model.sampler.chains:
                trace_data = chain.raw_trace[:, :, param_idx].flatten()
                all_samples.append(trace_data)
            combined_samples = np.concatenate(all_samples)
            acf = _compute_autocorr(combined_samples, max_lag=50)

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acf))),
                    y=acf,
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=3),
                    showlegend=False,
                ),
                row=subplot_idx,
                col=2,
            )

            # Posterior plot (column 3)
            samples = model.trace[:, param_idx]
            from scipy import stats

            kde = stats.gaussian_kde(samples)
            x_range = np.linspace(samples.min(), samples.max(), 100)
            density = kde(x_range)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=density,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#1f77b4", width=2),
                    fillcolor="#1f77b4",
                    opacity=0.5,
                    showlegend=False,
                ),
                row=subplot_idx,
                col=3,
            )

        # Update axes labels
        fig.update_xaxes(title_text="Iteration", row=subplot_idx, col=1)
        fig.update_xaxes(title_text="Lag", row=subplot_idx, col=2)
        fig.update_xaxes(title_text="Value", row=subplot_idx, col=3)

    # Update layout
    fig.update_layout(
        title="MCMC Diagnostics Dashboard",
        height=300 * n_subplots,
        hovermode="closest",
        font=dict(family="Arial, sans-serif"),
    )
    fig.update_layout(**kwargs)

    return fig


# Helper functions
def _compute_autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function using FFT.

    Parameters
    ----------
    x : np.ndarray
        1D array of samples.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray
        Autocorrelation values from lag 0 to max_lag.
    """
    n = len(x)
    # Demean
    x_centered = x - np.mean(x)

    # Pad to avoid circular correlation
    f = np.fft.fft(x_centered, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real

    # Normalize
    acf /= acf[0]

    # Return up to max_lag
    return acf[: min(max_lag + 1, n)]
