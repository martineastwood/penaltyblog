====================================
Bayesian Football Goal Models
====================================

The ``penaltyblog`` package provides powerful Bayesian alternatives to traditional Maximum Likelihood Estimation (MLE) models. These models use Markov Chain Monte Carlo (MCMC) sampling to estimate the full posterior distribution of model parameters, allowing you to account for parameter uncertainty in your predictions.

Bayesian Goal Model
===================

The ``BayesianGoalModel`` implements the Dixon-Coles methodology within a Bayesian framework. Instead of single point estimates for team strengths, it provides a distribution of possible values.

- **Idea**: Use MCMC (via an ensemble sampler) to sample from the posterior distribution of attack, defense, home advantage, and rho parameters.
- **Strengths**: Captures parameter uncertainty, provides full posterior distributions, avoids over-fitting in small datasets via priors.
- **Best for**: Small datasets, understanding uncertainty, or when you need more than just point estimates.

Hierarchical Bayesian Goal Model
================================

An advanced extension that automatically learns the 'variance' of the league's attack and defense strengths.

- **Idea**: The priors for team attack and defense strengths are themselves learned from the data (hierarchical priors).
- **Strengths**: Automatically adjusts the 'shrinkage' of team strengths towards the mean based on the league's overall competitiveness.
- **Best for**: Multi-league modeling or when you want the model to decide how much to "trust" individual team performances versus the league average.

Quick Example
=============

Fitting a Bayesian model follows the same consistent API as other ``penaltyblog`` models, with a few extra options for the MCMC sampler.

.. code-block:: python

    from penaltyblog.models import BayesianGoalModel

    # Initialize the model
    model = BayesianGoalModel(
        train["goals_home"],
        train["goals_away"],
        train["team_home"],
        train["team_away"],
    )

    # Fit using MCMC sampling
    # n_samples: number of samples per chain
    # burn: samples to discard at the start
    # n_chains: number of parallel chains
    # thin: thinning factor to reduce autocorrelation
    model.fit(n_samples=1000, burn=1000, n_chains=4, thin=5)

    # Predict probabilities for a fixture
    # This automatically integrates over the posterior distribution
    prediction = model.predict("Arsenal", "Manchester City")
    print(prediction.home_draw_away)

MCMC Diagnostics
================

When using Bayesian models, it is crucial to verify that the MCMC chains have converged. ``penaltyblog`` provides built-in diagnostics for R-hat (Gelman-Rubin) and Effective Sample Size (ESS).

.. code-block:: python

    # Get a DataFrame with R-hat and ESS for all parameters
    diagnostics = model.get_diagnostics()
    print(diagnostics)

Generally, you want **R-hat values close to 1.0** (typically < 1.1) and a sufficiently large **Effective Sample Size** for all parameters to ensure reliable inference.

Inspecting the Posterior
========================

After calling the ``fit()`` method, the ``trace_dict`` attribute contains the traces for all parameters. You can use these to visualize the uncertainty in team strengths.

.. code-block:: python

    # Access the trace dictionary
    print(model.trace_dict.keys())

    # Example: get the posterior samples for Arsenal's attack
    arsenal_att = model.trace_dict["attack_Arsenal"]

Diagnostic Plotting
===================

The Bayesian models provide rich interactive diagnostic plots to assess MCMC convergence and parameter distributions. All plotting methods return Plotly figures that can be displayed in Jupyter notebooks or saved to files.

Trace Plots
-----------

Trace plots show the evolution of parameter values across MCMC iterations. Well-converged chains should look like "fuzzy caterpillars" with no trends or drift.

.. code-block:: python

    # Plot trace for all parameters (default)
    # Shows: home_advantage, rho, sigma_* (if hierarchical),
    # and ALL team attack/defense parameters (overlaid)
    fig = model.plot_trace()
    fig.show()

    # Plot specific parameters
    fig = model.plot_trace(params=["home_advantage", "rho"])
    fig.show()

    # Combine chains instead of showing them separately
    fig = model.plot_trace(chains=False)
    fig.show()

**Team Parameter Overlays**: By default, all team attack parameters are overlaid on one subplot (with different colors per team), and all defense parameters on another. This creates compact, readable plots that make it easy to compare convergence across teams.

Autocorrelation Plots
---------------------

Autocorrelation plots help identify the thinning interval needed for independent samples. Autocorrelation should decay to near zero within a reasonable lag.

.. code-block:: python

    # Plot autocorrelation for all parameters
    fig = model.plot_autocorr()
    fig.show()

    # Customize maximum lag
    fig = model.plot_autocorr(max_lag=100)
    fig.show()

    # Plot specific parameters
    fig = model.plot_autocorr(params=["home_advantage"])
    fig.show()

Posterior Distribution Plots
-----------------------------

Visualize the marginal posterior distributions to see the uncertainty in parameter estimates.

.. code-block:: python

    # Plot posterior distributions (density plots)
    fig = model.plot_posterior()
    fig.show()

    # Use histograms instead of density plots
    fig = model.plot_posterior(kind="histogram")
    fig.show()

    # Plot specific parameters
    fig = model.plot_posterior(params=["home_advantage", "rho"])
    fig.show()

Convergence Diagnostics Plot
-----------------------------

Visualize R-hat and Effective Sample Size (ESS) for all parameters in a single plot.

.. code-block:: python

    # Plot convergence diagnostics
    fig = model.plot_convergence()
    fig.show()

The plot uses color coding:
- **Green**: Excellent convergence (R-hat < 1.01, ESS > 400)
- **Orange**: Acceptable (R-hat < 1.1, ESS > 100)
- **Red**: Poor convergence (needs more samples)

Comprehensive Diagnostic Dashboard
-----------------------------------

Create a multi-panel figure combining trace, autocorrelation, and posterior plots.

.. code-block:: python

    # Create comprehensive diagnostic dashboard
    fig = model.plot_diagnostics()
    fig.show()

This creates a 3-column layout with trace, autocorrelation, and posterior plots for each parameter, providing a complete view of MCMC performance.

Customizing Plots
-----------------

All plotting methods accept additional keyword arguments that are passed to Plotly's ``update_layout()`` method:

.. code-block:: python

    # Customize plot appearance using Plotly layout parameters
    fig = model.plot_trace(
        width=1200,
        height=800,
        title="Custom MCMC Trace Plot",
        showlegend=True
    )
    fig.show()

    # Save to file
    fig.write_html("diagnostics.html")
    fig.write_image("diagnostics.png")  # Requires kaleido

For a full list of available layout parameters, see the `Plotly documentation <https://plotly.com/python/reference/layout/>`_.
