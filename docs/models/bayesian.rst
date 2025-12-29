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
    results = model.fit(n_samples=1000, burn=1000, n_chains=4, thin=5)

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

The ``fit()`` method returns a dictionary containing the traces for all parameters. You can use these to visualize the uncertainty in team strengths.

.. code-block:: python

    # Access the trace dictionary
    print(model.trace_dict.keys())

    # Example: get the posterior samples for Arsenal's attack
    arsenal_att = model.trace_dict["attack_Arsenal"]
