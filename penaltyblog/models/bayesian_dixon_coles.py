import numpy as np
from scipy.stats import norm

from .bayesian_base import BaseBayesianModel
from .bayesian_loss import bayesian_dixon_coles_log_prob
from .football_probability_grid import FootballProbabilityGrid
from .loss import dixon_coles_loss_function
from .probabilities import compute_posterior_predictive_dixon_coles


# --- Top-level function for multiprocessing ---
def log_probability_wrapper(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
) -> float:
    """
    Top-level wrapper for Dixon-Coles log probability to support multiprocessing.

    Parameters:
    -----------
    params : np.ndarray
        Model parameters
    home_idx : np.ndarray
        Indices of home teams
    away_idx : np.ndarray
        Indices of away teams
    goals_home : np.ndarray
        Goals scored by home teams
    goals_away : np.ndarray
        Goals scored by away teams
    weights : np.ndarray
        Match weights
    n_teams : int
        Number of teams

    Returns:
    --------
    float
        Log probability value
    """
    return bayesian_dixon_coles_log_prob(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )


class BayesianDixonColesModel(BaseBayesianModel):
    """
    A Bayesian implementation of the Dixon-Coles model for football match prediction.

    This model extends the classic Dixon-Coles framework with full Bayesian inference
    using MCMC sampling. It estimates team attack and defense strengths, home field
    advantage, and goal correlation while providing uncertainty quantification through
    posterior distributions.

    Mathematical Model:
    ------------------
    The model assumes goals follow independent Poisson distributions with rates:

    λ_home = exp(attack_home - defense_away + hfa)
    λ_away = exp(attack_away - defense_home)

    Where:
    - attack_i ~ N(0, 1) are team attack strengths
    - defense_i ~ N(0, 1) are team defense strengths
    - hfa ~ N(0.25, 0.5²) is home field advantage
    - rho ~ N(0, 0.5²) is the Dixon-Coles correlation parameter

    The Dixon-Coles correlation adjusts for the dependent structure in low-scoring
    games (0-0, 0-1, 1-0, 1-1 results) where the assumption of independence between
    home and away goals may not hold.

    Bayesian Framework:
    ------------------
    The model uses:
    - Conjugate normal priors on team strengths for regularization
    - MCMC sampling via emcee for posterior inference
    - Parallel computation for efficient sampling
    - Posterior predictive distributions for match predictions

    Parameters:
    -----------
    goals_home : array-like
        Number of goals scored by home teams in each match
    goals_away : array-like
        Number of goals scored by away teams in each match
    teams_home : array-like
        Home team identifiers for each match
    teams_away : array-like
        Away team identifiers for each match
    weights : array-like, optional
        Match weights for temporal discounting (default: None, all weights = 1)

    Attributes:
    -----------
    n_teams : int
        Number of unique teams in the dataset
    teams : array
        Sorted array of unique team identifiers
    sampler : emcee.EnsembleSampler
        MCMC sampler object (after fitting)
    chain : ndarray
        Posterior samples from MCMC (shape: [n_samples, n_parameters])
    fitted : bool
        Whether the model has been fitted with MCMC

    Examples:
    ---------
    >>> # Basic usage
    >>> model = BayesianDixonColesModel(
    ...     goals_home=[2, 1, 0, 3],
    ...     goals_away=[1, 1, 2, 0],
    ...     teams_home=['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'],
    ...     teams_away=['Tottenham', 'Arsenal', 'Chelsea', 'Liverpool']
    ... )
    >>> model.fit(n_walkers=100, n_steps=2000, n_burn=500)
    >>> probs = model.predict_match('Arsenal', 'Chelsea')
    >>> print(f"Arsenal win probability: {probs.home_win_prob:.3f}")

    >>> # With temporal weighting
    >>> recent_weights = np.exp(-0.01 * np.arange(len(goals_home))[::-1])
    >>> model = BayesianDixonColesModel(
    ...     goals_home, goals_away, teams_home, teams_away,
    ...     weights=recent_weights
    ... )

    Notes:
    ------
    - Computational complexity scales as O(n_teams²) parameters
    - MCMC sampling can be computationally intensive for large datasets
    - Uses multiprocessing for parallel chain evaluation
    - Provides full uncertainty quantification unlike MLE approaches
    - Regularization through priors prevents overfitting

    References:
    -----------
    Dixon, M.J. and Coles, S.G. (1997) "Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market"
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        weights=None,
    ):
        """
        Initialize the Bayesian Dixon-Coles model with match data.

        Sets up the model structure, processes team identifiers, and initializes
        parameter vectors with sensible defaults. The model parameters will be
        estimated during the fitting process via MCMC sampling.

        Parameters:
        -----------
        goals_home : array-like
            Goals scored by home teams in each match. Must be non-negative integers.
        goals_away : array-like
            Goals scored by away teams in each match. Must be non-negative integers.
        teams_home : array-like
            Home team identifiers for each match. Can be strings or any hashable type.
        teams_away : array-like
            Away team identifiers for each match. Can be strings or any hashable type.
        weights : array-like, optional
            Weights for each match, typically used for temporal discounting where
            more recent matches have higher weights. Must be positive values.
            If None (default), all matches receive equal weight of 1.0.

        Raises:
        -------
        ValueError
            If input arrays have mismatched lengths or contain invalid values

        Notes:
        -----
        - All input arrays must have the same length (number of matches)
        - Team identifiers are automatically mapped to integer indices
        - Initial parameter values are set to reasonable defaults:
          * Attack/defense strengths: 0.0 (average)
          * Home field advantage: 0.25 (typical value)
          * Correlation parameter: -0.1 (slight negative correlation)
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        # Define the parameter vector shape and names for reference
        self._params = np.concatenate(
            (
                np.zeros(self.n_teams),  # Attack ratings
                np.zeros(self.n_teams),  # Defense ratings
                [0.25],  # home_advantage
                [-0.1],  # rho
            )
        )

    def _get_initial_params(self) -> np.ndarray:
        """
        Find smart starting parameters for MCMC sampling.

        This method fits a frequentist Poisson model first and appends a default
        rho value. This ensures the Bayesian model starts near the maximum
        likelihood estimate.

        Returns:
        --------
        np.ndarray
            Initial parameter vector containing team ratings, home advantage, and rho.
        """
        base_params = super()._get_initial_params()
        return np.concatenate([base_params, [-0.1]])

    def _get_log_probability_function(self) -> callable:
        """
        Return the model-specific log probability function.

        Returns:
        --------
        callable
            Log probability function for MCMC sampling
        """
        return log_probability_wrapper

    def _get_param_names(self) -> list[str]:
        """
        Return the names of all parameters in the model.

        Returns:
        --------
        list[str]
            List of parameter names for team strengths, home advantage, and rho
        """
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "rho"]
        )

    def _unpack_params(self, params) -> dict[str, np.ndarray]:
        """
        Unpack the flat parameter vector into named components.

        Converts the 1D parameter array used by the MCMC sampler into a structured
        dictionary for easier access to individual model components.

        Parameters:
        -----------
        params : array-like
            Flat parameter vector of length (2*n_teams + 2) containing:
            - params[0:n_teams]: attack strengths for all teams
            - params[n_teams:2*n_teams]: defense strengths for all teams
            - params[2*n_teams]: home field advantage (hfa)
            - params[2*n_teams+1]: Dixon-Coles correlation parameter (rho)

        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary with keys:
            - 'attack': array of attack strengths (length n_teams)
            - 'defense': array of defense strengths (length n_teams)
            - 'hfa': scalar home field advantage
            - 'rho': scalar Dixon-Coles correlation parameter

        Notes:
        ------
        This method is primarily used internally during MCMC sampling and
        prediction to convert between the flat representation needed for
        optimization and the structured representation needed for computation.
        """
        nt = self.n_teams
        return {
            "attack": params[:nt],
            "defense": params[nt : 2 * nt],
            "hfa": params[2 * nt],
            "rho": params[2 * nt + 1],
        }

    def _calculate_frequentist_metrics(self):
        """
        Calculate frequentist-style metrics (AIC, log-likelihood) using posterior mean.

        This method provides compatibility with frequentist models for comparison purposes.
        """
        # Extract parameters at posterior mean for AIC calculation
        p = self._unpack_params(self._params)

        # Calculate negative log-likelihood using existing loss function
        neg_ll = dixon_coles_loss_function(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            p["attack"],
            p["defense"],
            p["hfa"],
            p["rho"],
        )

        # Set frequentist metrics using posterior mean
        self.n_params = len(self._params)
        self.loglikelihood = -neg_ll
        self.aic = 2 * neg_ll + 2 * self.n_params

    def _calculate_log_likelihood(self, params) -> float:
        """
        Calculate total log-likelihood for given parameters.

        Parameters:
        -----------
        params : array-like
            Model parameters

        Returns:
        --------
        float
            Total log-likelihood
        """
        p = self._unpack_params(params)
        neg_ll = dixon_coles_loss_function(
            self.goals_home,
            self.goals_away,
            self.weights,
            self.home_idx,
            self.away_idx,
            p["attack"],
            p["defense"],
            p["hfa"],
            p["rho"],
        )
        return -neg_ll

    def _compute_probabilities(
        self, home_idx, away_idx, max_goals, normalize=True
    ) -> FootballProbabilityGrid:
        """
        Generate the posterior predictive distribution for a match.

        This method averages the predictions from all samples in the MCMC chain
        to produce a single, robust probability grid for the match outcome.

        Args:
            home_idx (int): The integer index of the home team.
            away_idx (int): The integer index of the away team.
            max_goals (int): The maximum number of goals to model.
            normalize (bool, optional): Whether to normalize the resulting
                probability grid. Defaults to True.

        Returns:
            FootballProbabilityGrid: A grid of probabilities for each possible
                scoreline, representing the posterior predictive distribution.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")

        grid_matrix, lam_h, lam_a = compute_posterior_predictive_dixon_coles(
            self.chain, home_idx, away_idx, self.n_teams, max_goals
        )

        return FootballProbabilityGrid(
            grid_matrix,
            lam_h,
            lam_a,
            normalize=normalize,
        )

    def __repr__(self):
        """
        Return a string representation of the fitted Bayesian Dixon-Coles model.

        Displays both frequentist (AIC) and Bayesian (WAIC) information criteria
        for comprehensive model comparison, along with team parameters and
        model fit statistics.
        """
        lines = ["Module: Penaltyblog", "", "Model: Bayesian Dixon-Coles", ""]

        if not self.fitted:
            lines.append("Status: Model not fitted")
            return "\n".join(lines)

        # Ensure all required attributes are available
        assert self.aic is not None
        assert self.loglikelihood is not None
        assert self.n_params is not None

        lines.extend(
            [
                f"Number of parameters: {self.n_params}",
                f"Log Likelihood: {round(self.loglikelihood, 3)}",
                f"AIC: {round(self.aic, 3)}",
                f"WAIC: {round(self.waic, 3)}",
                f"Effective parameters (p_WAIC): {round(self.p_waic, 3)}",
                "",
                "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defence"),
                "-" * 60,
            ]
        )

        p = self._unpack_params(self._params)
        for idx, team in enumerate(self.teams):
            lines.append(
                "{0: <20} {1:<20} {2:<20}".format(
                    team,
                    round(p["attack"][idx], 3),
                    round(p["defense"][idx], 3),
                )
            )

        lines.extend(
            [
                "-" * 60,
                f"Home Advantage: {round(p['hfa'], 3)}",
                f"Rho: {round(p['rho'], 3)}",
            ]
        )

        return "\n".join(lines)
