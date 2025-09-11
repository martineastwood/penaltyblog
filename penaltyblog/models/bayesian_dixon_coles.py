from multiprocessing import Pool

import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)

from .loss import dixon_coles_loss_function  # noqa
from .probabilities import (
    compute_dixon_coles_probabilities,
)


# --- Top-level function for multiprocessing ---
def log_probability(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
):
    """
    Compute the log posterior probability for the Dixon-Coles model.

    This function evaluates the log posterior probability (log prior + log likelihood)
    for a given parameter vector. It serves as the target function for MCMC sampling
    and is defined at module level to enable multiprocessing with emcee.

    Mathematical Details:
    --------------------
    log p(θ|data) = log p(data|θ) + log p(θ) + constant

    Where:
    - p(data|θ) is the Dixon-Coles likelihood
    - p(θ) combines priors on all parameters
    - θ = [attack, defense, hfa, rho] is the parameter vector

    Prior Specifications:
    --------------------
    - attack_i ~ N(0, 1) for i = 1,...,n_teams
    - defense_i ~ N(0, 1) for i = 1,...,n_teams
    - hfa ~ N(0.25, 0.5²)
    - rho ~ N(0, 0.5²)

    Parameter Bounds:
    ----------------
    - attack, defense ∈ [-5, 5] (prevents extreme values)
    - rho ∈ (-1, 1) (correlation constraint)
    - No bounds on hfa (can be negative for away advantage)

    Parameters:
    -----------
    params : array-like
        Parameter vector of length (2*n_teams + 2) containing:
        - params[0:n_teams]: attack strengths
        - params[n_teams:2*n_teams]: defense strengths
        - params[2*n_teams]: home field advantage
        - params[2*n_teams+1]: Dixon-Coles correlation parameter
    home_idx : array-like
        Integer indices of home teams for each match
    away_idx : array-like
        Integer indices of away teams for each match
    goals_home : array-like
        Goals scored by home teams
    goals_away : array-like
        Goals scored by away teams
    weights : array-like
        Match weights (for temporal discounting)
    n_teams : int
        Total number of unique teams

    Returns:
    --------
    float
        Log posterior probability. Returns -np.inf if parameters violate
        bounds or if likelihood computation fails.

    Notes:
    ------
    - Function must be pickle-able for multiprocessing
    - Includes automatic bounds checking with hard rejection
    - Uses regularizing priors to prevent overfitting
    - Robust to numerical issues in likelihood computation
    """
    # Unpack params
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    hfa = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    # Bounds check: return -inf for invalid parameter sets
    if not (-1 < rho < 1):
        return -np.inf
    if np.any(attack < -5) or np.any(attack > 5):
        return -np.inf
    if np.any(defense < -5) or np.any(defense > 5):
        return -np.inf

    # --- Log Priors ---
    attack_priors = np.sum(norm.logpdf(attack, 0, 1.0))
    defense_priors = np.sum(norm.logpdf(defense, 0, 1.0))
    hfa_prior = norm.logpdf(hfa, 0.25, 0.5)
    rho_prior = norm.logpdf(rho, 0, 0.5)
    log_prior_prob = attack_priors + defense_priors + hfa_prior + rho_prior

    # --- Log Likelihood (with NaN check) ---
    neg_log_likelihood = dixon_coles_loss_function(
        goals_home, goals_away, weights, home_idx, away_idx, attack, defense, hfa, rho
    )
    if not np.isfinite(neg_log_likelihood):
        return -np.inf

    log_likelihood = -neg_log_likelihood

    return log_prior_prob + log_likelihood


class BayesianDixonColesModel(BaseGoalsModel):
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
        self.sampler = None
        self.chain = None

    def _get_param_names(self):
        """
        Gets the ordered list of parameter names.

        Returns:
            list: A list of strings corresponding to the parameter vector.
        """
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "rho"]
        )

    def _unpack_params(self, params):
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
        dict
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

    def fit(self, n_walkers=None, n_steps=2000, n_burn=1000, initial_params=None):
        """
        Fit the Bayesian Dixon-Coles model using MCMC sampling.

        This method performs full Bayesian inference by sampling from the posterior
        distribution of model parameters using the emcee affine-invariant ensemble
        sampler. The process includes automatic initialization via MAP estimation,
        parallel chain evaluation, and post-processing of samples.

        Algorithm Overview:
        ------------------
        1. Initialize walkers around MAP estimate (if initial_params=None)
        2. Run MCMC chains in parallel using multiprocessing
        3. Discard burn-in samples and thin the remaining chain
        4. Store flattened posterior samples for prediction

        Parameters:
        -----------
        n_walkers : int, optional
            Number of MCMC walkers (chains) to run in parallel. Should be at least
            2 times the number of parameters for good mixing. If None, defaults to
            2 * (2*n_teams + 2). Recommended: 50-200 for most applications.
        n_steps : int, optional
            Total number of MCMC steps per walker. More steps provide better
            convergence and more precise posterior estimates. Default: 2000.
            Recommended: 1000-5000 depending on model complexity.
        n_burn : int, optional
            Number of initial samples to discard as burn-in. These samples allow
            chains to reach the target distribution from their starting positions.
            Should be sufficient for convergence (typically 25-50% of n_steps).
            Default: 1000.
        initial_params : array-like, optional
            Starting parameter values for MCMC chains. If provided, must have shape
            (2*n_teams + 2,). If None, automatically finds good starting point via
            MAP estimation using L-BFGS-B optimization. Default: None.

        Raises:
        -------
        ValueError
            If n_walkers < 2*n_parameters or if optimization fails to find valid
            starting parameters
        RuntimeError
            If MCMC sampling encounters numerical issues or fails to converge

        Notes:
        ------
        - Uses multiprocessing for parallel evaluation of log-probability
        - Chains are automatically thinned by factor of 15 to reduce storage
        - Progress display can be controlled via emcee parameters
        - Sets self.fitted = True upon successful completion
        - Convergence diagnostics should be checked after fitting

        Performance Tips:
        ----------------
        - For large datasets: reduce n_steps and increase n_walkers
        - For small datasets: increase n_steps for better mixing
        - Monitor acceptance rates (should be 20-50%)
        - Use convergence diagnostics (R-hat, effective sample size)

        Examples:
        ---------
        >>> # Quick fit for exploration
        >>> model.fit(n_walkers=50, n_steps=1000, n_burn=300)

        >>> # Production fit with good convergence
        >>> model.fit(n_walkers=100, n_steps=3000, n_burn=1000)

        >>> # Custom initialization
        >>> custom_params = np.concatenate([attack_init, defense_init, [0.3, -0.05]])
        >>> model.fit(initial_params=custom_params)
        """
        ndim = len(self._params)

        # Rule of thumb: use at least twice as many walkers as dimensions
        if n_walkers is None:
            n_walkers = 2 * ndim

        # Prepare the tuple of extra arguments for the log_probability function
        args = (
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
            self.n_teams,
        )

        if initial_params is None:
            # Find a good starting point using a quick optimization
            neg_log_prob = lambda p, *a: -log_probability(p, *a)

            def safe_neg_log_prob(p, *a):
                res = neg_log_prob(p, *a)
                return res if np.isfinite(res) else 1e12

            result = minimize(
                safe_neg_log_prob, self._params, args=args, method="L-BFGS-B"
            )
            initial_params = result.x

        # Initialize walkers in a small, random ball around the best-fit parameters
        pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)

        # Use the multiprocessing Pool to run in parallel
        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                n_walkers, ndim, log_probability, args=args, pool=pool
            )

            # Run the MCMC sampler
            self.sampler.run_mcmc(
                pos, n_steps, progress=False, skip_initial_state_check=True
            )

        # Store the flattened chain, discarding the burn-in phase and thinning
        self.chain = self.sampler.get_chain(discard=n_burn, thin=15, flat=True)
        self.fitted = True

    def _compute_probabilities(self, home_idx, away_idx, max_goals, normalize=True):
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

        param_samples = self.chain
        n_samples = len(param_samples)

        total_score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64)
        total_lambda_home = 0.0
        total_lambda_away = 0.0

        # Pre-allocate arrays for the in-place Cython function
        sample_score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)
        sample_lambda_home = np.empty(1, dtype=np.float64)
        sample_lambda_away = np.empty(1, dtype=np.float64)

        for params in param_samples:
            p = self._unpack_params(params)

            home_attack = p["attack"][home_idx]
            away_attack = p["attack"][away_idx]
            home_defense = p["defense"][home_idx]
            away_defense = p["defense"][away_idx]

            compute_dixon_coles_probabilities(
                home_attack,
                away_attack,
                home_defense,
                away_defense,
                p["hfa"],
                p["rho"],
                max_goals,
                sample_score_matrix,
                sample_lambda_home,
                sample_lambda_away,
            )

            total_score_matrix += sample_score_matrix.reshape(max_goals, max_goals)
            total_lambda_home += sample_lambda_home[0]
            total_lambda_away += sample_lambda_away[0]

        # Average the results over all samples
        avg_score_matrix = total_score_matrix / n_samples
        avg_lambda_home = total_lambda_home / n_samples
        avg_lambda_away = total_lambda_away / n_samples

        return FootballProbabilityGrid(
            avg_score_matrix,
            avg_lambda_home,
            avg_lambda_away,
            normalize=normalize,
        )
