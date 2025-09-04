from multiprocessing import Pool

import emcee
import numpy as np
from scipy.optimize import minimize

from .base_model import BaseGoalsModel
from .football_probability_grid import FootballProbabilityGrid
from .loss import random_intercept_loss_function
from .probabilities import compute_random_intercept_probabilities


# --- Vectorized helper functions for speed ---
def _compute_log_prior_vectorized(
    attack_offsets,
    defense_offsets,
    match_offsets,
    mu_attack,
    mu_defense,
    log_sigma_attack,
    log_sigma_defense,
    log_sigma_match,
    hfa,
    rho,
):
    """Vectorized computation of log priors using pure NumPy."""
    # Individual priors (avoid scipy.stats calls for simple distributions)
    mu_attack_prior = -0.5 * mu_attack**2  # N(0,1) log pdf up to constant
    mu_defense_prior = -0.5 * mu_defense**2

    # Exponential priors (log scale) - manual computation is faster
    sigma_attack = np.exp(log_sigma_attack)
    sigma_defense = np.exp(log_sigma_defense)
    sigma_match = np.exp(log_sigma_match)
    sigma_attack_prior = log_sigma_attack - sigma_attack  # Exponential(1) log pdf
    sigma_defense_prior = log_sigma_defense - sigma_defense
    sigma_match_prior = log_sigma_match - sigma_match

    # Normal priors for hfa and rho (manual computation)
    hfa_prior = -0.5 * ((hfa - 0.25) / 0.5) ** 2  # N(0.25, 0.5)
    rho_prior = -0.5 * (rho / 0.5) ** 2  # N(0, 0.5)

    # Vectorized offset priors - single operations on arrays
    attack_offset_priors = -0.5 * np.dot(attack_offsets, attack_offsets)
    defense_offset_priors = -0.5 * np.dot(defense_offsets, defense_offsets)
    match_offset_priors = -0.5 * np.dot(match_offsets, match_offsets)

    return (
        mu_attack_prior
        + mu_defense_prior
        + sigma_attack_prior
        + sigma_defense_prior
        + sigma_match_prior
        + hfa_prior
        + rho_prior
        + attack_offset_priors
        + defense_offset_priors
        + match_offset_priors
    )


# --- Top-level function for multiprocessing ---
def hierarchical_random_intercept_log_prob(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams, n_matches
):
    """
    Compute the log-posterior probability for the hierarchical Dixon-Coles model.

    This function evaluates the log-posterior probability (log-prior + log-likelihood)
    for the hierarchical Dixon-Coles model with per-match random intercepts.
    It's designed as a top-level function to work with multiprocessing in emcee.

    Parameters:
    -----------
    params : array-like
        Parameter vector containing:
        - attack_offsets[n_teams]: Standardized team attack strengths
        - defense_offsets[n_teams]: Standardized team defense strengths
        - match_intercept_offsets[n_matches]: Standardized match intercepts
        - mu_attack: Population mean attack strength
        - log_sigma_attack: Log scale parameter for attack strengths
        - mu_defense: Population mean defense strength
        - log_sigma_defense: Log scale parameter for defense strengths
        - log_sigma_match: Log scale parameter for match intercepts
        - hfa: Home field advantage
        - rho: Dixon-Coles correlation parameter
    home_idx : array-like
        Home team indices for each match
    away_idx : array-like
        Away team indices for each match
    goals_home : array-like
        Goals scored by home teams
    goals_away : array-like
        Goals scored by away teams
    weights : array-like
        Match weights (typically for temporal discounting)
    n_teams : int
        Number of teams in the dataset
    n_matches : int
        Number of matches in the dataset

    Returns:
    --------
    float
        Log-posterior probability. Returns -inf for invalid parameter values.

    Notes:
    ------
    - Uses non-centered parameterization for numerical stability
    - Includes bounds checking for all parameters
    - Applies conjugate priors where appropriate
    """
    # 1. Unpack all parameters
    # Team-level parameters
    attack_offsets = params[:n_teams]
    defense_offsets = params[n_teams : 2 * n_teams]

    # Match-level random intercept parameters
    match_intercept_offsets = params[2 * n_teams : 2 * n_teams + n_matches]

    # Hyperparameters
    hyper_params_start_idx = 2 * n_teams + n_matches
    mu_attack = params[hyper_params_start_idx]
    log_sigma_attack = params[hyper_params_start_idx + 1]
    mu_defense = params[hyper_params_start_idx + 2]
    log_sigma_defense = params[hyper_params_start_idx + 3]
    log_sigma_match = params[hyper_params_start_idx + 4]  # New hyperparameter
    hfa = params[hyper_params_start_idx + 5]
    rho = params[hyper_params_start_idx + 6]

    # 2. Bounds Check
    if not (0 < hfa < 1 and -1 < rho < 1):
        return -np.inf
    if not (-3 < mu_attack < 3 and -3 < mu_defense < 3):
        return -np.inf
    if not (-5 < log_sigma_attack < 2 and -5 < log_sigma_defense < 2):
        return -np.inf
    if not (-5 < log_sigma_match < 2):
        return -np.inf  # Bound for new hyperparameter

    # 3. Priors (optimized vectorized computation)
    log_prior_prob = _compute_log_prior_vectorized(
        attack_offsets,
        defense_offsets,
        match_intercept_offsets,
        mu_attack,
        mu_defense,
        log_sigma_attack,
        log_sigma_defense,
        log_sigma_match,
        hfa,
        rho,
    )

    # 4. Construct Final Parameters (Non-Centered Parameterization)
    epsilon = 1e-6
    sigma_attack = np.exp(log_sigma_attack) + epsilon
    sigma_defense = np.exp(log_sigma_defense) + epsilon
    sigma_match = np.exp(log_sigma_match) + epsilon

    attack = mu_attack + sigma_attack * attack_offsets
    defense = mu_defense + sigma_defense * defense_offsets
    match_intercepts = sigma_match * match_intercept_offsets

    # 5. Log Likelihood
    neg_log_likelihood = random_intercept_loss_function(
        goals_home,
        goals_away,
        weights,
        home_idx,
        away_idx,
        attack,
        defense,
        match_intercepts,
        hfa,
        rho,
    )
    if not np.isfinite(neg_log_likelihood):
        return -np.inf

    return log_prior_prob - neg_log_likelihood


class BayesianRandomInterceptModel(BaseGoalsModel):
    """
    A hierarchical Bayesian Dixon-Coles model with per-match random intercepts.

    This model extends the classic Dixon-Coles framework by incorporating a hierarchical
    structure with random intercepts for each match, allowing for match-specific variations
    in goal-scoring rates beyond what can be explained by team strengths alone.

    Mathematical Model:
    ------------------
    For match i between home team h and away team a:

    λ_h,i = exp(μ_attack + σ_attack * α_h - μ_defense - σ_defense * β_a + hfa + δ_i)
    λ_a,i = exp(μ_attack + σ_attack * α_a - μ_defense - σ_defense * β_h + δ_i)

    Where:
    - α_h, β_h ~ N(0, 1) are standardized team attack/defense offsets
    - δ_i ~ N(0, σ_match) are match-specific random intercepts
    - μ_attack, μ_defense are population-level means
    - σ_attack, σ_defense, σ_match are scale parameters
    - hfa is the home field advantage
    - rho is the Dixon-Coles correlation parameter

    The model uses a non-centered parameterization for efficient MCMC sampling
    and includes proper Bayesian priors on all hyperparameters.

    Parameters:
    -----------
    goals_home : array-like
        Number of goals scored by home teams
    goals_away : array-like
        Number of goals scored by away teams
    teams_home : array-like
        Home team identifiers
    teams_away : array-like
        Away team identifiers
    weights : array-like, optional
        Match weights for temporal discounting (default: None)

    Attributes:
    -----------
    n_matches : int
        Number of matches in the dataset
    sampler : emcee.EnsembleSampler
        MCMC sampler object (after fitting)
    chain : ndarray
        Posterior samples (after fitting)
    fitted : bool
        Whether the model has been fitted

    Examples:
    ---------
    >>> model = BayesianRandomInterceptModel(
    ...     goals_home=[1, 2, 0],
    ...     goals_away=[0, 1, 2],
    ...     teams_home=['Arsenal', 'Chelsea', 'Liverpool'],
    ...     teams_away=['Tottenham', 'Arsenal', 'Chelsea']
    ... )
    >>> model.fit(n_walkers=100, n_steps=2000)
    >>> probs = model.predict_match('Arsenal', 'Chelsea')

    Notes:
    ------
    - Fitting requires substantial computational time due to MCMC sampling
    - The model scales as O(n_teams + n_matches) parameters
    - Uses multiprocessing for parallel chain evaluation
    - Includes automatic initialization via MAP estimation
    """

    # Class constants for optimization
    DEFAULT_THIN_FACTOR = 15
    DEFAULT_MAX_PREDICTION_SAMPLES = 1000
    DEFAULT_PREDICTION_BATCH_SIZE = 100

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=None):
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)
        self.n_matches = len(goals_home)
        self._params = np.concatenate(
            (
                np.zeros(self.n_teams),  # attack_offsets
                np.zeros(self.n_teams),  # defense_offsets
                np.zeros(self.n_matches),  # match_intercept_offsets
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # mu_att, log_sig_att, mu_def, log_sig_def, log_sig_match
                [0.25],
                [-0.1],  # hfa, rho
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
            [f"attack_offset_{t}" for t in self.teams]
            + [f"defense_offset_{t}" for t in self.teams]
            + [f"match_intercept_{i}" for i in range(self.n_matches)]
            + [
                "mu_attack",
                "log_sigma_attack",
                "mu_defense",
                "log_sigma_defense",
                "log_sigma_match",
                "hfa",
                "rho",
            ]
        )

    def _unpack_params_for_pred(self, params):
        """
        Unpacks a single sample from the chain for prediction.
        Note: We only need the parameters for team strengths and the *distribution*
        of match intercepts, not the specific intercepts from the training data.
        """
        nt = self.n_teams
        nm = self.n_matches
        epsilon = 1e-6

        # We need to extract the hyperparameters to construct team strengths and the new match effect
        mu_attack = params[2 * nt + nm]
        sigma_attack = np.exp(params[2 * nt + nm + 1]) + epsilon
        mu_defense = params[2 * nt + nm + 2]
        sigma_defense = np.exp(params[2 * nt + nm + 3]) + epsilon
        sigma_match = np.exp(params[2 * nt + nm + 4]) + epsilon

        attack_offsets = params[:nt]
        defense_offsets = params[nt : 2 * nt]

        attack = mu_attack + sigma_attack * attack_offsets
        defense = mu_defense + sigma_defense * defense_offsets

        return {
            "attack": attack,
            "defense": defense,
            "sigma_match": sigma_match,
            "hfa": params[2 * nt + nm + 5],
            "rho": params[2 * nt + nm + 6],
        }

    def fit(
        self,
        n_walkers=None,
        n_steps=4000,
        n_burn=1000,
        initial_params=None,
        thin_factor=None,
    ):
        """
        Fit the hierarchical Bayesian model using MCMC sampling.

        This method uses the emcee ensemble sampler with multiprocessing to
        efficiently sample from the posterior distribution. If no initial parameters
        are provided, the method first performs MAP estimation for initialization.

        Parameters:
        -----------
        n_walkers : int, optional
            Number of MCMC walkers (default: 2 * n_parameters)
        n_steps : int, optional
            Total number of MCMC steps per walker (default: 4000)
        n_burn : int, optional
            Number of burn-in steps to discard (default: 1000)
        initial_params : array-like, optional
            Initial parameter values for MCMC chains. If None, uses MAP estimation
        thin_factor : int, optional
            Thinning factor for chain storage (default: 15)

        Notes:
        ------
        - Uses multiprocessing for parallel evaluation of log-probability
        - Chains are thinned to reduce autocorrelation and memory usage
        - Progress bar is displayed during sampling
        - Sets self.fitted = True upon completion

        Raises:
        -------
        ValueError
            If input data is invalid or optimization fails
        """
        ndim = len(self._params)
        if n_walkers is None:
            n_walkers = 2 * ndim

        args = (
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
            self.n_teams,
            self.n_matches,
        )

        if initial_params is None:

            def safe_neg_log_prob(p, *a):
                res = -hierarchical_random_intercept_log_prob(p, *a)
                return res if np.isfinite(res) else 1e12

            result = minimize(
                safe_neg_log_prob, self._params, args=args, method="L-BFGS-B"
            )
            initial_params = result.x

        pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)

        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                n_walkers,
                ndim,
                hierarchical_random_intercept_log_prob,
                args=args,
                pool=pool,
            )
            self.sampler.run_mcmc(pos, n_steps, progress=True)

        if thin_factor is None:
            thin_factor = self.DEFAULT_THIN_FACTOR

        self.chain = self.sampler.get_chain(discard=n_burn, thin=thin_factor, flat=True)
        self.fitted = True

    def _compute_probabilities(self, home_idx, away_idx, max_goals, normalize=True):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")

        param_samples = self.chain
        n_samples = len(param_samples)

        # Optimization: Subsample for faster predictions if we have many samples
        max_samples = self.DEFAULT_MAX_PREDICTION_SAMPLES
        if n_samples > max_samples:
            subsample_idx = np.random.choice(n_samples, max_samples, replace=False)
            param_samples = param_samples[subsample_idx]
            n_samples = max_samples

        total_score_matrix = np.zeros((max_goals, max_goals), dtype=np.float64)
        total_lambda_home, total_lambda_away = 0.0, 0.0

        # Pre-allocate arrays for the in-place Cython function
        sample_score_matrix = np.empty(max_goals * max_goals, dtype=np.float64)
        sample_lambda_home = np.empty(1, dtype=np.float64)
        sample_lambda_away = np.empty(1, dtype=np.float64)

        # Process in batches for memory efficiency
        batch_size = min(self.DEFAULT_PREDICTION_BATCH_SIZE, n_samples)

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_params = param_samples[i:batch_end]

            for params in batch_params:
                p = self._unpack_params_for_pred(params)

                # For a new prediction, we draw a new random intercept from the learned distribution
                new_match_intercept = np.random.normal(0, p["sigma_match"])

                home_attack = p["attack"][home_idx]
                away_attack = p["attack"][away_idx]
                home_defense = p["defense"][home_idx]
                away_defense = p["defense"][away_idx]

                compute_random_intercept_probabilities(
                    home_attack,
                    away_attack,
                    home_defense,
                    away_defense,
                    p["hfa"],
                    p["rho"],
                    new_match_intercept,
                    max_goals,
                    sample_score_matrix,
                    sample_lambda_home,
                    sample_lambda_away,
                )

                total_score_matrix += sample_score_matrix.reshape(max_goals, max_goals)
                total_lambda_home += sample_lambda_home[0]
                total_lambda_away += sample_lambda_away[0]

        avg_score_matrix = total_score_matrix / n_samples
        avg_lambda_home = total_lambda_home / n_samples
        avg_lambda_away = total_lambda_away / n_samples

        return FootballProbabilityGrid(
            avg_score_matrix, avg_lambda_home, avg_lambda_away, normalize=normalize
        )
