from abc import abstractmethod
from multiprocessing import Pool

import emcee
import numpy as np
from scipy.optimize import minimize

from .base_model import BaseGoalsModel


class BaseBayesianModel(BaseGoalsModel):
    """
    Base class for Bayesian football prediction models.

    This class provides common functionality for Bayesian models including:
    - MCMC sampling and chain processing
    - WAIC calculation and Bayesian metrics
    - Posterior prediction averaging
    - Parameter handling and validation

    Subclasses must implement:
    - _get_log_probability_function(): Return model-specific log probability function
    - _unpack_params(): Model-specific parameter unpacking
    - _get_param_names(): Return parameter names for the model
    - _compute_probabilities(): Model-specific probability computation

    Attributes:
        sampler (emcee.EnsembleSampler): MCMC sampler object (after fitting)
        chain (np.ndarray): Posterior samples from MCMC
        waic (float): Widely Applicable Information Criterion
        lppd (float): Log pointwise predictive density
        p_waic (float): Effective number of parameters for WAIC
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=None):
        """
        Initialize the base Bayesian model.

        Parameters:
        -----------
        goals_home : array-like
            Goals scored by home teams in each match
        goals_away : array-like
            Goals scored by away teams in each match
        teams_home : array-like
            Home team identifiers for each match
        teams_away : array-like
            Away team identifiers for each match
        weights : array-like, optional
            Match weights for temporal discounting
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)
        self.sampler = None
        self.chain = None
        self.waic = None
        self.lppd = None
        self.p_waic = None

    def fit(self, n_walkers=None, n_steps=2000, n_burn=1000, initial_params=None):
        """
        Fit the Bayesian model using MCMC sampling.

        This method performs full Bayesian inference by sampling from the posterior
        distribution of model parameters using the emcee affine-invariant ensemble
        sampler.

        Parameters:
        -----------
        n_walkers : int, optional
            Number of MCMC walkers. If None, defaults to 2 * number of parameters
        n_steps : int, optional
            Total number of MCMC steps per walker. Default: 2000
        n_burn : int, optional
            Number of initial samples to discard as burn-in. Default: 1000
        initial_params : array-like, optional
            Starting parameter values for MCMC chains. If None, automatically finds
            good starting point via MAP estimation

        Returns:
        --------
        None
        """
        ndim = len(self._params)

        # Rule of thumb: use at least twice as many walkers as dimensions
        if n_walkers is None:
            n_walkers = 2 * ndim

        # Get model-specific log probability function
        log_prob_func = self._get_log_probability_function()

        if initial_params is None:
            # Find a good starting point using optimization
            neg_log_prob = lambda p, *a: -log_prob_func(p, *a)

            def safe_neg_log_prob(p, *a):
                res = neg_log_prob(p, *a)
                return res if np.isfinite(res) else 1e12

            args = self._get_log_probability_args()
            result = minimize(
                safe_neg_log_prob, self._params, args=args, method="L-BFGS-B"
            )
            initial_params = result.x

        # Initialize walkers in a small, random ball around the best-fit parameters
        pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)

        # Run MCMC sampling
        self._run_mcmc_sampling(log_prob_func, pos, n_steps)

        # Store the flattened chain, discarding the burn-in phase and thinning
        self.chain = self.sampler.get_chain(discard=n_burn, thin=15, flat=True)

        # Calculate posterior mean parameters
        self._params = np.mean(self.chain, axis=0)

        # Calculate metrics using posterior mean
        self._calculate_fit_metrics()

        self.fitted = True

    def _run_mcmc_sampling(self, log_prob_func, pos, n_steps):
        """
        Run MCMC sampling with multiprocessing.

        Parameters:
        -----------
        log_prob_func : callable
            Log probability function
        pos : array-like
            Initial walker positions
        n_steps : int
            Number of MCMC steps

        Returns:
        --------
        None
        """
        args = self._get_log_probability_args()

        with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                self._get_n_walkers(),
                len(self._params),
                log_prob_func,
                args=args,
                pool=pool,
            )
            self.sampler.run_mcmc(
                pos, n_steps, progress=False, skip_initial_state_check=True
            )

    def _get_n_walkers(self):
        """
        Get the number of walkers for MCMC sampling.

        Returns:
        --------
        int
            Number of walkers (2 * number of parameters)
        """
        return 2 * len(self._params)

    def _calculate_fit_metrics(self):
        """
        Calculate AIC, WAIC, and log-likelihood for the Bayesian model.

        This method computes both frequentist (AIC) and Bayesian (WAIC) information
        criteria to provide comprehensive model comparison capabilities. AIC uses
        posterior mean parameters for consistency with frequentist models, while
        WAIC uses the full posterior distribution for proper Bayesian model comparison.
        """
        # Calculate frequentist metrics using posterior mean
        self._calculate_frequentist_metrics()

        # Calculate WAIC using full posterior distribution
        self._calculate_waic()

    def _calculate_frequentist_metrics(self):
        """
        Calculate frequentist-style metrics (AIC, log-likelihood) using posterior mean.

        This method provides compatibility with frequentist models for comparison purposes.
        """
        # This should be implemented by subclasses to calculate model-specific
        # frequentist metrics using the posterior mean parameters
        raise NotImplementedError(
            "Subclasses must implement _calculate_frequentist_metrics"
        )

    def _calculate_waic(self):
        """
        Calculate WAIC (Widely Applicable Information Criterion) using posterior samples.

        WAIC is a more Bayesian approach to model comparison that uses the full
        posterior distribution rather than just point estimates. It's computed as:
        WAIC = -2 * (lppd - p_WAIC)
        where lppd is the log pointwise predictive density and p_WAIC is the effective
        number of parameters.
        """
        param_samples = self.chain
        n_samples = len(param_samples)
        n_matches = len(self.goals_home)

        log_lik_samples = np.zeros((n_matches, n_samples))

        for i, params in enumerate(param_samples):
            total_log_lik = self._calculate_log_likelihood(params)
            log_lik_samples[:, i] = total_log_lik / n_matches

        max_log_lik = np.max(log_lik_samples, axis=1, keepdims=True)
        shifted_log_lik = log_lik_samples - max_log_lik
        lppd = np.sum(
            np.log(np.mean(np.exp(shifted_log_lik), axis=1)) + max_log_lik.flatten()
        )

        p_waic = np.sum(np.var(log_lik_samples, axis=1))

        self.waic = -2 * (lppd - p_waic)
        self.lppd = lppd
        self.p_waic = p_waic

    def _calculate_log_likelihood(self, params):
        """
        Calculate total log-likelihood for given parameters.

        This should be implemented by subclasses to provide model-specific
        log-likelihood calculation.

        Parameters:
        -----------
        params : array-like
            Model parameters

        Returns:
        --------
        float
            Total log-likelihood
        """
        raise NotImplementedError("Subclasses must implement _calculate_log_likelihood")

    def _get_log_probability_args(self):
        """
        Get arguments for the log probability function.

        Returns:
        --------
        tuple
            Arguments to pass to the log probability function
        """
        return (
            self.home_idx,
            self.away_idx,
            self.goals_home,
            self.goals_away,
            self.weights,
            self.n_teams,
        )

    @abstractmethod
    def _get_log_probability_function(self):
        """
        Return the model-specific log probability function.

        Returns:
        --------
        callable
            Log probability function for MCMC sampling
        """
        raise NotImplementedError(
            "Subclasses must implement _get_log_probability_function"
        )

    @abstractmethod
    def _unpack_params(self, params):
        """
        Unpack the flat parameter vector into named components.

        Parameters:
        -----------
        params : array-like
            Flat parameter vector

        Returns:
        --------
        dict
            Dictionary with named parameter components
        """
        raise NotImplementedError("Subclasses must implement _unpack_params")

    @abstractmethod
    def _get_param_names(self):
        """
        Return the parameter names for this model.

        Returns:
        --------
        list[str]
            List of parameter names
        """
        raise NotImplementedError("Subclasses must implement _get_param_names")

    @abstractmethod
    def _compute_probabilities(self, home_idx, away_idx, max_goals, normalize=True):
        """
        Compute the probability grid for a fixture.

        Parameters:
        -----------
        home_idx : int
            Index of home team
        away_idx : int
            Index of away team
        max_goals : int
            Maximum goals to consider
        normalize : bool, optional
            Whether to normalize probabilities

        Returns:
        --------
        FootballProbabilityGrid
            Probability grid for the match
        """
        raise NotImplementedError("Subclasses must implement _compute_probabilities")
