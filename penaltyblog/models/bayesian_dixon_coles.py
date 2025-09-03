from multiprocessing import Pool

import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)

from .loss import dixon_coles_loss_function
from .probabilities import (
    compute_dixon_coles_probabilities,
)


# --- Top-level function for multiprocessing ---
def log_probability(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
):
    """
    Calculates the log posterior probability for a given set of parameters.

    This is the target function for the MCMC sampler. It is defined at the
    top level so it can be pickled for multiprocessing. The log posterior is
    the sum of the log prior and the log likelihood.

    Args:
        params (np.ndarray): A single array containing all model parameters
            (attack, defense, hfa, rho).
        home_idx (np.ndarray): Integer indices for the home teams.
        away_idx (np.ndarray): Integer indices for the away teams.
        goals_home (np.ndarray): Goals scored by the home teams.
        goals_away (np.ndarray): Goals scored by the away teams.
        weights (np.ndarray): Weight for each match.
        n_teams (int): The total number of unique teams.

    Returns:
        float: The log posterior probability. Returns -inf if parameters
               are outside the defined bounds or if the likelihood is not finite.
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
    A regularized Bayesian Dixon-Coles model fitted in parallel with the emcee MCMC sampler.

    This model estimates team strengths (attack and defense) and other parameters
    (home-field advantage, goal correlation) by sampling from the posterior
    distribution using MCMC. This provides a robust estimation of uncertainty
    in the parameters.

    Attributes:
        goals_home (np.ndarray): Goals scored by the home teams for each match.
        goals_away (np.ndarray): Goals scored by the away teams for each match.
        teams_home (np.ndarray): Names of the home teams.
        teams_away (np.ndarray): Names of the away teams.
        weights (np.ndarray, optional): Weight for each match. Defaults to 1.
        sampler (emcee.EnsembleSampler): The MCMC sampler instance.
        chain (np.ndarray): The flattened, burned, and thinned MCMC chain of
            posterior samples.
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
        Initializes the BayesianDixonColesModel with match data.

        Args:
            goals_home (np.ndarray): Goals scored by the home teams.
            goals_away (np.ndarray): Goals scored by the away teams.
            teams_home (np.ndarray): The home team for each match.
            teams_away (np.ndarray): The away team for each match.
            weights (np.ndarray, optional): The weight for each match.
                Defaults to None, which assigns a weight of 1 to each match.
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
        Unpacks the flat parameter vector into a dictionary.

        Args:
            params (np.ndarray): The flat parameter vector.

        Returns:
            dict: A dictionary of named parameters.
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
        Fit the model using the emcee MCMC sampler in parallel.

        This method runs an MCMC simulation to sample from the posterior
        distribution of the model parameters. It begins by finding a good
        starting point via optimization (Maximum A Posteriori) and then
        initializes a set of "walkers" to explore the parameter space.

        Args:
            n_walkers (int, optional): The number of MCMC walkers. If None,
                defaults to twice the number of dimensions.
            n_steps (int, optional): The number of steps for each walker to
                take. Defaults to 2000.
            n_burn (int, optional): The number of "burn-in" steps to discard
                from the beginning of the chain. Defaults to 1000.
            initial_params (np.ndarray, optional): A specific starting point
                for the parameters. If None, a starting point is found via
                optimization. Defaults to None.
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
            self.sampler.run_mcmc(pos, n_steps, progress=False)

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
