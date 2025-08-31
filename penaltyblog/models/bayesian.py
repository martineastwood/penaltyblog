import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# --- Assumed Imports from your package ---
# These classes and functions should be available from your existing package structure.
from penaltyblog.models.base_model import BaseGoalsModel
from penaltyblog.models.football_probability_grid import (
    FootballProbabilityGrid,
)

from .loss import dixon_coles_loss_function
from .probabilities import (
    compute_dixon_coles_probabilities,
)


class EmceeDixonColesModel(BaseGoalsModel):
    """
    A regularized Bayesian Dixon-Coles model fitted with the emcee MCMC sampler.
    This provides robust Bayesian inference with minimal, easy-to-install dependencies.
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        weights=None,
    ):
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
        return (
            [f"attack_{t}" for t in self.teams]
            + [f"defense_{t}" for t in self.teams]
            + ["home_advantage", "rho"]
        )

    def _unpack_params(self, params):
        """Unpacks the parameter vector."""
        nt = self.n_teams
        return {
            "attack": params[:nt],
            "defense": params[nt : 2 * nt],
            "hfa": params[2 * nt],
            "rho": params[2 * nt + 1],
        }

    def _log_probability(self, params):
        """
        This is the function emcee will sample. It returns the log-probability
        (log-prior + log-likelihood).
        """
        p = self._unpack_params(params)

        # --- Define Bounds ---
        # emcee works by returning -inf for parameter sets outside the bounds.
        if not (0 < p["hfa"] < 1 and -1 < p["rho"] < 1):
            return -np.inf
        if np.any(p["attack"] < -5) or np.any(p["attack"] > 5):
            return -np.inf
        if np.any(p["defense"] < -5) or np.any(p["defense"] > 5):
            return -np.inf

        # --- Log Priors ---
        attack_priors = np.sum(norm.logpdf(p["attack"], 0, 1.0))
        defense_priors = np.sum(norm.logpdf(p["defense"], 0, 1.0))
        hfa_prior = norm.logpdf(p["hfa"], 0.25, 0.5)
        rho_prior = norm.logpdf(p["rho"], 0, 0.5)
        log_prior_prob = attack_priors + defense_priors + hfa_prior + rho_prior

        # --- Log Likelihood ---
        neg_log_likelihood = dixon_coles_loss_function(
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

        # If the loss function returns a non-finite value, the parameters are invalid.
        if not np.isfinite(neg_log_likelihood):
            return -np.inf

        log_likelihood = -neg_log_likelihood

        return log_prior_prob + log_likelihood

    def fit(self, n_walkers=None, n_steps=2000, n_burn=500, initial_params=None):
        """
        Fit the model using the emcee MCMC sampler.
        """
        ndim = len(self._params)

        if n_walkers is None:
            n_walkers = 2 * ndim

        if initial_params is None:
            # Find a good starting point using a quick optimization to find the MAP
            neg_log_prob = lambda p: -self._log_probability(p)

            def safe_neg_log_prob(p):
                res = neg_log_prob(p)
                return res if np.isfinite(res) else 1e12

            result = minimize(safe_neg_log_prob, self._params, method="L-BFGS-B")
            initial_params = result.x

        # Initialize walkers in a small, random ball around the best-fit parameters
        pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)

        # Set up the sampler
        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, self._log_probability)

        # Run the MCMC sampler
        self.sampler.run_mcmc(pos, n_steps, progress=False)

        # Store the flattened chain, discarding the burn-in phase and thinning
        self.chain = self.sampler.get_chain(discard=n_burn, thin=15, flat=True)
        self.fitted = True

    def _compute_probabilities(self, home_idx, away_idx, max_goals, normalize=True):
        """
        Generate the posterior predictive distribution by averaging predictions
        from the MCMC samples.
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
