import numpy as np

from .bayesian_base import BaseBayesianModel
from .bayesian_loss import bayesian_hierarchical_log_prob
from .football_probability_grid import FootballProbabilityGrid
from .loss import dixon_coles_loss_function
from .probabilities import compute_posterior_predictive_hierarchical


# --- Top-level function for multiprocessing ---
def hierarchical_log_probability_wrapper(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
):
    return bayesian_hierarchical_log_prob(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )


class BayesianHierarchicalModel(BaseBayesianModel):
    """
    A hierarchical Bayesian Dixon-Coles model fitted in parallel with emcee.

    Uses Non-Centered Parameterization (NCP) for efficient sampling and
    Cythonized likelihoods for speed.
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

        # Initialize with zeros; fit() will overwrite this with smart defaults
        self._params = np.concatenate(
            (
                np.zeros(self.n_teams),  # attack_offsets
                np.zeros(self.n_teams),  # defense_offsets
                [0.0],  # mu_attack
                [0.0],  # log_sigma_attack
                [0.0],  # mu_defense
                [0.0],  # log_sigma_defense
                [0.25],  # home_advantage
                [-0.1],  # rho
            )
        )

    def _get_log_probability_function(self):
        return hierarchical_log_probability_wrapper

    def _get_initial_params(self):
        """
        Smart initialization:
        1. Fit frequentist Poisson model (fast).
        2. Calculate mean/std of those ratings.
        3. Convert ratings into Hierarchical 'offsets' so the Bayesian model
           starts exactly where the Frequentist model finished.
        """
        # 1. Get Frequentist params [attack... defense... hfa]
        freq_params = super()._get_initial_params()

        attacks = freq_params[: self.n_teams]
        defenses = freq_params[self.n_teams : 2 * self.n_teams]
        hfa = freq_params[2 * self.n_teams]

        # 2. Derive Hyperparameters (Avoid div/0 with epsilon)
        mu_att = np.mean(attacks)
        sigma_att = np.std(attacks) + 1e-6

        mu_def = np.mean(defenses)
        sigma_def = np.std(defenses) + 1e-6

        # 3. Calculate Offsets (Reverse the NCP transformation)
        # rating = mu + sigma * offset  ->  offset = (rating - mu) / sigma
        att_offsets = (attacks - mu_att) / sigma_att
        def_offsets = (defenses - mu_def) / sigma_def

        # 4. Construct the full parameter vector
        return np.concatenate(
            [
                att_offsets,
                def_offsets,
                [mu_att, np.log(sigma_att)],
                [mu_def, np.log(sigma_def)],
                [hfa],
                [-0.1],  # rho default
            ]
        )

    def _unpack_params(self, params):
        """
        Unpack params and transform offsets back to real ratings
        for inspection (e.g. in __repr__).
        """
        nt = self.n_teams
        epsilon = 1e-6

        # Slice the vector
        attack_offsets = params[:nt]
        defense_offsets = params[nt : 2 * nt]
        mu_attack = params[2 * nt]
        sigma_attack = np.exp(params[2 * nt + 1]) + epsilon
        mu_defense = params[2 * nt + 2]
        sigma_defense = np.exp(params[2 * nt + 3]) + epsilon
        hfa = params[2 * nt + 4]
        rho = params[2 * nt + 5]

        # Transform offsets back to real ratings
        attack = mu_attack + sigma_attack * attack_offsets
        defense = mu_defense + sigma_defense * defense_offsets

        return {
            "attack": attack,
            "defense": defense,
            "hfa": hfa,
            "rho": rho,
            "mu_attack": mu_attack,
            "sigma_attack": sigma_attack,
        }

    def _get_param_names(self):
        return (
            [f"attack_offset_{t}" for t in self.teams]
            + [f"defense_offset_{t}" for t in self.teams]
            + [
                "mu_attack",
                "log_sigma_attack",
                "mu_defense",
                "log_sigma_defense",
                "home_advantage",
                "rho",
            ]
        )

    def _calculate_frequentist_metrics(self):
        # Uses unpack_params, so it automatically gets the "Real" ratings
        p = self._unpack_params(self._params)
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
        self.n_params = len(self._params)
        self.loglikelihood = -neg_ll
        self.aic = 2 * neg_ll + 2 * self.n_params

    def _calculate_log_likelihood(self, params):
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

    def _compute_probabilities(self, home_idx, away_idx, max_goals, normalize=True):
        """
        Generate posterior predictive distribution using optimized Cython engine.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")

        # Call the new hierarchical-specific Cython function
        grid_matrix, lam_h, lam_a = compute_posterior_predictive_hierarchical(
            self.chain, home_idx, away_idx, self.n_teams, max_goals
        )

        return FootballProbabilityGrid(grid_matrix, lam_h, lam_a, normalize=normalize)
