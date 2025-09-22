import numpy as np
from scipy.stats import expon, norm

from .bayesian_base import BaseBayesianModel
from .football_probability_grid import FootballProbabilityGrid
from .loss import dixon_coles_loss_function
from .probabilities import compute_dixon_coles_probabilities


# --- Top-level function for multiprocessing ---
def hierarchical_log_probability(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
):
    """
    Calculates the log posterior probability for the hierarchical model.

    This function uses a non-centered parameterization. Instead of sampling
    team strengths directly, we sample standardized offsets from a standard
    normal distribution, and then scale them by a group-level standard
    deviation and shift by a group-level mean.

    Args:
        params (np.ndarray): A single array containing all model parameters
            (offsets, hyperparameters, etc.).
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
    # 1. Unpack all parameters
    attack_offsets = params[:n_teams]
    defense_offsets = params[n_teams : 2 * n_teams]
    mu_attack = params[2 * n_teams]
    log_sigma_attack = params[2 * n_teams + 1]
    mu_defense = params[2 * n_teams + 2]
    log_sigma_defense = params[2 * n_teams + 3]
    hfa = params[2 * n_teams + 4]
    rho = params[2 * n_teams + 5]

    # 2. Bounds Check
    if not (-1 < rho < 1):
        return -np.inf
    if not (-3 < mu_attack < 3 and -3 < mu_defense < 3):
        return -np.inf
    if not (-5 < log_sigma_attack < 2 and -5 < log_sigma_defense < 2):
        return -np.inf

    # 3. Priors
    # Priors for hyperparameters
    mu_attack_prior = norm.logpdf(mu_attack, 0, 1)
    mu_defense_prior = norm.logpdf(mu_defense, 0, 1)
    sigma_attack_prior = expon.logpdf(np.exp(log_sigma_attack), scale=1)
    sigma_defense_prior = expon.logpdf(np.exp(log_sigma_defense), scale=1)
    hfa_prior = norm.logpdf(hfa, 0.25, 0.5)
    rho_prior = norm.logpdf(rho, 0, 0.5)

    # Priors for the standardized offsets
    attack_offset_priors = np.sum(norm.logpdf(attack_offsets, 0, 1))
    defense_offset_priors = np.sum(norm.logpdf(defense_offsets, 0, 1))

    log_prior_prob = (
        mu_attack_prior
        + mu_defense_prior
        + sigma_attack_prior
        + sigma_defense_prior
        + hfa_prior
        + rho_prior
        + attack_offset_priors
        + defense_offset_priors
    )

    # 4. Construct Final Team Strengths (Non-Centered Parameterization)
    epsilon = 1e-6
    sigma_attack = np.exp(log_sigma_attack) + epsilon
    sigma_defense = np.exp(log_sigma_defense) + epsilon
    attack = mu_attack + sigma_attack * attack_offsets
    defense = mu_defense + sigma_defense * defense_offsets

    # 5. Log Likelihood
    neg_log_likelihood = dixon_coles_loss_function(
        goals_home, goals_away, weights, home_idx, away_idx, attack, defense, hfa, rho
    )
    if not np.isfinite(neg_log_likelihood):
        return -np.inf

    return log_prior_prob - neg_log_likelihood


class BayesianHierarchicalModel(BaseBayesianModel):
    """
    A hierarchical Bayesian Dixon-Coles model fitted in parallel with emcee.

    This model assumes that each team's attack and defense parameters are
    drawn from a common league-wide distribution. This allows for more
    robust estimates, especially for teams with fewer matches, as it
    "shrinks" estimates towards the league average.

    Attributes:
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
        Initializes the HierarchicalDixonColesModel with match data.

        Args:
            goals_home (np.ndarray): Goals scored by the home teams.
            goals_away (np.ndarray): Goals scored by the away teams.
            teams_home (np.ndarray): The home team for each match.
            teams_away (np.ndarray): The away team for each match.
            weights (np.ndarray, optional): The weight for each match.
                Defaults to None, which assigns a weight of 1 to each match.
        """
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)
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
        """
        Return the model-specific log probability function.

        Returns:
        --------
        callable
            Log probability function for MCMC sampling
        """
        return hierarchical_log_probability

    def _get_param_names(self):
        """
        Gets the ordered list of parameter names for the MCMC chain.

        Returns:
            list: A list of strings corresponding to the parameter vector.
        """
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

    def _unpack_params(self, params):
        """
        Unpacks a single sample from the chain to construct final team strengths.

        This translates the non-centered parameters (offsets and sigmas) from
        an MCMC sample into the interpretable, final attack and defense ratings.

        Args:
            params (np.ndarray): A single sample vector from the MCMC chain.

        Returns:
            dict: A dictionary of the interpretable parameters: "attack",
                  "defense", "hfa", and "rho".
        """
        nt = self.n_teams
        epsilon = 1e-6
        attack_offsets = params[:nt]
        defense_offsets = params[nt : 2 * nt]
        mu_attack = params[2 * nt]
        sigma_attack = np.exp(params[2 * nt + 1]) + epsilon
        mu_defense = params[2 * nt + 2]
        sigma_defense = np.exp(params[2 * nt + 3]) + epsilon

        attack = mu_attack + sigma_attack * attack_offsets
        defense = mu_defense + sigma_defense * defense_offsets

        return {
            "attack": attack,
            "defense": defense,
            "hfa": params[2 * nt + 4],
            "rho": params[2 * nt + 5],
        }

    def _calculate_frequentist_metrics(self):
        """
        Calculate frequentist-style metrics (AIC, log-likelihood) using posterior mean.

        This method provides compatibility with frequentist models for comparison purposes.
        """
        # Extract final team strengths from posterior mean for AIC calculation
        p = self._unpack_params(self._params)

        # Calculate negative log-likelihood using Dixon-Coles loss
        # (hierarchical model uses same likelihood, just different priors)
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

    def _calculate_log_likelihood(self, params):
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

        avg_score_matrix = total_score_matrix / n_samples
        avg_lambda_home = total_lambda_home / n_samples
        avg_lambda_away = total_lambda_away / n_samples

        return FootballProbabilityGrid(
            avg_score_matrix, avg_lambda_home, avg_lambda_away, normalize=normalize
        )

    def __repr__(self):
        """
        Return a string representation of the fitted Bayesian Hierarchical model.

        Displays both frequentist (AIC) and Bayesian (WAIC) information criteria
        for comprehensive model comparison, along with team parameters and
        hierarchical hyperparameters.
        """
        lines = [
            "Module: Penaltyblog",
            "",
            "Model: Bayesian Hierarchical Dixon-Coles",
            "",
        ]

        if not self.fitted:
            lines.append("Status: Model not fitted")
            return "\n".join(lines)

        # Ensure all required attributes are available
        assert self.aic is not None
        assert self.loglikelihood is not None
        assert self.n_params is not None
        assert hasattr(
            self, "waic"
        ), "WAIC not calculated - model fitting may have failed"

        lines.extend(
            [
                f"Number of parameters: {self.n_params}",
                f"Log Likelihood: {round(self.loglikelihood, 3)}",
                f"AIC: {round(self.aic, 3)}",
                f"WAIC: {round(self.waic, 3)}",
                f"Effective parameters (p_WAIC): {round(self.p_waic, 3)}",
                "",
                "Hyperparameters:",
                "-" * 40,
            ]
        )

        # Extract hyperparameters
        nt = self.n_teams
        epsilon = 1e-6
        attack_offsets = self._params[:nt]
        defense_offsets = self._params[nt : 2 * nt]
        mu_attack = self._params[2 * nt]
        sigma_attack = np.exp(self._params[2 * nt + 1]) + epsilon
        mu_defense = self._params[2 * nt + 2]
        sigma_defense = np.exp(self._params[2 * nt + 3]) + epsilon
        hfa = self._params[2 * nt + 4]
        rho = self._params[2 * nt + 5]

        lines.extend(
            [
                f"Attack mean (μ_attack): {round(mu_attack, 3)}",
                f"Attack std (σ_attack): {round(sigma_attack, 3)}",
                f"Defense mean (μ_defense): {round(mu_defense, 3)}",
                f"Defense std (σ_defense): {round(sigma_defense, 3)}",
                f"Home Advantage: {round(hfa, 3)}",
                f"Rho: {round(rho, 3)}",
                "",
                "Team Parameters:",
                "{0: <20} {1:<20} {2:<20}".format("Team", "Attack", "Defense"),
                "-" * 60,
            ]
        )

        # Calculate final team strengths
        attack = mu_attack + sigma_attack * attack_offsets
        defense = mu_defense + sigma_defense * defense_offsets

        for idx, team in enumerate(self.teams):
            lines.append(
                "{0: <20} {1:<20} {2:<20}".format(
                    team,
                    round(attack[idx], 3),
                    round(defense[idx], 3),
                )
            )

        return "\n".join(lines)
