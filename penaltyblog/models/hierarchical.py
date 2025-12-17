import numpy as np

from .bayesian_base import BaseBayesianModel
from .bayesian_loss import bayesian_hierarchical_log_prob
from .football_probability_grid import FootballProbabilityGrid
from .loss import dixon_coles_loss_function
from .probabilities import compute_posterior_predictive_hierarchical


# --- Top-level function for multiprocessing ---
def hierarchical_log_probability_wrapper(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
) -> float:
    """
    Top-level wrapper for hierarchical log probability to support multiprocessing.

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
    return bayesian_hierarchical_log_prob(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )


class BayesianHierarchicalModel(BaseBayesianModel):
    """
    A hierarchical Bayesian Dixon-Coles model for football match prediction.

    This model extends the Dixon-Coles model by adding a hierarchical structure
    to team attack and defense ratings. Ratings are assumed to be drawn from
    common distributions (hyperpriors), which helps in regularizing the estimates,
    especially for teams with few matches in the dataset by "borrowing strength"
    from the rest of the league.

    The model uses Non-Centered Parameterization (NCP) to improve sampling
    efficiency by reducing dependencies between parameters and hyperparameters.
    It is fitted using the emcee MCMC sampler with Cythonized likelihoods for speed.

    Mathematical Model:
    ------------------
    The model assumes goals follow independent Poisson distributions with rates
    adjusted by the Dixon-Coles correlation (rho):

    λ_home = exp(attack_home - defense_away + hfa)
    λ_away = exp(attack_away - defense_home)

    The hierarchical structure is defined as:
    - attack_i = mu_attack + sigma_attack * attack_offset_i
    - defense_i = mu_defense + sigma_defense * defense_offset_i

    Priors:
    -------
    - attack_offset_i ~ N(0, 1)
    - defense_offset_i ~ N(0, 1)
    - mu_attack ~ N(0, 1)
    - mu_defense ~ N(0, 1)
    - sigma_attack ~ Exp(1)
    - sigma_defense ~ Exp(1)
    - hfa ~ N(0.25, 0.5²)
    - rho ~ N(0, 0.5²)

    Attributes:
    -----------
    n_teams : int
        Number of unique teams in the dataset
    teams : list
        List of team names in the order they appear in parameters
    sampler : emcee.EnsembleSampler
        The MCMC sampler object (available after calling .fit())
    chain : np.ndarray
        Posterior samples from MCMC (available after calling .fit())

    Examples:
    ---------
    >>> model = BayesianHierarchicalModel(
    ...     goals_home=[2, 1, 0, 3],
    ...     goals_away=[1, 1, 2, 0],
    ...     teams_home=['Arsenal', 'Chelsea', 'Liverpool', 'ManCity'],
    ...     teams_away=['Tottenham', 'Arsenal', 'Chelsea', 'Liverpool']
    ... )
    >>> model.fit(n_steps=2000, n_burn=1000)
    >>> probs = model.predict_match('Arsenal', 'Chelsea')
    >>> print(f"Arsenal win probability: {probs.home_win_prob:.3f}")
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
        Initialize the Bayesian Hierarchical model.

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
            Match weights for temporal discounting. Default: None
        """
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

    def _get_log_probability_function(self) -> callable:
        """
        Return the log probability function for the hierarchical model.

        Returns:
        --------
        callable
            Hierarchical log probability wrapper function
        """
        return hierarchical_log_probability_wrapper

    def _get_initial_params(self) -> np.ndarray:
        """
        Find smart starting parameters for MCMC sampling.

        This method fits a frequentist Poisson model first, then uses those
        ratings to initialize the hierarchical model's offsets and
        hyperparameters. This ensures the Bayesian model starts near the
        maximum likelihood estimate.

        Returns:
        --------
        np.ndarray
            Initial parameter vector containing attack/defense offsets,
            hyperparameters (mu/log_sigma), home advantage, and rho.
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

    def _unpack_params(self, params) -> dict[str, np.ndarray]:
        """
        Unpack the flat parameter vector and transform offsets back to real ratings.

        The hierarchical model samples in 'offset' space (NCP). This method
        converts those offsets back to the actual attack and defense ratings
        used in the Dixon-Coles likelihood.

        Parameters:
        -----------
        params : np.ndarray
            Flat parameter vector from the sampler

        Returns:
        --------
        dict
            Dictionary containing 'attack', 'defense', 'hfa', 'rho',
            'mu_attack', and 'sigma_attack'.
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
            "mu_defense": mu_defense,
            "sigma_defense": sigma_defense,
        }

    def _get_param_names(self) -> list[str]:
        """
        Return the names of all parameters in the model.

        Returns:
        --------
        list[str]
            List of parameter names including offsets and hyperparameters
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

    def _calculate_frequentist_metrics(self):
        """
        Calculate frequentist-style metrics (AIC, Log-Likelihood) using posterior mean.
        """
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

    def _calculate_log_likelihood(self, params) -> float:
        """
        Calculate total log-likelihood for a given parameter set.

        Parameters:
        -----------
        params : np.ndarray
            Model parameters

        Returns:
        --------
        float
            Log-likelihood value
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
        Generate posterior predictive distribution for a match fixture.

        Uses an optimized Cython engine to integrate over the posterior chain
        and produce a probability grid for the match score.

        Parameters:
        -----------
        home_idx : int
            Index of the home team
        away_idx : int
            Index of the away team
        max_goals : int
            Maximum number of goals to compute probabilities for
        normalize : bool, optional
            Whether to normalize the probability grid. Default: True

        Returns:
        --------
        FootballProbabilityGrid
            The computed probability grid for the match
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")

        # Call the new hierarchical-specific Cython function
        grid_matrix, lam_h, lam_a = compute_posterior_predictive_hierarchical(
            self.chain, home_idx, away_idx, self.n_teams, max_goals
        )

        return FootballProbabilityGrid(grid_matrix, lam_h, lam_a, normalize=normalize)
