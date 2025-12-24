import numpy as np

from .bayesian_base import BaseBayesianModel
from .bayesian_loss import bayesian_hierarchical_log_prob
from .football_probability_grid import FootballProbabilityGrid
from .loss import dixon_coles_loss_function
from .probabilities import compute_posterior_predictive_dixon_coles


def hierarchical_log_probability_wrapper(
    params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
) -> float:
    """
    Wrapper to catch NaN returns from the Cython hierarchical log prob function.
    """
    val = bayesian_hierarchical_log_prob(
        params, home_idx, away_idx, goals_home, goals_away, weights, n_teams
    )

    # The NaN Barrier: If math failed (overflow/underflow), reject the sample.
    if not np.isfinite(val):
        return -np.inf

    return val


class BayesianHierarchicalGoalModel(BaseBayesianModel):
    """
    Hierarchical Bayesian Dixon-Coles model.
    Learns the standard deviation (spread) of team strengths automatically.

    Structure:
    - Attack = Raw_Attack * Sigma_Attack
    - Defense = Raw_Defense * Sigma_Defense
    - Raw parameters are sampled from Normal(0, 1)
    - Sigmas are learned from the data
    """

    def __init__(self, goals_home, goals_away, teams_home, teams_away, weights=None):
        super().__init__(goals_home, goals_away, teams_home, teams_away, weights)

        # Initialize internal params structure for the base class
        # This is just a placeholder to set the length; fit() will overwrite it
        # Size: 2*n (Raw) + 2 (HFA, Rho) + 2 (Log Sigmas)
        self._params = np.zeros(2 * self.n_teams + 4)

    def _get_log_probability_function(self):
        return hierarchical_log_probability_wrapper

    def _get_n_walkers(self):
        # We have slightly more parameters, so we need more walkers
        return 8 * len(self._params)

    def _get_param_names(self) -> list[str]:
        """
        Return the parameter names corresponding to the MCMC chain.
        Note: The chain contains 'Raw' parameters and 'Log Sigmas'.
        """
        names = []
        # Raw Attack
        names.extend([f"raw_attack_{t}" for t in self.teams])
        # Raw Defense
        names.extend([f"raw_defense_{t}" for t in self.teams])
        # Global
        names.extend(["home_advantage", "rho"])
        # Hyperparameters
        names.extend(["log_sigma_attack", "log_sigma_defense"])
        return names

    def fit(self, **kwargs):
        # 1. Get frequentist params to initialize smartly
        freq_params = self._get_initial_params()  # [Att | Def | HFA | Rho]

        # 2. Estimate initial sigmas based on frequentist spread
        init_sigma_att = np.std(freq_params[: self.n_teams])
        init_sigma_def = np.std(freq_params[self.n_teams : 2 * self.n_teams])

        # Avoid zero division if freq model is weird
        if init_sigma_att < 0.1:
            init_sigma_att = 0.5
        if init_sigma_def < 0.1:
            init_sigma_def = 0.5

        # 3. Construct Hierarchical Initial Position
        # Transform Real -> Raw (Raw = Real / Sigma)
        raw_att = freq_params[: self.n_teams] / init_sigma_att
        raw_def = freq_params[self.n_teams : 2 * self.n_teams] / init_sigma_def
        hfa = freq_params[2 * self.n_teams]
        rho = -0.1

        # Pack into the structure expected by _bayesian_hierarchical_log_prob_c
        hierarchical_params = np.concatenate(
            [
                raw_att,
                raw_def,
                [hfa, rho],
                [np.log(init_sigma_att), np.log(init_sigma_def)],
            ]
        )

        # Update internal params so base class knows the dimensions
        self._params = hierarchical_params

        # Call parent fit
        super().fit(initial_params=hierarchical_params, **kwargs)

    def _unpack_params(self, params):
        """
        Converts the internal 'Raw' parameters to user-friendly 'Real' parameters.
        Used for __repr__ and AIC calculations.
        """
        n = self.n_teams

        # Extract Log Sigmas
        log_sigma_att = params[2 * n + 2]
        log_sigma_def = params[2 * n + 3]
        sigma_att = np.exp(log_sigma_att)
        sigma_def = np.exp(log_sigma_def)

        # Extract Raw
        raw_att = params[:n]
        raw_def = params[n : 2 * n]

        # Transform to Real
        real_att = raw_att * sigma_att
        real_def = raw_def * sigma_def

        return {
            "attack": real_att,
            "defense": real_def,
            "hfa": params[2 * n],
            "rho": params[2 * n + 1],
            "sigma_att": sigma_att,
            "sigma_def": sigma_def,
        }

    def _calculate_frequentist_metrics(self):
        """
        Calculates AIC based on the transformed (Real) parameters.
        """
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
        Generate the posterior predictive distribution.
        Transforms the chain from Raw to Real before passing to Cython helper.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")

        # 1. Get the Raw Chain
        chain = self.chain
        n_samples = chain.shape[0]
        n = self.n_teams

        # 2. Extract Sigmas (N, 1)
        sigma_att = np.exp(chain[:, 2 * n + 2])[:, np.newaxis]
        sigma_def = np.exp(chain[:, 2 * n + 3])[:, np.newaxis]

        # 3. Create temp Real chain for Cython
        # Cython expects: [Att | Def | HFA | Rho]
        real_chain = np.zeros((n_samples, 2 * n + 2))

        # 4. Transform Raw -> Real
        real_chain[:, :n] = chain[:, :n] * sigma_att
        real_chain[:, n : 2 * n] = chain[:, n : 2 * n] * sigma_def

        # 5. Copy Global Params
        real_chain[:, 2 * n] = chain[:, 2 * n]  # HFA
        real_chain[:, 2 * n + 1] = chain[:, 2 * n + 1]  # Rho

        # 6. Pass to Cython
        grid_matrix, lam_h, lam_a = compute_posterior_predictive_dixon_coles(
            real_chain, home_idx, away_idx, n, max_goals
        )

        return FootballProbabilityGrid(
            grid_matrix,
            lam_h,
            lam_a,
            normalize=normalize,
        )
