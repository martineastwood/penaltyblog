import pickle
import platform
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from penaltyblog.models.custom_types import (
    GoalInput,
    TeamInput,
    WeightInput,
)


# Platform-specific integer type for Cython compatibility
def _get_cython_long_dtype():
    """
    Get the correct NumPy dtype that matches Cython's 'long' type.

    On Windows, Cython's 'long' is 32-bit (int32).
    On Unix/Linux/macOS, Cython's 'long' is 64-bit (int64).
    """
    if platform.system() == "Windows":
        return np.int32
    else:
        return np.int64


class BaseGoalsModel(ABC):
    """
    Base class for football (soccer) goals prediction models.

    This abstract base class provides:
      - Input validation and conversion to NumPy arrays
      - Automatic team indexing for efficient lookups
      - Optional weighting of historical matches
      - Save/load persistence methods
      - A common interface for fitting and predicting
      - AIC and log-likelihood calculation after fitting

    Subclasses must implement:
      - `fit()`
      - `_compute_probabilities()`
      - `_get_param_names()`
    """

    def __init__(
        self,
        goals_home: GoalInput,
        goals_away: GoalInput,
        teams_home: TeamInput,
        teams_away: TeamInput,
        weights: WeightInput = None,
    ):
        """
        Initialise the BaseGoalsModel with match data.

        Parameters
        ----------
        goals_home : GoalInput
            Goals scored by the home team in each match.
        goals_away : GoalInput
            Goals scored by the away team in each match.
        teams_home : TeamInput
            Names of home teams for each match.
        teams_away : TeamInput
            Names of away teams for each match.
        weights : WeightInput, optional
            Match weights (e.g., from time decay). If None, all matches are weighted equally.

        Raises
        ------
        ValueError
            If the weight array length does not match the number of matches.
        """
        # Use platform-specific integer type for Cython compatibility
        cython_long_dtype = _get_cython_long_dtype()
        self.goals_home = np.asarray(goals_home, dtype=cython_long_dtype, order="C")
        self.goals_away = np.asarray(goals_away, dtype=cython_long_dtype, order="C")
        self.teams_home = np.asarray(teams_home, dtype=str, order="C")
        self.teams_away = np.asarray(teams_away, dtype=str, order="C")

        n_matches = len(self.goals_home)

        if weights is None:
            self.weights = np.ones(n_matches, dtype=np.double, order="C")
        else:
            self.weights = np.asarray(weights, dtype=np.double, order="C")
            if len(self.weights) != n_matches:
                raise ValueError(
                    "Weights array must have the same length as the number of matches."
                )

        self._validate_inputs(n_matches)
        self._setup_teams()

        self.fitted: bool = False
        self.aic: Optional[float] = None
        self._res: Optional[Any] = None
        self.n_params: Optional[int] = None
        self.loglikelihood: Optional[float] = None

    def _validate_inputs(self, n_matches: int):
        """
        Validate that input arrays have consistent lengths and valid values.

        Parameters
        ----------
        n_matches : int
            Number of matches provided.

        Raises
        ------
        ValueError
            If input lengths are inconsistent, goal counts are negative,
            or team arrays are empty.
        """
        if not (
            len(self.goals_away)
            == len(self.teams_home)
            == len(self.teams_away)
            == n_matches
        ):
            raise ValueError(
                "Input arrays for goals and teams must all have the same length."
            )
        if (self.goals_home < 0).any() or (self.goals_away < 0).any():
            raise ValueError("Goal counts must be non-negative.")
        if self.teams_home.size == 0 or self.teams_away.size == 0:
            raise ValueError("Team arrays must not be empty.")

    def _setup_teams(self):
        """
        Build team lookup structures.

        Creates:
          - A sorted list of unique teams
          - Mapping from team names to integer indices
          - Arrays of home/away indices for each match
        """
        self.teams = np.sort(
            np.unique(np.concatenate([self.teams_home, self.teams_away]))
        )
        self.n_teams = len(self.teams)
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        # Use platform-specific integer type for Cython compatibility
        cython_long_dtype = _get_cython_long_dtype()
        self.home_idx = np.array(
            [self.team_to_idx[t] for t in self.teams_home],
            dtype=cython_long_dtype,
            order="C",
        )
        self.away_idx = np.array(
            [self.team_to_idx[t] for t in self.teams_away],
            dtype=cython_long_dtype,
            order="C",
        )

    def save(self, filepath: str):
        """
        Save the fitted model to disk via pickle.

        Parameters
        ----------
        filepath : str
            File path to save to.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> Any:
        """
        Load a model from a pickle file.

        Parameters
        ----------
        filepath : str
            File path to load from.

        Returns
        -------
        Any
            Loaded model instance.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _fit(
        self,
        loss_function,
        params,
        constraints,
        bounds,
        minimizer_options,
        jac=None,
    ):
        """
        Optimise model parameters using `scipy.optimize.minimize`.

        Also sets:
          - `self.fitted`
          - `self.aic`
          - `self.loglikelihood`
          - `self._params`

        Parameters
        ----------
        loss_function : callable
            Function to minimise (e.g., negative log-likelihood).
        params : np.ndarray
            Initial parameter guess.
        constraints : dict or sequence of dict
            Optimisation constraints.
        bounds : sequence
            Parameter bounds.
        minimizer_options : dict
            Extra arguments passed to the optimiser.
        jac : callable, optional
            Gradient (Jacobian) function. If provided, may speed optimisation.

        Raises
        ------
        ValueError
            If optimisation fails.
        """
        options = {"maxiter": 1000, "disp": False}
        if minimizer_options is not None:
            options.update(minimizer_options)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._res = minimize(
                loss_function,
                params,
                jac=jac,
                constraints=constraints,
                bounds=bounds,
                options=options,
            )

        if not self._res.success:
            raise ValueError(f"Optimization failed with message: {self._res.message}")

        self._params = self._res["x"]
        self.n_params = len(self._params)
        self.loglikelihood = self._res["fun"] * -1
        self.aic = -2 * self.loglikelihood + 2 * self.n_params
        self.fitted = True

    @abstractmethod
    def fit(self, minimizer_options: Optional[dict] = None):
        """
        Fit the model to training data.

        Parameters
        ----------
        minimizer_options : dict, optional
            Options to pass to the optimiser.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _predict(self, home_team: str, away_team: str) -> tuple[int, int]:
        """
        Perform checks before predicting and return team indices.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.

        Returns
        -------
        tuple[int, int]
            Indices of home and away teams.

        Raises
        ------
        ValueError
            If model is not fitted or teams were not seen in training.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Both teams must have been in the training data.")

        home_idx = self.team_to_idx[home_team]
        away_idx = self.team_to_idx[away_team]
        return home_idx, away_idx

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 15,
        normalize: bool = True,
    ):
        """
        Predict outcome probabilities for a given fixture.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.
        max_goals : int, default 15
            Maximum goals per team to consider.
        normalize : bool, default True
            Whether to normalise the probability grid.

        Returns
        -------
        FootballProbabilityGrid
            Probability grid object for the match.
        """
        home_idx, away_idx = self._predict(home_team, away_team)
        return self._compute_probabilities(home_idx, away_idx, max_goals, normalize)

    @abstractmethod
    def _compute_probabilities(
        self, home_idx: int, away_idx: int, max_goals: int, normalize: bool = True
    ):
        """
        Compute the probability grid for a fixture.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_param_names(self) -> list[str]:
        """
        Return the parameter names for this model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_params(self) -> Dict[str, Any]:
        """
        Get the fitted model parameters as a dictionary.

        Returns
        -------
        dict
            Parameter names mapped to fitted values.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        param_names = self._get_param_names()
        return dict(zip(param_names, self._params))

    @property
    def params(self) -> Dict[str, Any]:
        """
        Fitted parameters as a property.

        Equivalent to `.get_params()`.

        Returns
        -------
        dict
            Parameter names mapped to fitted values.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        return self.get_params()

    @property
    def params_array(self) -> np.ndarray:
        """
        Return a read-only copy of the fitted parameter vector.

        This provides direct access to the underlying numpy array for
        downstream tools that need to perform numerical operations on
        the parameters (e.g., applying external adjustment factors).

        Returns
        -------
        np.ndarray
            Copy of the internal parameter array. Modifications do not
            affect the model.

        Raises
        ------
        ValueError
            If the model is not fitted.

        Notes
        -----
        Parameter layout: [attack_0, ..., attack_{n-1}, defense_0, ...,
                           defense_{n-1}, <model_specific_params>]

        Model-specific trailing parameters (use `param_indices()` for positions):

        - Poisson: [home_advantage]
        - DixonColes: [home_advantage, rho]
        - NegativeBinomial: [home_advantage, dispersion]
        - ZeroInflatedPoisson: [home_advantage, zero_inflation]
        - BivariatePoisson: [home_advantage, correlation]
        - WeibullCopula: [home_advantage, shape, kappa]

        Examples
        --------
        >>> model.fit()
        >>> p = model.params_array
        >>> home_attack = p[model.team_to_idx["Arsenal"]]
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")
        return self._params.copy()

    def param_indices(self) -> Dict[str, Any]:
        """
        Return indices for named parameter groups in the parameter array.

        This method provides a stable API for accessing parameter positions,
        enabling downstream code to work with model parameters without
        relying on internal implementation details.

        Returns
        -------
        dict
            Keys are parameter group names, values are indices or slices:
            - 'attack': slice for attack parameters (0 to n_teams)
            - 'defense': slice for defense parameters (n_teams to 2*n_teams)
            - Model-specific keys for trailing parameters (e.g., 'home_advantage', 'rho')

        Raises
        ------
        ValueError
            If the model is not fitted.

        Examples
        --------
        >>> model.fit()
        >>> idx = model.param_indices()
        >>> attacks = model.params_array[idx['attack']]
        >>> hfa = model.params_array[idx['home_advantage']]
        """
        if not self.fitted:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

        n = self.n_teams
        base_indices: Dict[str, Any] = {
            "attack": slice(0, n),
            "defense": slice(n, 2 * n),
        }
        base_indices.update(self._get_tail_param_indices())
        return base_indices

    @abstractmethod
    def _get_tail_param_indices(self) -> Dict[str, int]:
        """
        Return indices for model-specific trailing parameters.

        Subclasses must implement this to document their parameter layout.

        Returns
        -------
        dict
            Parameter names mapped to their index in the params array.
            Use negative indices for trailing parameters (e.g., -1, -2).
        """
        raise NotImplementedError("Subclasses must implement this method.")
