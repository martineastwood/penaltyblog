import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from penaltyblog.models.custom_types import (
    GoalInput,
    ParamsOutput,
    TeamInput,
    WeightInput,
)


class BaseGoalsModel(ABC):
    """
    Base class for football prediction models.

    Provides common functionality for football prediction models, including:
      - Input validation
      - Team setup (unique team list and mapping)
      - Model persistence (save/load)
      - Abstract methods for fit and predict
    """

    def __init__(
        self,
        goals_home: GoalInput,
        goals_away: GoalInput,
        teams_home: TeamInput,
        teams_away: TeamInput,
        weights: WeightInput = None,
    ):
        # Convert inputs to numpy arrays
        self.goals_home = np.asarray(goals_home, dtype=np.int64, order="C")
        self.goals_away = np.asarray(goals_away, dtype=np.int64, order="C")
        self.teams_home = np.asarray(teams_home, dtype=str, order="C")
        self.teams_away = np.asarray(teams_away, dtype=str, order="C")

        n_matches = len(self.goals_home)

        # Process weights: if None, create an array of 1s; else, validate its length
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

        # Common attributes for fitted state
        self.fitted: bool = False
        self.aic: Optional[float] = None
        self._res: Optional[Any] = None
        self.n_params: Optional[int] = None
        self.loglikelihood: Optional[float] = None

    def _validate_inputs(self, n_matches: int):
        """Validates that all inputs have consistent dimensions and values."""
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

        if len(self.weights) != n_matches:
            raise ValueError(
                "Weights array must have the same length as the number of matches."
            )

        if self.teams_home.size == 0 or self.teams_away.size == 0:
            raise ValueError("Team arrays must not be empty.")

    def _setup_teams(self):
        """Set up unique teams and mappings for fast lookup."""
        self.teams = np.sort(
            np.unique(np.concatenate([self.teams_home, self.teams_away]))
        )
        self.n_teams = len(self.teams)
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        self.home_idx = np.array(
            [self.team_to_idx[t] for t in self.teams_home], dtype=np.int64, order="C"
        )
        self.away_idx = np.array(
            [self.team_to_idx[t] for t in self.teams_away], dtype=np.int64, order="C"
        )

    def save(self, filepath: str):
        """
        Saves the model to a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file where the model will be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> Any:
        """
        Loads a model from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the model will be loaded.

        Returns
        -------
        Any
            An instance of the model.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def fit(self):
        """
        Fits the model to the data.

        Must be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def predict(self, home_team: str, away_team: str, max_goals: int = 15):
        """
        Predicts the probability of each scoreline for a given match.

        Must be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Returns the fitted parameters of the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def params(self) -> Dict[str, Any]:
        """
        Property to retrieve the fitted model parameters.
        Same as `get_params()`, but allows attribute-like access.

        Returns
        -------
        dict
            A dictionary containing attack, defense, home advantage, and correlation parameters.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        return self.get_params()
