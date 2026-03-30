from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class XTData:
    """
    Provider-agnostic schema wrapper for xT event data.

    Wraps a DataFrame with column name mappings so that ``XTModel``
    can work with any provider's column naming conventions.

    The ``is_success`` column has a consistent meaning across event types:
    for moves (passes, carries, etc.) it means the action was completed
    successfully; for shots it means a goal was scored.

    Canonical event types
    ---------------------
    Move families: ``pass``, ``carry``, ``throw_in``, ``goal_kick``,
    ``corner``, ``free_kick``

    Shot families: ``shot``, ``free_kick_shot``
    """

    events: pd.DataFrame
    x: str
    y: str
    event_type: str
    end_x: Optional[str] = None
    end_y: Optional[str] = None
    is_success: Optional[str] = None

    def __post_init__(self) -> None:
        required = [self.x, self.y, self.event_type]
        _validate_columns(self.events, required)

        if (self.end_x is None) != (self.end_y is None):
            raise ValueError("Both or neither end_x and end_y must be provided")

        optional = [
            c for c in [self.end_x, self.end_y, self.is_success] if c is not None
        ]
        if optional:
            _validate_columns(self.events, optional)

    @property
    def df(self) -> pd.DataFrame:
        """Return a normalized DataFrame with canonical column names."""
        return self._normalized_df()

    def _normalized_df(self) -> pd.DataFrame:
        rename: dict[str, str] = {
            self.x: "x",
            self.y: "y",
            self.event_type: "event_type",
        }
        if self.end_x is not None:
            rename[self.end_x] = "end_x"
        if self.end_y is not None:
            rename[self.end_y] = "end_y"
        if self.is_success is not None:
            rename[self.is_success] = "is_success"

        df = self.events.rename(columns=rename, errors="ignore")
        keep = list(rename.values())
        result = df[keep].copy()

        for col, fill in [
            ("end_x", float("nan")),
            ("end_y", float("nan")),
            ("is_success", True),
        ]:
            if col not in result.columns:
                result[col] = fill

        return result

    def map_events(
        self,
        event_map: Optional[Dict[str, str]] = None,
        success_map: Optional[Dict[str, bool]] = None,
    ) -> "XTData":
        """
        Return a new XTData with raw labels mapped to canonical labels.

        Parameters
        ----------
        event_map : dict, optional
            Mapping from raw event_type values to canonical types
            (e.g. ``{"Pass": "pass", "Shot": "shot", "Throw-in": "throw_in",
            "Corner kick": "corner"}``).
        success_map : dict, optional
            Mapping from raw is_success values to booleans
            (e.g. ``{"Complete": True, "Incomplete": False, "Goal": True,
            "Saved": False}``).
        """
        df = self._normalized_df()
        if event_map:
            df["event_type"] = df["event_type"].map(event_map).fillna(df["event_type"])
        if success_map:
            df["is_success"] = (
                df["is_success"].map(success_map).fillna(df["is_success"])
            )

        return XTData(
            events=df,
            x="x",
            y="y",
            event_type="event_type",
            end_x="end_x",
            end_y="end_y",
            is_success="is_success",
        )


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
