from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from ..matchflow.flow import Flow


def _validate_range(name: str, value: Tuple[float, float]) -> None:
    low, high = value
    if pd.isna(low) or pd.isna(high):
        raise ValueError(f"{name} cannot contain NaN")
    if high <= low:
        raise ValueError(f"{name} must be an increasing tuple (min, max)")


def _scale_to_100(series: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric - low) * 100.0 / (high - low)


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

    events: pd.DataFrame | Flow
    x: str
    y: str
    event_type: str
    end_x: Optional[str] = None
    end_y: Optional[str] = None
    is_success: Optional[str] = None
    x_range: Tuple[float, float] = (0.0, 100.0)
    y_range: Tuple[float, float] = (0.0, 100.0)

    def __post_init__(self) -> None:
        events_df = _coerce_events_dataframe(self.events)
        object.__setattr__(self, "events", events_df)

        required = [self.x, self.y, self.event_type]
        _validate_columns(events_df, required)

        if (self.end_x is None) != (self.end_y is None):
            raise ValueError("Both or neither end_x and end_y must be provided")

        optional = [
            c for c in [self.end_x, self.end_y, self.is_success] if c is not None
        ]
        if optional:
            _validate_columns(events_df, optional)

        _validate_range("x_range", self.x_range)
        _validate_range("y_range", self.y_range)

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

        x_low, x_high = self.x_range
        y_low, y_high = self.y_range
        result["x"] = _scale_to_100(result["x"], x_low, x_high)
        result["end_x"] = _scale_to_100(result["end_x"], x_low, x_high)
        result["y"] = _scale_to_100(result["y"], y_low, y_high)
        result["end_y"] = _scale_to_100(result["end_y"], y_low, y_high)

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
            x_range=(0.0, 100.0),
            y_range=(0.0, 100.0),
        )


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_events_dataframe(events: object) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events
    from ..matchflow.flow import Flow

    if isinstance(events, Flow):
        return events.to_pandas()
    raise TypeError("events must be a pandas DataFrame or penaltyblog.matchflow.Flow")
