"""Provider-agnostic xT event schema and data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..matchflow.flow import Flow


def _validate_range(name: str, value: tuple[float, float]) -> None:
    low, high = value
    if pd.isna(low) or pd.isna(high):
        raise ValueError(f"{name} cannot contain NaN")
    if high <= low:
        raise ValueError(f"{name} must be an increasing tuple (min, max)")


def _scale_to_100(series: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric - low) * 100.0 / (high - low)


@dataclass(frozen=True)
class XTEventSchema:
    """
    Column/range/label mapping for provider xT event data.

    This is the primary schema object used by
    :class:`~penaltyblog.xt.XTModel`
    in ``fit(..., schema=...)`` and ``score(..., schema=...)``.
    """

    x: str = "x"
    y: str = "y"
    event_type: str = "event_type"
    end_x: str | None = "end_x"
    end_y: str | None = "end_y"
    is_success: str | None = "is_success"
    x_range: tuple[float, float] = (0.0, 100.0)
    y_range: tuple[float, float] = (0.0, 100.0)
    event_type_map: dict[str, str] | None = None
    success_value_map: dict[Any, bool] | None = None

    def __post_init__(self) -> None:
        if (self.end_x is None) != (self.end_y is None):
            raise ValueError("Both or neither end_x and end_y must be provided")
        _validate_range("x_range", self.x_range)
        _validate_range("y_range", self.y_range)

    def apply(self, events: pd.DataFrame | "Flow") -> "XTData":
        """Return normalized :class:`XTData` using this schema."""
        data = XTData(
            events=events,
            x=self.x,
            y=self.y,
            event_type=self.event_type,
            end_x=self.end_x,
            end_y=self.end_y,
            is_success=self.is_success,
            x_range=self.x_range,
            y_range=self.y_range,
        )
        if self.event_type_map or self.success_value_map:
            data = data.map_events(
                event_type_map=self.event_type_map,
                success_value_map=self.success_value_map,
            )
        return data

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "x": self.x,
            "y": self.y,
            "event_type": self.event_type,
            "end_x": self.end_x,
            "end_y": self.end_y,
            "is_success": self.is_success,
            "x_range": [self.x_range[0], self.x_range[1]],
            "y_range": [self.y_range[0], self.y_range[1]],
            "event_type_map": self.event_type_map,
            "success_value_map": self.success_value_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "XTEventSchema":
        """Load schema from metadata dict."""
        return cls(
            x=str(data.get("x", "x")),
            y=str(data.get("y", "y")),
            event_type=str(data.get("event_type", "event_type")),
            end_x=data.get("end_x"),
            end_y=data.get("end_y"),
            is_success=data.get("is_success"),
            x_range=tuple(data.get("x_range", (0.0, 100.0))),
            y_range=tuple(data.get("y_range", (0.0, 100.0))),
            event_type_map=data.get("event_type_map"),
            success_value_map=data.get("success_value_map"),
        )


@dataclass(frozen=True)
class XTData:
    """
    Provider-agnostic schema wrapper for xT event data.

    Wraps a DataFrame with column name mappings so that
    :class:`XTModel`
    can work with any provider's column naming conventions.

    The ``is_success`` column has a consistent meaning across event types:
    for moves (passes, carries, etc.) it means the action was completed
    successfully; for shots it means a goal was scored.

    ``is_success`` should ideally be boolean.  Numeric ``0``/``1`` is also
    accepted.  Provider-specific string labels should be mapped explicitly
    with ``success_value_map`` in :meth:`map_events` before fitting or scoring.

    Parameters
    ----------
    events : pandas.DataFrame or penaltyblog.matchflow.Flow
        Raw event data from any provider.
    x : str
        Column name for the start x-coordinate of each event.
    y : str
        Column name for the start y-coordinate of each event.
    event_type : str
        Column containing event-type labels (e.g. ``"pass"``, ``"shot"``).
    end_x : str, optional
        Column name for the end x-coordinate of move events (pass, carry, …).
        Must be provided together with *end_y*, or omitted entirely.
    end_y : str, optional
        Column name for the end y-coordinate of move events.
    is_success : str, optional
        Column indicating completion (for moves) or goal (for shots).
        If omitted, all events are assumed to have succeeded.
    x_range : tuple of (float, float)
        The (min, max) coordinate range used by your provider for the x-axis.
        Default ``(0.0, 100.0)`` means coordinates are already normalised.
        For StatsBomb data use ``(0, 120)``; for Opta use ``(0, 100)``.
    y_range : tuple of (float, float)
        The (min, max) coordinate range used by your provider for the y-axis.
        Default ``(0.0, 100.0)``.  For StatsBomb data use ``(0, 80)``.

    Canonical event types
    ---------------------
    Move families: ``pass``, ``carry``, ``throw_in``, ``goal_kick``,
    ``corner``, ``free_kick``

    Shot families: ``shot``, ``free_kick_shot``

    Examples
    --------
    Create an :class:`XTData` object from a DataFrame that already uses
    canonical column names:

    >>> data = pb.xt.XTData(
    ...     events=df,
    ...     x="x",
    ...     y="y",
    ...     event_type="event_type",
    ...     end_x="end_x",
    ...     end_y="end_y",
    ...     is_success="is_success",
    ... )

    StatsBomb data with non-standard coordinate ranges:

    >>> data = pb.xt.XTData(
    ...     events=df,
    ...     x="location_x",
    ...     y="location_y",
    ...     event_type="type_name",
    ...     end_x="pass_end_x",
    ...     end_y="pass_end_y",
    ...     is_success="pass_outcome",
    ...     x_range=(0, 120),
    ...     y_range=(0, 80),
    ... )
    """

    events: pd.DataFrame | Flow
    x: str
    y: str
    event_type: str
    end_x: str | None = None
    end_y: str | None = None
    is_success: str | None = None
    x_range: tuple[float, float] = (0.0, 100.0)
    y_range: tuple[float, float] = (0.0, 100.0)

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

    @cached_property
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

        df = self.events.rename(columns=rename)
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
        event_type_map: dict[str, str] | None = None,
        success_value_map: dict[Any, bool] | None = None,
    ) -> "XTData":
        """
        Return a new :class:`XTData` with raw labels mapped to canonical labels.

        This is the recommended way to handle provider-specific event names and
        success indicators before passing data to
        :meth:`XTModel.fit` or :meth:`XTModel.score`.

        Parameters
        ----------
        event_type_map : dict, optional
            Mapping from raw ``event_type`` values to canonical xT event families.
            Any values not present in the map are kept as-is.
            Example: ``{"Pass": "pass", "Shot": "shot", "Throw-in": "throw_in"}``.
        success_value_map : dict, optional
            Mapping from raw ``is_success`` values to booleans.
            This is the recommended path for provider feeds that encode
            outcomes as strings.
            Example: ``{"Complete": True, "Incomplete": False, "Goal": True}``.

        Returns
        -------
        XTData
            A new :class:`XTData` instance with canonical column values.
            The original instance is not modified.
        """
        df = self._normalized_df()
        if event_type_map:
            df["event_type"] = (
                df["event_type"].map(event_type_map).fillna(df["event_type"])
            )
        if success_value_map:
            df["is_success"] = (
                df["is_success"].map(success_value_map).fillna(df["is_success"])
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


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Check the column-name arguments "
            "passed to XTData(...)."
        )


def _coerce_events_dataframe(events: object) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events
    from ..matchflow.flow import Flow

    if isinstance(events, Flow):
        return events.to_pandas()
    raise TypeError("events must be a pandas DataFrame or penaltyblog.matchflow.Flow")
