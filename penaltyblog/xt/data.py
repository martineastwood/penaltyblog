from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XTData:
    """
    Provider-agnostic schema wrapper for heterogeneous xT event data.

    Supports providers where pass-like actions and carry-like actions
    have separate destination columns, and goal/success are indicated
    by dedicated boolean columns.
    """

    events: pd.DataFrame
    x: str
    y: str
    event_type: str
    event_subtype: Optional[str] = None
    pass_end_x: Optional[str] = None
    pass_end_y: Optional[str] = None
    carry_end_x: Optional[str] = None
    carry_end_y: Optional[str] = None
    move_success: Optional[str] = None
    shot_goal: Optional[str] = None

    def __post_init__(self) -> None:
        required = [self.x, self.y, self.event_type]
        _validate_columns(self.events, required)

        for a, b, label in [
            (self.pass_end_x, self.pass_end_y, "pass_end"),
            (self.carry_end_x, self.carry_end_y, "carry_end"),
        ]:
            if (a is None) != (b is None):
                raise ValueError(
                    f"Both or neither {label}_x and {label}_y must be provided"
                )

        optional = [
            c
            for c in [
                self.event_subtype,
                self.pass_end_x,
                self.pass_end_y,
                self.carry_end_x,
                self.carry_end_y,
                self.move_success,
                self.shot_goal,
            ]
            if c is not None
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
        if self.event_subtype is not None:
            rename[self.event_subtype] = "event_subtype"
        if self.pass_end_x is not None:
            rename[self.pass_end_x] = "pass_end_x"
        if self.pass_end_y is not None:
            rename[self.pass_end_y] = "pass_end_y"
        if self.carry_end_x is not None:
            rename[self.carry_end_x] = "carry_end_x"
        if self.carry_end_y is not None:
            rename[self.carry_end_y] = "carry_end_y"
        if self.move_success is not None:
            rename[self.move_success] = "move_success"
        if self.shot_goal is not None:
            rename[self.shot_goal] = "shot_goal"

        df = self.events.rename(columns=rename, errors="ignore")
        keep = list(rename.values())
        result = df[keep].copy()

        for col, fill in [
            ("event_subtype", ""),
            ("pass_end_x", np.nan),
            ("pass_end_y", np.nan),
            ("carry_end_x", np.nan),
            ("carry_end_y", np.nan),
            ("move_success", True),
            ("shot_goal", False),
        ]:
            if col not in result.columns:
                result[col] = fill

        return result

    def map_events(
        self,
        event_map: Optional[Dict[str, str]] = None,
        subtype_map: Optional[Dict[str, str]] = None,
        success_map: Optional[Dict[str, bool]] = None,
        goal_map: Optional[Dict[str, bool]] = None,
    ) -> "XTData":
        """
        Return a new XTData with raw labels mapped to canonical labels.

        Parameters
        ----------
        event_map : dict, optional
            Mapping from raw event_type values to canonical families
            (e.g. ``{"Pass": "pass", "Shot": "shot", "Acceleration": "carry"}``).
        subtype_map : dict, optional
            Mapping from raw event_subtype values to canonical subtypes
            (e.g. ``{"carry": "carry"}``).
        success_map : dict, optional
            Mapping from raw move_success values to booleans
            (e.g. ``{"Complete": True, "Incomplete": False}``).
        goal_map : dict, optional
            Mapping from raw shot_goal values to booleans
            (e.g. ``{"Goal": True, "Saved": False, "Blocked": False}``).
        """
        df = self._normalized_df()
        if event_map:
            df["event_type"] = df["event_type"].map(event_map).fillna(df["event_type"])
        if subtype_map:
            original = df["event_subtype"].copy()
            mapped = original.map(subtype_map)
            df["event_subtype"] = mapped.fillna(original)
        if success_map:
            df["move_success"] = (
                df["move_success"].map(success_map).fillna(df["move_success"])
            )
        if goal_map:
            df["shot_goal"] = df["shot_goal"].map(goal_map).fillna(df["shot_goal"])

        return XTData(
            events=df,
            x="x",
            y="y",
            event_type="event_type",
            event_subtype="event_subtype",
            pass_end_x="pass_end_x",
            pass_end_y="pass_end_y",
            carry_end_x="carry_end_x",
            carry_end_y="carry_end_y",
            move_success="move_success",
            shot_goal="shot_goal",
        )


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
