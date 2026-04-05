"""Expected Threat (xT) model fitting, scoring, and I/O."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import XTData
from .io import load_xt_npz, save_xt_npz
from .plotting import plot_xt_surface

if TYPE_CHECKING:
    from ..matchflow.flow import Flow

_ALWAYS_IGNORE = {
    "penalty",
    "own_goal",
    "shot_against",
    "shootout",
    "postmatch_penalty",
    "penalty_kick",
}

_BOOL_TYPES = (bool, np.bool_)
_NUMERIC_TYPES = (int, float, np.integer, np.floating)


def _coerce_bool(series: pd.Series, default: bool) -> pd.Series:
    """Coerce a success column to boolean with strict validation.

    Accepted values:

    - native booleans and ``numpy.bool_``
    - numeric ``0`` or ``1`` (integer or float)
    - missing values, which fall back to ``default``

    String labels are rejected so provider-specific success values must be
    mapped explicitly via ``success_map`` before fitting or scoring.
    """
    # Fast path: already a boolean dtype (common after map_events or explicit casting)
    if series.dtype == bool or str(series.dtype) in ("bool", "boolean"):
        filled = series.fillna(default)
        return filled.astype(bool)

    # Fast path: integer column where all non-NaN values are 0 or 1
    if pd.api.types.is_integer_dtype(series.dtype):
        notna = series.notna()
        vals = series[notna]
        bad = ~vals.isin([0, 1])
        if bad.any():
            unique_bad = list(dict.fromkeys(repr(v) for v in vals[bad]))[:5]
            preview = ", ".join(unique_bad)
            raise ValueError(
                "Invalid values found in is_success: "
                f"{preview}. Expected booleans or numeric 0/1. "
                "If your provider uses labels such as 'Complete' or 'Goal', "
                "pass success_map={...} to XTData.map_events(...), XTModel.fit(...), "
                "or XTModel.score(...)."
            )
        result = pd.Series(default, index=series.index, dtype=bool)
        result[notna] = vals.astype(bool)
        return result

    # General path for object dtype (mixed types).
    # We use bulk masks + vectorised assignment rather than per-element .at[]
    # to avoid the per-element pandas label-lookup overhead.
    result = pd.Series(default, index=series.index, dtype=bool)
    notna_mask = series.notna()

    bool_mask = notna_mask & series.map(
        lambda v: isinstance(v, _BOOL_TYPES) if pd.notna(v) else False
    )
    numeric_mask = (
        notna_mask
        & ~bool_mask
        & series.map(
            lambda v: (
                (isinstance(v, _NUMERIC_TYPES) and not isinstance(v, _BOOL_TYPES))
                if pd.notna(v)
                else False
            )
        )
    )

    if bool_mask.any():
        result[bool_mask.to_numpy()] = series[bool_mask].astype(bool).to_numpy()

    invalid_numeric = pd.Series(False, index=series.index)
    if numeric_mask.any():
        as_float = series[numeric_mask].map(float)
        valid_01 = as_float.map(lambda v: np.isfinite(v) and v in {0.0, 1.0})
        if valid_01.any():
            valid_mask = numeric_mask.copy()
            valid_mask[numeric_mask] = valid_01
            result[valid_mask.to_numpy()] = (
                as_float[valid_01].astype(int).astype(bool).to_numpy()
            )
        invalid_numeric[numeric_mask] = ~valid_01

    invalid_mask = (notna_mask & ~bool_mask & ~numeric_mask) | invalid_numeric
    if invalid_mask.any():
        unique_values = list(dict.fromkeys(repr(v) for v in series[invalid_mask]))
        preview = ", ".join(unique_values[:5])
        if len(unique_values) > 5:
            preview += ", ..."
        raise ValueError(
            "Invalid values found in is_success: "
            f"{preview}. Expected booleans or numeric 0/1. "
            "If your provider uses labels such as 'Complete' or 'Goal', "
            "pass success_map={...} to XTData.map_events(...), XTModel.fit(...), "
            "or XTModel.score(...)."
        )

    return result


def _clip_coords(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 100.0)


def _coords_to_cell(x: np.ndarray, y: np.ndarray, l: int, w: int) -> np.ndarray:
    grid_x = np.floor(x * l / 100.0).astype(int)
    grid_y = np.floor(y * w / 100.0).astype(int)
    grid_x = np.clip(grid_x, 0, l - 1)
    grid_y = np.clip(grid_y, 0, w - 1)
    return grid_y * l + grid_x


def _safe_cell_index(x: np.ndarray, y: np.ndarray, l: int, w: int) -> np.ndarray:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    out = np.full(x.shape, -1, dtype=int)
    if mask.any():
        out[mask] = _coords_to_cell(x[mask], y[mask], l, w)
    return out


def _out_of_bounds_count(values: np.ndarray) -> int:
    mask = (~np.isnan(values)) & ((values < 0.0) | (values > 100.0))
    return int(mask.sum())


@dataclass
class XTModel:
    """
    Position-based Expected Threat (xT) model.

    Expected Threat (xT) quantifies how much a ball action — a pass, carry,
    throw-in, etc. — increases the probability of the attacking team scoring.
    A higher xT value means the action moved the ball to a more dangerous area.

    The pitch is divided into an ``n_cols × n_rows`` grid.  The model learns
    the probability of scoring from each cell by jointly solving a linear
    system that accounts for shot probability, goal probability, and movement
    between cells.

    Parameters
    ----------
    n_cols : int
        Number of columns in the pitch grid, left-to-right (default 16).
        More columns give finer horizontal resolution at the cost of needing
        more data to fill each cell reliably.
    n_rows : int
        Number of rows in the pitch grid, bottom-to-top (default 12).
    include_carries : bool
        Whether to include ``carry`` events when fitting (default ``True``).
    include_throw_ins : bool
        Whether to include ``throw_in`` events (default ``True``).
    include_goal_kicks : bool
        Whether to include ``goal_kick`` events (default ``True``).
    include_corners : bool
        Whether to include ``corner`` events (default ``True``).
    include_free_kicks : bool
        Whether to include ``free_kick`` and ``free_kick_shot`` events
        (default ``True``).
    transition_smoothing_k : float
        Shrinkage strength for per-family transition matrices (default 5.0).
        Higher values pull sparse families closer to the pooled baseline.
    coord_policy : {'warn', 'error', 'clip'}
        What to do when coordinates fall outside the expected 0–100 range:

        - ``'warn'`` *(default)* — emit a :class:`UserWarning` and clip.
        - ``'error'`` — raise a :exc:`ValueError`.
        - ``'clip'`` — silently clip to the valid range.

    Notes
    -----
    Coordinates are normalised to the range 0–100 internally.  If your
    provider uses a different scale (e.g. StatsBomb 0–120 × 0–80), pass
    ``x_range`` and ``y_range`` to :class:`~penaltyblog.xt.XTData` or to
    :meth:`fit` / :meth:`score` directly.

    Examples
    --------
    Fit a model and score events:

    >>> import pandas as pd
    >>> import penaltyblog as pb
    >>> df = pd.read_csv("events.csv")          # must include x, y, event_type
    >>> model = pb.xt.XTModel()
    >>> model.fit(df)
    >>> scored = model.score(df)
    >>> scored[["xt_start", "xt_end", "xt_added"]].head()

    Load a ready-to-use pretrained model instead of fitting your own:

    >>> model = pb.xt.load_pretrained_xt()
    >>> model.value_at(85, 50)   # xT near the penalty spot (0–100 coordinates)
    """

    n_cols: int = 16
    n_rows: int = 12
    include_carries: bool = True
    include_throw_ins: bool = True
    include_goal_kicks: bool = True
    include_corners: bool = True
    include_free_kicks: bool = True
    transition_smoothing_k: float = 5.0
    coord_policy: str = "warn"
    # Deprecated aliases kept for backwards compatibility
    l: int | None = field(default=None, repr=False)
    w: int | None = field(default=None, repr=False)
    fitted_: bool = field(default=False, init=False, repr=False)
    _fit_schema_: dict[str, object] | None = field(default=None, init=False, repr=False)

    # Attributes populated by fit() — declared here so type-checkers and IDEs
    # can see them without requiring a fitted instance.
    surface_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    shot_probability_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    goal_probability_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    move_probability_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    transition_matrix_: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    metadata_: dict[str, object] = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    included_move_families_: list[str] = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    included_shot_families_: list[str] = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Handle deprecated l/w aliases
        if self.l is not None:
            warnings.warn(
                "XTModel(l=...) is deprecated; use n_cols=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_cols = self.l
        if self.w is not None:
            warnings.warn(
                "XTModel(w=...) is deprecated; use n_rows=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_rows = self.w
        if self.n_cols <= 0 or self.n_rows <= 0:
            raise ValueError("n_cols and n_rows must be positive integers")
        if self.coord_policy not in {"warn", "error", "clip"}:
            raise ValueError("coord_policy must be one of: 'warn', 'error', 'clip'")

    def _validate_coords(self, df: pd.DataFrame, context: str) -> None:
        counts = {}
        for col in ["x", "y", "end_x", "end_y"]:
            arr = df[col].to_numpy(dtype=float, copy=False)
            count = _out_of_bounds_count(arr)
            if count > 0:
                counts[col] = count

        if not counts:
            return

        counts_str = ", ".join(
            f"{col}={count}" for col, count in sorted(counts.items())
        )
        msg = (
            f"xT {context} received coordinates outside expected 0..100; "
            f"out-of-bounds counts: {counts_str}. "
            "Set XTData(x_range=..., y_range=...) for provider-specific scaling."
        )
        if self.coord_policy == "error":
            raise ValueError(msg)
        if self.coord_policy == "warn":
            warnings.warn(f"{msg} Values will be clipped.", UserWarning, stacklevel=4)

    def _as_xtdata(
        self,
        data: XTData | pd.DataFrame | Flow,
        *,
        x: str = "x",
        y: str = "y",
        event_type: str = "event_type",
        end_x: str | None = "end_x",
        end_y: str | None = "end_y",
        is_success: str | None = "is_success",
        x_range: tuple[float, float] = (0.0, 100.0),
        y_range: tuple[float, float] = (0.0, 100.0),
        event_map: dict[str, str] | None = None,
        success_map: dict[str, bool] | None = None,
    ) -> XTData:
        if isinstance(data, XTData):
            if event_map or success_map:
                return data.map_events(event_map=event_map, success_map=success_map)
            return data
        if self._is_matchflow_flow(data):
            data = data.to_pandas()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be XTData, pandas DataFrame, or penaltyblog.matchflow.Flow"
            )

        resolved_end_x, resolved_end_y, resolved_is_success = self._resolve_columns(
            data,
            end_x=end_x,
            end_y=end_y,
            is_success=is_success,
        )

        xt_data = XTData(
            events=data,
            x=x,
            y=y,
            event_type=event_type,
            end_x=resolved_end_x,
            end_y=resolved_end_y,
            is_success=resolved_is_success,
            x_range=x_range,
            y_range=y_range,
        )
        if event_map or success_map:
            xt_data = xt_data.map_events(event_map=event_map, success_map=success_map)
        return xt_data

    @staticmethod
    def _resolve_columns(
        data: pd.DataFrame,
        *,
        end_x: str | None,
        end_y: str | None,
        is_success: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        resolved_end_x = end_x
        resolved_end_y = end_y
        resolved_is_success = is_success

        if (
            end_x == "end_x"
            and end_y == "end_y"
            and end_x not in data.columns
            and end_y not in data.columns
        ):
            resolved_end_x = None
            resolved_end_y = None

        if is_success == "is_success" and is_success not in data.columns:
            resolved_is_success = None

        return resolved_end_x, resolved_end_y, resolved_is_success

    def _classify_events(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify each row as move, shot, or ignore.

        Returns a Series of 'move', 'shot', or 'ignore'.
        """
        role = pd.Series("ignore", index=df.index, dtype=object)
        et = df["event_type"]

        # Always-included families
        role.loc[et == "pass"] = "move"
        role.loc[et == "shot"] = "shot"

        # Optional move families
        if self.include_carries:
            role.loc[et == "carry"] = "move"
        if self.include_throw_ins:
            role.loc[et == "throw_in"] = "move"
        if self.include_goal_kicks:
            role.loc[et == "goal_kick"] = "move"
        if self.include_free_kicks:
            role.loc[et == "free_kick"] = "move"
            role.loc[et == "free_kick_shot"] = "shot"
        if self.include_corners:
            role.loc[et == "corner"] = "move"

        # Always ignore
        for fam in _ALWAYS_IGNORE:
            role.loc[et == fam] = "ignore"

        return role

    def fit(
        self,
        data: XTData | pd.DataFrame | Flow,
        *,
        x: str = "x",
        y: str = "y",
        event_type: str = "event_type",
        end_x: str | None = "end_x",
        end_y: str | None = "end_y",
        is_success: str | None = "is_success",
        x_range: tuple[float, float] = (0.0, 100.0),
        y_range: tuple[float, float] = (0.0, 100.0),
        event_map: dict[str, str] | None = None,
        success_map: dict[str, bool] | None = None,
    ) -> "XTModel":
        """
        Fit one unified xT surface from all included attacking events.

        Column name mappings used here are saved and automatically replayed
        when you call :meth:`score` without arguments, so you only need to
        specify them once.

        Parameters
        ----------
        data : XTData, pandas.DataFrame, or penaltyblog.matchflow.Flow
            Event data to fit on. Coordinates are expected in the ranges given
            by ``x_range`` and ``y_range`` and are normalised internally to
            0–100.
        x, y : str
            Column names for the event start coordinates.
        event_type : str
            Column containing the event family labels.
        end_x, end_y : str, optional
            Column names for the end coordinates of move actions. If both
            default names are absent, they are treated as unavailable.
        is_success : str, optional
            Column indicating whether a move succeeded or a shot was a goal.
            If the default name is absent, it is treated as unavailable.
            Values should be boolean (or numeric 0/1). Provider-specific
            strings should be normalised explicitly via ``success_map``.
        x_range, y_range : tuple[float, float]
            Coordinate ranges for the incoming provider data. For example,
            StatsBomb-style coordinates can be passed as ``x_range=(0, 120)``
            and ``y_range=(0, 80)``.
        event_map : dict, optional
            Mapping from provider-specific event labels to canonical xT event
            families such as ``pass`` and ``shot``.
        success_map : dict, optional
            Mapping from provider-specific success labels to booleans.

        Returns
        -------
        XTModel
            The fitted model instance (``self``), so you can chain calls:
            ``XTModel().fit(df)``.

        Examples
        --------
        Fit on a DataFrame that already uses canonical column names:

        >>> model = pb.xt.XTModel()
        >>> model.fit(df)   # df must have x, y, event_type, end_x, end_y, is_success

        Fit on StatsBomb data with a custom coordinate range and label mapping:

        >>> model.fit(
        ...     df,
        ...     x="location_x",
        ...     y="location_y",
        ...     end_x="pass_end_location_x",
        ...     end_y="pass_end_location_y",
        ...     event_type="type_name",
        ...     is_success="pass_outcome_name",
        ...     x_range=(0, 120),
        ...     y_range=(0, 80),
        ...     event_map={"Pass": "pass", "Shot": "shot", "Carry": "carry"},
        ...     success_map={"Complete": True, "Incomplete": False,
        ...                  "Goal": True, "Saved": False},
        ... )
        """
        fit_schema: dict[str, object] | None = None
        tabular_data = data.to_pandas() if self._is_matchflow_flow(data) else data
        if isinstance(tabular_data, XTData):
            fit_schema = {
                "x": data.x,
                "y": data.y,
                "event_type": data.event_type,
                "end_x": data.end_x,
                "end_y": data.end_y,
                "is_success": data.is_success,
                "x_range": list(data.x_range),
                "y_range": list(data.y_range),
                "event_map": event_map or None,
                "success_map": success_map or None,
            }
        elif isinstance(tabular_data, pd.DataFrame):
            resolved_end_x, resolved_end_y, resolved_is_success = self._resolve_columns(
                tabular_data,
                end_x=end_x,
                end_y=end_y,
                is_success=is_success,
            )

            fit_schema = {
                "x": x,
                "y": y,
                "event_type": event_type,
                "end_x": resolved_end_x,
                "end_y": resolved_end_y,
                "is_success": resolved_is_success,
                "x_range": list(x_range),
                "y_range": list(y_range),
                "event_map": event_map or None,
                "success_map": success_map or None,
            }

        xt_data = self._as_xtdata(
            tabular_data,
            x=x,
            y=y,
            event_type=event_type,
            end_x=end_x,
            end_y=end_y,
            is_success=is_success,
            x_range=x_range,
            y_range=y_range,
            event_map=event_map,
            success_map=success_map,
        )
        if xt_data.is_success is None:
            raise ValueError(
                "Missing success information. Provide an is_success column "
                "(or mapping) where moves use completion status and shots use goal status."
            )
        df = xt_data.df.copy()

        # Convert coordinates to float
        for col in ["x", "y", "end_x", "end_y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self._validate_coords(df, context="fit")

        # Classify events
        role = self._classify_events(df)

        # is_success: for moves means completed, for shots means goal
        success_flags = _coerce_bool(df["is_success"], default=True)

        # Build masks
        valid_start = df["x"].notna() & df["y"].notna()
        valid_end = df["end_x"].notna() & df["end_y"].notna()

        shot_mask = (role == "shot") & valid_start
        goal_mask = shot_mask & success_flags
        # All move attempts (successful + failed) for the denominator
        all_move_mask = (role == "move") & valid_start
        # Only successful moves contribute transitions
        move_mask = all_move_mask & success_flags & valid_end

        if not shot_mask.any() and not move_mask.any():
            raise ValueError(
                "No relevant events to fit the model. Check that event_map maps "
                "your provider labels to canonical xT event types."
            )
        if not shot_mask.any():
            warnings.warn(
                "No shot events found in the training data. The fitted xT surface "
                "will be all zeros. Ensure your data includes shots (event_type='shot') "
                "or check that event_map correctly maps shot labels.",
                UserWarning,
                stacklevel=2,
            )

        num_cells = self.n_cols * self.n_rows

        # Clip and map to cells
        x_arr = _clip_coords(df["x"].to_numpy(dtype=float))
        y_arr = _clip_coords(df["y"].to_numpy(dtype=float))
        ex_arr = _clip_coords(
            np.where(df["end_x"].isna(), 0.0, df["end_x"].to_numpy(dtype=float))
        )
        ey_arr = _clip_coords(
            np.where(df["end_y"].isna(), 0.0, df["end_y"].to_numpy(dtype=float))
        )

        shot_idx = shot_mask.to_numpy()
        goal_idx = goal_mask.to_numpy()
        all_move_idx = all_move_mask.to_numpy()
        move_idx = move_mask.to_numpy()

        shot_cells = _coords_to_cell(
            x_arr[shot_idx], y_arr[shot_idx], self.n_cols, self.n_rows
        )
        goal_cells = _coords_to_cell(
            x_arr[goal_idx], y_arr[goal_idx], self.n_cols, self.n_rows
        )
        all_move_cells = _coords_to_cell(
            x_arr[all_move_idx], y_arr[all_move_idx], self.n_cols, self.n_rows
        )
        move_start_cells = _coords_to_cell(
            x_arr[move_idx], y_arr[move_idx], self.n_cols, self.n_rows
        )

        # Count matrices
        shot_count = np.bincount(shot_cells, minlength=num_cells).astype(float)
        goal_count = np.bincount(goal_cells, minlength=num_cells).astype(float)
        all_move_count = np.bincount(all_move_cells, minlength=num_cells).astype(float)
        successful_move_count = np.bincount(
            move_start_cells, minlength=num_cells
        ).astype(float)

        # Action probabilities
        # Denominator includes ALL moves (successful + failed) so that
        # failed moves consume probability without contributing transitions.
        # The gap (1 - shot_prob - move_prob) is the implicit turnover rate.
        total = shot_count + all_move_count
        shot_probability = np.divide(
            shot_count, total, out=np.zeros(num_cells), where=total > 0
        )
        move_probability = np.divide(
            successful_move_count, total, out=np.zeros(num_cells), where=total > 0
        )

        # Per-cell goal probability with beta-binomial smoothing
        alpha, beta = 1.0, 1.0
        goal_probability = (goal_count + alpha) / (shot_count + alpha + beta)

        # Per-family transition matrices with shrinkage
        #
        # Each move family gets its own transition matrix, but sparse
        # families (e.g. free kicks from a given cell) are smoothed
        # toward the pooled all-families transition to avoid noisy
        # single-observation rows dominating.
        #
        # For family f at cell i with n_f observations:
        #   T_f_smoothed[i] = (counts_f[i] + k * T_pooled[i]) / (n_f[i] + k)
        #
        # k controls the strength of the prior.  With k=5, you need ~5+
        # events of a family from a cell before its pattern meaningfully
        # diverges from the pooled baseline.
        #
        # The combined move-transition product is:
        #   MT = sum_f  diag(p_f) @ T_f_smoothed

        k = self.transition_smoothing_k

        event_types = df.loc[move_idx, "event_type"].to_numpy()
        move_start_arr = _coords_to_cell(
            x_arr[move_idx], y_arr[move_idx], self.n_cols, self.n_rows
        )
        move_end_arr = _coords_to_cell(
            ex_arr[move_idx], ey_arr[move_idx], self.n_cols, self.n_rows
        )

        # Pooled transition matrix (prior for per-family smoothing)
        pooled_tc = np.zeros((num_cells, num_cells), dtype=float)
        if len(move_start_arr) > 0:
            np.add.at(pooled_tc, (move_start_arr, move_end_arr), 1.0)
        pooled_rs = pooled_tc.sum(axis=1)
        T_pooled = np.zeros_like(pooled_tc)
        pooled_nz = pooled_rs > 0
        T_pooled[pooled_nz] = pooled_tc[pooled_nz] / pooled_rs[pooled_nz][:, None]

        MT = np.zeros((num_cells, num_cells), dtype=float)
        families_seen = set(event_types)

        for fam in families_seen:
            fam_sel = event_types == fam
            fam_starts = move_start_arr[fam_sel]
            fam_ends = move_end_arr[fam_sel]

            # Per-cell family count -> family action probability
            fam_count = np.bincount(fam_starts, minlength=num_cells).astype(float)
            fam_prob = np.divide(
                fam_count, total, out=np.zeros(num_cells), where=total > 0
            )

            # Family transition counts
            tc = np.zeros((num_cells, num_cells), dtype=float)
            np.add.at(tc, (fam_starts, fam_ends), 1.0)
            n_f = tc.sum(axis=1)  # per-cell family event count

            # Shrink toward pooled: (counts + k * T_pooled) / (n + k)
            prior = k * T_pooled
            T_f = np.zeros_like(tc)
            denom = n_f + k
            has_prior = denom > 0
            T_f[has_prior] = (tc[has_prior] + prior[has_prior]) / denom[has_prior][
                :, None
            ]

            MT += np.diag(fam_prob) @ T_f

        # Derive an effective combined transition matrix for storage
        # so that  diag(move_probability) @ transition_matrix ≈ MT
        transition_matrix = np.zeros((num_cells, num_cells), dtype=float)
        has_moves = move_probability > 0
        transition_matrix[has_moves] = (
            MT[has_moves] / move_probability[has_moves][:, None]
        )

        # Solve (I - MT) X = S
        S = shot_probability * goal_probability
        A = np.eye(num_cells) - MT

        cond = np.linalg.cond(A)
        if cond > 1e12:
            warnings.warn(
                f"xT transition matrix is ill-conditioned (condition number {cond:.2e}). "
                "The fitted surface may contain extreme values. Consider using more data "
                "or a larger grid to stabilise the solution.",
                UserWarning,
                stacklevel=2,
            )

        try:
            X = np.linalg.solve(A, S)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "xT solve failed due to a singular matrix. "
                "Check that the event data produces a valid system."
            ) from exc

        # Store results
        self.surface_ = X.reshape((self.n_rows, self.n_cols))
        self.shot_probability_ = shot_probability.reshape((self.n_rows, self.n_cols))
        self.goal_probability_ = goal_probability.reshape((self.n_rows, self.n_cols))
        self.move_probability_ = move_probability.reshape((self.n_rows, self.n_cols))
        self.transition_matrix_ = transition_matrix

        # Track which families were actually present
        self.included_move_families_ = sorted(df.loc[move_idx, "event_type"].unique())
        self.included_shot_families_ = sorted(df.loc[shot_idx, "event_type"].unique())

        self.metadata_ = {
            "artifact_version": 1,
            "model_type": "xt",
            "name": "xt_v1",
            "grid": [self.n_cols, self.n_rows],
            "include_carries": self.include_carries,
            "include_throw_ins": self.include_throw_ins,
            "include_goal_kicks": self.include_goal_kicks,
            "include_corners": self.include_corners,
            "include_free_kicks": self.include_free_kicks,
            "coord_policy": self.coord_policy,
            "coordinate_system": "0_100",
            "orientation": "left_to_right",
            "goal_probability_smoothing": {"alpha": alpha, "beta": beta},
            "included_move_families": self.included_move_families_,
            "included_shot_families": self.included_shot_families_,
            "fit_schema": fit_schema,
        }
        self._fit_schema_ = self.metadata_["fit_schema"]
        self.fitted_ = True
        return self

    def score(
        self,
        data: XTData | pd.DataFrame | Flow,
        *,
        x: str = "x",
        y: str = "y",
        event_type: str = "event_type",
        end_x: str | None = "end_x",
        end_y: str | None = "end_y",
        is_success: str | None = "is_success",
        x_range: tuple[float, float] = (0.0, 100.0),
        y_range: tuple[float, float] = (0.0, 100.0),
        event_map: dict[str, str] | None = None,
        success_map: dict[str, bool] | None = None,
    ) -> pd.DataFrame:
        """
        Score events by xT delta and return the annotated DataFrame.

        Returns a copy of the input with three new columns:

        - ``xt_start`` — xT value at the start location (set for all moves with
          valid start coordinates, including failed moves)
        - ``xt_end`` — xT value at the end location (NaN for failed/unended moves
          and shots)
        - ``xt_added`` — ``xt_end - xt_start`` (NaN for failed/unended moves and shots)

        .. note::
            ``xt_start`` is populated for **all** moves with valid start coordinates,
            regardless of whether the move succeeded.  This lets you measure the
            possession value that was risked on a failed pass, for example.
            ``xt_end`` and ``xt_added`` are only set for *successful* moves that
            also have valid end coordinates.

        When called after :meth:`fit`, the column mapping used during fitting
        is re-applied automatically — you can call ``model.score(df)`` with no
        extra arguments if ``df`` has the same structure as the training data.

        Parameters
        ----------
        data : XTData, pandas.DataFrame, or penaltyblog.matchflow.Flow
            Event data to score.
        x, y : str
            Column names for the event start coordinates.
        event_type : str
            Column containing the event family labels.
        end_x, end_y : str, optional
            Column names for the end coordinates of move actions. If both
            default names are absent, they are treated as unavailable.
        is_success : str, optional
            Column indicating whether a move succeeded or a shot was a goal.
            If the default name is absent, it is treated as unavailable.
            Values should be boolean (or numeric 0/1). Provider-specific
            strings should be normalized explicitly via ``success_map``.
        x_range, y_range : tuple[float, float]
            Coordinate ranges for the incoming provider data. For example,
            StatsBomb-style coordinates can be passed as ``x_range=(0, 120)``
            and ``y_range=(0, 80)``.
        event_map : dict, optional
            Mapping from provider-specific event labels to canonical xT event
            families such as ``pass`` and ``shot``.
        success_map : dict, optional
            Mapping from provider-specific success labels to booleans.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with ``xt_start``, ``xt_end``, and
            ``xt_added`` columns added.
        """
        self._ensure_fitted()
        tabular_data = data.to_pandas() if self._is_matchflow_flow(data) else data
        use_fitted_schema = (
            isinstance(tabular_data, pd.DataFrame)
            and self._fit_schema_ is not None
            and x == "x"
            and y == "y"
            and event_type == "event_type"
            and end_x == "end_x"
            and end_y == "end_y"
            and is_success == "is_success"
            and x_range == (0.0, 100.0)
            and y_range == (0.0, 100.0)
            and event_map is None
            and success_map is None
        )
        if use_fitted_schema:
            schema = self._fit_schema_
            x = str(schema["x"])
            y = str(schema["y"])
            event_type = str(schema["event_type"])
            end_x = schema["end_x"] if schema["end_x"] is None else str(schema["end_x"])
            end_y = schema["end_y"] if schema["end_y"] is None else str(schema["end_y"])
            is_success = (
                schema["is_success"]
                if schema["is_success"] is None
                else str(schema["is_success"])
            )
            x_range = tuple(schema["x_range"])
            y_range = tuple(schema["y_range"])
            event_map = schema["event_map"]
            success_map = schema["success_map"]

        xt_data = self._as_xtdata(
            tabular_data,
            x=x,
            y=y,
            event_type=event_type,
            end_x=end_x,
            end_y=end_y,
            is_success=is_success,
            x_range=x_range,
            y_range=y_range,
            event_map=event_map,
            success_map=success_map,
        )
        if xt_data.is_success is None:
            raise ValueError(
                "Missing success information. Provide an is_success column "
                "(or mapping) where moves use completion status and shots use goal status."
            )
        df = xt_data.df.copy()

        for col in ["x", "y", "end_x", "end_y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self._validate_coords(df, context="score")

        role = self._classify_events(df)

        success_flags = _coerce_bool(df["is_success"], default=True)

        # start_mask: all moves with valid start coordinates (includes failed moves)
        start_mask = (role == "move") & df["x"].notna() & df["y"].notna()
        # score_mask: only successful moves with valid end coordinates
        score_mask = (
            start_mask & success_flags & df["end_x"].notna() & df["end_y"].notna()
        )

        x_arr = _clip_coords(df["x"].to_numpy(dtype=float, na_value=np.nan))
        y_arr = _clip_coords(df["y"].to_numpy(dtype=float, na_value=np.nan))
        ex_arr = _clip_coords(df["end_x"].to_numpy(dtype=float, na_value=np.nan))
        ey_arr = _clip_coords(df["end_y"].to_numpy(dtype=float, na_value=np.nan))

        start_cells = _safe_cell_index(x_arr, y_arr, self.n_cols, self.n_rows)
        end_cells = _safe_cell_index(ex_arr, ey_arr, self.n_cols, self.n_rows)

        surface_flat = self.surface_.reshape(-1)

        n = len(df)
        xt_start = np.full(n, np.nan)
        xt_end = np.full(n, np.nan)
        xt_added = np.full(n, np.nan)

        # Populate xt_start for all moves with a valid start cell
        valid_start = start_mask.to_numpy() & (start_cells >= 0)
        if valid_start.any():
            xt_start[valid_start] = surface_flat[start_cells[valid_start]]

        # Populate xt_end and xt_added only for successful moves with valid end cell
        valid_score = score_mask.to_numpy() & (start_cells >= 0) & (end_cells >= 0)
        if valid_score.any():
            xt_end[valid_score] = surface_flat[end_cells[valid_score]]
            xt_added[valid_score] = xt_end[valid_score] - xt_start[valid_score]

        if isinstance(tabular_data, XTData):
            result = tabular_data.events.copy()
        else:
            result = tabular_data.copy()
        result["xt_start"] = xt_start
        result["xt_end"] = xt_end
        result["xt_added"] = xt_added
        return result

    @staticmethod
    def _is_matchflow_flow(data: object) -> bool:
        from ..matchflow.flow import Flow

        return isinstance(data, Flow)

    def value_at(self, x: float, y: float) -> float:
        """Return the xT value at a single (x, y) coordinate.

        The return value is a float in approximately the range 0–0.15,
        representing the probability that the current possession from this
        location will result in a goal before possession is lost.

        For querying many positions at once, use :meth:`values_at`.

        Parameters
        ----------
        x : float
            Horizontal position in normalised 0–100 coordinates (0 = own goal,
            100 = opponent goal).
        y : float
            Vertical position in normalised 0–100 coordinates.

        Returns
        -------
        float
            Expected Threat value at (x, y).

        Notes
        -----
        Coordinates are expected in the 0–100 normalised space used internally.
        If your data uses a different coordinate system (e.g. StatsBomb 0–120 × 0–80),
        pass ``x_range`` and ``y_range`` to :class:`XTData` when fitting or scoring
        to convert automatically — but pass already-normalised values here.

        Examples
        --------
        >>> model = pb.xt.load_pretrained_xt()
        >>> model.value_at(85, 50)   # near penalty spot — high xT
        >>> model.value_at(10, 50)   # own half — low xT
        """
        self._ensure_fitted()
        x_val = float(x)
        y_val = float(y)
        if (not np.isfinite(x_val)) or (not np.isfinite(y_val)):
            raise ValueError("x and y must be finite numbers")

        out_of_bounds = (
            (x_val < 0.0) or (x_val > 100.0) or (y_val < 0.0) or (y_val > 100.0)
        )
        if out_of_bounds and self.coord_policy == "error":
            raise ValueError("xT value_at received coordinates outside expected 0..100")
        if out_of_bounds and self.coord_policy == "warn":
            warnings.warn(
                "xT value_at received coordinates outside expected 0..100. "
                "Values will be clipped.",
                UserWarning,
                stacklevel=2,
            )

        x_arr = _clip_coords(np.array([x_val]))
        y_arr = _clip_coords(np.array([y_val]))
        cell = _coords_to_cell(x_arr, y_arr, self.n_cols, self.n_rows)[0]
        return float(self.surface_.reshape(-1)[cell])

    def values_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return xT values for arrays of (x, y) coordinates.

        Vectorised version of :meth:`value_at`.  Useful for building heatmaps
        or scoring a grid of positions without writing a Python loop.

        Parameters
        ----------
        x : array-like of float
            Horizontal positions in normalised 0–100 coordinates.
        y : array-like of float
            Vertical positions in normalised 0–100 coordinates.
            Must have the same length as *x*.

        Returns
        -------
        numpy.ndarray
            xT value for each (x, y) pair, shape ``(len(x),)``.

        Examples
        --------
        Build a full-pitch xT heatmap grid:

        >>> import numpy as np
        >>> xs = np.linspace(0, 100, 16)
        >>> ys = np.linspace(0, 100, 12)
        >>> xx, yy = np.meshgrid(xs, ys)
        >>> vals = model.values_at(xx.ravel(), yy.ravel())
        >>> heatmap = vals.reshape(12, 16)
        """
        self._ensure_fitted()
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same length")

        out_of_bounds = np.any(
            (x_arr < 0) | (x_arr > 100) | (y_arr < 0) | (y_arr > 100)
        )
        if out_of_bounds and self.coord_policy == "error":
            raise ValueError(
                "xT values_at received coordinates outside expected 0..100"
            )
        if out_of_bounds and self.coord_policy == "warn":
            warnings.warn(
                "xT values_at received coordinates outside expected 0..100. "
                "Values will be clipped.",
                UserWarning,
                stacklevel=2,
            )

        x_clipped = _clip_coords(x_arr)
        y_clipped = _clip_coords(y_arr)
        cells = _coords_to_cell(x_clipped, y_clipped, self.n_cols, self.n_rows)
        return self.surface_.reshape(-1)[cells]

    def save(self, path: str) -> None:
        """Save the fitted model to an ``.npz`` file.

        The file can be reloaded on any machine with :meth:`load` and shared
        freely — it contains all arrays needed to score new events.

        Parameters
        ----------
        path : str
            Destination file path.  A ``.npz`` extension is recommended but
            not required.

        Examples
        --------
        >>> model.save("my_xt_model.npz")
        >>> loaded = XTModel.load("my_xt_model.npz")
        """
        self._ensure_fitted()
        arrays = {
            "surface": self.surface_,
            "shot_probability": self.shot_probability_,
            "goal_probability": self.goal_probability_,
            "move_probability": self.move_probability_,
            "transition_matrix": self.transition_matrix_,
        }
        save_xt_npz(path, arrays, self.metadata_)

    @classmethod
    def load(cls, path: str) -> "XTModel":
        """Load a fitted model from an ``.npz`` file created by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` file previously written by :meth:`save`.

        Returns
        -------
        XTModel
            A fully fitted model instance ready for scoring, querying, and
            plotting — no further call to :meth:`fit` is needed.

        Raises
        ------
        ValueError
            If the file does not contain a valid xT model artifact.

        Examples
        --------
        >>> model = XTModel.load("my_xt_model.npz")
        >>> model.value_at(85, 50)
        """
        arrays, metadata = load_xt_npz(path)
        if metadata.get("model_type") != "xt":
            raise ValueError("Not an xT model artifact")

        grid = metadata.get("grid")
        if not (
            isinstance(grid, list)
            and len(grid) == 2
            and all(isinstance(dim, int) for dim in grid)
        ):
            raise ValueError(
                "Invalid xT model artifact: metadata.grid must be a two-item list "
                "of positive integers"
            )

        n_cols, n_rows = grid
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError(
                "Invalid xT model artifact: metadata.grid must contain positive integers"
            )
        model = cls(
            n_cols=n_cols,
            n_rows=n_rows,
            include_carries=metadata.get("include_carries", True),
            include_throw_ins=metadata.get("include_throw_ins", True),
            include_goal_kicks=metadata.get("include_goal_kicks", True),
            include_corners=metadata.get("include_corners", True),
            include_free_kicks=metadata.get("include_free_kicks", True),
            coord_policy=metadata.get("coord_policy", "warn"),
        )
        model.surface_ = arrays["surface"]
        model.shot_probability_ = arrays["shot_probability"]
        model.goal_probability_ = arrays["goal_probability"]
        model.move_probability_ = arrays["move_probability"]
        model.transition_matrix_ = arrays["transition_matrix"]

        # Validate shapes are consistent with the declared grid dimensions
        num_cells = n_cols * n_rows
        expected_grid = (n_rows, n_cols)
        expected_2d = (num_cells, num_cells)
        shape_errors = []
        for name, expected in [
            ("surface", expected_grid),
            ("shot_probability", expected_grid),
            ("goal_probability", expected_grid),
            ("move_probability", expected_grid),
            ("transition_matrix", expected_2d),
        ]:
            actual = arrays[name].shape
            if actual != expected:
                shape_errors.append(f"  '{name}': expected {expected}, got {actual}")
        if shape_errors:
            raise ValueError(
                "Invalid xT model artifact: array shapes do not match the "
                f"declared grid ({n_cols}×{n_rows} = {num_cells} cells):\n"
                + "\n".join(shape_errors)
            )
        model.metadata_ = metadata
        model.included_move_families_ = metadata.get("included_move_families", [])
        model.included_shot_families_ = metadata.get("included_shot_families", [])
        model._fit_schema_ = metadata.get("fit_schema")
        model.fitted_ = True
        return model

    def plot(self, pitch=None, **kwargs):
        """Plot the xT surface as a heatmap on a football pitch.

        Parameters
        ----------
        pitch : Pitch, optional
            A :class:`~penaltyblog.viz.pitch.Pitch` instance to draw on.
            If omitted, an Opta-style pitch is created automatically.
        **kwargs
            Additional keyword arguments forwarded to the heatmap layer:

            - ``colorscale`` — Plotly colorscale name or list (default: pitch theme).
            - ``opacity`` — float between 0 and 1 (default: pitch theme).
            - ``show_colorbar`` — bool, whether to show the colour scale bar
              (default ``False``).

        Returns
        -------
        Pitch
            The :class:`~penaltyblog.viz.pitch.Pitch` instance with the xT
            heatmap layer added.  Call ``.fig`` to obtain the underlying
            Plotly figure, or use ``.show()`` / ``.to_image()`` directly.

        Examples
        --------
        >>> pitch = model.plot()
        >>> pitch.fig.show()
        """
        self._ensure_fitted()
        return plot_xt_surface(
            self.surface_, self.n_cols, self.n_rows, pitch=pitch, **kwargs
        )

    def _ensure_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise ValueError(
                "Model is not fitted. Call .fit() first, or use "
                "load_pretrained_xt() to load a ready-to-use model."
            )
