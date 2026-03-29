from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import XTData
from .io import load_xt_npz, save_xt_npz
from .plotting import plot_xt_surface

_ALWAYS_IGNORE = {
    "penalty",
    "own_goal",
    "shot_against",
    "shootout",
    "postmatch_penalty",
    "penalty_kick",
}

_TRUTHY = frozenset(
    {
        "true",
        "1",
        "yes",
        "t",
        "y",
        "success",
        "successful",
        "complete",
        "completed",
        "accurate",
    }
)
_FALSY = frozenset(
    {
        "false",
        "0",
        "no",
        "f",
        "n",
        "fail",
        "failure",
        "incomplete",
        "unsuccessful",
        "inaccurate",
        "miss",
        "missed",
    }
)


def _coerce_bool(series: "pd.Series", default: bool) -> "pd.Series":
    """Coerce a column to boolean, handling strings like 'False' safely.

    Native booleans and numeric 0/1 pass through directly.
    String values are matched case-insensitively against known truthy/falsy
    labels. Unrecognised values and NaN fall back to *default*.
    """
    if series.dtype == bool:
        return series

    result = pd.Series(default, index=series.index, dtype=bool)
    notna = series.notna()

    if pd.api.types.is_numeric_dtype(series):
        result[notna] = series[notna].astype(float) != 0.0
        return result

    lowered = series[notna].astype(str).str.strip().str.lower()
    result[notna & lowered.isin(_TRUTHY)] = True
    result[notna & lowered.isin(_FALSY)] = False
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


@dataclass
class XTModel:
    """
    Position-based Expected Threat (xT) model.

    Fits one unified xT surface from all included attacking event families
    using the direct linear algebra formulation: ``(I - MT) X = S``.
    """

    l: int = 16
    w: int = 12
    include_carries: bool = True
    include_throw_ins: bool = True
    include_goal_kicks: bool = True
    include_corners: bool = True
    include_free_kicks: bool = True

    def __post_init__(self) -> None:
        if self.l <= 0 or self.w <= 0:
            raise ValueError("l and w must be positive")
        self.fitted_: bool = False

    def _classify_events(self, df: pd.DataFrame):
        """
        Classify each row as move, shot, or ignore.

        Returns (role, end_x, end_y) where role is a Series of
        'move', 'shot', or 'ignore', and end_x/end_y are the resolved
        end coordinates for move events.
        """
        role = pd.Series("ignore", index=df.index, dtype=object)
        end_x = pd.Series(np.nan, index=df.index, dtype=float)
        end_y = pd.Series(np.nan, index=df.index, dtype=float)

        et = df["event_type"]
        sub = df["event_subtype"].fillna("")

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

        # Ambiguous families: free_kick and corner can be move or shot
        if self.include_free_kicks:
            fk = et == "free_kick"
            role.loc[fk & (sub == "shot")] = "shot"
            role.loc[fk & (sub != "shot")] = "move"

        if self.include_corners:
            co = et == "corner"
            role.loc[co & (sub == "shot")] = "shot"
            role.loc[co & (sub != "shot")] = "move"

        # Always ignore
        for fam in _ALWAYS_IGNORE:
            role.loc[et == fam] = "ignore"

        # Resolve end coordinates for move events
        move_mask = role == "move"
        carry_mask = move_mask & (et == "carry")
        non_carry_move = move_mask & ~carry_mask

        # Non-carry moves use pass_end coordinates
        end_x.loc[non_carry_move] = df.loc[non_carry_move, "pass_end_x"]
        end_y.loc[non_carry_move] = df.loc[non_carry_move, "pass_end_y"]

        # Carries: prefer carry_end, fall back to pass_end
        if carry_mask.any():
            has_carry_end = df["carry_end_x"].notna() & df["carry_end_y"].notna()
            use_carry = carry_mask & has_carry_end
            use_pass = carry_mask & ~has_carry_end

            end_x.loc[use_carry] = df.loc[use_carry, "carry_end_x"]
            end_y.loc[use_carry] = df.loc[use_carry, "carry_end_y"]
            end_x.loc[use_pass] = df.loc[use_pass, "pass_end_x"]
            end_y.loc[use_pass] = df.loc[use_pass, "pass_end_y"]

        return role, end_x, end_y

    def fit(self, data: XTData) -> "XTModel":
        """
        Fit one unified xT surface from all included attacking events.
        """
        df = data.df.copy()

        # Convert coordinates to float
        for col in ["x", "y", "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Classify events
        role, resolved_end_x, resolved_end_y = self._classify_events(df)

        # Determine move success and goals (safe coercion for string data)
        move_success = _coerce_bool(df["move_success"], default=True)
        is_goal = _coerce_bool(df["shot_goal"], default=False)

        # Build masks
        valid_start = df["x"].notna() & df["y"].notna()
        valid_end = resolved_end_x.notna() & resolved_end_y.notna()

        shot_mask = (role == "shot") & valid_start
        goal_mask = shot_mask & is_goal
        move_mask = (role == "move") & move_success & valid_start & valid_end

        if not shot_mask.any() and not move_mask.any():
            raise ValueError("No relevant events to fit the model")

        num_cells = self.l * self.w

        # Clip and map to cells
        x_arr = _clip_coords(df["x"].to_numpy(dtype=float))
        y_arr = _clip_coords(df["y"].to_numpy(dtype=float))
        ex_arr = _clip_coords(
            np.where(resolved_end_x.isna(), 0.0, resolved_end_x.to_numpy(dtype=float))
        )
        ey_arr = _clip_coords(
            np.where(resolved_end_y.isna(), 0.0, resolved_end_y.to_numpy(dtype=float))
        )

        shot_idx = shot_mask.to_numpy()
        goal_idx = goal_mask.to_numpy()
        move_idx = move_mask.to_numpy()

        shot_cells = _coords_to_cell(x_arr[shot_idx], y_arr[shot_idx], self.l, self.w)
        goal_cells = _coords_to_cell(x_arr[goal_idx], y_arr[goal_idx], self.l, self.w)
        move_start_cells = _coords_to_cell(
            x_arr[move_idx], y_arr[move_idx], self.l, self.w
        )

        # Count matrices
        shot_count = np.bincount(shot_cells, minlength=num_cells).astype(float)
        goal_count = np.bincount(goal_cells, minlength=num_cells).astype(float)
        move_count = np.bincount(move_start_cells, minlength=num_cells).astype(float)

        # Action probabilities
        total = shot_count + move_count
        shot_probability = np.divide(
            shot_count, total, out=np.zeros(num_cells), where=total > 0
        )
        move_probability = np.divide(
            move_count, total, out=np.zeros(num_cells), where=total > 0
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

        transition_smoothing_k = 5.0

        event_types = df.loc[move_idx, "event_type"].to_numpy()
        move_start_arr = _coords_to_cell(
            x_arr[move_idx], y_arr[move_idx], self.l, self.w
        )
        move_end_arr = _coords_to_cell(
            ex_arr[move_idx], ey_arr[move_idx], self.l, self.w
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
            k = transition_smoothing_k
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

        try:
            X = np.linalg.solve(A, S)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "xT solve failed due to a singular matrix. "
                "Check that the event data produces a valid system."
            ) from exc

        # Store results
        self.surface_ = X.reshape((self.w, self.l))
        self.shot_probability_ = shot_probability.reshape((self.w, self.l))
        self.goal_probability_ = goal_probability.reshape((self.w, self.l))
        self.move_probability_ = move_probability.reshape((self.w, self.l))
        self.transition_matrix_ = transition_matrix

        # Track which families were actually present
        self.included_move_families_ = sorted(
            set(df.loc[move_idx, "event_type"].unique())
        )
        self.included_shot_families_ = sorted(
            set(df.loc[shot_idx, "event_type"].unique())
        )

        self.metadata_ = {
            "artifact_version": 1,
            "model_type": "xt",
            "name": "xt_v1",
            "grid": [self.l, self.w],
            "include_carries": self.include_carries,
            "include_throw_ins": self.include_throw_ins,
            "include_goal_kicks": self.include_goal_kicks,
            "include_corners": self.include_corners,
            "include_free_kicks": self.include_free_kicks,
            "coordinate_system": "0_100",
            "orientation": "left_to_right",
            "goal_probability_smoothing": {"alpha": alpha, "beta": beta},
            "included_move_families": self.included_move_families_,
            "included_shot_families": self.included_shot_families_,
        }
        self.fitted_ = True
        return self

    def score(self, data: XTData) -> pd.DataFrame:
        """
        Score successful move actions by xT delta.

        Returns a copy of the original DataFrame with ``xt_start``,
        ``xt_end``, and ``xt_added`` columns. Non-scoreable rows get NaN.
        """
        self._ensure_fitted()
        df = data.df.copy()

        for col in ["x", "y", "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        role, resolved_end_x, resolved_end_y = self._classify_events(df)

        move_success = _coerce_bool(df["move_success"], default=True)
        score_mask = (role == "move") & move_success

        x_arr = _clip_coords(df["x"].to_numpy(dtype=float, na_value=np.nan))
        y_arr = _clip_coords(df["y"].to_numpy(dtype=float, na_value=np.nan))
        ex_arr = _clip_coords(resolved_end_x.to_numpy(dtype=float, na_value=np.nan))
        ey_arr = _clip_coords(resolved_end_y.to_numpy(dtype=float, na_value=np.nan))

        start_cells = _safe_cell_index(x_arr, y_arr, self.l, self.w)
        end_cells = _safe_cell_index(ex_arr, ey_arr, self.l, self.w)

        surface_flat = self.surface_.reshape(-1)

        n = len(df)
        xt_start = np.full(n, np.nan)
        xt_end = np.full(n, np.nan)
        xt_added = np.full(n, np.nan)

        valid = score_mask.to_numpy() & (start_cells >= 0) & (end_cells >= 0)
        if valid.any():
            xt_start[valid] = surface_flat[start_cells[valid]]
            xt_end[valid] = surface_flat[end_cells[valid]]
            xt_added[valid] = xt_end[valid] - xt_start[valid]

        result = data.events.copy()
        result["xt_start"] = xt_start
        result["xt_end"] = xt_end
        result["xt_added"] = xt_added
        return result

    def value_at(self, x: float, y: float) -> float:
        """Return the xT value at the given (x, y) coordinate."""
        self._ensure_fitted()
        x_arr = _clip_coords(np.array([float(x)]))
        y_arr = _clip_coords(np.array([float(y)]))
        cell = _coords_to_cell(x_arr, y_arr, self.l, self.w)[0]
        return float(self.surface_.reshape(-1)[cell])

    def save(self, path: str) -> None:
        """Save the fitted model to an ``.npz`` file."""
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
        """Load a model from an ``.npz`` file."""
        arrays, metadata = load_xt_npz(path)
        if metadata.get("model_type") != "xt":
            raise ValueError("Not an xT model artifact")

        l, w = metadata.get("grid", [None, None])
        model = cls(
            l=l,
            w=w,
            include_carries=metadata.get("include_carries", True),
            include_throw_ins=metadata.get("include_throw_ins", True),
            include_goal_kicks=metadata.get("include_goal_kicks", True),
            include_corners=metadata.get("include_corners", True),
            include_free_kicks=metadata.get("include_free_kicks", True),
        )
        model.surface_ = arrays["surface"]
        model.shot_probability_ = arrays["shot_probability"]
        model.goal_probability_ = arrays["goal_probability"]
        model.move_probability_ = arrays["move_probability"]
        model.transition_matrix_ = arrays["transition_matrix"]
        model.metadata_ = metadata
        model.included_move_families_ = metadata.get("included_move_families", [])
        model.included_shot_families_ = metadata.get("included_shot_families", [])
        model.fitted_ = True
        return model

    def plot(self, pitch=None, **kwargs):
        """Plot the xT surface on a Pitch."""
        self._ensure_fitted()
        return plot_xt_surface(self.surface_, self.l, self.w, pitch=pitch, **kwargs)

    def _ensure_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise ValueError("Model is not fitted")
