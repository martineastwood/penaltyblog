from typing import Any, Callable, Dict, Optional

import pandas as pd


class PitchDimensions:
    # ─── the *target* drawing size ─────────────────────────────────────────
    DRAW_LENGTH: float = 105.0
    DRAW_WIDTH: float = 68.0

    def __init__(
        self,
        length: float,
        width: float,
        shapes: Optional[Dict[str, Dict[str, float]]] = None,
        _scale_fn: Optional[Callable] = None,
    ):
        """
        Initialize a PitchDimensions instance.

        Args:
            length: The length of the pitch in native units.
            width: The width of the pitch in native units.
            shapes: A dictionary of shapes to be used in the pitch.
            _scale_fn: A function to scale the shapes.
        """
        self.length = length
        self.width = width
        self.shapes = shapes or {}
        self._scale_fn = _scale_fn

    def __repr__(self) -> str:
        return f"<PitchDimensions length={self.length} width={self.width}>"

    # ─── accessors for draw size ────────────────────────────────────────────
    def get_draw_length(self) -> float:
        """
        Returns the draw length of the pitch.

        Returns:
            float: The draw length of the pitch.
        """
        return self.DRAW_LENGTH

    def get_draw_width(self) -> float:
        """
        Returns the draw width of the pitch.

        Returns:
            float: The draw width of the pitch.
        """
        return self.DRAW_WIDTH

    # ─── factory for known providers ────────────────────────────────────────
    @classmethod
    def from_provider(cls, provider: str) -> "PitchDimensions":
        """
        Returns a PitchDimensions instance for a known provider.

        Args:
            provider: The provider to use.

        Returns:
            PitchDimensions: A PitchDimensions instance for the provider.
        """
        p = provider.lower()
        if p == "statsbomb":
            return cls(
                length=120,
                width=80,
                shapes={
                    "penalty_area_left": {"x0": 0, "y0": 18, "x1": 18, "y1": 62},
                    "penalty_area_right": {"x0": 102, "y0": 18, "x1": 120, "y1": 62},
                    "six_yard_left": {"x0": 0, "y0": 30, "x1": 6, "y1": 50},
                    "six_yard_right": {"x0": 114, "y0": 30, "x1": 120, "y1": 50},
                    "penalty_spot_left": {"x": 12, "y": 40},
                    "penalty_spot_right": {"x": 108, "y": 40},
                    "center_circle": {"x": 60, "y": 40, "r": 9.15},
                    "halfway_line": {"x0": 60, "y0": 0, "x1": 60, "y1": 80},
                },
                _scale_fn=cls._scale_statsbomb,
            )
        elif p == "wyscout":
            return cls(
                length=100,
                width=100,
                shapes={
                    "penalty_area_left": {"x0": 0, "y0": 81, "x1": 16, "y1": 19},
                    "penalty_area_right": {"x0": 84, "y0": 19, "x1": 100, "y1": 81},
                    "six_yard_left": {"x0": 0, "y0": 63, "x1": 6, "y1": 37},
                    "six_yard_right": {"x0": 94, "y0": 37, "x1": 100, "y1": 63},
                    "penalty_spot_left": {"x": 10, "y": 50},
                    "penalty_spot_right": {"x": 90, "y": 50},
                    "center_circle": {"x": 50, "y": 50, "r": 9.15},
                    "halfway_line": {"x0": 50, "y0": 0, "x1": 50, "y1": 100},
                },
                _scale_fn=cls._scale_wyscout,
            )
        elif p == "opta":
            return cls(
                length=100,
                width=100,
                shapes={
                    "penalty_area_left": {"x0": 0, "y0": 21.1, "x1": 17, "y1": 78.9},
                    "penalty_area_right": {"x0": 83, "y0": 21.1, "x1": 100, "y1": 78.9},
                    "six_yard_left": {"x0": 0, "y0": 36.8, "x1": 5.8, "y1": 63.2},
                    "six_yard_right": {"x0": 94.2, "y0": 36.8, "x1": 100, "y1": 63.2},
                    "penalty_spot_left": {"x": 11.5, "y": 50},
                    "penalty_spot_right": {"x": 88.5, "y": 50},
                    "center_circle": {"x": 50, "y": 50, "r": 9.15},
                    "halfway_line": {"x0": 50, "y0": 0, "x1": 50, "y1": 100},
                },
                _scale_fn=cls._scale_opta,
            )
        elif p == "metrica":
            return cls(length=1, width=1, _scale_fn=cls._scale_default)
        else:
            raise ValueError(f"Unsupported provider: {provider!r}")

    # ─── public API ─────────────────────────────────────────────────────────
    def apply_coordinate_scaling(
        self, df: pd.DataFrame, x: str = "x", y: str = "y"
    ) -> pd.DataFrame:
        """
        Returns a new DataFrame with x,y scaled into the
        DRAW_LENGTH×DRAW_WIDTH plotting coords.

        Args:
            df: The DataFrame to scale.
            x: The column name for the x coordinate.
            y: The column name for the y coordinate.

        Returns:
            pd.DataFrame: A new DataFrame with x,y scaled into the
            DRAW_LENGTH×DRAW_WIDTH plotting coords.
        """
        if not self._scale_fn:
            return df.copy()
        return self._scale_fn(self, df.copy(), x, y)

    def scaled_shapes(
        self,
        target_length: Optional[float] = None,
        target_width: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns `self.shapes` with every number mapped into the
        DRAW_LENGTH×DRAW_WIDTH plotting space (or an override).

        Args:
            target_length: The target length of the pitch.
            target_width: The target width of the pitch.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary of shapes with every number mapped into the
            DRAW_LENGTH×DRAW_WIDTH plotting space (or an override).
        """

        L = target_length or self.DRAW_LENGTH
        W = target_width or self.DRAW_WIDTH

        def sx(v: float) -> float:
            # horizontal scale
            return v * (L / self.length)

        def sy(v: float, invert: bool = False) -> float:
            # vertical scale (optionally inverted)
            if invert:
                v = self.width - v
            return v * (W / self.width)

        # on providers whose native origin is top-left, we'll invert Y
        invert_y = self._scale_fn in {
            PitchDimensions._scale_wyscout,
            PitchDimensions._scale_statsbomb,
        }

        out: Dict[str, Dict[str, float]] = {}
        for name, shape in self.shapes.items():
            scaled: Dict[str, float] = {}
            for k, v in shape.items():
                if k == "r":
                    # radius always scales with horizontal axis
                    scaled[k] = sx(v)
                elif k.startswith("x"):
                    scaled[k] = sx(v)
                elif k.startswith("y"):
                    scaled[k] = sy(v, invert=invert_y)
                else:
                    scaled[k] = v
            out[name] = scaled
        return out

    def arc_radius(self, target_length: Optional[float] = None) -> float:
        """
        Returns the arc radius of the pitch.

        Args:
            target_length: The target length of the pitch.

        Returns:
            float: The arc radius of the pitch.
        """
        L = target_length or self.DRAW_LENGTH
        return 9.15 * (L / self.length)

    def apply_coordinate_scaling_raw(
        self,
        xs: list[float],
        ys: list[float],
    ) -> tuple[list[float], list[float]]:
        """
        Scale x and y coordinate lists into the DRAW_LENGTH × DRAW_WIDTH space.

        Mirrors apply_coordinate_scaling(), but operates on plain lists.

        Returns:
            tuple: (scaled_xs, scaled_ys)
        """
        if not self._scale_fn:
            return xs, ys

        scale_x = lambda x: x * (self.DRAW_LENGTH / self.length)
        scale_y = lambda y: y * (self.DRAW_WIDTH / self.width)

        # Wyscout & StatsBomb flip the y axis
        if self._scale_fn in {self._scale_wyscout, self._scale_statsbomb}:
            scale_y = lambda y: (self.width - y) * (self.DRAW_WIDTH / self.width)

        return [scale_x(x) for x in xs], [scale_y(y) for y in ys]

    # ─── private scaling functions ──────────────────────────────────────────
    @staticmethod
    def _scale_default(self, df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        """
        Scales x and y coordinates using a simple linear scaling.

        Args:
            self: The PitchDimensions instance.
            df: The DataFrame to scale.
            x: The column name for the x coordinate.
            y: The column name for the y coordinate.

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        df[x] = df[x] * (self.DRAW_LENGTH / self.length)
        df[y] = df[y] * (self.DRAW_WIDTH / self.width)
        return df

    @staticmethod
    def _scale_wyscout(self, df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        """
        Scales x and y coordinates using Wyscout's coordinate system.

        Wyscout's coordinates have the origin at the top-left corner of the pitch,
        so we need to flip the y axis.

        Args:
            self: The PitchDimensions instance.
            df: The DataFrame to scale.
            x: The column name for the x coordinate.
            y: The column name for the y coordinate.

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        df[x] = df[x] * (self.DRAW_LENGTH / self.length)
        df[y] = (self.width - df[y]) * (self.DRAW_WIDTH / self.width)
        return df

    @staticmethod
    def _scale_statsbomb(self, df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        """
        Scales x and y coordinates using StatsBomb's coordinate system.

        StatsBomb's coordinates have the origin at the top-left corner of the
        pitch, so we need to flip the y axis.

        Args:
            self: The PitchDimensions instance.
            df: The DataFrame to scale.
            x: The column name for the x coordinate.
            y: The column name for the y coordinate.

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        df[x] = df[x] * (self.DRAW_LENGTH / self.length)
        df[y] = (self.width - df[y]) * (self.DRAW_WIDTH / self.width)
        return df

    @staticmethod
    def _scale_opta(self, df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        """
        Scales x and y coordinates using Opta's coordinate system.

        Opta's coordinates have the origin at the center of the pitch, so we
        don't need to flip the y axis like we do with Wyscout and StatsBomb.

        Args:
            self: The PitchDimensions instance.
            df: The DataFrame to scale.
            x: The column name for the x coordinate.
            y: The column name for the y coordinate.

        Returns:
            pd.DataFrame: The scaled DataFrame.
        """
        df[x] = df[x] * (self.DRAW_LENGTH / self.length)
        df[y] = df[y] * (self.DRAW_WIDTH / self.width)
        return df
