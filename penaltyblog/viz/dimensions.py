class PitchDimensions:
    def __init__(
        self,
        length,
        width,
        shapes=None,
        scale_coordinates=None,
    ):
        self.length = length
        self.width = width
        self.shapes = shapes or {}
        self._scale_coordinates_fn = scale_coordinates

    @classmethod
    def from_provider(cls, provider: str):
        provider = provider.lower()

        if provider == "statsbomb":
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
                    "center_circle": {"x": 60, "y": 40, "r": 10},
                    "halfway_line": {"x0": 60, "y0": 0, "x1": 60, "y1": 80},
                },
                scale_coordinates=cls._scale_coordinates_statsbomb,
            )

        elif provider == "wyscout":
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
                    "center_circle": {"x": 50, "y": 50, "r": 10},
                    "halfway_line": {"x0": 50, "y0": 0, "x1": 50, "y1": 100},
                },
                scale_coordinates=cls._scale_coordinates_wyscout,
            )

        elif provider == "metrica":
            return cls(
                length=1, width=1, scale_coordinates=cls._scale_coordinates_default
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _scale_coordinates_default(self, df, x="x", y="y"):
        df = df.copy()
        df[x] = df[x] * (105 / self.length)
        df[y] = df[y] * (68 / self.width)
        return df

    @staticmethod
    def _scale_coordinates_wyscout(self, df, x="x", y="y"):
        df = df.copy()
        df[x] = df[x] * (105 / self.length)
        df[y] = (self.width - df[y]) * (68 / self.width)
        return df

    @staticmethod
    def _scale_coordinates_statsbomb(self, df, x="x", y="y"):
        df = df.copy()
        df[x] = df[x] * (105 / self.length)
        df[y] = (self.width - df[y]) * (68 / self.width)
        return df

    def apply_coordinate_scaling(self, df, x="x", y="y"):
        if self._scale_coordinates_fn is None:
            return df
        return self._scale_coordinates_fn(self, df, x, y)

    def scaled_shapes(self, target_length=105, target_width=68):
        def scale_x(x):
            return x * (target_length / self.length)

        def scale_y(y):
            if self._scale_coordinates_fn == self._scale_coordinates_wyscout:
                y = self.width - y
            return y * (target_width / self.width)

        def scale_r(r):
            return r * (target_width / self.width)

        scaled = {}
        for key, shape in self.shapes.items():
            if isinstance(shape, dict):
                scaled_shape = {}
                for k, v in shape.items():
                    if k == "r":
                        scaled_shape[k] = scale_r(v)
                    elif "x" in k:
                        scaled_shape[k] = scale_x(v)
                    elif "y" in k:
                        scaled_shape[k] = scale_y(v)
                    else:
                        scaled_shape[k] = v
                scaled[key] = scaled_shape
            elif isinstance(shape, tuple):
                x, y = shape
                scaled[key] = (scale_x(x), scale_y(y))
            else:
                raise ValueError(f"Unsupported shape format: {shape}")
        return scaled

    def __repr__(self):
        return f"<PitchDimensions length={self.length} width={self.width}>"
