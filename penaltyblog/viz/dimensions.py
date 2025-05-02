# penaltyblog/viz/dimensions.py


class PitchDimensions:
    def __init__(
        self,
        length,
        width,
        penalty_area_length=18,
        penalty_area_width=44,
        six_yard_box_length=6,
        six_yard_box_width=20,
        penalty_spot_distance=12,
        center_circle_radius=10,
    ):
        self.length = length
        self.width = width
        self.penalty_area_length = penalty_area_length
        self.penalty_area_width = penalty_area_width
        self.six_yard_box_length = six_yard_box_length
        self.six_yard_box_width = six_yard_box_width
        self.penalty_spot_distance = penalty_spot_distance
        self.center_circle_radius = center_circle_radius

    @classmethod
    def from_provider(cls, provider: str):
        provider = provider.lower()

        if provider == "statsbomb":
            return cls(
                length=120,
                width=80,
                penalty_area_length=18,
                penalty_area_width=44,
                six_yard_box_length=6,
                six_yard_box_width=20,
                penalty_spot_distance=12,
                center_circle_radius=10,
            )

        elif provider in ["opta", "wyscout"]:
            return cls(
                length=105,
                width=68,
                penalty_area_length=16.5,
                penalty_area_width=40.3,
                six_yard_box_length=5.5,
                six_yard_box_width=18.32,
                penalty_spot_distance=11,
                center_circle_radius=9.15,
            )

        elif provider in ["tracab", "skillcorner", "secondspectrum"]:
            return cls(
                length=105,
                width=68,
                penalty_area_length=16.5,
                penalty_area_width=40.3,
                six_yard_box_length=5.5,
                six_yard_box_width=18.32,
                penalty_spot_distance=11,
                center_circle_radius=9.15,
            )

        elif provider == "metrica":
            # normalized 0–1 pitch, common in computer vision
            return cls(
                length=1,
                width=1,
                penalty_area_length=0.157,
                penalty_area_width=0.375,
                six_yard_box_length=0.05,
                six_yard_box_width=0.25,
                penalty_spot_distance=0.092,
                center_circle_radius=0.087,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def custom(cls, **kwargs):
        return cls(**kwargs)

    def to_dict(self):
        return vars(self)

    def __repr__(self):
        return f"<PitchDimensions length={self.length} width={self.width}>"
