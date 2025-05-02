# penaltyblog/viz/theme.py


class Theme:
    PRESETS = {
        "white": {
            "pitch_color": "white",
            "line_color": "black",
            "marker_color": "blue",
        },
        "green": {
            "pitch_color": "#a8bc95",
            "line_color": "white",
            "marker_color": "white",
        },
        "black": {
            "pitch_color": "black",
            "line_color": "white",
            "marker_color": "white",
        },
        "grey": {
            "pitch_color": "#e5e5e5",
            "line_color": "black",
            "marker_color": "black",
        },
        "blue": {
            "pitch_color": "#1b2838",
            "line_color": "white",
            "marker_color": "white",
        },
    }

    def __init__(self, name="white"):
        self.name = name
        self.styles = self.PRESETS.get(name, self.PRESETS["white"])

    @property
    def pitch_color(self):
        return self.styles["pitch_color"]

    @property
    def line_color(self):
        return self.styles["line_color"]

    @property
    def marker_color(self):
        return self.styles["marker_color"]

    @classmethod
    def from_dict(cls, style_dict):
        theme = cls()
        theme.styles = style_dict
        return theme
