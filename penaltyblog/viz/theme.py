# penaltyblog/viz/theme.py


class Theme:
    # ––– Curated, designer themes –––––––––––––––––––––––––––––––––––––––––––––
    PRESETS = {
        "classic": {
            "pitch_color": "#a8bc95",  # grass green
            "line_color": "#ffffff",  # field lines
            "marker_color": "#1f77b4",  # blue accents
            "heatmap_colorscale": "Viridis",
            "heatmap_opacity": 0.6,
            "font_family": "Roboto, Arial, sans-serif",
            "line_width": 1.5,
            "marker_size": 10,
            "spot_size": 8,
        },
        "night": {
            "pitch_color": "#0f1e2e",  # deep navy
            "line_color": "#90caf9",  # light blue
            "marker_color": "#ffca28",  # amber
            "heatmap_colorscale": "Blues",
            "heatmap_opacity": 0.5,
            "font_family": "Montserrat, Helvetica, sans-serif",
            "line_width": 1.8,
            "marker_size": 12,
            "spot_size": 9,
        },
        "retro": {
            "pitch_color": "#f4ecd8",  # cream
            "line_color": "#6d4c41",  # brown
            "marker_color": "#d32f2f",  # brick red
            "heatmap_colorscale": "Cividis",
            "heatmap_opacity": 0.5,
            "font_family": "Courier New, monospace",
            "line_width": 1.2,
            "marker_size": 9,
            "spot_size": 7,
        },
        "minimal": {
            "pitch_color": "#ffffff",  # white
            "line_color": "#444444",  # dark grey
            "marker_color": "#e07a5f",  # medium grey
            "heatmap_colorscale": "Inferno",
            "heatmap_opacity": 0.8,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.0,
            "marker_size": 8,
            "spot_size": 6,
        },
    }

    # A fallback if a theme doesn’t specify a font
    DEFAULT_FONT_FAMILY = "Arial, sans-serif"

    def __init__(self, name: str = "classic"):
        self.name = name
        # look up the preset (or fall back to "classic")
        self.styles = self.PRESETS.get(name, self.PRESETS["classic"])

    # ––– Simple accessors for your plot code –––––––––––––––––––––––––––––––––
    @property
    def pitch_color(self) -> str:
        return self.styles["pitch_color"]

    @property
    def line_color(self) -> str:
        return self.styles["line_color"]

    @property
    def marker_color(self) -> str:
        return self.styles["marker_color"]

    @property
    def heatmap_colorscale(self) -> str:
        return self.styles["heatmap_colorscale"]

    @property
    def heatmap_opacity(self) -> float:
        return self.styles["heatmap_opacity"]

    @property
    def line_width(self) -> float:
        return self.styles["line_width"]

    @property
    def marker_size(self) -> float:
        return self.styles["marker_size"]

    @property
    def spot_size(self) -> float:
        return self.styles["spot_size"]

    @property
    def font_family(self) -> str:
        # if the theme forgot to specify it, use our default fallback
        return self.styles.get("font_family", self.DEFAULT_FONT_FAMILY)

    @classmethod
    def from_dict(cls, style_dict: dict) -> "Theme":
        """Create a custom theme on the fly from your own style dict."""
        t = cls()
        t.styles = style_dict
        return t
