# penaltyblog/viz/theme.py


class Theme:
    PRESETS = {
        "classic": {
            "pitch_color": "#a8bc95",  # grass green
            "line_color": "#ffffff",  # field lines
            "marker_color": "#1f77b4",  # blue accents
            "heatmap_colorscale": "Viridis",
            "heatmap_opacity": 0.6,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.5,
            "marker_size": 10,
            "spot_size": 8,
            "hover_bgcolor": "rgba(255,255,255,0.9)",  # light on green
            "hover_font_color": "#1f1f1f",
            "hover_border_color": "rgba(0,0,0,0.2)",
            "hover_font_family": "Helvetica Neue, Arial, sans-serif",
            "hover_font_size": 16,
            "title_margin": 50,
            "subtitle_margin": 30,
            "subnote_margin": 50,
            "subtitle_font_size": 16,
            "subnote_font_size": 14,
        },
        "night": {
            "pitch_color": "#0f1e2e",  # deep navy
            "line_color": "#90caf9",  # light blue
            "marker_color": "#ffca28",  # amber
            "heatmap_colorscale": "Blues",
            "heatmap_opacity": 0.5,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.8,
            "marker_size": 12,
            "spot_size": 8,
            "hover_bgcolor": "rgba(255,255,255,0.15)",  # subtle white
            "hover_font_color": "white",
            "hover_border_color": "rgba(144,202,249,0.6)",
            "hover_font_family": "Helvetica Neue, Arial, sans-serif",
            "hover_font_size": 16,
            "title_margin": 50,
            "subtitle_margin": 30,
            "subnote_margin": 50,
            "subtitle_font_size": 16,
            "subnote_font_size": 14,
        },
        "retro": {
            "pitch_color": "#f4ecd8",  # cream
            "line_color": "#6d4c41",  # brown
            "marker_color": "#d32f2f",  # brick red
            "heatmap_colorscale": "Cividis",
            "heatmap_opacity": 0.5,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.2,
            "marker_size": 9,
            "spot_size": 8,
            "hover_bgcolor": "rgba(109,76,65,0.85)",  # matches brown
            "hover_font_color": "white",
            "hover_border_color": "rgba(255,255,255,0.2)",
            "hover_font_family": "Helvetica Neue, Arial, sans-serif",
            "hover_font_size": 16,
            "title_margin": 50,
            "subtitle_margin": 30,
            "subnote_margin": 50,
            "subtitle_font_size": 16,
            "subnote_font_size": 14,
        },
        "minimal": {
            "pitch_color": "#ffffff",  # white
            "line_color": "#444444",  # dark grey
            "marker_color": "#e07a5f",  # accent
            "heatmap_colorscale": "Inferno",
            "heatmap_opacity": 0.8,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.0,
            "marker_size": 8,
            "spot_size": 6,
            "hover_bgcolor": "rgba(50,50,50,0.8)",
            "hover_font_color": "white",
            "hover_border_color": "rgba(255,255,255,0.2)",
            "hover_font_family": "Helvetica Neue, Arial, sans-serif",
            "hover_font_size": 16,
            "title_margin": 50,
            "subtitle_margin": 30,
            "subnote_margin": 50,
            "subtitle_font_size": 16,
            "subnote_font_size": 14,
        },
        "turf": {
            "pitch_color": "#7a9c72",  # rich turf green
            "line_color": "#ffffff",  # crisp white lines
            "marker_color": "#ffca28",
            "heatmap_colorscale": "Greens",  # consistent with the green theme
            "heatmap_opacity": 0.6,
            "font_family": "Helvetica Neue, Arial, sans-serif",
            "line_width": 1.4,
            "marker_size": 10,
            "spot_size": 8,
            "hover_bgcolor": "rgba(40,60,40,0.9)",
            "hover_font_color": "white",
            "hover_border_color": "rgba(255,255,255,0.2)",
            "hover_font_family": "Helvetica Neue, Arial, sans-serif",
            "hover_font_size": 15,
            "title_margin": 50,
            "subtitle_margin": 30,
            "subnote_margin": 50,
            "subtitle_font_size": 16,
            "subnote_font_size": 14,
        },
    }

    # A fallback if a theme doesn’t specify a font
    DEFAULT_FONT_FAMILY = "Helvetica Neue, Arial, sans-serif"

    def __init__(self, name: str = "minimal"):
        """
        Initialize a Theme from a preset name.

        Parameters
        ----------
        name : str, optional
            The name of the theme preset to use. If not provided, defaults to "minimal".

        Notes
        -----
        If the theme name is not recognized, the class will fall back to "minimal".
        """
        self.name = name
        # look up the preset (or fall back to "classic")
        self.styles = self.PRESETS.get(name, self.PRESETS["minimal"])

    # ––– Simple accessors for your plot code –––––––––––––––––––––––––––––––––
    @property
    def pitch_color(self) -> str:
        """
        The color of the pitch background.

        Returns
        -------
        str
            The hex color code for the pitch background.
        """
        return self.styles["pitch_color"]

    @property
    def line_color(self) -> str:
        """
        The color of the pitch lines.

        Returns
        -------
        str
            The hex color code for the pitch lines.
        """
        return self.styles["line_color"]

    @property
    def marker_color(self) -> str:
        """
        The color of the pitch markers.

        Returns
        -------
        str
            The hex color code for the pitch markers.
        """
        return self.styles["marker_color"]

    @property
    def heatmap_colorscale(self) -> str:
        """
        The color scale used for heatmaps.

        Returns
        -------
        str
            The name of the color scale used for heatmaps.
        """
        return self.styles["heatmap_colorscale"]

    @property
    def heatmap_opacity(self) -> float:
        """
        The opacity of the heatmap.

        Returns
        -------
        float
            The opacity of the heatmap.
        """
        return self.styles["heatmap_opacity"]

    @property
    def line_width(self) -> float:
        """
        The width of the pitch lines.

        Returns
        -------
        float
            The width of the pitch lines.
        """
        return self.styles["line_width"]

    @property
    def marker_size(self) -> float:
        """
        The size of the pitch markers.

        Returns
        -------
        float
            The size of the pitch markers.
        """
        return self.styles["marker_size"]

    @property
    def spot_size(self) -> float:
        """
        The size of the pitch spots.

        Returns
        -------
        float
            The size of the pitch spots.
        """
        return self.styles["spot_size"]

    @property
    def font_family(self) -> str:
        """
        The font family used for text.

        Returns
        -------
        str
            The font family used for text.
        """
        # if the theme forgot to specify it, use our default fallback
        return self.styles.get("font_family", self.DEFAULT_FONT_FAMILY)

    @property
    def hover_font_color(self) -> str:
        """
        The color of the hover font.

        Returns
        -------
        str
            The color of the hover font.
        """
        return self.styles.get("hover_font_color", "black")

    @property
    def hover_border_color(self) -> str:
        """
        The color of the border of the hover box.

        Returns
        -------
        str
            The color of the border of the hover box.
        """
        return self.styles.get("hover_border_color", "#ccc")

    @property
    def hover_font_family(self) -> str:
        """
        The font family used for hover text.

        Returns
        -------
        str
            The font family used for hover text.
        """
        return self.styles.get("hover_font_family", "Arial, sans‐serif")

    @property
    def hover_font_size(self) -> float:
        """
        The font size of the hover text.

        Returns
        -------
        float
            The font size of the hover text.
        """
        return self.styles.get("hover_font_size", 11)

    @property
    def title_margin(self) -> float:
        """
        The margin for the title.

        Returns
        -------
        float
            The margin for the title.
        """
        return self.styles.get("title_margin", 50)

    @property
    def subtitle_margin(self) -> float:
        """
        The margin for the subtitle.

        Returns
        -------
        float
            The margin for the subtitle.
        """
        return self.styles.get("subtitle_margin", 30)

    @property
    def subnote_margin(self) -> float:
        """
        The margin for the subnote.

        Returns
        -------
        float
            The margin for the subnote.
        """
        return self.styles.get("subnote_margin", 50)

    @property
    def subtitle_font_size(self) -> float:
        """
        The font size for the subtitle.

        Returns
        -------
        float
            The font size for the subtitle.
        """
        return self.styles.get("subtitle_font_size", 16)

    @property
    def subnote_font_size(self) -> float:
        """
        The font size for the subnote.

        Returns
        -------
        float
            The font size for the subnote.
        """
        return self.styles.get("subnote_font_size", 14)

    @property
    def hover_bgcolor(self) -> str:
        return self.styles.get("hover_bgcolor", "rgba(50,50,50,0.8)")

    @classmethod
    def from_dict(cls, style_dict: dict, base: str = "minimal") -> "Theme":
        """
        Create a Theme instance from a dictionary of style settings.

        Parameters
        ----------
        style_dict : dict
            A dictionary containing style settings.
        base : str, optional
            The name of a preset theme to use as a base, by default "minimal".

        Returns
        -------
        Theme
            A Theme instance with the specified style settings.
        """
        base_styles = cls.PRESETS.get(base, {})
        merged = {**base_styles, **style_dict}
        t = cls()
        t.styles = merged
        return t
