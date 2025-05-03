from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as _go
import plotly.io as pio
from matplotlib import colors as _mcolors

from .dimensions import PitchDimensions
from .plotting import plot_arrows as plot_arrows
from .plotting import plot_comets as plot_comets
from .plotting import plot_heatmap
from .plotting import plot_scatter as plot_scatter
from .theme import Theme


class Pitch:
    AXIS_MARGIN = 5

    def __init__(
        self,
        provider: str = "statsbomb",
        width: int = 800,
        height: int = 600,
        theme: str = "minimal",
        orientation: str = "horizontal",
        title: Optional[str] = None,
        show_axis: bool = False,
        show_legend: bool = False,
        show_spots: bool = True,
    ):
        # public settings
        self.width = width
        self.height = height
        self.show_axis = show_axis
        self.show_legend = show_legend
        self.show_spots = show_spots
        self.orientation = orientation
        self.title = title

        # theme + dimensions
        self.theme = Theme(theme)
        self.dim = (
            PitchDimensions.from_provider(provider)
            if isinstance(provider, str)
            else provider
        )

        self.L = self.dim.get_draw_length()
        self.W = self.dim.get_draw_width()

        # figure
        self.fig = go.Figure()
        self._draw_base_pitch()

    def _draw_base_pitch(self) -> None:
        """Draw pitch boundary, boxes, center circle, arcs, optional spots."""
        lc = self.theme.line_color
        shapes = []
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)

        # 1) Border
        shapes.append(self._rect(0, 0, self.L, self.W, lc))

        # 2) Boxes & halfway
        for key in (
            "halfway_line",
            "penalty_area_left",
            "penalty_area_right",
            "six_yard_left",
            "six_yard_right",
        ):
            if b := scaled.get(key):
                shapes.append(self._rect(b["x0"], b["y0"], b["x1"], b["y1"], lc))

        # 3) Center circle
        if c := scaled.get("center_circle"):
            shapes.append(self._circle(c["x"], c["y"], c["r"], lc))

        # apply shapes + axis config
        self.fig.update_layout(
            shapes=shapes,
            width=self.width,
            height=self.height,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=self.theme.pitch_color,
            title=self.title,
            showlegend=self.show_legend,
            font=dict(
                family=self.theme.font_family,
                color=self.theme.line_color,
            ),
            title_font=dict(
                family=self.theme.font_family,
                color=self.theme.line_color,
            ),
            paper_bgcolor=self.theme.pitch_color,
        )
        self.fig.update_xaxes(
            range=[-5, 110],
            scaleanchor="y",
            constrain="domain",
            fixedrange=True,
            showticklabels=self.show_axis,
            visible=self.show_axis,
            showgrid=False,
            zeroline=False,
        )
        self.fig.update_yaxes(
            range=[-5, 73],
            constrain="domain",
            fixedrange=True,
            showticklabels=self.show_axis,
            visible=self.show_axis,
            showgrid=False,
            zeroline=False,
        )

        # 4) arcs + spots
        self._draw_penalty_arcs()
        if self.show_spots:
            self._draw_spots()

    def _rect(
        self, x0: float, y0: float, x1: float, y1: float, color: str
    ) -> Dict[str, Any]:
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color))

    def _circle(self, cx, cy, r, color) -> Dict[str, Any]:
        return dict(
            type="circle",
            x0=cx - r,
            y0=cy - r,
            x1=cx + r,
            y1=cy + r,
            line=dict(color=color),
        )

    def _draw_penalty_arcs(self) -> None:
        """
        Draw the two penalty-area semicircles that exactly meet the box line.
        """
        lc = self.theme.line_color
        raw = self.dim.shapes
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)
        r_plot = 9.15 * (self.L / self.dim.length)  # 9.15m scaled

        def draw(spot_key: str, area_key: str, left: bool) -> None:
            spot = scaled[spot_key]
            area = scaled[area_key]
            dx = (area["x1"] - spot["x"]) if left else (spot["x"] - area["x0"])
            θ_max = np.degrees(np.arccos(dx / r_plot))
            angles = (
                np.linspace(-θ_max, +θ_max, 50)
                if left
                else np.linspace(180 - θ_max, 180 + θ_max, 50)
            )
            th = np.radians(angles)
            xs = spot["x"] + r_plot * np.cos(th)
            ys = spot["y"] + r_plot * np.sin(th)

            self.fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=lc),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if "penalty_spot_left" in raw and "penalty_area_left" in raw:
            draw("penalty_spot_left", "penalty_area_left", left=True)
        if "penalty_spot_right" in raw and "penalty_area_right" in raw:
            draw("penalty_spot_right", "penalty_area_right", left=False)

    def _draw_spots(self) -> None:
        """Add center‐spot and penalty‐spots as small markers."""
        lc = self.theme.line_color
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)
        for key in ("penalty_spot_left", "penalty_spot_right", "center_circle"):
            if p := scaled.get(key):
                self.fig.add_trace(
                    go.Scatter(
                        x=[p["x"]],
                        y=[p["y"]],
                        mode="markers",
                        marker=dict(size=6, color=lc),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    def _prepare_hover(
        self, df, x: str, y: str, hover: Optional[str], tooltip_orig: bool
    ) -> Tuple[Any, Optional[str]]:
        """
        Scale (x,y), and if tooltip_orig is True, build a `hover_text` column
        showing the original (rounded) coords plus any hover‐label.
        """
        df2 = self.dim.apply_coordinate_scaling(df.copy(), x=x, y=y)
        if tooltip_orig and x in df.columns and y in df.columns:
            xs = df[x].round(1).astype(str)
            ys = df[y].round(1).astype(str)
            if hover:
                df2["hover_text"] = df[hover].astype(str) + " (" + xs + ", " + ys + ")"
            else:
                df2["hover_text"] = "(" + xs + ", " + ys + ")"
            return df2, "hover_text"
        elif hover:
            df2["hover_text"] = df[hover].astype(str)
            return df2, "hover_text"
        else:
            return df2, None

    def plot_scatter(
        self,
        df,
        x: str = "x",
        y: str = "y",
        hover: Optional[str] = None,
        tooltip_original: bool = True,
        **kwargs,
    ) -> None:
        df2, hover_col = self._prepare_hover(df, x, y, hover, tooltip_original)
        plot_scatter(
            self.fig,
            df2,
            x=x,
            y=y,
            hover=hover_col,
            color=self.theme.marker_color,
            **kwargs,
        )

    def plot_arrows(
        self,
        df,
        x: str = "x",
        y: str = "y",
        x_end: str = "x2",
        y_end: str = "y2",
        hover: Optional[str] = None,
        tooltip_original: bool = False,
        color: Optional[str] = None,
        **kwargs,
    ) -> None:
        df2, hover_col = self._prepare_hover(df, x, y, hover, tooltip_original)
        df2[[x_end, y_end]] = self.dim.apply_coordinate_scaling(
            df[[x_end, y_end]].copy(), x=x_end, y=y_end
        )
        plot_arrows(
            self.fig,
            df2,
            x=x,
            y=y,
            x_end=x_end,
            y_end=y_end,
            color=color or self.theme.marker_color,
            hover=hover_col,
            **kwargs,
        )

    def plot_comets(
        self,
        df,
        x: str = "x",
        y: str = "y",
        x_end: str = "x2",
        y_end="y2",
        color=None,
        width: int = 3,
        segments: int = 12,
        fade: bool = True,
        hover=None,
        tooltip_original=False,
        **kwargs,
    ):
        """
        Plot comet-style trails: fading line segments from (x,y) to (x_end,y_end).
        """
        # 1) scale both start and end coordinates
        # copy so we don’t clobber the user’s DF
        df2 = df.copy()
        df2 = self.dim.apply_coordinate_scaling(df2, x=x, y=y)
        df2[[x_end, y_end]] = self.dim.apply_coordinate_scaling(
            df[[x_end, y_end]].copy(), x=x_end, y=y_end
        )

        # 2) prepare hover-text column if user wants original coords in tooltip
        if tooltip_original and (x in df.columns and y in df.columns):
            hover_txt = (
                df[hover].astype(str)
                + " ("
                + df[x].round(1).astype(str)
                + ", "
                + df[y].round(1).astype(str)
                + ")"
                if hover
                else "("
                + df[x].round(1).astype(str)
                + ", "
                + df[y].round(1).astype(str)
                + ")"
            )
            df2["__hover__"] = hover_txt
            hover_key = "__hover__"
        elif hover:
            df2["__hover__"] = df[hover].astype(str)
            hover_key = "__hover__"
        else:
            hover_key = None

        # 3) default to marker color (usually blue), not the pitch line color
        if color is None:
            color = self.theme.marker_color

        # 4) draw the comets
        # inline helper so you don’t need an external import:
        def _to_rgb(c):
            c = c.strip()
            if c.startswith(("rgba(", "rgb(")):
                vals = c[c.find("(") + 1 : c.find(")")].split(",")
                return tuple(map(int, vals[:3]))
            else:
                return tuple(int(255 * x) for x in _mcolors.to_rgb(c))

        r0, g0, b0 = _to_rgb(color)

        for _, row in df2.iterrows():
            x0, y0 = row[x], row[y]
            x1, y1 = row[x_end], row[y_end]
            txt = row[hover_key] if hover_key else None

            for i in range(segments):
                t0, t1 = i / segments, (i + 1) / segments
                xs = [x0 + (x1 - x0) * t0, x0 + (x1 - x0) * t1]
                ys = [y0 + (y1 - y0) * t0, y0 + (y1 - y0) * t1]
                alpha = t1 if fade else 1.0
                rgba = f"rgba({r0},{g0},{b0},{alpha:.2f})"

                self.fig.add_trace(
                    _go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=rgba, width=width),
                        hovertext=txt if (txt and i == segments - 1) else None,
                        hoverinfo="text" if (txt and i == segments - 1) else "skip",
                        showlegend=False,
                        **kwargs,
                    )
                )

    # inside penaltyblog/viz/pitch.py

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        bins: tuple[int, int] = (10, 8),
        show_colorbar: bool = False,
        colorscale: str | None = None,
        opacity: float | None = None,
        **kwargs,
    ) -> None:
        """
        Draw a 2D density (heatmap) inside the 105×68 pitch,
        using theme defaults for colourscale & opacity.
        """
        # 1) Scale raw coords into DRAW_LENGTH×DRAW_WIDTH
        df_scaled = self.dim.apply_coordinate_scaling(df, x=x, y=y)

        # 2) Decide on colourscale & opacity (theme first, then user override)
        cs = colorscale or self.theme.heatmap_colorscale
        op = opacity if opacity is not None else self.theme.heatmap_opacity

        # 3) Delegate to your helper
        plot_heatmap(
            self.fig,
            df_scaled,
            x=x,
            y=y,
            length=self.dim.get_draw_length(),
            width=self.dim.get_draw_width(),
            bins=bins,
            colorscale=cs,
            opacity=op,
            show_colorbar=show_colorbar,
            **kwargs,
        )

    def show(self) -> None:
        pio.show(self.fig, config=dict(displaylogo=False))
