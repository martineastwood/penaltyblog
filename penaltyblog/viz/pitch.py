import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .dimensions import PitchDimensions
from .plotting import plot_arrows, plot_comets, plot_heatmap, plot_scatter
from .theme import Theme


class Pitch:
    def __init__(
        self,
        width=800,
        height=600,
        theme="white",
        provider="statsbomb",
        orientation="horizontal",
        title=None,
        show_axis=False,
        show_legend=False,
        show_spots=True,
    ):
        self.width = width
        self.height = height
        self.theme = Theme(theme) if not isinstance(theme, Theme) else theme
        self.dimensions = (
            PitchDimensions.from_provider(provider)
            if not isinstance(provider, PitchDimensions)
            else provider
        )
        self.orientation = orientation
        self.title = title
        self.show_axis = show_axis
        self.show_legend = show_legend
        self.show_spots = show_spots

        self.fig = go.Figure()
        self._draw_pitch()

    def _draw_pitch(self):
        d = self.dimensions
        line_color = self.theme.line_color
        shapes = []

        # Get all scaled shapes based on 105x68 drawing space
        scaled_shapes = d.scaled_shapes(target_length=105, target_width=68)

        # Pitch boundary (drawn using scaled pitch space directly)
        shapes.append(
            dict(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color=line_color))
        )

        if "center_circle" in scaled_shapes:
            c = scaled_shapes["center_circle"]
            shapes.append(
                dict(
                    type="circle",
                    x0=c["x"] - c["r"],
                    y0=c["y"] - c["r"],
                    x1=c["x"] + c["r"],
                    y1=c["y"] + c["r"],
                    line=dict(color=line_color),
                )
            )

        if "halfway_line" in scaled_shapes:
            box = scaled_shapes["halfway_line"]
            shapes.append(
                dict(
                    type="rect",
                    x0=box["x0"],
                    y0=box["y0"],
                    x1=box["x1"],
                    y1=box["y1"],
                    line=dict(color=line_color),
                )
            )

        if "penalty_area_left" in scaled_shapes:
            box = scaled_shapes["penalty_area_left"]
            shapes.append(
                dict(
                    type="rect",
                    x0=box["x0"],
                    y0=box["y0"],
                    x1=box["x1"],
                    y1=box["y1"],
                    line=dict(color=line_color),
                )
            )

        if "penalty_area_right" in scaled_shapes:
            box = scaled_shapes["penalty_area_right"]
            shapes.append(
                dict(
                    type="rect",
                    x0=box["x0"],
                    y0=box["y0"],
                    x1=box["x1"],
                    y1=box["y1"],
                    line=dict(color=line_color),
                )
            )

        if "six_yard_left" in scaled_shapes:
            box = scaled_shapes["six_yard_left"]
            shapes.append(
                dict(
                    type="rect",
                    x0=box["x0"],
                    y0=box["y0"],
                    x1=box["x1"],
                    y1=box["y1"],
                    line=dict(color=line_color),
                )
            )

        if "six_yard_right" in scaled_shapes:
            box = scaled_shapes["six_yard_right"]
            shapes.append(
                dict(
                    type="rect",
                    x0=box["x0"],
                    y0=box["y0"],
                    x1=box["x1"],
                    y1=box["y1"],
                    line=dict(color=line_color),
                )
            )

        # Apply layout and aspect ratio lock
        self.fig.update_layout(
            shapes=shapes,
            width=self.width,
            height=self.height,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=self.theme.pitch_color,
            title=self.title,
            showlegend=self.show_legend,
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

        self._add_penalty_arcs()

    def _add_penalty_arcs(self):
        d = self.dimensions
        c = self.theme.line_color
        shapes = d.shapes

        if not all(k in shapes for k in ["penalty_spot_left", "penalty_spot_right"]):
            return

        def scale_x(x):
            return x * (105 / d.length)

        def scale_y(y):
            if d.apply_coordinate_scaling == d._scale_coordinates_wyscout:
                y = d.width - y
            return y * (68 / d.width)

        def draw_arc(spot_raw, theta_range):
            x_raw = spot_raw["x"]
            y_raw = spot_raw["y"]
            x = scale_x(x_raw)
            y = scale_y(y_raw)
            r = 10 * (105 / d.length)  # Arc radius scaled using pitch length

            theta = np.radians(np.linspace(*theta_range, 50))
            x_arc = x + r * np.cos(theta)
            y_arc = y + r * np.sin(theta)

            self.fig.add_trace(
                go.Scatter(
                    x=x_arc,
                    y=y_arc,
                    mode="lines",
                    line=dict(color=c),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Left arc (facing right)
        draw_arc(shapes["penalty_spot_left"], (308, 412))

        # Right arc (facing left)
        draw_arc(shapes["penalty_spot_right"], (128, 232))

        if self.show_spots:
            self._add_spots()

    def _add_spots(self):
        d = self.dimensions
        c = self.theme.line_color
        scaled_shapes = d.scaled_shapes()

        spots = []
        for key in ["penalty_spot_left", "penalty_spot_right", "center_circle"]:
            if key not in scaled_shapes:
                continue

            x = scaled_shapes[key]["x"]
            y = scaled_shapes[key]["y"]

            spots.append(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(size=6, color=c),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for trace in spots:
            self.fig.add_trace(trace)

    def plot_scatter(
        self, df, x="x", y="y", hover=None, tooltip_original=True, **kwargs
    ):
        df_scaled = self.dimensions.apply_coordinate_scaling(df, x=x, y=y)

        if tooltip_original and (x in df.columns and y in df.columns):
            if hover:
                df_scaled["hover_text"] = (
                    df[hover].astype(str)
                    + " ("
                    + df[x].round(1).astype(str)
                    + ", "
                    + df[y].round(1).astype(str)
                    + ")"
                )
            else:
                df_scaled["hover_text"] = (
                    "("
                    + df[x].round(1).astype(str)
                    + ", "
                    + df[y].round(1).astype(str)
                    + ")"
                )
            hover = "hover_text"
        elif hover:
            df_scaled["hover_text"] = df[hover]
            hover = "hover_text"
        else:
            hover = None

        plot_scatter(
            self.fig,
            df_scaled,
            x="x",
            y="y",
            hover=hover,
            color=self.theme.marker_color,
            **kwargs,
        )

    def plot_heatmap(self, df, x="x", y="y", **kwargs):
        plot_heatmap(
            self.fig,
            df,
            length=self.dimensions.length,
            width=self.dimensions.width,
            **kwargs,
        )

    def plot_arrows(
        self,
        df,
        x="x",
        y="y",
        x_end="x2",
        y_end="y2",
        color=None,
        tooltip_original=False,
        hover=None,
        **kwargs,
    ):
        df_scaled = self.dimensions.apply_coordinate_scaling(df, x=x, y=y)
        df_scaled[[x_end, y_end]] = self.dimensions.apply_coordinate_scaling(
            df[[x_end, y_end]], x=x_end, y=y_end
        )

        if tooltip_original and (x in df.columns and y in df.columns):
            df_scaled["hover_text"] = (
                df[hover]
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
            hover = "hover_text"
        elif hover:
            df_scaled["hover_text"] = df[hover]
            hover = "hover_text"
        else:
            hover = None

        if color is None:
            color = self.theme.marker_color

        plot_arrows(
            self.fig,
            df_scaled,
            x=x,
            y=y,
            x_end=x_end,
            y_end=y_end,
            color=color,
            hover=hover,
            **kwargs,
        )

    def plot_comets(
        self,
        df,
        x="x",
        y="y",
        x_end="x2",
        y_end="y2",
        hover=None,
        tooltip_original=False,
        **kwargs,
    ):
        df_scaled = self.dimensions.apply_coordinate_scaling(df.copy(), x=x, y=y)
        df_scaled[[x_end, y_end]] = self.dimensions.apply_coordinate_scaling(
            df[[x_end, y_end]].copy(), x=x_end, y=y_end
        )

        if tooltip_original and (x in df.columns and y in df.columns):
            df_scaled["hover_text"] = (
                df[hover]
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
            hover = "hover_text"
        elif hover:
            df_scaled["hover_text"] = df[hover]
            hover = "hover_text"
        else:
            hover = None

        plot_comets(
            self.fig,
            df_scaled,
            x=x,
            y=y,
            color=self.theme.line_color,
            hover=hover,
            **kwargs,
        )

    def show(self):
        pio.show(self.fig, config=dict(displaylogo=False))
