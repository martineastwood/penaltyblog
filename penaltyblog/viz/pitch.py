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

        self.fig = go.Figure()
        self._draw_pitch()

    def _draw_pitch(self):
        d = self.dimensions
        line_color = self.theme.line_color
        shapes = []

        # Pitch boundary
        shapes.append(
            dict(
                type="rect",
                x0=0,
                y0=0,
                x1=d.length,
                y1=d.width,
                line=dict(color=line_color),
            )
        )

        # Center line
        shapes.append(
            dict(
                type="line",
                x0=d.length / 2,
                y0=0,
                x1=d.length / 2,
                y1=d.width,
                line=dict(color=line_color),
            )
        )

        # Center circle
        shapes.append(
            dict(
                type="circle",
                x0=d.length / 2 - d.center_circle_radius,
                y0=d.width / 2 - d.center_circle_radius,
                x1=d.length / 2 + d.center_circle_radius,
                y1=d.width / 2 + d.center_circle_radius,
                line=dict(color=line_color),
            )
        )

        # Left penalty area
        shapes.append(
            dict(
                type="rect",
                x0=0,
                y0=(d.width - d.penalty_area_width) / 2,
                x1=d.penalty_area_length,
                y1=(d.width + d.penalty_area_width) / 2,
                line=dict(color=line_color),
            )
        )

        # Right penalty area
        shapes.append(
            dict(
                type="rect",
                x0=d.length - d.penalty_area_length,
                y0=(d.width - d.penalty_area_width) / 2,
                x1=d.length,
                y1=(d.width + d.penalty_area_width) / 2,
                line=dict(color=line_color),
            )
        )

        # Left six-yard box
        shapes.append(
            dict(
                type="rect",
                x0=0,
                y0=(d.width - d.six_yard_box_width) / 2,
                x1=d.six_yard_box_length,
                y1=(d.width + d.six_yard_box_width) / 2,
                line=dict(color=line_color),
            )
        )

        # Right six-yard box
        shapes.append(
            dict(
                type="rect",
                x0=d.length - d.six_yard_box_length,
                y0=(d.width - d.six_yard_box_width) / 2,
                x1=d.length,
                y1=(d.width + d.six_yard_box_width) / 2,
                line=dict(color=line_color),
            )
        )

        # Layout update
        self.fig.update_layout(
            shapes=shapes,
            xaxis=dict(
                showgrid=False,
                range=[-5, d.length + 5],
                zeroline=False,
                showticklabels=self.show_axis,
                visible=self.show_axis,
            ),
            yaxis=dict(
                showgrid=False,
                range=[-5, d.width + 5],
                zeroline=False,
                showticklabels=self.show_axis,
                visible=self.show_axis,
            ),
            width=self.width,
            height=self.height,
            plot_bgcolor=self.theme.pitch_color,
            title=self.title,
            showlegend=self.show_legend,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        self.fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=self.show_axis,
            visible=self.show_axis,
            range=[-5, self.dimensions.length + 5],
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True,
        )

        self.fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=self.show_axis,
            visible=self.show_axis,
            range=[-5, self.dimensions.width + 5],
            constrain="domain",
            fixedrange=True,
        )

        self._add_penalty_arcs()

    def _add_penalty_arcs(self):
        d = self.dimensions
        color = self.theme.line_color
        radius = d.center_circle_radius
        spot = d.penalty_spot_distance

        # Left arc
        theta_left = np.radians(np.linspace(308, 412, 50))
        x_left = spot + radius * np.cos(theta_left)
        y_left = d.width / 2 + radius * np.sin(theta_left)

        self.fig.add_trace(
            go.Scatter(
                x=x_left,
                y=y_left,
                mode="lines",
                line=dict(color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Right arc
        theta_right = np.radians(np.linspace(128, 232, 50))
        x_right = d.length - spot + radius * np.cos(theta_right)
        y_right = d.width / 2 + radius * np.sin(theta_right)

        self.fig.add_trace(
            go.Scatter(
                x=x_right,
                y=y_right,
                mode="lines",
                line=dict(color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def plot_scatter(self, df, **kwargs):
        plot_scatter(self.fig, df, color=self.theme.marker_color, **kwargs)

    def plot_heatmap(self, df, **kwargs):
        plot_heatmap(
            self.fig,
            df,
            length=self.dimensions.length,
            width=self.dimensions.width,
            **kwargs,
        )

    def plot_arrows(self, df, color=None, **kwargs):
        if color is None:
            color = self.theme.marker_color
        plot_arrows(self.fig, df, color=color, **kwargs)

    def plot_comets(self, df, **kwargs):
        plot_comets(self.fig, df, color=self.theme.line_color, **kwargs)

    def show(self):
        pio.show(self.fig, config=dict(displaylogo=False))
