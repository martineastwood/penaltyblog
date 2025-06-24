from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def plot_scatter(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    *,
    color: str = "blue",
    size: int = 10,
    text: Optional[str] = None,
    hover: Optional[str] = None,
    showlegend: bool = False,
    **kwargs: Any,
) -> None:
    """
    Add a scatter trace of points (and optional labels) to `fig`.

    Parameters
    ----------
    fig
      Plotly Figure to add to.
    df
      DataFrame containing at least columns `x`, `y`.
    x, y
      Column names for coordinates.
    color
      Marker color.
    size
      Marker size.
    text
      Column name whose values are rendered as inside‐marker labels.
    hover
      Column name whose values populate the hover‐tooltip.
    showlegend
      Whether to include this trace in the legend.
    **kwargs
      Passed on to go.Scatter.
    """
    mode = "markers+text" if text else "markers"
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode=mode,
            marker=dict(size=size, color=color),
            text=df[text] if text else None,
            hovertext=df[hover] if hover else None,
            hoverinfo="text" if hover or text else "skip",
            showlegend=showlegend,
            **kwargs,
        )
    )


def plot_heatmap(
    fig,
    df,
    x="x",
    y="y",
    length=120,
    width=80,
    bins=(10, 8),
    colorscale="Viridis",
    show_colorbar: bool = False,
    opacity: float = 0.6,
    **kwargs,
):

    hover_template = f"x: %{{x:.1f}}<br>" f"y: %{{y:.1f}}<br>" "z: %{z}<extra></extra>"

    fig.add_trace(
        go.Histogram2d(
            x=df[x],
            y=df[y],
            xbins=dict(start=0, end=length, size=length / bins[0]),
            ybins=dict(start=0, end=width, size=width / bins[1]),
            colorscale=colorscale,
            colorbar=dict(title="Density"),
            showscale=show_colorbar,
            opacity=opacity,
            hovertemplate=hover_template,
            **kwargs,
        )
    )


def plot_arrows(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str,
    y: str,
    x_end: str,
    y_end: str,
    *,
    color: str = "black",
    width: float = 2,
    arrowhead: int = 2,
    arrowsize: float = 1.0,
    hover: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Draw directional arrows (annotations) from (x,y) to (x_end,y_end).

    Parameters
    ----------
    fig
        Plotly Figure to add arrows to.
    df
        DataFrame with start/end coordinate columns.
    x, y, x_end, y_end
        Column names for arrow tail and head positions.
    color
        Arrow color.
    width
        Arrow line width.
    arrowhead
        Arrowhead style (0–7).
    arrowsize
        Arrowhead size multiplier.
    hover
        Optional column name whose value appears on hover at the arrow head.
    **kwargs
        Passed through to fig.add_annotation.
    """
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row[x_end],
            y=row[y_end],
            ax=row[x],
            ay=row[y],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=arrowsize,
            arrowwidth=width,
            arrowcolor=color,
            opacity=1.0,
            hovertext=str(row[hover]) if hover else None,
            **kwargs,
        )


def plot_comets(
    fig,
    df,
    x="x",
    y="y",
    x_end="x2",
    y_end="y2",
    color="blue",  # default can be overridden by caller
    width=3,
    segments=12,
    fade=True,
    hover=None,
):
    """
    Plots comet-style directional lines using fading segments.
    """

    # Precompute an (r,g,b) tuple from whatever color the user passed
    def make_rgb_tuple(c):
        c = c.strip()
        if c.startswith(("rgba(", "rgb(")):
            # strip off rgba/(), split, take first 3 ints
            vals = c[c.find("(") + 1 : c.find(")")].split(",")
            r, g, b = map(int, vals[:3])
        else:
            # named color → convert via matplotlib
            r, g, b = [int(255 * x) for x in mcolors.to_rgb(c)]
        return r, g, b

    base_r, base_g, base_b = make_rgb_tuple(color)

    for _, row in df.iterrows():
        x0, y0 = row[x], row[y]
        x1, y1 = row[x_end], row[y_end]
        hover_text = str(row[hover]) if hover else None

        for i in range(segments):
            t0 = i / segments
            t1 = (i + 1) / segments

            xs = [x0 + (x1 - x0) * t0, x0 + (x1 - x0) * t1]
            ys = [y0 + (y1 - y0) * t0, y0 + (y1 - y0) * t1]

            alpha = t1 if fade else 1.0
            rgba = f"rgba({base_r},{base_g},{base_b},{alpha:.2f})"

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=rgba, width=width),
                    hovertext=hover_text if (hover and i == segments - 1) else None,
                    hoverinfo="text" if (hover and i == segments - 1) else "skip",
                    showlegend=False,
                )
            )


def plot_kde(
    fig,
    df,
    x: str = "x",
    y: str = "y",
    length: float = 105,
    width: float = 68,
    grid_size: int = 100,
    bandwidth: float | None = None,
    colorscale: str = "Viridis",
    opacity: float = 0.6,
    show_colorbar: bool = False,
    **kwargs,
):
    """
    Plot a smooth 2D kernel‐density estimate over the pitch.

    Parameters
    ----------
    fig
        A plotly.graph_objects.Figure
    df
        DataFrame containing at least columns `x`, `y` (already in plot units).
    x, y
        Column names for coordinates.
    length, width
        Plot extents (usually 105×68).
    grid_size
        Number of grid points per axis (higher = smoother, slower).
    bandwidth
        'bw_method' passed to scipy.stats.gaussian_kde; if None, defaults to Silverman’s rule.
    colorscale
        Plotly colorscale name.
    opacity
        Heatmap opacity.
    show_colorbar
        Whether to show the colorbar.
    **kwargs
        Additional args passed to go.Heatmap.
    """
    # 1. Build evaluation grid
    xs = np.linspace(0, length, grid_size)
    ys = np.linspace(0, width, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    # 2. Fit KDE on the raw points
    values = np.vstack([df[x], df[y]])
    kde = gaussian_kde(values, bw_method=bandwidth)

    # 3. Evaluate density on grid, then reshape
    zz = kde(grid_coords)
    zz = zz.reshape(grid_size, grid_size)

    # 4. Draw as a smooth heatmap
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=zz,
            colorscale=colorscale,
            opacity=opacity,
            showscale=show_colorbar,
            **kwargs,
        )
    )
