"""Plotting utilities for xT surfaces."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..viz.pitch import Pitch


def plot_xt_surface(
    surface: np.ndarray,
    n_cols: int,
    n_rows: int,
    pitch: Optional[Pitch] = None,
    **kwargs,
) -> Pitch:
    """
    Plot an xT surface as a heatmap on a football pitch.

    Parameters
    ----------
    surface : numpy.ndarray
        xT surface array of shape ``(n_rows, n_cols)``.
    n_cols : int
        Number of columns in the grid (left-to-right).
    n_rows : int
        Number of rows in the grid (bottom-to-top).
    pitch : Pitch, optional
        A :class:`~penaltyblog.viz.pitch.Pitch` instance to draw on.
        If omitted, an Opta-style pitch is created automatically.
    **kwargs
        Optional keyword arguments:

        - ``colorscale`` — Plotly colorscale name or list.
        - ``opacity`` — float between 0 and 1.
        - ``show_colorbar`` — bool, whether to show the colour scale bar
          (default ``False``).

    Returns
    -------
    Pitch
        The :class:`~penaltyblog.viz.pitch.Pitch` instance with the xT
        heatmap layer added.

    Notes
    -----
    The xT surface is defined in normalised 0–100 coordinates.
    Provider-specific coordinate systems should be scaled before plotting.
    """
    if pitch is None:
        pitch = Pitch(provider="opta")

    x_centers = (np.arange(n_cols) + 0.5) * (100.0 / n_cols)
    y_centers = (np.arange(n_rows) + 0.5) * (100.0 / n_rows)

    x_scaled, y_scaled = pitch.dim.apply_coordinate_scaling_raw(
        x_centers.tolist(), y_centers.tolist()
    )
    x_plot, y_plot = pitch._apply_orientation_raw(x_scaled, y_scaled)

    colorscale = kwargs.pop("colorscale", None)
    opacity = kwargs.pop("opacity", None)

    trace = go.Heatmap(
        z=surface,
        x=x_plot,
        y=y_plot,
        colorscale=colorscale or pitch.theme.heatmap_colorscale,
        opacity=opacity if opacity is not None else pitch.theme.heatmap_opacity,
        showscale=kwargs.pop("show_colorbar", False),
        hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>xT: %{z:.3f}<extra></extra>",
    )

    if hasattr(pitch, "_add_layer"):
        pitch._add_layer("xt", trace)
    else:
        pitch.fig.add_trace(trace)

    return pitch
