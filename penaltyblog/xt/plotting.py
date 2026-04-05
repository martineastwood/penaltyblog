from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from penaltyblog.viz.pitch import Pitch


def plot_xt_surface(
    surface: np.ndarray,
    l: int,
    w: int,
    pitch: Optional[Pitch] = None,
    **kwargs,
):
    """
    Plot an xT surface on a Pitch using a plotly Heatmap.

    If ``pitch`` is omitted, an Opta-style pitch is created. The xT surface is
    defined in normalized 0..100 coordinates, so provider-specific coordinate
    systems should be scaled before plotting.

    Returns the Pitch instance.
    """
    if pitch is None:
        pitch = Pitch(provider="opta")

    x_centers = (np.arange(l) + 0.5) * (100.0 / l)
    y_centers = (np.arange(w) + 0.5) * (100.0 / w)

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
