import numpy as np
import plotly.graph_objects as go


def plot_scatter(
    fig, df, x="x", y="y", hover=None, text=None, color="blue", size=10, **kwargs
):
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers+text" if text else "markers",
            text=df[text] if text else None,
            hovertext=df[hover] if hover else None,
            marker=dict(size=size, color=color),
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
    bins=(30, 20),
    colorscale="Viridis",
    **kwargs,
):
    fig.add_trace(
        go.Histogram2d(
            x=df[x],
            y=df[y],
            xbins=dict(start=0, end=length, size=length / bins[0]),
            ybins=dict(start=0, end=width, size=width / bins[1]),
            colorscale=colorscale,
            colorbar=dict(title="Density"),
            **kwargs,
        )
    )


def plot_arrows(
    fig,
    df,
    x,
    y,
    x_end,
    y_end,
    color="black",
    width=2,
    arrowhead=2,
    arrowsize=1,
    **kwargs,
):
    """
    Plot clean directional arrows using Plotly shapes (non-interactive).

    Parameters:
    - fig: Plotly figure to draw on
    - df: DataFrame with x, y, x2, y2 columns
    - arrowhead: Plotly arrowhead style (0–7)
    - arrowsize: Scale of arrowhead
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
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=width,
            arrowcolor=color,
            opacity=1,
            standoff=0,
            hovertext="",
            text="",  # No label
        )


def plot_comets(
    fig,
    df,
    x="x",
    y="y",
    x_end="x2",
    y_end="y2",
    color="blue",
    width=3,
    segments=12,
    fade=True,
):
    """
    Plots comet-style directional lines using fading segments.

    Parameters:
    - fig: Plotly figure to draw on
    - df: DataFrame with start and end positions
    - x, y: start position columns
    - x_end, y_end: end position columns
    - color: base RGB color (e.g. 'blue', or 'rgb(0,0,255)')
    - width: max line width
    - segments: number of segments to split line into
    - fade: if True, opacity increases toward destination
    """
    for _, row in df.iterrows():
        x0, y0 = row[x], row[y]
        x1, y1 = row[x_end], row[y_end]

        for i in range(segments):
            t0 = i / segments
            t1 = (i + 1) / segments

            x_start = x0 + (x1 - x0) * t0
            y_start = y0 + (y1 - y0) * t0
            x_stop = x0 + (x1 - x0) * t1
            y_stop = y0 + (y1 - y0) * t1

            alpha = t1 if fade else 1.0
            rgba = (
                f"rgba(0, 0, 255, {alpha:.2f})"
                if "rgb" not in color and "rgba" not in color
                else color
            )

            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_stop],
                    y=[y_start, y_stop],
                    mode="lines",
                    line=dict(color=rgba, width=width),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
