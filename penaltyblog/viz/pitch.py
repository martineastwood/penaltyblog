from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde

from .dimensions import PitchDimensions
from .theme import Theme


def _layered(default_layer: str):
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        def wrapper(
            self,
            *args,
            return_trace: bool = False,
            layer: Optional[str] = None,
            **kwargs,
        ):
            items = fn(self, *args, **kwargs)
            if items is None:
                return None
            if not isinstance(items, list):
                items = [items]
            if return_trace:
                return items if len(items) > 1 else items[0]
            layer_name = layer or default_layer
            for it in items:
                self._add_layer(layer_name, it)

        return wrapper

    return decorator


class Pitch:
    AXIS_MARGIN = 5

    def __init__(
        self,
        provider: Union[str, PitchDimensions] = "statsbomb",
        width: int = 800,
        height: int = 600,
        theme: str = "minimal",
        orientation: str = "horizontal",
        view: Union[
            str, Tuple[float, float], Tuple[float, float, float, float]
        ] = "full",
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        subnote: Optional[str] = None,
        show_axis: bool = False,
        show_legend: bool = False,
        show_spots: bool = True,
    ):
        """
        provider: 'statsbomb'|'wyscout'|'opta'|'metrica' or a PitchDimensions instance
        orientation: 'horizontal' or 'vertical'
        view: 'full'|'left'|'right'|'top'|'bottom' or (x0,x1) or (x0,x1,y0,y1) in native units
        """
        # Public settings
        self.width = width
        self.height = height
        self.show_axis = show_axis
        self.show_legend = show_legend
        self.show_spots = show_spots
        self.orientation = orientation
        self.view = view
        self.title = title
        self.subtitle = subtitle
        self.subnote = subnote

        # Theme + dimensions
        self.theme = Theme(theme)
        self.dim = (
            PitchDimensions.from_provider(provider)
            if isinstance(provider, str)
            else provider
        )
        self.L = self.dim.get_draw_length()
        self.W = self.dim.get_draw_width()

        # Layers & figure
        self.layers: Dict[str, List[Any]] = {}
        self.fig = go.Figure()

        # Draw
        self._draw_base_pitch()

    def _rect(self, x0, y0, x1, y1, color) -> dict:
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color))

    def _circle(self, cx, cy, r, color) -> dict:
        return dict(
            type="circle",
            x0=cx - r,
            y0=cy - r,
            x1=cx + r,
            y1=cy + r,
            line=dict(color=color),
        )

    def _add_layer(self, layer: str, item: Any) -> None:
        self.layers.setdefault(layer, []).append(item)

        # If it's a Plotly trace, add as a trace…
        if hasattr(item, "to_plotly_json"):
            self.fig.add_trace(item)
        # …otherwise assume it's an annotation dict
        elif isinstance(item, dict) and "xref" in item and "yref" in item:
            existing = list(self.fig.layout.annotations or ())
            existing.append(item)
            self.fig.update_layout(annotations=existing)
        else:
            # fallback (shouldn’t happen)
            self.fig.add_trace(item)

    def _apply_orientation(self, df: pd.DataFrame, x="x", y="y") -> pd.DataFrame:
        if self.orientation == "vertical":
            df = df.copy()
            df["__tmp"] = df[x]
            df[x] = df[y]
            df[y] = self.L - df["__tmp"]
            df.drop(columns="__tmp", inplace=True)
        return df

    def _prepare_hover(
        self, df: pd.DataFrame, x: str, y: str, hover: Optional[str], tooltip_orig: bool
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        df2 = self.dim.apply_coordinate_scaling(df.copy(), x=x, y=y)
        df2 = self._apply_orientation(df2, x=x, y=y)

        if tooltip_orig and x in df.columns and y in df.columns:
            xs = df[x].round(1).astype(str)
            ys = df[y].round(1).astype(str)
            lbl = (df[hover].astype(str) + " ") if hover else ""
            df2["hover_text"] = lbl + "(" + xs + ", " + ys + ")"
            return df2, "hover_text"
        elif hover:
            df2["hover_text"] = df[hover].astype(str)
            return df2, "hover_text"
        else:
            return df2, None

    def _compute_view_window(self) -> Tuple[float, float, float, float]:
        m = self.AXIS_MARGIN

        # Named views
        if isinstance(self.view, str):
            L, W = self.L, self.W
            halves = {
                "full": (-m, L + m, -m, W + m),
                "left": (-m, L / 2 + m, -m, W + m),
                "right": (L / 2 - m, L + m, -m, W + m),
                "bottom": (-m, L + m, -m, W / 2 + m),
                "top": (-m, L + m, W / 2 - m, W + m),
            }
            if self.view not in halves:
                raise ValueError(f"Unknown view: {self.view!r}")
            x0, x1, y0, y1 = halves[self.view]

        # Custom slice (2-tuple): x0,x1 in native, full Y
        elif isinstance(self.view, tuple) and len(self.view) == 2:
            orig_x0, orig_x1 = self.view
            orig_y0, orig_y1 = 0, self.dim.width

            df_c = pd.DataFrame({"x": [orig_x0, orig_x1], "y": [orig_y0, orig_y1]})
            df_c = self.dim.apply_coordinate_scaling(df_c, x="x", y="y")
            df_c = self._apply_orientation(df_c, x="x", y="y")
            xs, ys = df_c["x"], df_c["y"]
            x0, x1 = xs.min() - m, xs.max() + m
            y0, y1 = ys.min() - m, ys.max() + m

        # Custom slice (4-tuple): x0,x1,y0,y1 in native
        elif isinstance(self.view, tuple) and len(self.view) == 4:
            orig_x0, orig_x1, orig_y0, orig_y1 = self.view
            df_c = pd.DataFrame({"x": [orig_x0, orig_x1], "y": [orig_y0, orig_y1]})
            df_c = self.dim.apply_coordinate_scaling(df_c, x="x", y="y")
            df_c = self._apply_orientation(df_c, x="x", y="y")
            xs, ys = df_c["x"], df_c["y"]
            x0, x1 = xs.min() - m, xs.max() + m
            y0, y1 = ys.min() - m, ys.max() + m

        else:
            raise ValueError(f"`view` must be str or 2-/4-tuple, got {self.view!r}")

        return x0, x1, y0, y1

    def _draw_base_pitch(self) -> None:
        lc = self.theme.line_color
        shapes: List[Dict[str, Any]] = []
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)

        # pitch outline
        shapes.append(self._rect(0, 0, self.L, self.W, lc))

        # boxes + halfway
        for key in (
            "halfway_line",
            "penalty_area_left",
            "penalty_area_right",
            "six_yard_left",
            "six_yard_right",
        ):
            if b := scaled.get(key):
                shapes.append(self._rect(b["x0"], b["y0"], b["x1"], b["y1"], lc))

        # center circle
        if c := scaled.get("center_circle"):
            shapes.append(self._circle(c["x"], c["y"], c["r"], lc))

        # apply view window
        x0, x1, y0, y1 = self._compute_view_window()
        self.fig.update_layout(
            shapes=shapes,
            width=self.width,
            height=self.height,
            margin=dict(
                l=0,
                r=0,
                t=self.theme.title_margin + (self.theme.subtitle_margin or 0),
                b=self.theme.subnote_margin or 0,
            ),
            plot_bgcolor=self.theme.pitch_color,
            paper_bgcolor=self.theme.pitch_color,
            showlegend=self.show_legend,
            font=dict(family=self.theme.font_family, color=lc),
            title=(
                dict(text=self.title, x=0.5, xanchor="center") if self.title else None
            ),
            hoverlabel=dict(
                bgcolor=self.theme.hover_bgcolor,
                font_family=self.theme.hover_font_family,
                font_size=self.theme.hover_font_size,
                font_color=self.theme.hover_font_color,
                bordercolor=self.theme.hover_border_color,
            ),
        )
        self.fig.update_xaxes(
            range=[x0, x1],
            scaleanchor="y",
            constrain="domain",
            fixedrange=True,
            showgrid=False,
            zeroline=False,
            showticklabels=self.show_axis,
            visible=self.show_axis,
        )
        self.fig.update_yaxes(
            range=[y0, y1],
            constrain="domain",
            fixedrange=True,
            showgrid=False,
            zeroline=False,
            showticklabels=self.show_axis,
            visible=self.show_axis,
        )

        self._draw_penalty_arcs()
        if self.show_spots:
            self._draw_spots()

    def _draw_penalty_arcs(self) -> None:
        lc = self.theme.line_color
        raw = self.dim.shapes
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)
        r = 9.15 * (self.L / self.dim.length)

        def draw(spot_key: str, area_key: str, left: bool):
            s, a = scaled[spot_key], scaled[area_key]
            dx = (a["x1"] - s["x"]) if left else (s["x"] - a["x0"])
            θ = np.degrees(np.arccos(dx / r))
            angs = (
                np.linspace(-θ, +θ, 50) if left else np.linspace(180 - θ, 180 + θ, 50)
            )
            th = np.radians(angs)
            xs = s["x"] + r * np.cos(th)
            ys = s["y"] + r * np.sin(th)

            df_arc = pd.DataFrame({"x": xs, "y": ys})
            df_arc = self._apply_orientation(df_arc, x="x", y="y")

            trace = go.Scatter(
                x=df_arc["x"],
                y=df_arc["y"],
                mode="lines",
                line=dict(color=lc),
                hoverinfo="skip",
                showlegend=False,
            )
            self._add_layer("arcs", trace)

        if "penalty_spot_left" in raw and "penalty_area_left" in raw:
            draw("penalty_spot_left", "penalty_area_left", True)
        if "penalty_spot_right" in raw and "penalty_area_right" in raw:
            draw("penalty_spot_right", "penalty_area_right", False)

    def _draw_spots(self) -> None:
        lc = self.theme.line_color
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)
        for key in ("penalty_spot_left", "penalty_spot_right", "center_circle"):
            if p := scaled.get(key):
                df_pt = pd.DataFrame({"x": [p["x"]], "y": [p["y"]]})
                df_pt = self._apply_orientation(df_pt, x="x", y="y")

                trace = go.Scatter(
                    x=df_pt["x"],
                    y=df_pt["y"],
                    mode="markers",
                    marker=dict(size=self.theme.spot_size, color=lc),
                    hoverinfo="skip",
                    showlegend=False,
                )
                self._add_layer("spots", trace)

    def set_layer_visibility(self, layer: str, visible: bool = True) -> None:
        """Sets visibility for all items in a named layer."""
        if layer not in self.layers:
            raise ValueError(f"Layer {layer!r} does not exist.")

        # Show or hide traces and annotations by (re)adding or removing them
        # Traces:
        remaining = []
        for trace in self.fig.data:
            if trace not in self.layers.get(layer, []):
                remaining.append(trace)
        if visible:
            # add back any hidden traces
            for item in self.layers[layer]:
                if hasattr(item, "to_plotly_json"):
                    if item not in self.fig.data:
                        self.fig.add_trace(item)
        else:
            # hide: set new data without the layer's traces
            self.fig.data = tuple(remaining)

        # Annotations:
        annots = list(self.fig.layout.annotations or [])
        if visible:
            for item in self.layers[layer]:
                if isinstance(item, dict) and "showarrow" in item:
                    if item not in annots:
                        annots.append(item)
        else:
            annots = [a for a in annots if a not in self.layers[layer]]

        self.fig.update_layout(annotations=annots)

    def remove_layer(self, name: str) -> None:
        """Completely removes a layer from the figure and internal layer storage.

        Args:
            name: The name of the layer to remove

        Raises:
            ValueError: If the specified layer does not exist
        """
        if name not in self.layers:
            raise ValueError(f"Layer {name!r} does not exist.")

        # First hide the layer (removes traces and annotations from figure)
        self.set_layer_visibility(name, visible=False)

        # Then remove the layer from internal storage
        del self.layers[name]

    def show(self) -> None:
        """Display the figure."""
        pio.show(self.fig, config=dict(displaylogo=False))

    def save(
        self,
        filename: str,
        format: Optional[str] = None,
        scale: float = 1.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save the figure to a file.

        Args:
            filename: Path to save the figure to. If format is not specified, it will be inferred from the file extension.
            format: The format to save as. One of 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps'.
                    If not provided, the format is inferred from the filename extension.
            scale: Scale factor to use when exporting. Defaults to 1.0.
            width: The width of the exported image in layout pixels. If not provided, uses the current figure width.
            height: The height of the exported image in layout pixels. If not provided, uses the current figure height.
            **kwargs: Additional keyword arguments passed to plotly.io.write_image.

        Note:
            This function requires the kaleido package to be installed.
            You can install it with: pip install kaleido

        Example:
            ```python
            pitch = Pitch()
            # Add visualizations to the pitch
            pitch.save('my_pitch.png')  # Save as PNG
            pitch.save('my_pitch.svg')  # Save as SVG
            pitch.save('my_pitch.pdf', scale=2.0)  # Save as PDF with 2x resolution
            ```
        """
        # Use provided dimensions or fall back to current figure dimensions
        w = width if width is not None else self.width
        h = height if height is not None else self.height

        # If format is not specified, try to infer from filename
        if format is None:
            import os

            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            if ext in ["png", "jpg", "jpeg", "webp", "svg", "pdf", "eps"]:
                format = ext
            else:
                raise ValueError(
                    f"Could not infer format from filename extension '{ext}'. "
                    f"Please specify format explicitly."
                )

        # Save the figure
        self.fig.write_image(
            filename, format=format, scale=scale, width=w, height=h, **kwargs
        )

    def set_layer_order(self, order: List[str]) -> None:
        """Reorder plotting layers according to provided sequence."""
        missing = [l for l in order if l not in self.layers]
        if missing:
            raise ValueError(f"Layers not found: {missing}")
        # Build new ordered layers dict
        new_layers: Dict[str, List[Any]] = {l: self.layers[l] for l in order}
        # Append any layers not in the specified order
        for l in self.layers:
            if l not in new_layers:
                new_layers[l] = self.layers[l]
        self.layers = new_layers
        # Rebuild traces
        self.fig.data = ()
        for items in self.layers.values():
            for item in items:
                if hasattr(item, "to_plotly_json"):
                    self.fig.add_trace(item)
        # Rebuild annotations
        annots: List[Any] = []
        for items in self.layers.values():
            for item in items:
                if isinstance(item, dict) and "xref" in item and "yref" in item:
                    annots.append(item)
        self.fig.update_layout(annotations=annots)

    @_layered("scatter")
    def plot_scatter(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        hover: Optional[str] = None,
        tooltip_original: bool = True,
        size: int = 10,
        color: Optional[str] = None,
    ) -> go.Scatter:
        df2, hv = self._prepare_hover(df, x, y, hover, tooltip_original)
        mk = dict(size=size, color=color or self.theme.marker_color)
        return go.Scatter(
            x=df2[x],
            y=df2[y],
            mode="markers",
            marker=mk,
            hovertext=(df2[hv] if hv else None),
            hoverinfo=("text" if hv else "skip"),
            showlegend=False,
        )

    @_layered("heatmap")
    def plot_heatmap(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        bins: Tuple[int, int] = (10, 8),
        show_colorbar: bool = False,
        colorscale: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> go.Histogram2d:
        df2 = self.dim.apply_coordinate_scaling(df.copy(), x=x, y=y)
        df2 = self._apply_orientation(df2, x, y)
        cs = colorscale or self.theme.heatmap_colorscale
        op = opacity if opacity is not None else self.theme.heatmap_opacity
        return go.Histogram2d(
            x=df2[x],
            y=df2[y],
            xbins=dict(start=0, end=self.L, size=self.L / bins[0]),
            ybins=dict(start=0, end=self.W, size=self.W / bins[1]),
            colorscale=cs,
            opacity=op,
            showscale=show_colorbar,
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z}<extra></extra>",
        )

    @_layered("kde")
    def plot_kde(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        grid_size: int = 100,
        show_colorbar: bool = False,
        colorscale: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> go.Heatmap:
        df2 = self.dim.apply_coordinate_scaling(df.copy(), x=x, y=y)
        df2 = self._apply_orientation(df2, x, y)
        xs = np.linspace(0, self.L, grid_size)
        ys = np.linspace(0, self.W, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.vstack([xx.ravel(), yy.ravel()])
        vals = np.vstack([df2[x], df2[y]])
        zz = gaussian_kde(vals)(coords).reshape(grid_size, grid_size)
        return go.Heatmap(
            x=xs,
            y=ys,
            z=zz,
            colorscale=colorscale or self.theme.heatmap_colorscale,
            opacity=opacity or self.theme.heatmap_opacity,
            showscale=show_colorbar,
        )

    @_layered("comets")
    def plot_comets(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        x_end: str = "x2",
        y_end: str = "y2",
        color: Optional[str] = None,
        width: int = 6,
        segments: int = 12,
        fade: bool = True,
        hover: Optional[str] = None,
        tooltip_original: bool = False,
    ) -> List[go.Scatter]:
        df2, hv = self._prepare_hover(df, x, y, hover, tooltip_original)
        tmp = df[[x_end, y_end]].copy().rename(columns={x_end: x, y_end: y})
        tmp = self.dim.apply_coordinate_scaling(tmp, x=x, y=y)
        tmp = self._apply_orientation(tmp, x=x, y=y)
        df2[x_end], df2[y_end] = tmp[x], tmp[y]
        col = color or self.theme.marker_color

        def to_rgb(c: str):
            c = c.strip()
            if c.startswith(("rgba(", "rgb(")):
                vals = c[c.find("(") + 1 : c.find(")")].split(",")
                return tuple(map(int, vals[:3]))
            import matplotlib.colors as mcol

            return tuple(int(255 * x) for x in mcol.to_rgb(c))

        r0, g0, b0 = to_rgb(col)
        traces: List[go.Scatter] = []
        for _, row in df2.iterrows():
            x0, y0, x1, y1 = row[x], row[y], row[x_end], row[y_end]
            txt = row[hv] if hv else None
            for i in range(segments):
                t0, t1 = i / segments, (i + 1) / segments
                xs = [x0 + (x1 - x0) * t0, x0 + (x1 - x0) * t1]
                ys = [y0 + (y1 - y0) * t0, y0 + (y1 - y0) * t1]
                alpha = t1 if fade else 1.0
                rgba = f"rgba({r0},{g0},{b0},{alpha:.2f})"
                traces.append(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=rgba, width=width),
                        hovertext=(txt if (txt and i == segments - 1) else None),
                        hoverinfo=("text" if (txt and i == segments - 1) else "skip"),
                        showlegend=False,
                    )
                )
        return traces

    @_layered("arrows")
    def plot_arrows(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        x_end: str = "x2",
        y_end: str = "y2",
        hover: Optional[str] = None,
        *,
        color: Optional[str] = None,
        arrowhead: int = 2,
        arrowsize: float = 1.0,
        width: float = 2.0,
    ) -> List[dict]:
        df2, hv = self._prepare_hover(df, x, y, hover, False)
        tmp = df[[x_end, y_end]].copy().rename(columns={x_end: x, y_end: y})
        tmp = self.dim.apply_coordinate_scaling(tmp, x=x, y=y)
        tmp = self._apply_orientation(tmp, x, y)
        df2[x_end], df2[y_end] = tmp[x], tmp[y]
        col = color or self.theme.marker_color
        annots: List[dict] = []
        for _, row in df2.iterrows():
            annots.append(
                dict(
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
                    arrowcolor=col,
                    opacity=1.0,
                    hovertext=(str(row[hv]) if hv else None),
                )
            )
        return annots
