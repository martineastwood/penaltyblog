import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcol
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde

from ..matchflow.helpers import resolve_path
from .dimensions import PitchDimensions
from .flow_support import normalize_path, to_records
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
        width: int = 600,
        height: int = 500,
        theme: Union[str, Theme] = "minimal",
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
        Initialize a Pitch instance.

        Parameters
        ----------
        provider : str or PitchDimensions, optional
            The pitch dimensions provider. Can be a string for a built-in
            provider (e.g. "statsbomb", "wyscout", etc.), or a custom
            PitchDimensions instance.
        width : int, optional
            The width of the visualization in pixels.
        height : int, optional
            The height of the visualization in pixels.
        theme : str or Theme, optional
            The visual theme for the pitch. Can be a string for a built-in
            theme (e.g. "minimal", "classic", etc.), or a custom Theme
            instance.
        orientation : str, optional
            The orientation of the pitch. Either "horizontal" or "vertical".
        view : str or tuple of floats, optional
            The view mode for the pitch. Can be one of the following:

            - "full": Show the full pitch.
            - "left": Show the left half of the pitch.
            - "right": Show the right half of the pitch.
            - "top": Show the top half of the pitch (if orientation is
              "vertical").
            - "bottom": Show the bottom half of the pitch (if orientation is
              "vertical").
            - (x0, x1): Show the portion of the pitch between x0 and x1.
            - (x0, x1, y0, y1): Show the portion of the pitch between x0 and x1
              (horizontally) and y0 and y1 (vertically).
        title : str or None, optional
            The title of the visualization.
        subtitle : str or None, optional
            The subtitle of the visualization.
        subnote : str or None, optional
            The subnote of the visualization.
        show_axis : bool, optional
            Whether to show axis labels.
        show_legend : bool, optional
            Whether to show a legend.
        show_spots : bool, optional
            Whether to show penalty spots.

        Attributes
        ----------
        layers : dict
            A dictionary mapping layer names to lists of traces.
        fig : go.Figure
            The underlying Plotly figure.
        """
        # Public settings
        self.width: float = width
        self.height: float = height
        self.show_axis = show_axis
        self.show_legend = show_legend
        self.show_spots = show_spots
        self.orientation = orientation
        self.view = view
        self.title = title
        self.subtitle = subtitle
        self.subnote = subnote

        # Theme + dimensions
        self.theme = theme if isinstance(theme, Theme) else Theme(theme)
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

    def _apply_orientation(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
    ) -> pd.DataFrame:
        if self.orientation == "vertical":
            df = df.copy()
            df[[x, y]] = df[[y, x]]  # swap the two columns
        return df

    def _apply_orientation_raw(
        self,
        xs: list[float],
        ys: list[float],
    ) -> tuple[list[float], list[float]]:
        """
        Convert *lists* of x- and y-coordinates from the “native” horizontal
        system to the plotting system.

        Horizontal  (default) → leave as-is
        Vertical               → 90 ° rotation *anti-clockwise*
                                 x' = y
                                 y' = x
        """
        if getattr(self, "orientation", "horizontal") == "vertical":
            return ys, xs
        return xs, ys

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
        self,
        data,
        x: str,
        y: str,
        hover: Optional[str],
        tooltip_original: bool = True,
    ):
        # Normalize dot/index paths

        x, y = normalize_path(x), normalize_path(y)
        hover = normalize_path(hover) if hover else None
        fields = [x, y]
        if hover:
            fields.append(hover)

        # Convert Flow or DataFrame to records with dot path resolution
        records = to_records(data, fields)

        # Extract x, y
        xs = [r[x] if x in r else resolve_path(r, x) for r in records]
        ys = [r[y] if y in r else resolve_path(r, y) for r in records]

        # Determine hovertext
        if hover:
            hover_text = []
            for r in records:
                if hover in r and r[hover] is not None:
                    hover_text.append(r[hover])
                else:
                    hover_text.append(resolve_path(r, hover, default=""))
        elif tooltip_original:
            hover_text = [str(r) for r in records]
        else:
            hover_text = None

        df_like = {"x": xs, "y": ys}
        if hover_text:
            df_like["hover"] = hover_text

        return df_like, "hover" if hover_text else None

    # ────────────────────────────────────────────────────────────────────
    def _compute_view_window(self) -> tuple[float, float, float, float]:
        """
        Return axis limits (x0, x1, y0, y1) for the requested `view`,
        fully respecting vertical or horizontal orientation.
        """
        m = self.AXIS_MARGIN

        # ── Pre-defined keyword views ───────────────────────────────────
        if isinstance(self.view, str):
            # Effective extents once orientation is applied
            if self.orientation == "vertical":
                Lx, Ly = self.W, self.L  # x-span, y-span after rotation
            else:
                Lx, Ly = self.L, self.W

            halves = {
                "full": (-m, Lx + m, -m, Ly + m),
                "left": (-m, Lx / 2 + m, -m, Ly + m),
                "right": (Lx / 2 - m, Lx + m, -m, Ly + m),
                "bottom": (-m, Lx + m, -m, Ly / 2 + m),
                "top": (-m, Lx + m, Ly / 2 - m, Ly + m),
            }
            if self.view not in halves:
                raise ValueError(f"Unknown view: {self.view!r}")
            return halves[self.view]

        # ── Numeric custom slices ───────────────────────────────────────
        # 2-tuple  → (x0,x1) with full height
        # 4-tuple  → (x0,x1,y0,y1)
        if isinstance(self.view, tuple):
            if len(self.view) == 2:
                orig = (*self.view, 0.0, self.dim.width)
            elif len(self.view) == 4:
                orig = self.view
            else:
                raise ValueError(
                    f"`view` tuple must have 2 or 4 elements, got {len(self.view)} elements. "
                    f"Use (x0, x1) for horizontal zoom or (x0, x1, y0, y1) for rectangular zoom."
                )

            # Type assertion to help the type checker understand that orig is always a 4-tuple here
            assert (
                len(orig) == 4
            ), f"orig should always be a 4-tuple, got {len(orig)} elements"
            orig_x0, orig_x1, orig_y0, orig_y1 = orig
            df_c = pd.DataFrame({"x": [orig_x0, orig_x1], "y": [orig_y0, orig_y1]})
            df_c = self.dim.apply_coordinate_scaling(df_c, x="x", y="y")
            df_c = self._apply_orientation(df_c, x="x", y="y")
            xs, ys = df_c["x"], df_c["y"]

            return (
                xs.min() - m,
                xs.max() + m,
                ys.min() - m,
                ys.max() + m,
            )

        raise ValueError(f"`view` must be str or 2-/4-tuple, got {self.view!r}")

    def _draw_base_pitch(self) -> None:
        """
        Draws the base pitch using the theme's styling parameters. The pitch is
        drawn using the theme's line color and line width, and the dimensions
        are calculated based on the theme's provider and the plot's dimensions.
        """
        lc = self.theme.line_color
        shapes: List[Dict[str, Any]] = []
        # 1) get your provider‐scaled shapes
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)

        # helper to apply your existing raw‐swap orientation
        def orient(x: float, y: float) -> Tuple[float, float]:
            xs, ys = self._apply_orientation_raw([x], [y])
            return xs[0], ys[0]

        # ── Pitch outline ─────────────────────────────────────────────────────────
        x0_o, y0_o = orient(0, 0)
        x1_o, y1_o = orient(self.L, self.W)
        shapes.append(self._rect(x0_o, y0_o, x1_o, y1_o, lc))

        # ── Boxes + halfway line ──────────────────────────────────────────────────
        for key in (
            "halfway_line",
            "penalty_area_left",
            "penalty_area_right",
            "six_yard_left",
            "six_yard_right",
        ):
            if b := scaled.get(key):
                x0_o, y0_o = orient(b["x0"], b["y0"])
                x1_o, y1_o = orient(b["x1"], b["y1"])
                shapes.append(self._rect(x0_o, y0_o, x1_o, y1_o, lc))

        # ── Center circle ─────────────────────────────────────────────────────────
        if c := scaled.get("center_circle"):
            cx_o, cy_o = orient(c["x"], c["y"])
            r_o = c["r"]  # radius already scaled horizontally
            shapes.append(self._circle(cx_o, cy_o, r_o, lc))

        # ── Apply view window & margins ───────────────────────────────────────────
        x0, x1, y0, y1 = self._compute_view_window()
        top_margin = self.theme.title_margin if self.title else 0
        top_margin += self.theme.subtitle_margin if self.subtitle else 0
        bottom_margin = self.theme.subnote_margin if self.subnote else 0
        self._y_domain_height = (y1 - y0) / max(y1 - y0, x1 - x0)

        self.fig.update_layout(
            shapes=shapes,
            width=self.width,
            height=self.height,
            margin=dict(l=0, r=0, t=top_margin, b=bottom_margin),
            plot_bgcolor=self.theme.pitch_color,
            paper_bgcolor=self.theme.pitch_color,
            showlegend=self.show_legend,
            font=dict(family=self.theme.font_family, color=lc),
            title=(
                dict(text="<b>" + self.title + "</b>", x=0.5, xanchor="center")
                if self.title
                else None
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

        # ── The rest of your drawing ───────────────────────────────────────────────
        self._draw_penalty_arcs()
        if self.show_spots:
            self._draw_spots()

        self._draw_titles_and_notes()

    # pitch.py ───────────────────────────────────────────────────────────────
    def _draw_penalty_arcs(self) -> None:
        """
        Draws the penalty arcs for both left and right penalty areas on the pitch.
        This method calculates the arc based on the scaled dimensions of the pitch
        and adds it to the plot as a line. The arcs are oriented correctly based on
        the current pitch orientation.
        """
        lc = self.theme.line_color
        scaled = self.dim.scaled_shapes(target_length=self.L, target_width=self.W)

        # official radius (already scaled on the length axis)
        r = 9.15 * (self.L / self.dim.length)

        # ------------------------------------------------------------------
        def build_arc(spot_key: str, area_key: str, is_left_goal: bool) -> None:
            """
            spot_key  : "penalty_spot_left" / "penalty_spot_right"
            area_key  : "penalty_area_left" / "penalty_area_right"
            is_left_goal : True if the goal is on the *left* end in raw coords
            """
            spot = scaled[spot_key]
            area = scaled[area_key]

            sx_raw, sy_raw = spot["x"], spot["y"]

            # boundary of the penalty area nearest the goal line
            bx_raw = area["x1"] if is_left_goal else area["x0"]

            # horizontal geometry -------------------------------------------------
            dx = abs(bx_raw - sx_raw)
            theta = np.degrees(np.arccos(dx / r))  # half-angle span

            if is_left_goal:  # open towards the right
                angs = np.linspace(-theta, +theta, 60)
            else:  # open towards the left
                angs = np.linspace(180 - theta, 180 + theta, 60)

            th = np.radians(angs)
            xs_h = sx_raw + r * np.cos(th)
            ys_h = sy_raw + r * np.sin(th)

            # rotate/swap into current orientation --------------------------------
            xs, ys = self._apply_orientation_raw(xs_h, ys_h)

            # add to figure -------------------------------------------------------
            self._add_layer(
                "arcs",
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=lc),
                    hoverinfo="skip",
                    showlegend=False,
                ),
            )

        # ------------------------------------------------------------------
        if {"penalty_spot_left", "penalty_area_left"} <= scaled.keys():
            build_arc("penalty_spot_left", "penalty_area_left", is_left_goal=True)

        if {"penalty_spot_right", "penalty_area_right"} <= scaled.keys():
            build_arc("penalty_spot_right", "penalty_area_right", is_left_goal=False)

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
        """
        Set the visibility of a layer of traces and annotations.

        Args:
            layer: The name of the layer to show or hide
            visible: Whether to show (True) or hide (False) the layer. Defaults to True.
        """
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
        """
        Render the Pitch figure in a Jupyter notebook or as an HTML file.

        This method uses Plotly's built-in `pio.show()` function to display the
        figure. All keyword arguments are passed through to `pio.show()`.
        """
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
        """
        Reorder plotting layers according to provided sequence.

        Args:
            order: List of layer names in the desired order.
        """
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

    def _draw_titles_and_notes(self) -> None:
        """
        Draws the title, subtitle, and subnote annotations on the plot using the
        theme's styling parameters. The positions are calculated based on the theme's
        margin settings and the plot's dimensions.
        """
        annotations = list(self.fig.layout.annotations or [])

        top_y = 1 + (self.theme.title_margin or 0) / self.height
        sub_y = top_y - (self.theme.subtitle_margin or 20) / self.height
        note_y = 0 - (self.theme.subnote_margin or 20) / self.height

        if self.subtitle:
            annotations.append(
                dict(
                    text=self.subtitle,
                    x=0.5,
                    y=sub_y,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                    showarrow=False,
                    font=dict(
                        size=self.theme.subtitle_font_size or 14,
                        family=self.theme.font_family,
                        color=self.theme.line_color,
                    ),
                )
            )

        if self.subnote:
            annotations.append(
                dict(
                    text=self.subnote,
                    x=0.5,
                    y=note_y,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(
                        size=self.theme.subnote_font_size or 12,
                        family=self.theme.font_family,
                        color=self.theme.line_color,
                    ),
                )
            )

        self.fig.update_layout(annotations=annotations)

    @_layered("scatter")
    def plot_scatter(
        self,
        data,
        x: str = "x",
        y: str = "y",
        hover: Optional[str] = None,
        tooltip_original: bool = True,
        size: int = 10,
        color: Optional[str] = None,
    ) -> go.Scatter:
        """
        Plot a scatter of points on the pitch.

        Args:
            data: Data to plot. Must contain columns for x and y coordinates.
            x: The column name for x coordinates. Defaults to "x".
            y: The column name for y coordinates. Defaults to "y".
            hover: The column name for hover text. If not specified, will display
                the original data row as hover text if `tooltip_original` is True.
            tooltip_original: If True (default), show original data row as hover
                text if `hover` is not specified. Otherwise, show no hover text.
            size: The size of the markers. Defaults to 10.
            color: The color of the markers. If not specified, uses the theme's
                marker color.

        Returns:
            A Scatter trace that has been added to the Pitch figure.
        """
        data_dict, hv = self._prepare_hover(data, x, y, hover, tooltip_original)
        x_raw, y_raw = data_dict["x"], data_dict["y"]
        x_scaled, y_scaled = self.dim.apply_coordinate_scaling_raw(x_raw, y_raw)
        x_plot, y_plot = self._apply_orientation_raw(x_scaled, y_scaled)

        marker = dict(size=size, color=color or self.theme.marker_color)

        return go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers",
            marker=marker,
            hovertext=(data_dict[hv] if hv else None),
            hoverinfo="text" if hv else "skip",
            showlegend=False,
        )

    @_layered("heatmap")
    def plot_heatmap(
        self,
        data,
        x: str = "x",
        y: str = "y",
        bins: Tuple[int, int] = (10, 8),
        show_colorbar: bool = False,
        colorscale: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> go.Histogram2d:
        """
        Plot a heatmap on the pitch.

        Args:
            data: Data to plot. Must contain columns for x and y coordinates.
            x: The column name for x coordinates. Defaults to "x".
            y: The column name for y coordinates. Defaults to "y".
            bins: Tuple of bin counts for x and y dimensions. Defaults to (10, 8).
            show_colorbar: Whether to show the colorbar. Defaults to False.
            colorscale: The colorscale to use. If not specified, uses the theme's
                heatmap colorscale.
            opacity: The opacity of the heatmap. If not specified, uses the theme's
                heatmap opacity.

        Returns:
            A Histogram2d trace that has been added to the Pitch figure.
        """
        data_dict, _ = self._prepare_hover(data, x, y, None, False)
        x_raw = data_dict["x"]
        y_raw = data_dict["y"]
        x_scaled, y_scaled = self.dim.apply_coordinate_scaling_raw(x_raw, y_raw)
        x_oriented, y_oriented = self._apply_orientation_raw(x_scaled, y_scaled)
        cs = colorscale or self.theme.heatmap_colorscale
        op = opacity if opacity is not None else self.theme.heatmap_opacity
        cb = dict(
            lenmode="fraction",
            len=self._y_domain_height,
            yanchor="middle",
            y=0.5,
        )
        return go.Histogram2d(
            x=x_oriented,
            y=y_oriented,
            xbins=dict(start=0, end=self.L, size=self.L / bins[0]),
            ybins=dict(start=0, end=self.W, size=self.W / bins[1]),
            colorscale=cs,
            opacity=op,
            showscale=show_colorbar,
            hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z}<extra></extra>",
            colorbar=cb,
        )

    @_layered("kde")
    def plot_kde(
        self,
        data,
        x: str = "x",
        y: str = "y",
        grid_size: int = 100,
        show_colorbar: bool = False,
        colorscale: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> go.Heatmap:
        """
        Plot a kernel density estimate (KDE) on the pitch.

        Args:
            data: Data to plot. Must contain columns for x and y coordinates.
            x: The column name for x coordinates. Defaults to "x".
            y: The column name for y coordinates. Defaults to "y".
            grid_size: The number of points to use for the grid. Defaults to 100.
            show_colorbar: Whether to show the colorbar. Defaults to False.
            colorscale: The colorscale to use. If not specified, uses the theme's
                heatmap colorscale.
            opacity: The opacity of the KDE. If not specified, uses the theme's
                heatmap opacity.

        Returns:
            A Heatmap trace that has been added to the Pitch figure.
        """
        data_dict, _ = self._prepare_hover(data, x, y, None, False)
        x_raw = data_dict["x"]
        y_raw = data_dict["y"]
        x_scaled, y_scaled = self.dim.apply_coordinate_scaling_raw(x_raw, y_raw)
        x_oriented, y_oriented = self._apply_orientation_raw(x_scaled, y_scaled)
        xs = np.linspace(0, self.L, grid_size)
        ys = np.linspace(0, self.W, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.vstack([xx.ravel(), yy.ravel()])
        vals = np.vstack([x_oriented, y_oriented])

        try:
            zz = gaussian_kde(vals)(coords).reshape(grid_size, grid_size)
        except np.linalg.LinAlgError:
            warnings.warn(
                "KDE computation failed due to singular covariance matrix. Falling back to simple density heatmap."
            )

            H, _, _ = np.histogram2d(
                vals[0],
                vals[1],
                bins=[grid_size, grid_size],
                range=[[0, self.L], [0, self.W]],
            )
            # Smooth the histogram a bit
            from scipy.ndimage import gaussian_filter

            zz = gaussian_filter(H, sigma=1.0)
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
        data,
        x: str = "x",
        y: str = "y",
        x_end: str = "x2",
        y_end: str = "y2",
        color: Optional[str] = None,
        width: int = 6,
        segments: int = 24,
        fade: bool = True,
        hover: Optional[str] = None,
        tooltip_original: bool = False,
    ) -> List[go.Scatter]:
        """
        Plot a comet trace on the pitch.

        Args:
            data: Data to plot. Must contain columns for x, y, x2, and y2 coordinates.
            x: The column name for x coordinates. Defaults to "x".
            y: The column name for y coordinates. Defaults to "y".
            x_end: The column name for x2 coordinates. Defaults to "x2".
            y_end: The column name for y2 coordinates. Defaults to "y2".
            color: The color of the comet trace. If not specified, uses the theme's
                comet color.
            width: The width of the comet trace. Defaults to 6.
            segments: The number of segments to use for the comet trace. Defaults to 24.
            fade: Whether to fade the comet trace. Defaults to True.
            hover: The column name for hover text. If not specified, will display
                the original data row as hover text if `tooltip_original` is True.
            tooltip_original: If True (default), show original data row as hover
                text if `hover` is not specified. Otherwise, show no hover text.

        Returns:
            A list of Scatter traces that have been added to the Pitch figure.
        """
        start_dict, hv = self._prepare_hover(data, x, y, hover, tooltip_original)
        end_dict, _ = self._prepare_hover(data, x_end, y_end, None, False)

        # Prepare raw start coordinates
        xs0_raw, ys0_raw = start_dict["x"], start_dict["y"]
        xs0, ys0 = self.dim.apply_coordinate_scaling_raw(xs0_raw, ys0_raw)
        xs0, ys0 = self._apply_orientation_raw(xs0, ys0)

        # Prepare raw end coordinates
        xs1_raw, ys1_raw = end_dict["x"], end_dict["y"]
        xs1, ys1 = self.dim.apply_coordinate_scaling_raw(xs1_raw, ys1_raw)
        xs1, ys1 = self._apply_orientation_raw(xs1, ys1)

        # Combine start and end coordinates and hover
        hv_list = start_dict.get(hv) if hv and hv in start_dict else [None] * len(xs0)
        col = color or self.theme.marker_color

        def to_rgb(c: str):
            c = c.strip()
            if c.startswith(("rgba(", "rgb(")):
                vals = c[c.find("(") + 1 : c.find(")")].split(",")
                return tuple(map(int, vals[:3]))
            return tuple(int(255 * x) for x in mcol.to_rgb(c))

        col = color or self.theme.marker_color
        r0, g0, b0 = to_rgb(col)
        traces: List[go.Scatter] = []
        for i, (x0, y0, x1, y1) in enumerate(zip(xs0, ys0, xs1, ys1)):
            txt = hv_list[i]
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
        data,
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
        """
        Plot arrows on the pitch to represent directional movement from a
        starting point (x, y) to an ending point (x2, y2).

        Parameters
        ----------
        data : DataFrame or similar
            The data containing start and end coordinates for the arrows.
        x : str, optional
            The column name for the starting x-coordinates, by default "x".
        y : str, optional
            The column name for the starting y-coordinates, by default "y".
        x_end : str, optional
            The column name for the ending x-coordinates, by default "x2".
        y_end : str, optional
            The column name for the ending y-coordinates, by default "y2".
        hover : str, optional
            The column name for the hover text to be displayed, by default None.
        color : str, optional
            The color of the arrows, by default None which uses the theme's marker color.
        arrowhead : int, optional
            The style of the arrowhead, by default 2.
        arrowsize : float, optional
            The size of the arrowhead, by default 1.0.
        width : float, optional
            The width of the arrow line, by default 2.0.

        Returns
        -------
        List[dict]
            A list of dictionaries representing the arrow annotations.
        """

        # Get starting and ending points
        start_dict, hv = self._prepare_hover(data, x, y, hover, tooltip_original=False)
        end_dict, _ = self._prepare_hover(
            data, x_end, y_end, None, tooltip_original=False
        )

        # Coordinate scaling + orientation
        xs, ys = self.dim.apply_coordinate_scaling_raw(start_dict["x"], start_dict["y"])
        xs, ys = self._apply_orientation_raw(xs, ys)

        x2s, y2s = self.dim.apply_coordinate_scaling_raw(end_dict["x"], end_dict["y"])
        x2s, y2s = self._apply_orientation_raw(x2s, y2s)

        hovertexts = start_dict.get(hv, [None] * len(xs)) if hv else [None] * len(xs)

        col = color or self.theme.marker_color
        annots: List[dict] = []

        for i in range(len(xs)):
            annots.append(
                dict(
                    x=x2s[i],
                    y=y2s[i],
                    ax=xs[i],
                    ay=ys[i],
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
                    hovertext=(
                        str(hovertexts[i]) if hovertexts[i] is not None else None
                    ),
                )
            )

        return annots
