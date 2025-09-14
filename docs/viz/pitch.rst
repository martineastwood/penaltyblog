==================
Pitch Visualization
==================

The ``Pitch`` class is a powerful and flexible interface for rendering football (soccer) pitch visualizations using ``plotly``. It supports multiple pitch dimensions (e.g., StatsBomb, Wyscout), configurable themes, and layered plotting of various visual elements (scatter, heatmap, arrows, comets, etc.).

📐 Initialization
=================

.. code-block:: python

   Pitch(
       provider="statsbomb",     # or "wyscout", "opta", "metrica", or a PitchDimensions instance
       width=800,
       height=600,
       theme="minimal",          # e.g., "classic", "night", "retro", "turf"
       orientation="horizontal", # or "vertical"
       view="full",              # or "left", "right", "top", "bottom", or (x0, x1), (x0, x1, y0, y1)
       title=None,
       subtitle=None,
       subnote=None,
       show_axis=False,
       show_legend=False,
       show_spots=True,
   )

Common Use
----------

.. code-block:: python

   from penaltyblog.viz import Pitch

   pitch = Pitch(provider="statsbomb", theme="night", orientation="horizontal")

🖼 Base Features
================

- Draws pitch lines, boxes, center circle, and optional penalty spots.
- Automatically applies orientation (``horizontal`` vs ``vertical``).
- Supports custom zoom via view (e.g., ``"left"``, ``(30, 90)``, ``(0, 120, 18, 62)``).
- Built-in styling presets (``theme``) to suit different visual aesthetics.

🔁 Layers
=========

Visual elements are grouped into named layers (e.g., ``"scatter"``, ``"heatmap"``, ``"arrows"``). This allows for flexible visibility control, ordering, and removal.

Layer methods
-------------

.. code-block:: python

   pitch.set_layer_visibility("arrows", visible=False)
   pitch.set_layer_order(["scatter", "arrows", "heatmap"])
   pitch.remove_layer("heatmap")

🧰 Plotting Methods
===================

Each method adds visual elements to the figure. Use ``return_trace=True`` to get the underlying Plotly trace(s) instead of adding to the layer.

plot_scatter(...)
-----------------

Plots individual points.

.. code-block:: python

   pitch.plot_scatter(data, x="x", y="y", hover="player_name")

- ``hover``: field for hover text.
- ``size``: marker size.
- ``color``: marker color.

plot_heatmap(...)
-----------------

Creates a 2D histogram of point density.

.. code-block:: python

   pitch.plot_heatmap(data, x="x", y="y", bins=(20, 14), show_colorbar=True)

- ``bins``: (x, y) bin count.
- ``colorscale``: override theme colorscale.
- ``opacity``: override opacity.

plot_kde(...)
-------------

Smooth kernel density estimate over the pitch.

.. code-block:: python

   pitch.plot_kde(data, x="x", y="y", grid_size=100)

- Automatically falls back to histogram+blur if KDE fails.
- Output is a Plotly ``go.Heatmap``.

plot_comets(...)
----------------

Draws trails ("comets") from (x, y) to (x2, y2).

.. code-block:: python

   pitch.plot_comets(data, x="start_x", y="start_y", x_end="end_x", y_end="end_y")

- ``segments``: how many segments per trail.
- ``fade``: True to fade out the trail.
- ``hover``: shown only at trail head.

plot_arrows(...)
----------------

Draws arrows using Plotly annotations.

.. code-block:: python

   pitch.plot_arrows(data, x="start_x", y="start_y", x_end="end_x", y_end="end_y")

- ``arrowhead``: arrowhead shape.
- ``arrowsize``: arrowhead size.
- ``width``: arrow width.
- ``color``: arrow color.
- ``hover``: shown at arrow tip.

🖼 Display & Export
===================

``pitch.show()``
----------------

Renders the figure in a browser or notebook.

``pitch.save(...)``
-------------------

Saves the pitch as an image (requires ``kaleido``).

.. code-block:: python

   pitch.save("output.svg")  # Format inferred
   pitch.save("output.pdf", scale=2.0)  # Higher resolution

Arguments:

- ``format``: 'png', 'svg', 'pdf', etc.
- ``scale``: output resolution multiplier.
- ``width`` / ``height``: override layout size.

📏 Supported Dimensions
=======================

The provider argument supports:

+--------------+-------------+--------------+------------+
| Provider     | Origin      | Native Units | Dimensions |
+==============+=============+==============+============+
| ``statsbomb``| Top-left    | 120 × 80     | meters     |
+--------------+-------------+--------------+------------+
| ``wyscout``  | Top-left    | 100 × 100    | percent    |
+--------------+-------------+--------------+------------+
| ``opta``     | Top-left    | 100 × 100    | percent    |
+--------------+-------------+--------------+------------+
| ``metrica``  | Bottom-left | 1.0 × 1.0    | normalized |
+--------------+-------------+--------------+------------+

All are automatically scaled to a consistent 105 × 68 drawing space.

🎨 Themes
=========

Themes define color schemes, fonts, sizes, and line styles. Available themes:

- ``"classic"``: green pitch, white lines.
- ``"night"``: navy background, bright accents.
- ``"retro"``: cream + brown tones.
- ``"minimal"``: white pitch, dark lines.
- ``"turf"``: deep green, amber markers.

Custom themes
-------------

.. code-block:: python

   from penaltyblog.viz import Theme

   custom = Theme.from_dict({
       "pitch_color": "#ffffff",
       "line_color": "#444444",
       "marker_color": "#e07a5f",
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
   })

   pitch = Pitch(theme=custom)

🧪 Advanced Usage
=================

- ``return_trace=True``: retrieve Plotly object instead of plotting.
- ``orientation="vertical"``: flips pitch orientation.
- ``view=(30, 90)``: zoom into a region.
- Plot multiple layers and toggle them interactively.

💡 Tips
=======

- Use ``.set_layer_visibility()`` for interactive toggling in notebooks or dashboards.
- Use ``.set_layer_order()`` to control stacking (e.g., heatmap behind scatter).
- Works seamlessly with ``Flow`` objects or Pandas DataFrames.
