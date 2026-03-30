Expected Threat (xT)
====================

This module implements a **position-based Expected Threat (xT)** model for
football event data. The implementation follows the direct linear algebra
formulation:

.. math::

   X = S + MTX

which is solved as:

.. math::

   (I - MT) X = S

This is the **direct solver** approach (no iterative convergence path).

Key characteristics
-------------------

- Position-based xT on a normalized ``0-100`` pitch.
- Grid discretization with a practical default of ``16x12``.
- **One unified xT surface** from all included attacking event families.
- Passes, carries, throw-ins, goal kicks, corners, and free kicks
  are treated as **move** actions.
- Shots and direct free-kick shots are treated as **shot** actions.
- Each move family maintains its own transition matrix, with sparse
  families shrunk toward the pooled baseline (see
  :ref:`per-family transitions <per-family-transitions>` below).
- Failed actions act as an implicit **turnover discount** — they consume
  probability without contributing transitions, so distant cells have
  lower xT (see :ref:`turnover discount <turnover-discount>` below).
- Successful moves are scored by the delta ``xT(end) - xT(start)``.
- Goal probability is estimated **per cell** with light beta-binomial smoothing.
- Plotting integrates with :class:`penaltyblog.viz.Pitch`.
- Provider-specific coordinate ranges can be normalized into the internal
  ``0-100`` xT coordinate system via :class:`~penaltyblog.xt.XTData`.
- Out-of-bounds coordinate handling is explicit and configurable via
  ``XTModel(coord_policy=...)``.

Supported event families
------------------------

Move events:

- ``pass`` — always included
- ``carry`` — included by default (``include_carries=True``)
- ``throw_in`` — included by default (``include_throw_ins=True``)
- ``goal_kick`` — included by default (``include_goal_kicks=True``)
- ``free_kick`` — included by default (``include_free_kicks=True``)
- ``corner`` — included by default (``include_corners=True``)

Shot events:

- ``shot`` — always included
- ``free_kick_shot`` — included when ``include_free_kicks=True``

Ignored events:

- ``penalty``, ``penalty_kick``, ``own_goal``, ``shot_against``,
  ``shootout``, ``postmatch_penalty``
- Any event not classifiable into a move or shot

.. _turnover-discount:

Turnover discount
-----------------

The denominator for action probabilities includes **all move attempts**
(successful and failed), not just successful ones. This means:

.. math::

   \text{move\_prob}(i) = \frac{\text{successful\_moves}(i)}{\text{shots}(i) + \text{all\_moves}(i)}

The gap :math:`1 - \text{shot\_prob} - \text{move\_prob}` is the per-cell
probability of losing possession without progressing the ball. This acts
as a natural discount: cells far from goal need many successful transitions
to reach a shooting position, and each step has a chance of failure that
compounds multiplicatively through the linear solve.

Without this, the xT surface has an artificially high floor in the
defensive half because ``move_prob ≈ 1.0`` everywhere and value propagates
freely across the pitch.

.. _per-family-transitions:

Per-family transitions
----------------------

Rather than pooling all move events into a single transition matrix, the
model builds a **separate transition matrix per move family**. This means
a throw-in from a cell near the touchline has a different destination
distribution from an open-play pass originating in the same cell.

To handle sparse families (e.g. free kicks, which may have very few
observations from any single cell), each family's transition row is
**shrunk toward the pooled transition**:

.. math::

   T_f^{smooth}(i) = \frac{counts_f(i) + k \cdot T_{pooled}(i)}{n_f(i) + k}

With the default ``k = 5``, a family needs roughly 5+ events from a cell
before its pattern meaningfully diverges from the pooled baseline.

The combined move-transition product used in the solve is:

.. math::

   MT = \sum_f \operatorname{diag}(p_f) \cdot T_f^{smooth}

where :math:`p_f(cell)` is the fraction of all actions (shots + all move
attempts) from that cell that are successful moves of family *f*.

Provider-agnostic schema
------------------------

The :class:`~penaltyblog.xt.XTData` class wraps a DataFrame with column
name mappings. A single ``is_success`` column has consistent meaning
across event types: for moves it means the action was completed
successfully; for shots it means a goal was scored.

``XTData`` handles boolean coercion safely — string values like
``"False"``, ``"Incomplete"``, or ``"0"`` are correctly interpreted
rather than treated as truthy non-empty strings.

Coordinate ranges and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, xT uses a normalized ``0-100`` pitch for both axes.
If your provider uses different ranges (for example ``x=0..120``,
``y=0..80``), declare them in ``XTData`` and the data is scaled once
during normalization:

.. code-block:: python

   data = XTData(
       events=df,
       x="x",
       y="y",
       event_type="event_type",
       end_x="end_x",
       end_y="end_y",
       is_success="is_success",
       x_range=(0, 120),
       y_range=(0, 80),
   )

Coordinate validation policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After normalization, ``XTModel`` validates coordinates before clipping.
Use ``coord_policy`` to control behavior when values fall outside ``0..100``:

- ``"warn"`` (default): emit a warning and clip.
- ``"error"``: raise ``ValueError``.
- ``"clip"``: silently clip.

This applies in both ``fit`` and ``score``.

Usage
-----

Fit on custom data
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from penaltyblog.xt import XTModel, XTData

   data = XTData(
       events=df,
       x="location_x",
       y="location_y",
       event_type="type_primary",
       end_x="end_location_x",
       end_y="end_location_y",
       is_success="is_successful",
       x_range=(0, 120),  # optional (default is (0, 100))
       y_range=(0, 80),   # optional (default is (0, 100))
   ).map_events(
       event_map={
           "Pass": "pass",
           "Throw-in": "throw_in",
           "Goal kick": "goal_kick",
           "Corner kick": "corner",
           "Free kick pass": "free_kick",
           "Free kick shot": "free_kick_shot",
           "Shot": "shot",
           "Carry": "carry",
       },
       success_map={
           "Complete": True,
           "Incomplete": False,
           "Goal": True,
           "Saved": False,
           "Blocked": False,
       },
   )

   xt = XTModel(
       l=16,
       w=12,
       include_carries=True,
       include_throw_ins=True,
       include_goal_kicks=True,
       include_corners=True,
       include_free_kicks=True,
       coord_policy="warn",  # "warn" | "error" | "clip"
   )
   xt.fit(data)
   scored = xt.score(data)

Load a pretrained surface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from penaltyblog.xt import load_pretrained_xt

   model = load_pretrained_xt(name="default")

The bundled ``"default"`` artifact is fit on ~14 million events
across multiple seasons, including passes, throw-ins, free kicks, corners,
and shots taken from the big five European leagues.

Scoring
^^^^^^^

Scoring uses **successful move actions only** across all included families:

.. code-block:: python

   scored = model.score(data)
   scored[["xt_start", "xt_end", "xt_added"]]

Plotting
^^^^^^^^

Plotting uses :class:`penaltyblog.viz.Pitch` under the hood:

.. code-block:: python

   model.plot()

Notes
-----

- Each move family has its own transition matrix, shrunk toward the pooled
  baseline for sparse cells.
- Failed actions act as a turnover discount — they reduce ``move_prob``
  without contributing transitions.
- Goal probability is estimated per cell with light smoothing.
- Each event type maps unambiguously to a role — e.g. ``corner``
  is always a move, ``free_kick_shot`` is always a shot.
