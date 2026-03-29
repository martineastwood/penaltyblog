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
- Passes, carries, throw-ins, goal kicks, corner passes, and free-kick
  passes are pooled as **move** actions.
- Shots, direct free-kick shots, and corner shots are treated as **shot**
  actions.
- Successful moves are scored by the delta ``xT(end) - xT(start)``.
- Failed actions are ignored.
- Goal probability is estimated **per cell** with light beta-binomial smoothing.
- Plotting integrates with :class:`penaltyblog.viz.Pitch`.

Supported event families
------------------------

Move events (pooled into one transition process):

- ``pass`` — always included
- ``carry`` — included by default (``include_carries=True``)
- ``throw_in`` — included by default (``include_throw_ins=True``)
- ``goal_kick`` — included by default (``include_goal_kicks=True``)
- ``free_kick`` (pass-like) — included by default (``include_free_kicks=True``)
- ``corner`` (pass-like) — included by default (``include_corners=True``)

Shot events:

- ``shot`` — always included
- ``free_kick`` (shot) — included when ``include_free_kicks=True``
- ``corner`` (shot) — included when ``include_corners=True``

Ignored events:

- ``penalty``, ``own_goal``, ``shot_against``, ``shootout``
- Any event not classifiable into a move or shot

Provider-agnostic schema
------------------------

The :class:`~penaltyblog.xt.XTData` class wraps a DataFrame with column
mappings, supporting providers where passes and carries have different
destination columns, and goals are indicated by a dedicated boolean column.

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
       event_subtype="type_secondary",
       pass_end_x="pass_end_location_x",
       pass_end_y="pass_end_location_y",
       carry_end_x="carry_end_location_x",
       carry_end_y="carry_end_location_y",
       move_success="pass_accurate",
       shot_goal="shot_is_goal",
   ).map_events(
       event_map={
           "Pass": "pass",
           "Throw-in": "throw_in",
           "Goal kick": "goal_kick",
           "Corner": "corner",
           "Free kick": "free_kick",
           "Shot": "shot",
           "Acceleration": "carry",
       },
       subtype_map={"carry": "carry"},
   )

   xt = XTModel(
       l=16,
       w=12,
       include_carries=True,
       include_throw_ins=True,
       include_goal_kicks=True,
       include_corners=True,
       include_free_kicks=True,
   )
   xt.fit(data)
   scored = xt.score(data)

Load a pretrained surface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from penaltyblog.xt import load_pretrained_xt

   model = load_pretrained_xt(name="default")

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

- This is a v1 implementation focused on simplicity and portability.
- All included move families are pooled into a single transition process.
- No action-value, turnover, or possession-chain model is included.
- Goal probability is estimated per cell with light smoothing.
- Ambiguous event types (``free_kick``, ``corner``) are classified as
  move or shot based on ``event_subtype``.
