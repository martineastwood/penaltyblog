==============================
Advanced Pipeline Operations
==============================

Beyond basic filtering and assignment, Flow provides advanced operations for manipulating, combining, and analyzing structured data at scale.

These tools help you:

- Sort by xG or timestamp
- Join datasets (e.g. match metadata + events)
- Eliminate duplicates
- Sample subsets for debugging
- Combine multiple flows

🔃 Sorting and Ordering
=======================

``.sort_by()`` – Sort Records
-----------------------------

Sort records by one or more fields:

.. code-block:: python

   from flow import Flow, where_equals
   from pprint import pprint

   sorted_events = Flow(events).sort_by("timestamp")

Sort shots by shot_xg, descending:

.. code-block:: python

   shots = (
       Flow(events)
       .filter(where_equals("type_name", "Shot"))
       .sort_by("shot_xg", ascending=False)
   )

   pprint(shots.head(1))

Sort by multiple fields:

.. code-block:: python

   Flow(events).sort_by(["team_name", "type_name"], ascending=False)

.. note::
   Sorting loads the full flow into memory.

📏 Limiting Results
===================

Use ``.limit(n)`` or ``.head(n)`` to take the first N records:

.. code-block:: python

   top_5 = Flow(events).limit(5)

🎯 Sampling
===========

``.sample_n()`` – Random N Records
----------------------------------

.. code-block:: python

   sample = Flow(events).sample_n(3, seed=42)

``.sample_fraction(p)`` – Fractional Sampling
---------------------------------------------

.. code-block:: python

   sample = Flow(events).sample_fraction(0.2, seed=1)  # 20% chance per row

🤝 Joining Datasets
===================

You can combine two Flow objects based on common keys, similar to a SQL join, with the ``.join()`` function.

.. code-block:: python

   flow.join(
       other: "Flow",
       on: Union[str, List[str], None] = None,
       left_on: Union[str, List[str], None] = None,
       right_on: Union[str, List[str], None] = None,
       how: Literal["left", "right", "outer", "inner", "anti"] = "left",
       lsuffix: str = "",
       rsuffix: str = "_right",
       type_coercion: Literal["strict", "auto", "string"] = "strict",
   )

**Key join Parameters:**

- ``other``: The other ``Flow`` object to join with.
- ``on``, ``left_on``, ``right_on``: The key(s) to join on.
    - Use ``on="field_name"`` if the key has the same name in both flows.
    - Use ``left_on="left_field"`` and ``right_on="right_field"`` if the key names are different.
- ``how``: The type of join to perform.
    - ``left``: (Default) Keep all records from the left ``Flow``, and add matching data from the right.
    - ``inner``: Keep only records where the key exists in both flows.
    - ``outer``: Keep all records from both flows, filling in missing data with None.
    - ``right``: Keep all records from the right ``Flow``.
    - ``anti``: Keep only the records from the left ``Flow`` that do not have a match in the right ``Flow``.
- ``type_coercion``: How to handle join keys of different types (e.g., 123 vs "123"). Default is ``"strict"`` (must be the same type). Use ``"auto"`` for smart coercion.

.. code-block:: python

   events_records = [
       {"event_id": 1, "player_id": 101, "action": "Shot"},
       {"event_id": 2, "player_id": 102, "action": "Pass"},
   ]
   players_records = [
       {"id": 101, "name": "Bukayo Saka"},
       {"id": 102, "name": "Martin Ødegaard"},
   ]

   events_flow = pb.Flow.from_records(events_records)
   players_flow = pb.Flow.from_records(players_records)

   # Join the two flows to add the player's name to each event
   enriched_flow = events_flow.join(
       players_flow,
       left_on="player_id",
       right_on="id",
       how="left"
   )

⚠️ Notes on ``.join()``
-----------------------

- The right-hand Flow is fully materialized in memory.

➕ Combining Flows
==================

Use ``.concat()`` to merge multiple flows:

.. code-block:: python

   combined = flow1.concat(flow2, flow3)

🚫 Handling Duplicates
======================

``.distinct()`` – Drop Duplicates
---------------------------------

Drop exact or partial duplicates:

.. code-block:: python

   unique_events = Flow(events).distinct()

   deduped = Flow(events).distinct("player_name", "type_name", keep="first")

Options for keep:

- "first" (default)
- "last"
- False → removes all duplicates

🧾 Extracting Unique Field Values
=================================

``.distinct("field")`` for unique values
----------------------------------------

.. code-block:: python

   unique_players = Flow(events).distinct("player_name")

For combinations:

.. code-block:: python

   unique = Flow(events).distinct("team_name", "type_name")

.. note::
   Internally tracks key combinations so be careful on large datasets with high cardinality.

🧪 Example: Join Events with Match Info
=======================================

.. code-block:: python

   events = Flow(events)
   matches = Flow(matches)

   enriched = events.join(matches, on="match_id", how="left")
   pprint(enriched.head(1))

.. code-block:: python

   {
       'event_id': 1,
       'match_id': 123,
       'type_name': 'Pass',
       'player_name': 'Kevin De Bruyne',
       'team_name': 'Manchester City',
       'competition_name': 'Premier League',   # from match metadata
       'match_date': '2023-10-08'
   }

🧠 Summary
==========

Flow's advanced operations let you:

- Sort and rank streams
- Sample intelligently
- Merge datasets using joins
- Deduplicate messy input
- Combine multiple sources

These tools are built for working with real-world, irregular JSON records - not just clean flat tables.

📥 Next: Saving and Exporting Data
==================================

In the next guide, we'll look at writing flows to disk using ``.to_jsonl()``, ``.to_json()``, and ``.to_pandas()`` for final output or reporting.
