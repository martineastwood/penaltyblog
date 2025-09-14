=======================================
Basic Pipelines: Transforming Your Data
=======================================

Once you've loaded your data into a ``Flow``, the next step is usually to clean, reshape, and enrich it.

Flow provides familiar methods like ``.filter()``, ``.assign()``, ``.select()``, and ``.explode()``, designed to work **lazily** over nested JSON records - no flattening or DataFrame conversion required.

📦 Example Records
==================

.. code-block:: python

   from penaltyblog.matchflow import Flow

   sample_records = [
       {
           "event_id": 1,
           "match_id": 123,
           "period": 1,
           "timestamp": "00:01:30.500",
           "type_name": "Pass",
           "player_name": "Kevin De Bruyne",
           "location": [60.1, 40.3],
           "pass_recipient_name": "Erling Haaland",
           "pass_outcome_name": "Complete",
       },
       {
           "event_id": 2,
           "type_name": "Shot",
           "player_name": "Erling Haaland",
           "location": [85.5, 50.2],
           "shot_xg": 0.05,
           "shot_outcome_name": "Goal",
       },
       # More records...
   ]

   flow = Flow.from_records(sample_records)

🎯 Selecting Fields with ``.select()``
=======================================

.. code-block:: python

   player_locations = flow.select("player_name", "location").head(1)
   print(player_locations)

.. code-block:: python

   [{'player_name': 'Kevin De Bruyne', 'location': [60.1, 40.3]}]

🧠 Accessing Nested Fields
===========================

.. code-block:: python

   example = [{"a": {"b": {"c": 1}}}]
   flow = Flow.from_records(example).select("a.b.c")
   print(flow.head(1))

.. code-block:: python

   [{'a': {'b': {'c': 1}}}]

🧹 Handling Dotted Keys
=======================

Option 1 — Flatten the record
-----------------------------

If your keys contain dots (e.g. "player.info.name.full"), you can:

.. code-block:: python

   flow.flatten().select("player.info.name.full")

Option 2 — Rename and assign
----------------------------

.. code-block:: python

   flow.rename(**{"player.info": "player_info"})
       .assign(name_full=lambda r: r["player_info"].get("name.full"))
       .select("name_full")

🔍 Filtering Records
====================

Basic filter using a lambda
----------------------------

.. code-block:: python

   shots = flow.filter(lambda r: r.get("type_name") == "Shot")
   print(shots.select("player_name", "shot_outcome_name").collect())

Using predicate helpers
-----------------------

.. code-block:: python

   from penaltyblog.matchflow import where_equals

   goals = flow.filter(
       where_equals("type_name", "Shot"),
       where_equals("shot_outcome_name", "Goal"),
       where_equals("player_name", "Erling Haaland")
   )
   print(goals.select("player_name", "shot_xg").collect())

✍️ Assigning Fields with ``.assign()``
=======================================

.. code-block:: python

   half_flow = flow.assign(
       half=lambda r: "First" if r.get("period") == 1 else "Second"
   )
   print(half_flow.select("player_name", "half").head(1))

You can also overwrite fields:

.. code-block:: python

   uppercase_flow = flow.assign(
       player_name=lambda r: r.get("player_name", "").upper()
   )

🔀 Renaming Fields with ``.rename()``
=====================================

.. code-block:: python

   renamed = flow.rename(
       match_id="id",
       type_name="event_type"
   ).select("id", "event_type")

   print(renamed.head(1))

🎈 Exploding Lists with ``.explode()``
======================================

.. code-block:: python

   example = [{
       "event_id": 30,
       "players": ["Player X", "Player Y"],
       "roles": ["Passer", "Receiver"]
   }]

   exploded = Flow.from_records(example).explode("players", "roles")
   pprint(exploded.collect())

.. code-block:: python

   [{'event_id': 30, 'players': 'Player X', 'roles': 'Passer'},
    {'event_id': 30, 'players': 'Player Y', 'roles': 'Receiver'}]

🎯 Splitting Arrays with ``.split_array()``
============================================

.. code-block:: python

   split = flow.split_array("location", into=["x", "y"]).select("x", "y").head(1)
   print(split)

.. code-block:: python

   [{'x': 60.1, 'y': 40.3}]

🧮 Accessing Array Elements by Index
====================================

If a field is a list (like coordinates or player IDs), you can extract specific values using dot notation with a numeric index:

.. code-block:: python

   record = {"player": "Kevin De Bruyne", "location": [60.1, 40.3]}
   flow = Flow.from_records([record])

To get just the X or Y value from location:

.. code-block:: python

   flow.select("location.0", "location.1").collect()

.. code-block:: python

   [{'location': {'0': 60.1, '1': 40.3}}]

.. note::
   The numeric indexes are treated like nested keys internally, so "location.0" means "first element of location".

If you want those values as top-level fields, just rename them:

.. code-block:: python

   xy = (
       flow.select("location.0", "location.1")
           .rename(**{
               "location.0": "x",
               "location.1": "y"
           })
   )

   print(xy.collect())

.. code-block:: python

   [{'location': {}, 'x': 60.1, 'y': 40.3}]

✅ Summary
==========

These methods form the building blocks of most Flow pipelines:

- ``.select()`` to pick fields
- ``.filter()`` to narrow your data
- ``.assign()`` to compute new columns
- ``.rename()`` to simplify field names
- ``.explode()`` to unpack lists
- ``.split_array()`` to handle coordinate fields

You chain these operations lazily and collect results only when you're ready.

.. code-block:: python

   flow = (
       Flow.from_records(sample_records)
       .filter(where_equals("type_name", "Shot"))
       .assign(xg_bin=lambda r: "High" if r.get("shot_xg", 0) > 0.1 else "Low")
       .select("player_name", "xg_bin")
       .show(3)
   )

🚀 Next: Grouping and Summaries
===============================

In the next section, we'll cover ``.group_by()`` and ``.summary()`` to compute aggregates - like total xG per player or matc
