====================================
Schema Validation and Type Casting
====================================

Working with messy football data often means handling nested structures, missing fields, and inconsistent types.

MatchFlow provides several tools to help you explore, infer, and enforce schemas as your pipeline evolves.

🔍 Quick Field Exploration: ``.keys()``
========================================

Use ``.keys()`` to quickly inspect the set of fields present in your data. This scans a sample of records, flattens nested structures, and returns a set of unique field names.

.. code-block:: python

   flow = Flow.from_jsonl("match_events.jsonl")
   fields = flow.keys()

   print(fields)
   # {'type.name', 'player.name', 'location', 'shot.statsbomb_xg', ...}

You can control the number of records sampled:

.. code-block:: python

   flow.keys(limit=10)

- Only inspects field names.
- Does not infer types.
- Useful for quickly exploring raw data.

🧮 Full Schema Inference: ``.schema()``
=======================================

For a deeper look at both field names and data types, use ``.schema()``.

.. code-block:: python

   schema = flow.schema()
   print(schema)
   # {'type.name': str, 'player.name': str, 'location': list, 'shot.statsbomb_xg': float, ...}

- Samples the first 100 records by default (you can override with ``n=``).
- Supports nested fields via dot notation.
- Helps you understand the structure before casting.

🎯 Type Casting: ``.cast()``
============================

You can cast fields to specific types or functions using ``.cast()``:

.. code-block:: python

   flow = flow.cast(
       minute=int,
       second=int,
       shot_xg=float
   )

- Takes keyword arguments where keys are field names (dot notation supported) and values are casting functions or types.
- If casting fails, original value is kept (safe fallback).

🚦 Full Schema Enforcement: ``.with_schema()``
==============================================

For full control, you can use ``.with_schema()`` to:

- Cast fields
- Optionally enforce strict type safety
- Optionally drop fields not in the schema

.. code-block:: python

   from datetime import datetime

   def parse_datetime(dt_str):
       return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

   flow.with_schema({
       "team.name": str,
       "score": int,
       "timestamp": parse_datetime
   }, strict=True, drop_extra=True)

- ``strict=True`` will raise an error on casting failure.
- ``drop_extra=True`` will retain only fields listed in the schema.

This is useful when you want to fully sanitize your data before downstream analysis or modeling.
