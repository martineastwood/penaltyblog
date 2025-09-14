=====================================
Utility, Inspection & Interoperability
=====================================

Flow gives you flexible tools to inspect, debug, and branch your data pipelines, as well as connect to external libraries like pandas. These tools help you:

- ✅ Peek into streams
- ✅ Materialize for reuse
- ✅ Split pipelines
- ✅ Convert to pandas for export or further analysis

🔍 Inspecting Your Flow
=======================

``.first()`` – Get the First Record
-----------------------------------

.. code-block:: python

   flow.first()

Returns the first record (or None if empty).

.. warning::
   This materializes the full flow under the hood. For lightweight preview, use .head() instead.

``.head(n)`` – Peek at the First N Records
-------------------------------------------

.. code-block:: python

   flow.head(3)

Returns a new Flow with just the first n records. Safe for previewing.

``.show(n, format="table")`` – Peek at the First N Records
-----------------------------------------------------------

.. code-block:: python

   flow.show(3)

Prints the first ``n`` records in a prettier format. If ``format`` is "table", prints a table. If ``format`` is "record", prints the raw list of dicts.

``.is_empty()`` – Check if Flow is Empty
----------------------------------------

.. code-block:: python

   if flow.is_empty():
       print("No records")

Efficiently checks for the presence of at least one record.

``.keys(limit=100)`` – Explore Schema
-------------------------------------

.. code-block:: python

   flow.keys()
   # → {'player.name', 'location', 'shot_xg', ...}

Looks at a sample of records and returns the union of top-level (or flattened) keys.

``len(flow)`` – Count Records
-----------------------------

.. code-block:: python

   print(len(flow))

Materializes and counts records.

``.schema(n=100)`` – Infer Types
--------------------------------

.. code-block:: python

   Flow(...).schema()
   # => {'shot_xg': float, 'player.name': str}

Internally flattens records and maps keys to their types.

``.explain(optimize=None, compare=False)`` – Visualize the Plan
---------------------------------------------------------------

.. code-block:: python

   Flow(...).filter(...).assign(...).explain()

.. code-block:: python

   flow = (
       Flow.statsbomb.events(16023)
       .filter(where_equals("type.name", "Shot"))
       .group_by("player.name")
       .summary({"n_shots": ("count", "shot")})
       .sort_by("n_shots", ascending=False)
       .limit(3)
   )
   flow.explain()

.. code-block:: bash

   === Plan ===
      1. from_statsbomb  {'source': 'events', 'args': {'match_id': 16023, 'include_360_metrics': False, 'creds': {'user': None, 'passwd': None}}}
      2. filter          {'predicate': <FieldPredicate: type.name>}
      3. group_by        {'keys': ['player.name']}
      4. group_summary   {'agg': <function FlowGroup.summary.<locals>.agg_func at 0x13ac305e0>, 'group_keys': ['player.name']}
      5. sort            {'keys': ['n_shots'], 'ascending': [False]}
      6. limit           {'count': 3}

Shows the steps in your DAG as text. If ``optimize`` is True, shows the optimized plan. If ``compare`` is True, shows both the raw and optimized plans side by side.

``.plot_plan(optimize=None, compare=False)`` – Visualize the Plan
-----------------------------------------------------------------

.. code-block:: python

   Flow(...).filter(...).assign(...).plot_plan()

Plots the steps in your DAG. If ``optimize`` is True, shows the optimized plan. If ``compare`` is True, shows both the raw and optimized plans side by side.

``.profile(optimize=None, fmt="table")`` – Profile the Flow
-----------------------------------------------------------

.. code-block:: python

   flow.profile()

Profiles each step in the plan. Returns a report of (step_index, op_name, time_s, rows_emitted). If ``fmt`` is "table", prints a table. If ``fmt`` is "records", returns the raw list of dicts.

.. code-block:: python

   flow = (
      Flow.from_jsonl("data.jsonl", optimize=True)
          .filter(lambda r: r["x"] > 0)
          .group_by("x")
          .summary({"sum_x": ("sum","x")})
   )
   flow.profile()

.. code-block:: bash

   |  # | op           |   sec |   rows |
   |---:|:-------------|------:|-------:|
   |  1 | from_jsonl   | 0.015 | 100000 |
   |  2 | filter       | 0.020 |  90000 |
   |  3 | group_by     | 0.050 |     10 |
   |  4 | group_summary| 0.002 |     10 |

📦 Materializing Data
=====================

Flow is lazy by default. Use these methods to "force" evaluation.

``.collect(optimize=None, progress=None, total_records=None)`` - Convert to List
-------------------------------------------------------------------------------

Fully materializes the flow into a list of records.

This method executes the entire flow pipeline and returns all records as a list. It is typically used when you need to load the data into memory for downstream processing, visualization, or export.

.. code-block:: python

   records = Flow(...).collect()

**Parameters**

- ``optimize``:
    - Whether to apply plan optimizations before execution.
    - If ``True``, applies optimizations to improve execution efficiency.
    - If ``False``, runs the plan exactly as constructed.
    - If ``None`` (default), uses the optimization setting specified when the Flow was created.
- ``progress``:
    - Enables progress bars during execution.
    - "input": displays progress while reading source data (before transformations).
    - "output": displays progress after transformations during final materialization.
    - None (default): disables progress bars.
- ``total_records``:
    - Expected total number of records (used for progress bar display).
    - If not provided, progress bars will fall back to indeterminate mode.

``.cache()`` – Materialize Once
-------------------------------

.. code-block:: python

   flow.cache()

Materializes the current records into memory and gives you a new Flow from that result. This is useful when you want to reuse the same records multiple times without re-executing the pipeline.

🧩 Custom Logic: ``.pipe()``
============================

``.map(func)`` – Transform Records
----------------------------------

Applies a function to each record, replacing it with the returned dict.

.. code-block:: python

   flow = flow.map(lambda r: {"name": r["player"]["name"], "x": r.get("x")})

If ``func(record)`` returns ``None``, the record is skipped.

.. note::
   Use ``.map()`` when you want to remap the entire record. Use ``.assign()`` to add or update fields while keeping the rest intact.

``.pipe(func)`` – Branch Into Custom Logic
------------------------------------------

Use ``.pipe()`` to cleanly encapsulate multi-step logic in a function:

.. code-block:: python

   def filter_shots(flow):
       return flow.filter(lambda r: r.get("type") == "Shot")

   Flow.from_folder("data/").pipe(filter_shots).select("player.name", "shot_xg")

🧩 Interop with Other Tools
===========================

``.to_pandas()`` – Convert to DataFrame
---------------------------------------

.. code-block:: python

   df = Flow(...).flatten().to_pandas()

Converts the flow to a pandas DataFrame. This is useful for exporting to CSV, Excel, or other tools.

✅ Summary
==========

+------------------+------------------------------------------+
| Method           | Purpose                                  |
+==================+==========================================+
| ``.head(n)``     | Get first ``n`` records                 |
+------------------+------------------------------------------+
| ``.first()``     | First record or ``None``                |
+------------------+------------------------------------------+
| ``.show(n)``     | Print first ``n`` records               |
+------------------+------------------------------------------+
| ``.is_empty()``  | Check if Flow yields any data           |
+------------------+------------------------------------------+
| ``.keys()``      | Discover fields                         |
+------------------+------------------------------------------+
| ``.schema()``    | Infer field types                       |
+------------------+------------------------------------------+
| ``.explain()``   | Visualize DAG plan as text              |
+------------------+------------------------------------------+
| ``.plot_plan()`` | Visualize DAG plan                      |
+------------------+------------------------------------------+
| ``.map()``       | Transform records completely            |
+------------------+------------------------------------------+
| ``.pipe()``      | Encapsulate logic or interop with pandas|
+------------------+------------------------------------------+
| ``.collect()``   | Materialize to list                     |
+------------------+------------------------------------------+
| ``.cache()``     | Materialize once and cache in memory    |
+------------------+------------------------------------------+
| ``.profile()``   | Profile each step in the plan           |
+------------------+------------------------------------------+
| ``.to_pandas()`` | Convert to DataFrame                    |
+------------------+------------------------------------------+
