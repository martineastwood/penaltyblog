============================
Grouping and Aggregating Data
============================

After cleaning and transforming individual records, the next step in data analysis is often to summarize information across different groups.

For example:

- Count the number of shots per team
- Compute the average xG per player
- Sum the number of passes per zone

Flow provides powerful tools for these group-based operations.

🔀 Grouping Records: ``.group_by(...)``
=======================================

Use ``.group_by(...)`` to define how records should be grouped. It takes one or more field names and returns a ``FlowGroup`` object - a pipeline for grouped records.

.. code-block:: python

   from penaltyblog.matchflow import Flow

   sample_records = [
       {"event_id": 1, "type_name": "Pass", "team_name": "Manchester City", "player_name": "Kevin De Bruyne"},
       {"event_id": 2, "type_name": "Shot", "team_name": "Manchester City", "player_name": "Erling Haaland", "shot_xg": 0.05},
       {"event_id": 3, "type_name": "Duel", "team_name": "Manchester City", "player_name": "Rodri"},
       {"event_id": 4, "type_name": "Pass", "team_name": "Manchester City", "player_name": "Kevin De Bruyne"},
       {"event_id": 5, "type_name": "Shot", "team_name": "Arsenal", "player_name": "Bukayo Saka", "shot_xg": 0.01},
   ]

   flow = Flow.from_records(sample_records)

   grouped = flow.group_by("team_name")

You can group by multiple keys:

.. code-block:: python

   flow.group_by("type_name", "player_name")

📊 Aggregating Groups with ``.summary(...)``
=============================================

Once you have a group, use ``.summary(...)`` to compute one or more aggregations per group.

Example: Sum xG per team
------------------------

.. code-block:: python

   result = (
       flow
       .group_by("team_name")
       .summary(total_xg=("shot_xg", "sum"))
   )

   print(result.collect())

Example: Shots and xG per player
--------------------------------

.. code-block:: python

   from penaltyblog.matchflow import where_equals

   player_summary = (
       flow
       .filter(where_equals("type_name", "Shot"))
       .group_by("player_name")
       .summary(
           total_xg=("shot_xg", "sum"),
           number_of_shots="count"
       )
   )

   print(player_summary.collect())

⚙️ Built-in Aggregation Functions
=================================

Flow supports many built-in aggregators:

- ``count``, ``sum``, ``mean``, ``min``, ``max``, ``median``, ``std``, ``var``
- ``first``, ``last``, ``mode``, ``range``, ``nunique``
- ``all``, ``any``, ``prod``
- Custom callables or lambdas

🧪 Custom Aggregation Example
=============================

Want to calculate a custom stat, like shots on target %?

.. code-block:: python

   def shot_accuracy(rows):
       outcomes = ["Goal", "Saved"]
       shots = [r for r in rows if r.get("shot_outcome_name")]
       if not shots:
           return 0
       return 100 * sum(r["shot_outcome_name"] in outcomes for r in shots) / len(shots)

Apply it:

.. code-block:: python

   result = (
       flow
       .filter(where_equals("type_name", "Shot"))
       .group_by("player_name")
       .summary(sot_percentage=shot_accuracy)
   )

   print(result.collect())

🔄 Rolling and Time-Based Aggregations
======================================

In addition to simple group summaries, Flow provides powerful tools to aggregate events across time windows. These are especially useful for event-based data like football matches, where you'd like to calculate rolling metrics or fixed interval summaries.

🔁 Rolling Summaries: .rolling_summary(...)
-------------------------------------------

Rolling summaries compute an aggregation for each row, based on a moving window of previous rows. This is useful for things like:

- xG over the previous 5 minutes
- Cumulative passes in the last 10 events
- Momentum metrics that change throughout a match

.. code-block:: python

   from datetime import timedelta

   result = (
       flow
       .filter(where_equals("type_name", "Shot"))
       .assign(
           timestamp=lambda r: timedelta(
               minutes=r["minute"],
               seconds=r.get("second", 0)
           )
       )
       .group_by("team_name")
       .sort_by("timestamp")  # Important: sort within groups!
       .rolling_summary(
           window="5m",
           time_field="timestamp",
           aggregators={
               "xg": ("sum", "shot_xg"),
               "shots": ("count", "shot_xg")
           }
       )
       .select("team_name", "timestamp", "xg", "shots")
   )

   result.show()

- The window can be either a time string (``"5m"``, ``"30s"``, ``"1h"``) or an integer (number of rows).
- Always ``.sort_by()`` on your time field after grouping: this ensures the rolling window works as intended.
- The function emits one row per input row, where each aggregation is computed over the previous window of rows.

⏱️ Fixed Time Buckets: ``.time_bucket(...)``
--------------------------------------------

If you want non-overlapping, regular time intervals (e.g. total xG every 5 minutes), use ``.time_bucket()``. This partitions your data into uniform windows and computes aggregates for each.

.. code-block:: python

   result = (
       flow
       .filter(where_equals("type_name", "Shot"))
       .assign(
           timestamp=lambda r: timedelta(
               minutes=r["minute"],
               seconds=r.get("second", 0)
           )
       )
       .group_by("team_name")
       .time_bucket(
           freq="5m",
           time_field="timestamp",
           label="left",   # bucket labeled at start of interval
           aggregators={
               "xg": ("sum", "shot_xg"),
               "shots": ("count", "shot_xg")
           }
       )
       .select("team_name", "bucket", "xg", "shots")
   )

   result.show()

- ``freq`` defines the bucket size (e.g. ``"10s"``,  ``"5m"``).
- You don't need to sort beforehand - ``.time_bucket()`` handles sorting internally.
- The label argument controls whether the bucket timestamp refers to the start (``"left"``) or end (``"right"``) of each window.
- The output field for the bucket timestamp defaults to "bucket", but can be renamed via ``bucket_name="..."``.

📈 Aggregating the Whole Dataset
================================

You can call ``.summary()`` directly on a ``Flow`` to compute dataset-wide aggregates without grouping.

.. code-block:: python

   summary = (
       flow
       .filter(where_equals("type_name", "Shot"))
       .summary(
           total_xg=("shot_xg", "sum"),
           avg_xg=("shot_xg", "mean"),
           total_shots="count"
       )
   )

   print(summary.head(1))

✅ Summary
==========

- ``.group_by()`` groups records by field(s)
- ``.summary()`` applies aggregations to each group (or full dataset)
- You can mix built-in aggregators with custom functions
- Grouped flows return regular Flow objects - chain ``.select()``, ``.sort_by()``, etc.

👉 Next Up: Joining Flows
=========================

Learn how to join datasets together - like linking events with players or match metadata - using ``.join()``.
