===============================
Using Flow with StatsBomb Data
===============================

Flow includes a built-in integration with the StatsBomb API, making it easy to stream structured football data directly into your pipelines.

Rather than loading everything upfront, Flow wraps the API as **lazy operations** - each call builds a plan that fetches the data only when needed (e.g., on ``.collect()`` or ``.to_pandas()``).

⚙️ Setup
========

Ensure your **StatsBomb credentials** are set as environment variables if you're using private access:

.. code-block:: bash

   export SB_USERNAME="your_username"
   export SB_PASSWORD="your_password"

🚀 Getting Started
==================

.. code-block:: python

   from penaltyblog.matchflow import Flow

   # Fetch all competitions
   comps = Flow.statsbomb.competitions()

   for comp in comps.head(3):
       print(comp)

All API calls return a ``Flow``, so you can apply all usual transformations like ``.filter()``, ``.select()``, ``.assign()``, etc.

🔍 Available Endpoints
======================

+-----------------------------------------------------+------------------------------------+
| Method                                              | Description                        |
+=====================================================+====================================+
| ``.competitions()``                                 | All competitions available via API |
+-----------------------------------------------------+------------------------------------+
| ``.matches(competition_id, season_id)``             | Matches for a specific season      |
+-----------------------------------------------------+------------------------------------+
| ``.events(match_id)``                               | All events in a match              |
+-----------------------------------------------------+------------------------------------+
| ``.lineups(match_id)``                              | Lineups and formation for a match  |
+-----------------------------------------------------+------------------------------------+
| ``.player_match_stats(match_id)``                   | Player-level stats for a match     |
+-----------------------------------------------------+------------------------------------+
| ``.player_season_stats(competition_id, season_id)`` | Player stats over a season         |
+-----------------------------------------------------+------------------------------------+
| ``.team_match_stats(match_id)``                     | Team stats for a match             |
+-----------------------------------------------------+------------------------------------+
| ``.team_season_stats(competition_id, season_id)``   | Team stats over a season           |
+-----------------------------------------------------+------------------------------------+

All of these return a lazy Flow

🧪 Example: Shots in a Match
============================

.. code-block:: python

   from penaltyblog.matchflow import Flow, where_equals

   shots = (
       Flow.statsbomb.events(match_id=19716)
       .filter(where_equals("type.name", "Shot"))
       .select("player.name", "location", "shot.outcome.name")
   )

   for shot in shots.head(3):
       print(shot)

🧼 Filtering & Transforming
===========================

Because Flow supports deep access to nested fields, you can work directly with StatsBomb's JSON structure without needing to flatten first:

.. code-block:: python

   from penaltyblog.matchflow import Flow, where_equals

   top_scorers = (
       Flow.statsbomb.player_season_stats(competition_id=43, season_id=106)
       .filter(lambda r: r["goals"] >= 5)
       .select("player.name", "team.name", "goals")
   )

🐢 Lazy Until Needed
====================

Remember, nothing is downloaded or processed until you **materialize the flow**:

- ``.collect()`` → fetches all records
- ``.to_pandas()`` → fetches and converts to DataFrame
- ``.head(n)`` → fetches just the first n records

.. code-block:: python

   df = Flow.statsbomb.competitions().to_pandas()
   print(df)

🔒 Authenticated Access
=======================

All API methods accept a creds dictionary, or you can use environment variables:

.. code-block:: python

   Flow.statsbomb.events(match_id=123, creds={"user": "...", "passwd": "..."})

🧠 Tips
=======

- Useful for clubs or analysts already using StatsBomb data
- Flows can be joined with your internal data or flattened and saved
- Try ``.flatten().to_jsonl()`` to export clean JSONL for later

📝 Summary
==========

Flow's StatsBomb integration:

- ✅ Keeps your data structured
- ✅ Streams on demand (not loaded eagerly)
- ✅ Integrates with full Flow pipeline tools
- ✅ Works with both open and authenticated endpoints
