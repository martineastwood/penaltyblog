===========================
Using Flow with Opta Data
===========================

Flow includes a built-in integration with the Stats Perform (Opta) API, making it easy to stream structured football data directly into your pipelines.

Rather than loading everything upfront, Flow wraps the API as **lazy operations** - each call builds a plan that fetches the data only when needed (e.g., on ``.collect()`` or ``.to_pandas()``).

⚙️ Setup
========

Ensure your **Opta credentials** are set as environment variables:

.. code-block:: bash

   export OPTA_AUTH_KEY="your_auth_key"
   export OPTA_RT_MODE="b"

🚀 Getting Started
==================

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta

   # Fetch all areas
   areas = opta.areas()

   for area in areas.head(3):
       print(area)

All API calls return a ``Flow``, so you can apply all usual transformations like ``.filter()``, ``.select()``, ``.assign()``, etc.

🔍 Available Endpoints
======================

+----------------------------------------------------------+---------+------------------------------------------------+
| Method                                                   | Feed ID | Description                                    |
+==========================================================+=========+================================================+
| ``.tournament_calendars(...)``                           | OT2     | All tournament calendars available via API     |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.venues(...)``                                         | OT3     | All venues available via API                   |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.areas([area_uuid])``                                  | OT4     | All areas available via API                    |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.tournament_schedule(tournament_calendar_uuid, ...)``  | MA0     | Matches for a specific season                  |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.matches(...)``                                        | MA1     | All matches available via API                  |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.match(fixture_uuid, ...)``                            | MA1     | A single match                                 |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.match_stats_player(fixture_uuids, ...)``              | MA2     | Player-level stats for a match                 |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.match_stats_team(fixture_uuids, ...)``                | MA2     | Team-level stats for a match                   |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.events(fixture_uuid, ...)``                           | MA3     | All events in a match                          |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.pass_matrix(fixture_uuid, ...)``                      | MA4     | Pass matrix and average formation data         |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.possession(fixture_uuid, ...)``                       | MA5     | Possession and territorial advantage data      |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.player_career(...)``                                  | PE2     | Player career data                             |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.referees(...)``                                       | PE3     | All referees available via API                 |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.rankings(tournament_calendar_uuid, ...)``             | PE4     | Rankings data for players, teams, and games    |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.injuries(...)``                                       | PE7     | All injuries available via API                 |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.teams(...)``                                          | TM1     | All teams available via API                    |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.team_standings(tournament_calendar_uuid, ...)``       | TM2     | League table and standings data                |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.squads(...)``                                         | TM3     | All squads available via API                   |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.player_season_stats(tmcl_uuid, ctst_uuid, ...)``      | TM4     | Player stats over a season                     |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.team_season_stats(tmcl_uuid, ctst_uuid, ...)``        | TM4     | Team stats over a season                       |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.transfers(...)``                                      | TM7     | Player transfer data                           |
+----------------------------------------------------------+---------+------------------------------------------------+
| ``.contestant_participation(contestant_uuid, ...)``      | TM16    | Contestant participation data                  |
+----------------------------------------------------------+---------+------------------------------------------------+

All of these return a lazy Flow

🧪 Example: Referees in a Tournament
====================================

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta

   referees = (
       opta.referees(tournament_calendar_uuid="51r6ph2woavlbbpk8f29nynf8")
       .select("firstName", "lastName", "nationality")
   )

   for referee in referees.head(3):
       print(referee)

🧼 Filtering & Transforming
===========================

Because Flow supports deep access to nested fields, you can work directly with Opta's JSON structure without needing to flatten first:

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta

   english_referees = (
       opta.referees(tournament_calendar_uuid="51r6ph2woavlbbpk8f29nynf8")
       .filter(lambda r: r["nationality"] == "England")
       .select("firstName", "lastName")
   )

🐢 Lazy Until Needed
====================

Remember, nothing is downloaded or processed until you **materialize the flow**:

- ``.collect()`` → fetches all records
- ``.to_pandas()`` → fetches and converts to DataFrame
- ``.head(n)`` → fetches just the first n records

.. code-block:: python

   df = opta.areas().to_pandas()
   print(df)

🔒 Authenticated Access
=======================

All API methods accept a creds dictionary, or you can use environment variables. They also accept a `proxies` argument for routing requests through a proxy.

.. code-block:: python

   proxies = {
       'http': 'socks5h://localhost:9090',
       'https://': 'socks5h://localhost:9090'
   }

   data = opta.tournament_calendars(
       status="all",
       proxies=proxies
   ).collect()

.. code-block:: python

   opta.referees(tournament_calendar_uuid="51r6ph2woavlbbpk8f29nynf8", creds={"auth_key": "...", "rt_mode": "..."})

🧠 Tips
=======

- Useful for clubs or analysts already using Opta data
- Flows can be joined with your internal data or flattened and saved
- Try ``.flatten().to_jsonl()`` to export clean JSONL for later

📝 Summary
==========

Flow's Opta integration:

- ✅ Keeps your data structured
- ✅ Streams on demand (not loaded eagerly)
- ✅ Integrates with full Flow pipeline tools
- ✅ Works with both open and authenticated endpoints

.. _opta-helpers:

Opta Helpers
============

The ``penaltyblog.matchflow.opta_helpers`` module provides helper functions to simplify common filtering tasks when working with Opta event data. These helpers allow you to filter by human-readable names instead of remembering specific Opta ID codes.

Filtering by Event Type
-----------------------

Use ``where_opta_event()`` to filter events by their name, like "Pass" or "Shot". The helper automatically looks up the correct ``typeId``.

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta
   from penaltyblog.matchflow.opta_helpers import where_opta_event

   # Get all shots for a match
   shots = (
       opta.events(fixture_uuid="some_match_id")
       .filter(where_opta_event("Shot"))
   )

   # You can also filter for multiple event types
   passes_and_shots = (
       opta.events(fixture_uuid="some_match_id")
       .filter(where_opta_event(["Pass", "Shot"]))
   )


Filtering by Qualifier
----------------------

Use ``where_opta_qualifier()`` to filter events that have a specific qualifier. You can check for the presence of a qualifier or for a qualifier with a specific value.

**Checking for Presence**

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta
   from penaltyblog.matchflow.opta_helpers import where_opta_qualifier

   # Get all penalty shots
   penalty_shots = (
       opta.events(fixture_uuid="some_match_id")
       .filter(where_opta_event("Shot"))
       .filter(where_opta_qualifier("Penalty"))
   )

**Checking for a Specific Value**

.. code-block:: python

   from penaltyblog.matchflow.contrib import opta
   from penaltyblog.matchflow.opta_helpers import where_opta_qualifier

   # Get all shots from the "Danger Zone"
   danger_zone_shots = (
       opta.events(fixture_uuid="some_match_id")
       .filter(where_opta_event("Shot"))
       .filter(where_opta_qualifier("Zone", "Danger Zone"))
   )


Exploring Available Mappings
----------------------------

To see all available event and qualifier names that you can use with the helpers, use the ``get_opta_mappings()`` function.

.. code-block:: python

   from penaltyblog.matchflow.opta_helpers import get_opta_mappings

   mappings = get_opta_mappings()

   print("Available Event Types:")
   for event in mappings["events"]:
       print(f"  ID: {event['id']:3d} | Name: {event['name']}")

   print("\nAvailable Qualifier Types:")
   for qualifier in mappings["qualifiers"]:
       print(f"  ID: {qualifier['id']:3d} | Name: {qualifier['name']}")

This will return a dictionary containing all event and qualifier names and their corresponding IDs. The mappings include comprehensive football event data such as:

**Key Event Types:**
- Pass (1), Offside Pass (2), Take On (3), Foul (4)
- Save (10), Clearance (12), Miss (13), Post (14), Attempt Saved (15), Goal (16)
- Card (17), Substitutions (18, 19), Interception (8), Tackle (7)
- And many more specialized events (80+ total event types)

**Key Qualifier Types:**
- Long Ball (1), Cross (2), Head Pass (3), Through Ball (4)
- Penalty (9), Handball (10), Various card types (31-33)
- Pitch zones (e.g. Small box - Centre (16), Box - Right (63))
- Shot locations (76-87), Save types (173-183), VAR-related qualifiers (329-336)
- And hundreds of detailed qualifiers for specific situations

The helper functions automatically handle the case-insensitive lookup, so you can use human-readable names like "Shot", "Pass", "Penalty", "Zone" etc. in your filters without needing to remember the specific Opta IDs.

.. autoclass:: penaltyblog.matchflow.contrib.opta.Opta
   :members:
   :undoc-members:
   :show-inheritance:
