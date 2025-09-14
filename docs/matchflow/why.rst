====================================================
Why Nested Data Isn't a Problem - It's the Point
====================================================

**TL;DR: Most football data pipelines flatten JSON into tables too early, losing structure and flexibility. Flow is a new engine that lets you explore and transform nested data directly - without premature normalization.**

In football (soccer) analytics, the default approach to working with data is to flatten it into tables. Whether it's passing networks, xG chains, or player event logs, we often reach for ``pandas`` or SQL to bring structure to chaos. But what if we're flattening too early - or even unnecessarily?

The ``Flow`` engine in ``penaltyblog`` takes a different path. Instead of reducing everything to rigid tables, it treats nested JSON as a first-class citizen, not a problem to fix, but a structure to embrace.

This article explores why embracing nested data opens up powerful new workflows, especially for clubs and analysts working with real-world, messy, event-based football data.

🌎 The Nature of Football Data
===============================

Football data is inherently nested:

- A "pass" might contain a start and end location, a pressure flag, and a list of tags.
- A "shot" might include multiple qualifiers, an assist type, and a freeze frame of defenders.
- A "match" contains players, teams, events, substitutions, and metadata - all deeply structured.

When we flatten this data:

- We lose structure.
- We risk key collisions (e.g. player.name vs team.name).
- We make it harder to model and reason about the game.

Flatten too early, and you invite brittle pipelines and constant cleanup.

🌬️ Why Nested Data is a Feature
=================================

1. It reflects the real world
-----------------------------

Nested structures mirror the natural hierarchy of football:

- Matches contain events
- Events have players, contexts, and outcomes
- Actions have tags, timestamps, and spatial data

Keeping this structure lets you work with the game as it's played, not just as rows in a table.

2. It's schema-flexible
-----------------------

Different providers (Opta, StatsBomb, Wyscout) use different formats. Trying to flatten these into a single table leads to endless exceptions.

A pipeline that embraces nesting can adapt:

.. code-block:: python

   flow.select("player.name", "location.x", "location.y")

Without caring if ``player`` is a dict or a flat field. This gives ``Flow`` the ability to **ingest, normalize, and transform** without overfitting.

3. It's analysis-friendly
-------------------------

Flattening forces premature decisions:

- Do I include all tags or just the first?
- How do I encode location - tuple, string, x/y?
- What if a freeze-frame includes 10 defenders?

Keeping the data nested lets you:

- Loop through freeze frames when needed
- Extract only meaningful tags
- Plot raw coordinates without munging

**You defer decisions until they actually matter.**

🧪 Why Not Just Use ``pandas.json_normalize()``?
=================================================

``pandas.json_normalize()`` is great for one-off tasks but in real pipelines, it quickly breaks down:

- It expects a consistent structure: nested fields that exist across every row.
- It can't easily handle mixed types or optional nesting (e.g. missing tags, variable-length freeze frames).
- Deep normalization requires fiddly recursive logic or multiple passes.
- You often end up with dozens of flattened columns like ``player.name``, ``player.id``, ``location.0``, ``location.1`` with no easy way to recombine them.

In short: ``pandas.json_normalize()`` is great for flattening once, but can be brittle, opaque, and hard to iterate with. ``Flow`` lets you stay close to the original structure, transform incrementally, and only flatten when you're ready.

🌟 Flow: A Query Engine for Nested JSON
=========================================

**Flow is not a DataFrame - it's a stream-first, nested-data engine designed for irregular, event-based JSON.** Instead of flattening your data, it lets you query and transform nested records directly, without writing normalization boilerplate or discarding structure.

- Just point ``Flow`` at your folder of JSON files
- Chain transformations lazily (filter, assign, group_by)
- Select nested fields naturally

Output to dashboards, notebooks, or summaries without needing pandas.

.. code-block:: python

   from penaltyblog.matchflow import Flow, where_equals

   flow = (
       Flow.from_folder("data/events/")
       .filter(where_equals("type", "Shot"))
       .assign(xT=lambda r: model.predict(r))
       .select("player.name", "xT", "location")
       .to_json("shots.json")
   )

This turns your raw event data into a **queryable, schema-aware** stream, not a rigid table.

📊 When Flattening Still Helps
===============================

Of course, flattening still plays an important role - just not always at the beginning. Use it when:

- You're building reports or exports for BI tools
- You've standardized your schema
- You need fast vectorized ops (e.g. model training)

Even then, with ``Flow`` you can defer flattening until the end:

.. code-block:: python

   flow.filter(...).flatten().to_pandas()

🚀 Final Thought: Let the Structure Work For You
=================================================

Football is complex and your data should be allowed to be, too.

With ``Flow``, you don't need to flatten first or second-guess your structure. You explore data as it is, shape it as needed, and only normalize when you're ready.

🧪 Try It Out
==============

If you've ever felt like your data tools were fighting the structure of football data, give Flow a try:

.. code-block:: bash

   pip install penaltyblog

Start with your StatsBomb data, or one of the included examples. Keep your data nested. Flatten only when you're ready.

I'd love feedback, edge cases, or ideas, especially if you break it. That's the whole point of a v1.
