====================
Introduction to Flow
====================

*A lazy, schema-aware pipeline for nested football data*

In football analytics, a lot of data comes as **deeply nested JSON** - think event data, match metadata, freeze frames, and tracking points.

Yet most tools flatten this structure too early, forcing everything into rigid tables. This leads to brittle pipelines, excessive cleanup, and premature decisions about schema.

**Flow** takes a different approach: it treats nested JSON as a first-class citizen. It lets you build clear, chainable pipelines over structured records without needing to normalize everything first.

🧠 What is Flow?
=================

Flow is a lightweight query engine for nested data. It gives you:

- Lazy, chainable operations: ``filter()``, ``assign()``, ``group_by()``, etc.
- Natural access to nested fields (``"player.name"``, ``"location.x"``)
- Reusable, explainable pipelines
- Outputs to JSONL, pandas, or disk - but only when you ask

Under the hood, Flow builds a **plan** - a list of transformation steps - and doesn't execute anything until you call ``.collect()`` or ``.to_pandas()``.

🧪 Example: Filter and Transform Shots
=======================================

.. code-block:: python

   from penaltyblog.matchflow import Flow, where_equals, where_gt

   flow = (
       Flow.from_folder("data/events/")
       .filter(
           where_equals("type.name", "Shot"),
           where_gt("shot.stats.xG", 0.2)
       )
       .assign(xg_label=lambda r: "High xG" if r["shot"]["stats"]["xG"] > 0.5 else "Low xG")
       .select("player.name", "team.name", "xg_label")
   )

   flow.show(5)

Nothing is computed until the end, you're building a lazy pipeline, not evaluating data immediately.

⚙️ Lazy Execution: Nothing Happens Until You Ask
=================================================

Flow's operations are lazy. Every method adds a step to the internal plan:

.. code-block:: text

   Flow(...) → .filter(...) → .assign(...) → .select(...)

But no records are actually processed until you:

- Call ``.collect()`` → get a list of records
- Call ``.to_pandas()`` → build a DataFrame
- Call ``.to_jsonl()`` → write to disk
- Use a loop: ``for row in flow``

🧊 Reuse and Caching
====================

Flows are built to be reusable. You can run ``.collect()`` multiple times, and even inspect the pipeline with ``.explain()``:

.. code-block:: python

   flow.explain()
   # Shows a step-by-step plan of your pipeline

If your data source is expensive (e.g. API or big JSONL), cache it:

.. code-block:: python

   flow = Flow.from_jsonl("events.jsonl").filter(...)

   cached = flow.cache()  # Runs once, stores the results

   df = cached.to_pandas()
   head = cached.head(3)

``.cache()`` materializes the current records into memory and gives you a new Flow from that result.

🧠 A Different Way of Thinking
==============================

Flow is not a dataframe.

It's a pipeline builder for nested JSON - more like SQL or Spark, but designed for Python and football analytics.

You don't flatten your data until you're ready.

You don't write repetitive dict lookups or munging code.

You don't worry about rows with missing tags or optional fields.

You just write clear pipelines.

⚠️ Notes on Mutability
======================

Flow may modify records in-place for performance.

- If you care about preserving your original data, use ``copy.deepcopy(data)`` before passing it in.
- Or call ``.cache()`` or ``.materialize()`` to freeze the state into a new memory-backed Flow.

🚀 Summary: Why Use Flow?
=========================

Flow is designed for working with real-world football data:

- ✅ Natural access to nested fields
- ✅ Lazy evaluation with reusability
- ✅ Built-in filter helpers (``where_equals``, ``where_in``, ``where_gt`` etc)
- ✅ Outputs to JSONL, pandas, or JSON
- ✅ Keeps pipelines readable and composable

If you're flattening your data just to load it into pandas, Flow lets you skip that step, and work with the structure as-is.

🛠️ Coming Soon: flowz Format
=============================

I'm actively working on a fast, binary format (``.flowz``) for even faster loading, predicate pushdown, and indexing. For now, JSON and JSONL are fully supported.

💬 Try it and break it
======================

.. code-block:: bash

   pip install penaltyblog

then

.. code-block:: python

   from penaltyblog.matchflow import Flow

   Flow.from_folder("data/")
       .filter(...)
       .select(...)
       .show()

If something doesn't work, or you're fighting the shape of your data, please open an issue or drop a note. That's the point of v1.
