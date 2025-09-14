=============================================
Best Practices, Performance & Troubleshooting
=============================================

Flow is designed for **clarity**, **composability**, and **structured JSON pipelines**. But to use it effectively — especially on large or semi-structured data — you need to understand how Flow executes and when data is consumed.

🧠 Think in DAGs, Not DataFrames
================================

Flow builds a **deferred plan** of steps (like a DAG). Nothing runs until you collect results:

.. code-block:: python

   flow = Flow.from_folder("data/").filter(...).assign(...).select(...)

At this point, **no data has been read**.

🚨 When Execution Happens
=========================

Flow starts processing only when you:

- ``.collect()``
- ``.to_pandas()``
- ``.to_json()``, ``.to_jsonl()``
- Iterate over the Flow
- Call ``.first()``, ``.keys()``, ``len()``

⚠️ When Materialization Happens
===============================

Certain operations require the **full dataset** and will materialize in memory:

+---------------+-----------------------------+
| Operation     | Reason                      |
+===============+=============================+
| ``.group_by()`` | Groups must be fully built  |
+---------------+-----------------------------+
| ``.summary()``  | Aggregates need full access |
+---------------+-----------------------------+
| ``.sort_by()``  | Requires sorting all rows   |
+---------------+-----------------------------+
| ``.join()``     | Right side is pre-loaded    |
+---------------+-----------------------------+
| ``.pivot()``    | Reshapes after aggregation  |
+---------------+-----------------------------+
| ``.limit()``    | Buffers to truncate         |
+---------------+-----------------------------+
| ``.cache()``    | Explicit full collection    |
+---------------+-----------------------------+

You can always call ``.explain()`` to see where materialization occurs:

.. code-block:: python

   flow.explain()

🧪 Inspect Safely
==================

You can preview without consuming the full plan:

.. code-block:: python

   Flow.from_jsonl("match.jsonl").head(3)

``.head(n)`` adds a ``.limit()`` and returns the first ``n`` results via ``.collect()``. It's a safe way to preview data.

🔁 Fork Pipelines Naturally
============================

.. code-block:: python

   f = Flow.from_jsonl("match.jsonl")

   attacks = f.filter(lambda r: r["team"] == "Arsenal")
   defence = f.filter(lambda r: r["team"] == "Manchester City")

Because the pipeline is just a plan, each branch is safe and isolated.

🧰 Use ``.pipe()`` for Debugging or Custom Steps
================================================

You can insert custom logic mid-pipeline with ``.pipe()``:

.. code-block:: python

   def peek(flow):
       print(flow.head(3))
       return flow

   Flow.from_jsonl("match.jsonl").pipe(peek).filter(...)

🔄 Pure Functions = Safer Pipelines
===================================

Since ``.map()`` and ``.assign()`` modify records, avoid side effects or mutating shared input.

Prefer using ``.from_records(copy.deepcopy(data))`` if you're passing mutable records from outside.

💡 Performance Tips
===================

- Prefer ``.from_jsonl()`` over ``.from_json()`` for large files
- Minimize ``.sort_by()`` or ``.group_by()`` until late in pipeline
- Use ``.filter()`` early to reduce data as soon as possible
- Avoid flattening too early. Use ``.select()`` to access nested fields instead

🧠 Summary
==========

+---------------------------+------------------------------------------+
| Principle                 | Recommendation                           |
+===========================+==========================================+
| Inspection                | Use ``.head(n)`` to preview              |
+---------------------------+------------------------------------------+
| Debugging                 | Use ``.pipe()`` for custom hooks         |
+---------------------------+------------------------------------------+
| Materialization Awareness | Use ``.explain()`` to understand plan    |
+---------------------------+------------------------------------------+
| Filtering Early           | Always filter before heavy ops           |
+---------------------------+------------------------------------------+

Flow gives you a structured, schema-aware, and composable pipeline for working with JSON, especially valuable when you want to defer flattening and stay close to raw data.
