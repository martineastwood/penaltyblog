==================
Query Optimization
==================

MatchFlow includes a **built-in query optimizer** that transparently rewrites your pipeline to improve performance while preserving semantics.

In general:

- ✅ safer pipelines
- ✅ faster execution
- ✅ smaller intermediate data
- ✅ better scalability

🚀 When Optimization Happens
============================

By default, flows are unoptimized:

.. code-block:: python

   flow = Flow.from_jsonl("match_events.jsonl")

To enable optimization, pass ``optimize=True``:

.. code-block:: python

   flow = Flow.from_jsonl("match_events.jsonl", optimize=True)

Or explicitly at collect-time:

.. code-block:: python

   flow.collect(optimize=True)

Any visualization (``explain()``, ``plot_plan()``, etc.) can also show optimized plans.

🧠 What The Optimizer Does
==========================

The optimizer currently performs **conservative rule-based** rewrites, including:

+--------------------------------+-----------------------------------------------------------------------------+
| Optimization                   | Description                                                                 |
+================================+=============================================================================+
| **Filter Pushdown**            | Moves ``filter()`` earlier to reduce data earlier                          |
+--------------------------------+-----------------------------------------------------------------------------+
| **Limit Pushdown**             | Moves ``limit()`` closer to source                                         |
+--------------------------------+-----------------------------------------------------------------------------+
| **Select/Drop Pushdown**       | Drops unused fields as early as safely possible                            |
+--------------------------------+-----------------------------------------------------------------------------+
| **Map/Assign Fusion**          | Merges consecutive ``map()``, ``assign()``, ``filter()`` into a single fused step |
+--------------------------------+-----------------------------------------------------------------------------+
| **Redundant Step Elimination** | Removes unnecessary repeated ``drop()``, ``dropna()``                      |
+--------------------------------+-----------------------------------------------------------------------------+
| **Rolling Validation**         | Warns if rolling summaries lack prior ``sort()`` step                      |
+--------------------------------+-----------------------------------------------------------------------------+

🧐 Example
==========

Consider the following flow:

.. code-block:: python

   flow = (
       Flow.from_jsonl("match_events.jsonl")
       .assign(team_name=lambda r: r["team"]["name"])
       .filter(lambda r: r["type"]["name"] == "Pass")
       .select("minute", "second", "team_name")
       .limit(100)
   )

Without optimization:

.. code-block:: bash

   from_jsonl → assign → filter → select → limit

With optimization:

.. code-block:: bash

   from_jsonl → filter → assign → select → limit

- The ``filter()`` is pushed earlier.
- The ``assign()`` and ``select()`` are reordered.
- The ``limit()`` is moved earlier.
- Fewer rows flow through the pipeline.

🔍 Explain Your Plans
=====================

You can always inspect both raw and optimized plans:

.. code-block:: python

   flow.explain(compare=True)

Or visualize them:

.. code-block:: python

   flow.plot_plan(compare=True)

🚫 What The Optimizer Does Not Do (yet...)
==========================================

- Complex join reordering
- Predicate simplification
- Cost-based optimization
- Group-by optimizations

The optimizer is designed to be **safe-by-default**: it will only reorder steps when correctness can be statically guaranteed.

⚙ Optimizer Safety Model
========================

MatchFlow applies **conservative optimizations** to preserve correctness when working with arbitrary user-defined functions:

✅ Safe to optimize:
  Operations like ``select()``, ``drop()``, ``limit()``, ``filter()`` (when independent), ``sort()``, ``group_by()``, and other structural plan steps.

🚫 Not assumed safe to reorder:
  - ``map()``
  - ``assign()``
  - any user-defined ``filter()`` with non-trivial predicates
  - any lambdas or custom functions

🔒 Why conservative?
  Unlike SQL engines, MatchFlow makes no assumptions about:

  - Commutativity: e.g. ``map()`` and ``filter()`` may not commute.
  - Determinism: user functions may depend on external state, random values, timestamps, etc.
  - Purity: functions may have side-effects or depend on execution order.

⚠ Fusion:
  - Consecutive ``map()`` / ``assign()`` / ``filter()`` steps may be fused together at plan build time (syntactic fusion).
  - Fusion never involves reordering; it only combines adjacent steps for efficiency.

🔬 Invariant:
  The execution semantics of any user-specified Flow remain the same under optimization, unless steps were fused at creation time.

Summary
=======

+---------------------------+---------------------------------------+
| You Write                 | Optimizer Makes Fast                  |
+===========================+=======================================+
| **Declarative pipelines** | Minimal and efficient execution plans |
+---------------------------+---------------------------------------+
| **Readable code**         | Faster runtime                        |
+---------------------------+---------------------------------------+
| **Safe transformations**  | Transparent optimization              |
+---------------------------+---------------------------------------+
