MatchFlow
==========

**MatchFlow** is a lightweight toolkit for working with structured football data, especially nested JSON like StatsBomb event files or match-level logs. Whether you're building quick explorations or full pipelines, MatchFlow helps you work directly with deeply structured data using a clean, lazy, and chainable API.

What is MatchFlow?
------------------

Flow is not a DataFrame, it's a **stream-first query engine** built for irregular, event-based football data.

You can:

- Load JSON, JSONL, or entire folders of match data
- Filter and transform records lazily with ``.filter()``, ``.assign()``, ``.select()``
- Group and summarize using ``.group_by()`` + ``.summary()``
- Join datasets, explode lists, split arrays, pivot rows
- Work with nested data without flattening too early
- Chain steps fluently, materialize only when ready

All transformations are **lazy**; nothing runs until you ask for results with ``.collect()``, ``.to_pandas()``, ``.to_jsonl()`` etc.

Guide Index
-----------

.. list-table:: Guide Index
   :widths: 25 75
   :header-rows: 1

   * - Section
     - Description
   * - :doc:`why`
     - Why working with nested football data needs a new tool
   * - :doc:`introduction`
     - Introduction to MatchFlow
   * - :doc:`basic_pipeline`
     - Filtering, assigning, selecting, and shaping your data
   * - :doc:`grouping_and_aggregating`
     - Summarizing by team, player, period, and more
   * - :doc:`advanced`
     - Sorting, ranking, joining, pivoting, deduplicating
   * - :doc:`file_io`
     - Working with JSON, JSONL, folders, glob patterns
   * - :doc:`inspection`
     - Exploring structure, peeking at records, debugging
   * - :doc:`best_practices`
     - Materialization, memory, performance, clean code
   * - :doc:`predicates`
     - Reusable filters like ``where_equals()``, ``and_()``
   * - :doc:`statsbomb`
     - Streaming data directly from the StatsBomb API

Example
-------

.. code-block:: python

   from penaltyblog.matchflow import Flow, where_equals

   # Load and filter StatsBomb shots
   flow = (
      Flow.statsbomb.events(match_id=19716)
      .filter(where_equals("type.name", "Shot"))
      .select("player.name", "location", "shot.statsbomb_xg")
   )

   for shot in flow.head(5):
      print(shot)

Notes
-----

.. note::

    This is the first public release of ``penaltyblog.matchflow``.
    It’s already powering real-world workflows, but edge cases may still surface.

    **If you find bugs, performance issues, or have feature ideas then please open an issue or reach out.**
    Your feedback makes this better for everyone.

Ready to Flow?
--------------

Pick a section from the guide above, or jump in with ``.from_jsonl()``, ``.from_folder()``, or ``.statsbomb.events()``  and start building your pipeline.

Need help? Ask questions, file issues, or suggest improvements any time.

.. toctree::
   :hidden:

   why
   introduction
   basic_pipeline
   grouping_and_aggregating
   advanced
   file_io
   inspection
   best_practices
   predicates
   statsbomb
