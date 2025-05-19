MatchFlow
=================================

**MatchFlow** is a lightweight toolkit for working with structured football data, especially JSON files like
StatsBomb events or match-level logs.

Whether you're loading a single file, streaming a folder of matches, or building a custom metric from event-level data,
**MatchFlow** gives you a powerful yet simple toolkit:
chainable operations like ``.filter()``, ``.assign()``, ``.explode()``, ``.group_by()``, ``.summary()``, and ``.join()`` - all built
to work lazily and efficiently on your data.

You can group by player, team, or match period; aggregate or de-duplicate records; and inspect your data at any step.
When you're ready, materialize your results with ``.collect()`` or ``.to_pandas()``, or export directly to JSON/JSONL.

Whether you're exploring a single match or scaling up analysis across thousands, **MatchFlow** helps you build clean,
composable pipelines - fast.

.. note::
   This is the first public release of ``penaltyblog.matchflow``. It's already powering a variety of real-world workflows,
   but edge cases may still surface. If you spot anything surprising or have suggestions, your feedback is very welcome -
   it helps improve the toolkit for everyone.


.. toctree::
   :maxdepth: 1
   :caption: Examples:

   why
   introduction
   basic_pipeline
   grouping_and_aggregating
   advanced
   file_io
   inspection_interop
   best_practices
   helpers
   statsbomb
