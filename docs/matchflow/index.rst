MatchFlow
=================================

**MatchFlow** is a lightweight toolkit for working with football data stored as JSON.

Whether you’re loading StatsBomb match files, streaming multiple JSON logs from a folder or glob, or prototyping new metrics on tagged event streams,
**MatchFlow** gives you lazy, composable pipelines - `filter`, `assign`, `explode`, `group_by`, `summary`, `join`, and more - that only load data into memory when you need it.

For group-based analyses (e.g. per team, per player, per half), **MatchFlow** lets you aggregate, de-duplicate, head / tail your data.

Also, you can seamlessly convert your results to pandas DataFrames or export to JSON/JSONL files for downstream plotting, machine learning, or sharing with teammates.

Streamline your analytics pipelines with **MatchFlow**.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   introduction
   basic_pipeline
   grouping_and_aggregating
   advanced
   file_io
   inspection_interop
   parallel
   best_practices
