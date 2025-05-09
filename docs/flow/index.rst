Flow
=================================

**Flow** is a lightweight toolkit for working with football data stored as JSON.

Whether you’re loading StatsBomb match files, streaming multiple JSON logs from a folder or glob, or prototyping new metrics on tagged event streams,
**Flow** gives you lazy, composable pipelines - `filter`, `assign`, `explode`, `group_by`, `summary`, `join`, and more - that only load data into memory when you need it.

For group-based analyses (e.g. per team, per player, per half), **Flow** lets you aggregate, de-duplicate, head / tail your data.

Also, you can seamlessly convert your results to pandas DataFrames or export to JSON/JSONL files for downstream plotting, machine learning, or sharing with teammates.

Streamline your analytics pipelines with **Flow**.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   introduction
   basic_pipeline
   grouping_and_aggregating
   file_io
   advanced
   inspection_interop
   parallel
   best_practices
