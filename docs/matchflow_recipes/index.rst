Matchflow Examples
=============================================

This section shows real-world examples of using **MatchFlow** to explore and analyze StatsBomb event data.
Each recipe is a focused, end-to-end workflow: loading data, transforming it with `Flow`, and visualizing
or exporting the results.

You'll find patterns for:

- Filtering and transforming events
- Calculating per-player or per-team summaries
- Visualizing shots, passes, and duels
- Combining multiple matches
- Building datasets for machine learning or reporting

These notebooks are designed to be simple, to give you a quick feel for how MatchFlow works.

.. list-table:: Guide Index
   :widths: 25 75
   :header-rows: 1

   * - Section
     - Description
   * - :doc:`06_statsbomb_api`
     - Streaming data directly from the StatsBomb API
   * - :doc:`01_xg_by_player`
     - Grouping and aggregating stats
   * - :doc:`02_shot_accuracy_by_team`
     - Using custom aggregation functions
   * - :doc:`03_pass_map`
     - Accessing values in arrays
   * - :doc:`04_cumulative_xg_by_time`
     - Calculating grouped cumulative stats
   * - :doc:`05_touchmap`
     - Using a pipe to apply a custom function to the Flow


.. toctree::
   :hidden:

   06_statsbomb_api
   01_xg_by_player
   02_shot_accuracy_by_team
   03_pass_map
   04_cumulative_xg_by_time
   05_touchmap
