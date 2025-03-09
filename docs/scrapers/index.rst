Scrapers
==============

The scrapers provide a consistent interface for extracting football data from multiple online sources, including FBRef, Understat, football-data.co.uk, and Club Elo.

Each scraper returns data in a standardized DataFrame format, ensuring uniform column names and structures across all sources.

Additionally, they support optional team name normalization (e.g., converting "Man United" to "Manchester United"), making it easy to merge datasets from different providers.

This consistency allows seamless integration and analysis, enabling users to combine data effortlessly without worrying about formatting discrepancies.

You can get a list of available competitions for each data source by calling the scraper's `list_competitions()` function.

.. code:: python

   import penaltyblog as pb

   pb.scraper.Understat.list_competitions()


See the examples below for more details on how to use the individual scrapers.


.. toctree::
   :maxdepth: 1
   :caption: Examples:

   fbref
   clubelo
   footballdata
   understat
