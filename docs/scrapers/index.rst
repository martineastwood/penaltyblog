Scrapers
==============

The scrapers provide a consistent wrapper around many online data sources, such as football-data.co.uk, Club Elo and ESPN.

Each scrapers returns the data as a dataframe with consistent column names, mapping of team names to make them consistent
(e.g. Man United Vs Manchester United) and IDs to allow different data sources to be joined together.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   espn
   clubelo
   footballdata
