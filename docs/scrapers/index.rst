Scrapers
==============

The scrapers provide a consistent wrapper around many online data sources, such as Understat, football-data.co.uk, Club Elo and ESPN.

Each scrapers returns the data as a dataframe with consistent column names, optional mapping of team names to make them consistent
(e.g. Man United Vs Manchester United) and IDs to allow different data sources to be joined together.

You can get a list of available competitions for each data source by calling the scraper's `list_competitions()` function

.. code:: python

   import penaltyblog as pb

   pd.scraper.Understat.list_competitions()


See the examples below for more details on how to use the scrapers.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   espn
   clubelo
   footballdata
   understat
   sofifa
