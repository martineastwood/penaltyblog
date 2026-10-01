Football Charts
===============

The Football Charts scraper loads completed match results from
`football-charts.com <https://www.football-charts.com/>`_. It includes
full-time and half-time scores, first-goal timing, and leagues that are not
covered by some of the other data providers.

The API provides the current and previous season without a key. Older seasons
may require a paid key. An API key can be passed explicitly or supplied using
the ``FC_API_KEY`` environment variable.

List Football Charts leagues
----------------------------

.. code-block:: python

   import penaltyblog as pb

   leagues = pb.scrapers.FootballCharts.list_leagues()
   leagues.head()

Use a canonical penaltyblog competition
----------------------------------------

Where a Football Charts league has a canonical penaltyblog mapping, use the
same competition and season interface as the other scrapers:

.. code-block:: python

   scraper = pb.scrapers.FootballCharts(
       "ENG Premier League",
       "2025-2026",
   )
   fixtures = scraper.get_fixtures()

Use an FC-native league code
----------------------------

For lower divisions, women's leagues, or other competitions without a
canonical mapping, use the league code returned by ``list_leagues()``:

.. code-block:: python

   scraper = pb.scrapers.FootballCharts.from_league(
       "germany3",
       "2025-2026",
   )
   fixtures = scraper.get_fixtures()

The returned dataframe follows the standard scraper fixture columns and also
includes ``first_goal_minute`` and ``footballcharts_id``. Football Charts
results do not include bookmaker odds. Attribution is available through
``fixtures.attrs["attribution"]``.
