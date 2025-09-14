==============
Colley Ratings
==============

Colley Ratings are a mathematically sound, unbiased method for ranking football teams, relying exclusively on match outcomes rather than scores or margins of victory.

Unlike some rating systems, the Colley method is specifically designed to be objective, using only wins, losses, and draws without subjective weighting.

Its mathematical simplicity and stability make it highly suitable for quickly evaluating team performance, objectively comparing team strengths, and identifying discrepancies between actual team quality and public perception or bookmaker odds.

Basic Usage
===========

.. code-block:: python

   import penaltyblog as pb

   # Load your match data
   fbd = pb.scrapers.FootballData("ENG Premier League", "2021-2022")
   df = fbd.get_fixtures()

Colley Ratings Including Tied Scorelines
=========================================

By default, the Colley method treats draws as partial wins for both teams (0.5 each):

.. code-block:: python

   colley = pb.ratings.Colley(
       df["goals_home"],
       df["goals_away"],
       df["team_home"],
       df["team_away"]
   )
   ratings = colley.get_ratings()
   print(ratings.head(10))

Example output:

.. code-block:: text

              team    rating
   0     Liverpool  1.904762
   1      Man City  1.892857
   2       Chelsea  1.791667
   3     Tottenham  1.708333
   4       Arsenal  1.672619
   5    Man United  1.654762
   6      Brighton   1.64881
   7  Crystal Palace     1.625
   8      West Ham  1.619048
   9     Leicester  1.607143

Colley Ratings Excluding Tied Scorelines
=========================================

You can exclude draws from the calculation by setting ``include_draws=False``. This focuses purely on decisive results:

.. code-block:: python

   colley = pb.ratings.Colley(
       df["goals_home"],
       df["goals_away"],
       df["team_home"],
       df["team_away"],
       include_draws=False,
   )
   ratings = colley.get_ratings()
   print(ratings.head(10))

Example output:

.. code-block:: text

              team    rating
   0     Liverpool  0.809524
   1      Man City  0.809524
   2       Chelsea  0.678571
   3     Tottenham  0.630952
   4       Arsenal  0.607143
   5    Man United  0.547619
   6      West Ham   0.52381
   7      Brighton  0.511905
   8     Leicester       0.5
   9  Crystal Palace  0.488095

Key Features
============

- **Objective**: Uses only match results, not subjective assessments
- **Stable**: Mathematical foundation prevents extreme fluctuations
- **Flexible**: Can include or exclude drawn matches
- **Fast**: Efficient computation suitable for regular updates
- **Unbiased**: No home field advantage or margin of victory weighting
