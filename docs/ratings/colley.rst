==============
Colley Ratings
==============

The Colley rating system is a sophisticated mathematical approach to team ranking that was originally developed by Wesley Colley for college football.

The method uses linear algebra to solve a system of equations based on team performance, creating ratings that reflect relative strength while maintaining mathematical rigor and objectivity.
What sets Colley apart from other rating systems is its focus purely on wins, losses, and draws, completely ignoring score margins, which eliminates potential bias from "running up the score" or other strategic considerations.
The system works by constructing a matrix equation where each team's rating is determined by their win-loss record relative to their opponents' strength.

This creates a self-reinforcing system where beating strong teams boosts your rating more than beating weak teams, while the mathematical framework ensures the ratings converge to a stable, unique solution.
The elegance of the Colley method lies in its ability to produce meaningful rankings with relatively simple inputs, making it both computationally efficient and resistant to manipulation.

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
