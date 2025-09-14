==============
Colley Ratings
==============

.. raw:: html

   <a href="https://colab.research.google.com/drive/1gjQASy0Ge_I_qdsJaBDfGTiwRzXqCISW?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
   <br><br>

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

            .. list-table:: Colley Ratings Example
               :header-rows: 1

               * - team
                 - rating
               * - Liverpool
                 - 1.904762
               * - Man City
                 - 1.892857
               * - Chelsea
                 - 1.791667
               * - Tottenham
                 - 1.708333
               * - Arsenal
                 - 1.672619
               * - Man United
                 - 1.654762
               * - Brighton
                 - 1.64881
               * - Crystal Palace
                 - 1.625
               * - West Ham
                 - 1.619048
               * - Leicester
                 - 1.607143

Colley Ratings Excluding Tied Scorelines
=========================================

You can exclude draws from the calculation by setting ``include_draws=False``.
This focuses purely on decisive results:

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

.. list-table:: Colley Ratings Example (Excluding Draws)
   :header-rows: 1

   * - team
     - rating
   * - Liverpool
     - 0.809524
   * - Man City
     - 0.809524
   * - Chelsea
     - 0.678571
   * - Tottenham
     - 0.630952
   * - Arsenal
     - 0.607143
   * - Man United
     - 0.547619
   * - West Ham
     - 0.52381
   * - Brighton
     - 0.511905
   * - Leicester
     - 0.5
   * - Crystal Palace
     - 0.488095

Key Features
============

- **Objective**: Uses only match results, not subjective assessments
- **Stable**: Mathematical foundation prevents extreme fluctuations
- **Flexible**: Can include or exclude drawn matches
- **Fast**: Efficient computation suitable for regular updates
- **Unbiased**: No home field advantage or margin of victory weighting

Interactive Example
===================

For a comprehensive, hands-on demonstration of the Colley rating system, try the interactive Colab notebook.
The notebook walks you through loading match data, calculating ratings, and visualizing the results.
You can modify the code, experiment with different parameters, and see how the ratings change in real-time.

.. raw:: html

   <a href="https://colab.research.google.com/drive/1gjQASy0Ge_I_qdsJaBDfGTiwRzXqCISW?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
