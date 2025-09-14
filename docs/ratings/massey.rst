==============
Massey Ratings
==============

.. raw:: html

   <a href="https://colab.research.google.com/drive/1d_WPJwQgrogeSI9oIO9fY8s18CPPZ8nL?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
   <br><br>

Overview
--------

The Massey rating system is a linear algebraic approach to team ranking developed by Kenneth Massey that gained prominence in college sports, particularly as part of the NCAA's Bowl Championship Series (BCS) computer rankings.

Unlike systems that rely solely on win-loss records, the Massey method incorporates point differentials to create a more nuanced assessment of team strength.

The system works by setting up a system of linear equations where each team's rating is determined by their average point differential against opponents, weighted by the strength of those opponents.

This creates a mathematically elegant solution that rewards teams not just for winning, but for winning convincingly against strong competition. The Massey ratings are particularly valued for their ability to handle strength-of-schedule adjustments automatically — beating a strong team by a small margin can be rated higher than blowing out a weak team, making it especially useful for ranking teams across different conferences or leagues with varying competitive levels.

Offensive and Defensive Ratings
------------------------------

A distinctive feature of the Massey rating system is its ability to decompose overall team strength into separate offensive and defensive components.

The offensive rating represents a team's ability to score points above average, while the defensive rating reflects their ability to prevent opponents from scoring. This decomposition is mathematically derived from the same linear system that produces the overall ratings, making it a natural byproduct rather than an ad-hoc addition.

These component ratings provide valuable insights into team composition, for example, identifying whether a team's success comes from a dominant offense, stifling defense, or balanced strength on both sides of the ball. This granular analysis is particularly useful for coaches, analysts, and bettors who want to understand stylistic matchups and predict how different team strengths and weaknesses might interact in head-to-head competition.


Basic Usage
===========

.. code-block:: python

   import penaltyblog as pb

   # Load your match data
   fbd = pb.scrapers.FootballData("ENG Premier League", "2021-2022")
   df = fbd.get_fixtures()

Calculate Ratings
=================

.. code-block:: python

    massey = pb.ratings.Massey(
        df["goals_home"],
        df["goals_away"],
        df["team_home"],
        df["team_away"]
    )

   ratings = massey.get_ratings()

   ratings.head(10)
   print(ratings.head(10))

Example output:

.. list-table:: Massey Ratings Example
    :header-rows: 1

    * - team
      - rating
      - offence
      - defence
    * - Liverpool
      - 1.125
      - 1.51133
      - -0.38633
    * - Arsenal
      - 0.875
      - 1.052997
      - -0.177997
    * - Man City
      - 0.7
      - 1.146053
      - -0.446053
    * - Chelsea
      - 0.525
      - 0.933553
      - -0.408553
    * - Newcastle
      - 0.525
      - 1.044664
      - -0.519664
    * - Bournemouth
      - 0.3
      - 0.779386
      - -0.479386
    * - Nott'm Forest
      - 0.3
      - 0.779386
      - -0.479386
    * - Brentford
      - 0.225
      - 1.005775
      - -0.780775
    * - Aston Villa
      - 0.175
      - 0.78633
      - -0.61133
    * - Brighton
      - 0.175
      - 1.008553
      - -0.833553

Offensive and Defensive Ratings
=============================================

.. code-block:: python

    print("Offense Ratings:")
    display(ratings[["team", "offence"]].head(10))

    print("\nDefence Ratings:")
    display(ratings[["team", "defence"]].head(10))

.. list-table:: Massey Ratings Example - Offense
    :header-rows: 1

    * - team
      - offence
    * - Liverpool
      - 1.51133
    * - Arsenal
      - 1.052997
    * - Man City
      - 1.146053
    * - Chelsea
      - 0.933553
    * - Newcastle
      - 1.044664
    * - Bournemouth
      - 0.779386
    * - Nott'm Forest
      - 0.779386
    * - Brentford
      - 1.005775
    * - Aston Villa
      - 0.78633
    * - Brighton
      - 1.008553


.. list-table:: Massey Ratings Example - Defensive
    :header-rows: 1

    * - team
      - defence
    * - Liverpool
      - -0.38633
    * - Arsenal
      - -0.177997
    * - Man City
      - -0.446053
    * - Chelsea
      - -0.408553
    * - Newcastle
      - -0.519664
    * - Bournemouth
      - -0.479386
    * - Nott'm Forest
      - -0.479386
    * - Brentford
      - -0.780775
    * - Aston Villa
      - -0.61133
    * - Brighton
      - -0.833553

Key Features
================

- **Comprehensive**: Incorporates both wins/losses and point differentials for richer analysis
- **Decomposable**: Automatically generates separate offensive and defensive ratings alongside overall team strength
- **Strength-of-schedule aware**: Ratings adjust based on opponent quality, rewarding performance against strong competition
- **Mathematically rigorous**: Based on linear algebra with unique, stable solutions
- **Predictive**: Component ratings can forecast expected point spreads for future matchups
- **Margin-sensitive**: Distinguishes between narrow victories and blowouts, providing more nuanced rankings
- **Scalable**: Handles leagues of any size with automatic strength-of-schedule adjustments

Interactive Example
===================

For a comprehensive, hands-on demonstration of the Massey rating system, try the interactive Colab notebook.
The notebook walks you through loading match data, calculating ratings, and visualizing the results.
You can modify the code, experiment with different parameters, and see how the ratings change in real-time.

.. raw:: html

   <a href="https://colab.research.google.com/drive/1d_WPJwQgrogeSI9oIO9fY8s18CPPZ8nL?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
