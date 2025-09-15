Pi Ratings
==========

.. raw:: html

   <a href="https://colab.research.google.com/drive/12qEDCNYG-FFHOJ_kURe0cm80sScandyh?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
   <br><br>

The Pi rating system is a dynamic football-specific rating method developed by Constantinou & Fenton that addresses the key limitations of traditional systems like Elo when applied to football.

Pi Ratings are built on three key principles:

- score margins matter (a 5-0 win should receive a greater rating boost than a 1-0 win)
- home and away ratings are separate (maintaining distinct ratings for home and away performances)
- recent performance is more important (incorporating a learning rate that ensures recent matches influence ratings more strongly than older results)

Unlike Elo ratings which only consider wins, losses, and draws without regard to margin of victory, Pi Ratings incorporate goal differences to provide a more nuanced assessment of team strength.

Each team begins with a rating of 0, representing the level of an average team, and the system is zero-centered, meaning when one team gains rating points, the other loses the same amount. This makes ratings truly relative and allows for meaningful comparisons across teams, with a rating of +1.0 indicating a team is one goal better than average.

The system has demonstrated superior predictive accuracy compared to Elo in football contexts while remaining computationally efficient and interpretable.

You can read more about the Pi rating system in my blog post: https://pena.lt/y/2025/04/14/pi-ratings-the-smarter-way-to-rank-football-teams/


.. code-block:: python

    import penaltyblog as pb

    pi = pb.ratings.PiRatingSystem()


New Teams Default to a Rating of Zero
-------------------------------------

.. code-block:: python

    pi.get_team_rating("Team A"), pi.get_team_rating("Team B")

.. code-block:: none

    (0.0, 0.0)

Predict Match Results
--------------------

.. code-block:: python

    pi.calculate_match_probabilities("Team A", "Team B")

.. code-block:: text

    {
        'home_win': np.float64(0.3085375387259869),
        'draw': np.float64(0.38292492254802624),
        'away_win': np.float64(0.3085375387259869)
    }

Update Ratings
--------------

.. code-block:: python

    goal_diff = 3  # Team A wins 3-0
    pi.update_ratings("Team A", "Team B", goal_diff)

Get New Ratings
---------------

.. code-block:: python

    pi.get_team_rating("Team A")

.. code-block:: none

    0.11538461538461539

Interactive Example
-------------------

For a comprehensive, hands-on demonstration of the Pi rating system, try the interactive Colab notebook.
The notebook walks you through loading match data, calculating ratings, and visualizing the results.
You can modify the code, experiment with different parameters, and see how the ratings change in real-time.

.. raw:: html

   <a href="https://colab.research.google.com/drive/12qEDCNYG-FFHOJ_kURe0cm80sScandyh?usp=sharing" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
   </a>
   <br>
