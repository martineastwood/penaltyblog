Pi Ratings
==========

.. code-block:: python

    import penaltyblog as pb

.. code-block:: python

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

.. code-block:: none

    {'home_win': np.float64(0.3085375387259869),
     'draw': np.float64(0.38292492254802624),
     'away_win': np.float64(0.3085375387259869)}

Update Ratings
--------------

.. code-block:: python

    pi.update_ratings("Team A", "Team B", 3)

Get New Ratings
---------------

.. code-block:: python

    pi.get_team_rating("Team A")

.. code-block:: none

    0.11538461538461539
