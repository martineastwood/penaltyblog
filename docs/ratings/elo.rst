Elo Ratings
===========

.. code-block:: python

    import penaltyblog as pb

.. code-block:: python

    elo = pb.ratings.Elo()

New Teams Default to 1500 Elo
-----------------------------

.. code-block:: python

    elo.get_team_rating("Team A"), elo.get_team_rating("Team B")

.. code-block:: none

    (1500.0, 1500.0)

Predict Match Results
--------------------

.. code-block:: python

    elo.calculate_match_probabilities("Team A", "Team B")

.. code-block:: none

    {'home_win': np.float64(0.5060806246811322),
     'draw': np.float64(0.20932932618252026),
     'away_win': np.float64(0.28459004913634756)}

Update Ratings
--------------

.. code-block:: python

    elo.update_ratings("Team A", "Team B", 0)

Get New Ratings
---------------

.. code-block:: python

    elo.get_team_rating("Team A")

.. code-block:: none

    1507.1987000039423
