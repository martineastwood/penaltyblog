.. _api_docs:

API Documentation
=================

``Poisson Goal Model``
--------------------------

.. autoclass:: penaltyblog.models.PoissonGoalsModel
    :members:
    :private-members:
    :member-order: bysource

``Dixon and Coles Goal Model``
-------------------------------

.. autoclass:: penaltyblog.models.DixonColesGoalModel
    :members:
    :private-members:
    :member-order: bysource


``Dixon and Coles Goal Model With Rue and Salvesen Adjustment``
----------------------------------------------------------------

.. autoclass:: penaltyblog.models.RueSalvesenGoalModel
    :members:
    :private-members:
    :member-order: bysource


``Calculate Implied Betting Odds``
----------------------------------

.. automodule:: penaltyblog.implied
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: multiplicative

|

.. autofunction:: additive

|

.. autofunction:: power

|

.. autofunction:: shin

|

.. autofunction:: differential_margin_weighting

|

.. autofunction:: odds_ratio

``Get data from footballdata.co.uk``
---------------------------------------

.. automodule:: penaltyblog.scrapers.footballdata.FootballData
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: list_competitions

|

.. autofunction:: get_fixtures


``Get data from clubelo.com``
-------------------------------

.. automodule:: penaltyblog.scrapers.clubelo.ClubElo
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: get_elo_by_date

|

.. autofunction:: get_team_names

|

.. autofunction:: get_elo_by_team


``Get data from fantasy.premierleague.com``
--------------------------------------------------------------

.. automodule:: penaltyblog.fpl
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: get_current_gameweek

|

.. autofunction:: get_gameweek_info

|

.. autofunction:: get_player_id_mappings

|

.. autofunction:: get_player_data

|

.. autofunction:: get_player_history

|

.. autofunction:: get_rankings

|

.. autofunction:: get_entry_picks_by_gameweek

|

.. autofunction:: get_entry_transfers



``Calculate Massey Ratings``
------------------------------

.. automodule:: penaltyblog.ratings.massey.Massey
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: get_ratings


``Calculate Colley Ratings``
------------------------------

.. automodule:: penaltyblog.ratings.colley.Colley
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: get_ratings


``Useful Metrics``
-------------------

.. automodule:: penaltyblog.utilities
    :members:
    :special-members:
    :exclude-members: __weakref__
    :member-order: bysource

.. autofunction:: rps
